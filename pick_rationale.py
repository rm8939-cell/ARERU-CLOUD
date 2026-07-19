"""本命・対抗・穴馬の判断根拠（信頼度・理由・コメント）を組み立てる。"""
from __future__ import annotations

import math
import re
from typing import Any


def _clamp(x: float, a: float = 0.0, b: float = 100.0) -> float:
    return float(max(a, min(b, x)))


def stars_for_score(score: float | int) -> str:
    filled = max(1, min(5, (int(round(float(score))) + 19) // 20))
    return "★" * filled + "☆" * (5 - filled)


def _mark(ok: bool | None, mid: bool = False) -> str:
    if ok is True:
        return "○"
    if mid:
        return "△"
    return "－"


def _fmt_pct(v) -> str:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"{float(v):.1f}%"
    except (TypeError, ValueError):
        return "—"


def _aptitude_text(mark: str, good: str, mid: str = "まずまず", weak: str = "不明") -> str:
    if mark == "○":
        return good
    if mark == "△":
        return mid
    return weak


def _split_materials(text: str) -> list[str]:
    s = str(text or "").strip()
    if not s or s in ("特記なし", "総合評価", "—", "なし", "nan"):
        return []
    parts = [p.strip() for p in re.split(r"[/／\|｜、,]", s) if p.strip()]
    return list(dict.fromkeys(parts))


def calc_horse_confidence(
    *,
    role: str,
    n_sample: int,
    win_pct: float | None,
    place_pct: float | None,
    idx_rank: int | None,
    field_n: int,
    dist_mark: str,
    course_mark: str,
    surface_mark: str,
    lap_fit: float | None,
    last3f: float | None,
    pace_fit: float | None,
    jockey_score: float | None,
    trouble: float | None,
    nar_scale: bool,
    plus_n: int,
    minus_n: int,
) -> float:
    """馬単位のAI信頼度 0〜100。"""
    score = 28.0
    score += min(18.0, max(0, n_sample) * 3.6)
    if win_pct is not None:
        if 8 <= win_pct <= 42:
            score += 12
        elif win_pct > 55:
            score -= 10
        elif win_pct < 5:
            score -= 6
    if place_pct is not None and win_pct is not None and place_pct >= win_pct * 1.6:
        score += 6
    if idx_rank is not None and field_n > 0:
        if idx_rank <= 2:
            score += 8
        elif idx_rank <= max(3, field_n // 4):
            score += 4
        elif idx_rank >= field_n - 1:
            score -= 4
    for m in (dist_mark, course_mark, surface_mark):
        if m == "○":
            score += 4
        elif m == "△":
            score += 1
    if lap_fit is not None:
        score += (lap_fit - 50) * 0.12
    if last3f is not None:
        score += (last3f - 50) * 0.10
    if pace_fit is not None:
        score += (pace_fit - 50) * 0.10
    if jockey_score is not None:
        score += max(-4.0, min(6.0, jockey_score))
    if trouble is not None and trouble >= 0.35:
        score += 2
    if nar_scale:
        score -= 8
    score += min(6, plus_n) * 1.2
    score -= min(5, minus_n) * 1.5
    if role == "本命":
        score += 2
    elif role == "穴馬":
        score -= 2
    return round(_clamp(score, 8, 95), 1)


def build_reason_rows(
    *,
    idx_rank: int | None,
    field_n: int,
    dist_mark: str,
    course_mark: str,
    surface_mark: str,
    surface: str,
    lap_label: str,
    lap_fit: float | None,
    last3f: float | None,
    last3_rank: int | None,
    pace_label: str,
    pace_fit: float | None,
    style: str,
    jockey_score: float | None,
    blood_score: float | None,
    trouble: float | None,
    nar_note: str,
    win_pct: float | None,
    quinella_pct: float | None,
    place_pct: float | None,
) -> list[dict[str, str]]:
    """【本命にした理由】行リスト {項目, 評価, 説明}。"""
    rows: list[dict[str, str]] = []

    if idx_rank is not None:
        tip = (
            f"近走指数は出走{field_n}頭中{idx_rank}位"
            + ("で上位評価" if idx_rank <= 3 else ("で中位" if idx_rank <= field_n // 2 else "でやや下位"))
        )
        rows.append({"項目": "近走指数順位", "評価": f"{idx_rank}位/{field_n}", "説明": tip})
    else:
        rows.append({"項目": "近走指数順位", "評価": "—", "説明": "指数順位の算出待ち"})

    rows.append({
        "項目": "距離適性",
        "評価": dist_mark,
        "説明": _aptitude_text(dist_mark, "同距離帯で好走実績あり", "距離実績は限定的", "距離適性データ不足"),
    })
    rows.append({
        "項目": "コース適性",
        "評価": course_mark,
        "説明": _aptitude_text(course_mark, "同コース実績がプラス", "コース実績は限定的", "コース適性データ不足"),
    })
    surf_name = surface or "馬場"
    rows.append({
        "項目": "馬場適性",
        "評価": surface_mark,
        "説明": _aptitude_text(
            surface_mark,
            f"{surf_name}での好走傾向",
            f"{surf_name}適性はまずまず",
            f"{surf_name}適性はデータ不足",
        ),
    })

    lap_desc = lap_label or "平均ペース適性"
    if lap_fit is not None and lap_fit >= 58:
        lap_desc += "（想定ペースと好相性）"
    elif lap_fit is not None and lap_fit <= 42:
        lap_desc += "（想定ペースと噛み合いにくい）"
    rows.append({"項目": "ラップ適性", "評価": lap_label or "—", "説明": lap_desc})

    if last3f is not None:
        rank_txt = f"（上がり順位{last3_rank}位）" if last3_rank else ""
        if last3f >= 62:
            tip = f"末脚評価{last3f:.0f}点と高く{rank_txt}"
        elif last3f <= 40:
            tip = f"末脚評価{last3f:.0f}点で物足りない{rank_txt}"
        else:
            tip = f"末脚評価{last3f:.0f}点で平均帯{rank_txt}"
        rows.append({
            "項目": "上がり順位",
            "評価": f"{last3_rank}位" if last3_rank else f"{last3f:.0f}点",
            "説明": tip.strip(),
        })
    else:
        rows.append({"項目": "上がり順位", "評価": "—", "説明": "上がり評価データ不足"})

    pace = pace_label or "平均"
    if pace_fit is not None and pace_fit >= 58:
        tip = f"想定{pace}ペースに対し{style or '脚質'}が噛み合う"
        mark = "好相性"
    elif pace_fit is not None and pace_fit <= 42:
        tip = f"想定{pace}ペースでは{style or '脚質'}がやや不利"
        mark = "やや不利"
    else:
        tip = f"想定{pace}ペースに対し標準的な相性"
        mark = "標準"
    rows.append({"項目": "展開との相性", "評価": mark, "説明": tip})

    if jockey_score is not None:
        if jockey_score >= 2.5:
            tip, mark = "この開催での騎手成績が上振れ", "○"
        elif jockey_score <= -2:
            tip, mark = "騎手成績がやや下振れ", "△"
        else:
            tip, mark = "騎手補正は中立", "－"
        rows.append({"項目": "騎手相性", "評価": mark, "説明": tip})
    else:
        rows.append({"項目": "騎手相性", "評価": "—", "説明": "騎手成績データ不足"})

    if blood_score is not None and abs(blood_score - 50) >= 4:
        mark = "○" if blood_score >= 55 else "△"
        tip = "血統傾向が条件と合致" if blood_score >= 55 else "血統面は条件と微妙"
    else:
        mark, tip = "－", "血統データは未取得のため中立評価"
    rows.append({"項目": "血統適性", "評価": mark, "説明": tip})

    if trouble is not None and trouble >= 0.35:
        rows.append({
            "項目": "前走不利補正",
            "評価": "補正+",
            "説明": "前走に不利の気配があり、巻き返し余地を加点",
        })
    else:
        rows.append({
            "項目": "前走不利補正",
            "評価": "なし",
            "説明": "前走不利としての特別補正なし",
        })

    if nar_note:
        rows.append({"項目": "地方→中央補正", "評価": "補正済", "説明": nar_note})
    else:
        rows.append({
            "項目": "地方→中央補正",
            "評価": "対象外",
            "説明": "中央実績中心で地方補正なし",
        })

    rows.append({
        "項目": "シミュレーション勝率",
        "評価": _fmt_pct(win_pct),
        "説明": "段階シミュレーションによる単勝圏の到達確率",
    })
    rows.append({
        "項目": "連対率",
        "評価": _fmt_pct(quinella_pct),
        "説明": "2着以内に入る仮想確率",
    })
    rows.append({
        "項目": "複勝率",
        "評価": _fmt_pct(place_pct),
        "説明": "3着以内に入る仮想確率",
    })
    return rows


def build_plus_minus(
    plus: list[str],
    minus: list[str],
    *,
    role: str,
    win_pct: float | None,
    place_pct: float | None,
    idx_rank: int | None,
    dist_mark: str,
    course_mark: str,
    surface_mark: str,
    last3f: float | None,
    pace_fit: float | None,
    market_value: bool,
) -> tuple[list[str], list[str]]:
    plus_out = list(dict.fromkeys([p for p in plus if p]))
    minus_out = list(dict.fromkeys([m for m in minus if m]))

    def _add_plus(msg: str):
        if msg not in plus_out:
            plus_out.append(msg)

    def _add_minus(msg: str):
        if msg not in minus_out:
            minus_out.append(msg)

    if idx_rank is not None and idx_rank <= 2:
        _add_plus(f"近走指数{idx_rank}位の上位評価")
    if dist_mark == "○":
        _add_plus("距離適性が合う")
    if course_mark == "○":
        _add_plus("コース適性が合う")
    if surface_mark == "○":
        _add_plus("馬場適性が合う")
    if last3f is not None and last3f >= 62:
        _add_plus("上がり性能が高い")
    if pace_fit is not None and pace_fit >= 58:
        _add_plus("想定展開との相性が良い")
    if place_pct is not None and place_pct >= 40:
        _add_plus("複勝圏の厚みがある")
    if market_value:
        _add_plus("現在オッズがAI適正より割安")
    if role == "穴馬" and win_pct is not None and win_pct >= 12:
        _add_plus("人気薄でもシミュレーション上位")

    if idx_rank is not None and idx_rank >= 8:
        _add_minus("近走指数がやや見劣り")
    if last3f is not None and last3f <= 40:
        _add_minus("上がりが物足りない")
    if pace_fit is not None and pace_fit <= 42:
        _add_minus("想定展開と噛み合いにくい")
    if win_pct is not None and win_pct < 8 and role == "本命":
        _add_minus("勝率自体は高くない")

    plus_out = plus_out[:5]
    minus_out = minus_out[:3]
    while len(plus_out) < 3:
        filler = {
            0: "総合指数とシミュレーションのバランスが良い",
            1: "条件面の大きな欠点が見当たらない",
            2: "相手関係を踏まえても残せる内容",
        }[len(plus_out)]
        if filler not in plus_out:
            plus_out.append(filler)
        else:
            break
    if not minus_out:
        minus_out = ["特記すべき大きな不安は少ない"]
    return plus_out[:5], minus_out[:3]


def build_ai_comment(
    *,
    role: str,
    horse: str,
    conf: float,
    idx_rank: int | None,
    field_n: int,
    dist_mark: str,
    course_mark: str,
    surface_mark: str,
    lap_label: str,
    last3f: float | None,
    pace_fit: float | None,
    style: str,
    win_pct: float | None,
    place_pct: float | None,
    plus: list[str],
    minus: list[str],
    nar_note: str,
) -> str:
    """なぜ本命/対抗/穴なのかを100〜200字で要約。"""
    role_word = {"本命": "本命", "対抗": "対抗", "穴馬": "穴評価"}.get(role, role)
    bits: list[str] = []

    if idx_rank is not None and field_n:
        bits.append(f"近走指数は{field_n}頭中{idx_rank}位")
    apt = []
    if dist_mark == "○":
        apt.append("距離")
    if course_mark == "○":
        apt.append("コース")
    if surface_mark == "○":
        apt.append("馬場")
    if apt:
        bits.append("・".join(apt) + "適性が合う")
    if last3f is not None and last3f >= 60:
        bits.append("末脚も使える")
    if pace_fit is not None and pace_fit >= 58:
        bits.append(f"{style or '脚質'}が想定展開と好相性")
    elif lap_label:
        bits.append(f"ラップ面は{lap_label}")
    if win_pct is not None:
        bits.append(f"シミュレーション勝率{_fmt_pct(win_pct)}")
    if place_pct is not None and place_pct >= 35:
        bits.append(f"複勝率{_fmt_pct(place_pct)}と崩れにくい")
    if nar_note:
        bits.append("地方実績は中央換算済み")
    if plus:
        bits.append(plus[0])
    if minus and minus[0] != "特記すべき大きな不安は少ない":
        bits.append(f"一方で{minus[0]}点は残る")

    core = "、".join(bits[:5])
    if role == "本命":
        head = f"「{horse}」を{role_word}としたのは、"
    elif role == "対抗":
        head = f"「{horse}」を{role_word}に置いたのは、"
    else:
        head = f"「{horse}」を{role_word}としたのは、"
    tail = f"。信頼度{conf:.0f}点の内容として提示します。"
    text = head + core + tail
    # 100〜200字に収める
    if len(text) < 100:
        text = text.replace("。信頼度", "ため、相手関係を踏まえても残せる判断です。信頼度")
    if len(text) > 200:
        text = text[:197] + "…"
    # 最低でもある程度の長さを確保
    if len(text) < 80:
        text = (
            f"「{horse}」を{role_word}としたのは、指数・適性・シミュレーションを総合した評価だからです。"
            f"数値だけでなく条件面の整合も加味し、信頼度{conf:.0f}点として提示します。"
        )
    return text


def build_pick_rationale(
    *,
    role: str,
    horse: str,
    row: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    pace: dict[str, Any] | None = None,
    idx_rank: int | None = None,
    field_n: int = 0,
    last3_rank: int | None = None,
    n_sample: int = 3,
    win_pct: float | None = None,
    quinella_pct: float | None = None,
    place_pct: float | None = None,
    market_odds: float | None = None,
    fair_odds: float | None = None,
    reason_text: str = "",
    existing_plus: str | list[str] | None = None,
    existing_minus: str | list[str] | None = None,
    lap_label: str = "",
    lap_fit: float | None = None,
    pace_fit: float | None = None,
    style: str = "",
) -> dict[str, Any]:
    """エンジン／表示共通で使う根拠パック。"""
    row = row or {}
    pr = profile or {}
    pace = pace or {}

    plus_list = (
        list(existing_plus) if isinstance(existing_plus, list)
        else _split_materials(existing_plus or "")
    )
    minus_list = (
        list(existing_minus) if isinstance(existing_minus, list)
        else _split_materials(existing_minus or "")
    )
    plus_list.extend(pr.get("plus") or [])
    minus_list.extend(pr.get("minus") or [])
    plus_list = list(dict.fromkeys(plus_list))
    minus_list = list(dict.fromkeys(minus_list))

    reasons_blob = str(reason_text or row.get("理由") or "") + " / " + " / ".join(plus_list + minus_list)
    dist_mark = "○" if ("距離適性" in plus_list or "距離適性" in reasons_blob) else (
        "△" if "距離" in reasons_blob else "－"
    )
    course_mark = "○" if ("コース適性" in plus_list or "コース適性" in reasons_blob) else "－"
    surface = str(pr.get("surface") or "")
    surface_mark = "－"
    for tag in ("芝適性", "ダ適性", "ダート適性"):
        if tag in plus_list or tag in reasons_blob:
            surface_mark = "○"
            break
    if surface_mark == "－" and surface and f"{surface}適性" in reasons_blob:
        surface_mark = "○"

    last3f = pr.get("last3f")
    if last3f is None:
        try:
            last3f = float(row.get("上がり評価")) if row.get("上がり評価") not in (None, "", "—") else None
        except (TypeError, ValueError):
            last3f = None
    jockey_score = pr.get("jockey")
    trouble = pr.get("trouble")
    blood_score = pr.get("blood")
    nar_scale = "地方実績を中央換算" in reasons_blob or "地方実績中心" in reasons_blob
    nar_note = ""
    if "地方実績を中央換算" in reasons_blob:
        nar_note = "地方実績を中央水準へ換算して評価"
    elif "地方実績中心" in reasons_blob:
        nar_note = "地方実績中心のため期待値を抑制"

    market_value = False
    if market_odds and fair_odds and fair_odds > 0 and market_odds > fair_odds * 1.12:
        market_value = True

    if not lap_label:
        lap_label = str(pr.get("lap_label") or row.get("ラップ適性") or "平均ペース適性")
    if lap_fit is None:
        lap_fit = pr.get("lap_fit")
    if pace_fit is None:
        pace_fit = pr.get("pace_fit")
    if not style:
        style = str(pr.get("style_label") or "")
        # 展開相性文字列から脚質抽出
        if not style and row.get("展開相性"):
            style = str(row.get("展開相性")).split("・")[0]

    plus_items, minus_items = build_plus_minus(
        plus_list,
        minus_list,
        role=role,
        win_pct=win_pct,
        place_pct=place_pct,
        idx_rank=idx_rank,
        dist_mark=dist_mark,
        course_mark=course_mark,
        surface_mark=surface_mark,
        last3f=float(last3f) if last3f is not None else None,
        pace_fit=float(pace_fit) if pace_fit is not None else None,
        market_value=market_value,
    )

    conf = calc_horse_confidence(
        role=role,
        n_sample=n_sample,
        win_pct=win_pct,
        place_pct=place_pct,
        idx_rank=idx_rank,
        field_n=field_n or 12,
        dist_mark=dist_mark,
        course_mark=course_mark,
        surface_mark=surface_mark,
        lap_fit=float(lap_fit) if lap_fit is not None else None,
        last3f=float(last3f) if last3f is not None else None,
        pace_fit=float(pace_fit) if pace_fit is not None else None,
        jockey_score=float(jockey_score) if jockey_score is not None else None,
        trouble=float(trouble) if trouble is not None else None,
        nar_scale=nar_scale,
        plus_n=len(plus_items),
        minus_n=len(minus_items),
    )

    reason_rows = build_reason_rows(
        idx_rank=idx_rank,
        field_n=field_n or 12,
        dist_mark=dist_mark,
        course_mark=course_mark,
        surface_mark=surface_mark,
        surface=surface or ("芝" if "芝" in reasons_blob else ("ダ" if "ダ" in reasons_blob else "")),
        lap_label=lap_label,
        lap_fit=float(lap_fit) if lap_fit is not None else None,
        last3f=float(last3f) if last3f is not None else None,
        last3_rank=last3_rank,
        pace_label=str(pace.get("想定ペース") or ""),
        pace_fit=float(pace_fit) if pace_fit is not None else None,
        style=style,
        jockey_score=float(jockey_score) if jockey_score is not None else None,
        blood_score=float(blood_score) if blood_score is not None else None,
        trouble=float(trouble) if trouble is not None else None,
        nar_note=nar_note,
        win_pct=win_pct,
        quinella_pct=quinella_pct,
        place_pct=place_pct,
    )

    comment = build_ai_comment(
        role=role,
        horse=horse,
        conf=conf,
        idx_rank=idx_rank,
        field_n=field_n or 12,
        dist_mark=dist_mark,
        course_mark=course_mark,
        surface_mark=surface_mark,
        lap_label=lap_label,
        last3f=float(last3f) if last3f is not None else None,
        pace_fit=float(pace_fit) if pace_fit is not None else None,
        style=style,
        win_pct=win_pct,
        place_pct=place_pct,
        plus=plus_items,
        minus=minus_items,
        nar_note=nar_note,
    )

    return {
        "AI信頼度スコア": conf,
        "AI信頼度": stars_for_score(conf),
        "判断根拠": reason_rows,
        "プラス材料一覧": plus_items,
        "不安材料一覧": minus_items,
        "プラス材料": " / ".join(plus_items),
        "不安材料": " / ".join(minus_items),
        "AIコメント": comment,
        "距離適性": dist_mark,
        "コース適性": course_mark,
        "馬場適性": surface_mark,
        "ラップ適性": lap_label,
        "上がり評価": round(float(last3f), 0) if last3f is not None else None,
        "上がり順位": last3_rank,
        "展開相性": (
            f"{style}・"
            + ("好相性" if pace_fit is not None and pace_fit >= 58 else (
                "やや不利" if pace_fit is not None and pace_fit <= 42 else "標準"
            ))
        ) if style else (row.get("展開相性") or "標準"),
        "騎手相性": next((r["評価"] for r in reason_rows if r["項目"] == "騎手相性"), "—"),
        "血統適性": next((r["評価"] for r in reason_rows if r["項目"] == "血統適性"), "—"),
        "前走不利補正": next((r["評価"] for r in reason_rows if r["項目"] == "前走不利補正"), "なし"),
        "地方→中央補正": next((r["評価"] for r in reason_rows if r["項目"] == "地方→中央補正"), "対象外"),
    }


def horse_grade(score) -> str:
    """馬のAI評価 S〜D。"""
    try:
        v = float(score)
    except (TypeError, ValueError):
        return "C"
    if v >= 78:
        return "S"
    if v >= 62:
        return "A"
    if v >= 48:
        return "B"
    if v >= 34:
        return "C"
    return "D"


def build_compact_bullets(card: dict, limit: int = 4) -> list[str]:
    """詳細根拠をユーザー向けの短い箇条書きへ圧縮。"""
    bullets: list[str] = []

    def add(msg: str):
        msg = str(msg or "").strip()
        if msg and msg not in bullets:
            bullets.append(msg)

    plus = card.get("プラス材料一覧") or _split_materials(card.get("プラス材料") or "")
    why = card.get("判断根拠") or []
    why_map = {str(x.get("項目")): x for x in why if isinstance(x, dict)}

    # 優先マッピング（短く・分かりやすく）
    last3 = why_map.get("上がり順位") or {}
    if str(last3.get("評価", "")).endswith("位") or "高" in str(last3.get("説明", "")):
        add("上がり3F上位")
    elif any("上がり" in str(p) for p in plus):
        add("上がり3F上位")

    dist = str((why_map.get("距離適性") or {}).get("評価") or card.get("距離適性") or "")
    course = str((why_map.get("コース適性") or {}).get("評価") or card.get("コース適性") or "")
    surface = str((why_map.get("馬場適性") or {}).get("評価") or card.get("馬場適性") or "")
    if dist == "○" and course == "○":
        add("同条件成績◎")
    elif dist == "○" or course == "○" or surface == "○":
        add("同条件成績○")

    pace = why_map.get("展開との相性") or {}
    if str(pace.get("評価")) == "好相性" or "好相性" in str(card.get("展開相性") or ""):
        add("展開が向く")
    elif "不利" in str(pace.get("評価") or card.get("展開相性") or ""):
        add("展開はやや不利")

    trouble = why_map.get("前走不利補正") or {}
    if "補正" in str(trouble.get("評価") or "") or any("前走不利" in str(p) for p in plus):
        add("前走不利補正")

    if any("複勝" in str(p) for p in plus):
        add("複勝圏が厚い")
    if any("割安" in str(p) for p in plus):
        add("オッズ妙味あり")
    if any("指数" in str(p) for p in plus):
        add("近走指数上位")
    if any("騎手" in str(p) for p in plus):
        add("騎手相性◎")
    if any("地方" in str(p) for p in plus) or "補正済" in str((why_map.get("地方→中央補正") or {}).get("評価") or ""):
        add("地方→中央補正済")

    # 足りなければプラス材料を短文化
    shorten = {
        "上がり評価高": "上がり3F上位",
        "複勝圏安定": "複勝圏が厚い",
        "市場より割安": "オッズ妙味あり",
        "距離適性が合う": "距離適性◎",
        "コース適性が合う": "コース適性◎",
        "馬場適性が合う": "馬場適性◎",
        "想定展開との相性が良い": "展開が向く",
        "前走不利の可能性": "前走不利補正",
        "騎手補正+": "騎手相性◎",
        "近走指数1位の上位評価": "近走指数上位",
        "上がり性能が高い": "上がり3F上位",
        "人気薄でもシミュレーション上位": "穴妙味あり",
        "総合指数とシミュレーションのバランスが良い": "総合評価上位",
        "条件面の大きな欠点が見当たらない": "欠点が少ない",
        "相手関係を踏まえても残せる内容": "相手関係でも残せる",
    }
    for p in plus:
        if len(bullets) >= limit:
            break
        key = str(p)
        if key in shorten:
            add(shorten[key])
        elif len(key) <= 10:
            add(key)

    # 足りなければ定型で埋める（重複で add が空振りしても無限ループしない）
    for filler in ("総合評価上位", "欠点が少ない", "相手関係でも残せる"):
        if len(bullets) >= min(3, limit):
            break
        add(filler)
    return bullets[:limit]


def build_display_picks(record: dict) -> list[dict]:
    """画面表示用: ◎本命 / ○対抗 / ☆注目馬 の3頭のみ。"""
    cards = [c for c in (record.get("ピックカード一覧") or []) if isinstance(c, dict)]
    by_role = {str(c.get("役割")): c for c in cards}
    ordered: list[tuple[str, str, dict | None]] = [
        ("本命", "◎", by_role.get("本命")),
        ("対抗", "○", by_role.get("対抗")),
    ]
    spotlight = by_role.get("注目馬")
    if spotlight is None:
        for c in cards:
            if c.get("役割") == "穴馬":
                spotlight = c
                break
    ordered.append(("注目馬", "☆", spotlight))

    out = []
    for role, mark, card in ordered:
        if not card:
            continue
        try:
            conf = float(card.get("AI信頼度スコア"))
        except (TypeError, ValueError):
            try:
                conf = float(card.get("AI評価") or 50)
            except (TypeError, ValueError):
                conf = 50.0
        grade = horse_grade(conf)
        ban = str(card.get("馬番表示") or card.get("馬番") or "").strip()
        name = str(card.get("馬名") or "").strip()
        line = f"{mark}{ban} {name}".strip() if ban else f"{mark} {name}".strip()
        out.append({
            "役割": role,
            "印": mark,
            "馬番表示": ban,
            "馬名": name,
            "表示行": line,
            "AI評価": grade,
            "AI信頼度スコア": round(conf, 1),
            "要点": build_compact_bullets(card, limit=4),
            "カード": card,
        })
    return out[:3]


def enrich_pick_card(card: dict, race: dict | None = None) -> dict:
    """既存CSVの薄いカードを表示用に充実（再予想なしでもUIを満たす）。"""
    if not isinstance(card, dict):
        return card
    race = race or {}
    role = str(card.get("役割") or "本命")
    horse = str(card.get("馬名") or "")
    try:
        idx_rank = int(float(card.get("近走指数順位"))) if card.get("近走指数順位") not in (None, "", "—") else None
    except (TypeError, ValueError):
        idx_rank = None
    try:
        win_pct = float(card.get("勝率")) if card.get("勝率") not in (None, "", "—") else None
    except (TypeError, ValueError):
        win_pct = None
    try:
        q_pct = float(card.get("連対率")) if card.get("連対率") not in (None, "", "—") else None
    except (TypeError, ValueError):
        q_pct = None
    try:
        p_pct = float(card.get("複勝率")) if card.get("複勝率") not in (None, "", "—") else None
    except (TypeError, ValueError):
        p_pct = None
    try:
        mo = float(card.get("単勝オッズ")) if card.get("単勝オッズ") not in (None, "", "—") else None
    except (TypeError, ValueError):
        mo = None
    try:
        fo = float(card.get("AI適正オッズ")) if card.get("AI適正オッズ") not in (None, "", "—") else None
    except (TypeError, ValueError):
        fo = None

    n_sample = 3
    reasons = str(race.get("本命理由") or "") + " / " + str(card.get("不安材料") or "")
    if "サンプル少" in reasons:
        n_sample = 1
    elif "履歴少" in reasons:
        n_sample = 2
    try:
        if race.get("本命データ件数") not in (None, "", "—"):
            n_sample = int(float(race.get("本命データ件数")))
    except (TypeError, ValueError):
        pass

    # フィールド頭数の近似（表示時は正確な出走頭数が無いため）
    field_n = 14
    if idx_rank is not None:
        field_n = max(field_n, int(idx_rank) + 2)

    pack = build_pick_rationale(
        role=role,
        horse=horse,
        row=card,
        idx_rank=idx_rank,
        field_n=field_n,
        n_sample=n_sample,
        win_pct=win_pct,
        quinella_pct=q_pct,
        place_pct=p_pct,
        market_odds=mo,
        fair_odds=fo,
        reason_text=str(race.get("本命理由") or "") + " / " + str(card.get("人気以上に評価した理由") or ""),
        existing_plus=card.get("プラス材料"),
        existing_minus=card.get("不安材料"),
        lap_label=str(card.get("ラップ適性") or ""),
        style=str(card.get("展開相性") or "").split("・")[0],
    )
    # 既存の明示値は優先して残す
    for k in ("距離適性", "コース適性", "ラップ適性", "上がり評価", "展開相性", "勝率", "連対率", "複勝率"):
        if card.get(k) not in (None, "", "—"):
            pack[k] = card.get(k) if k not in pack or pack.get(k) in (None, "", "—") else pack[k]
            if k in ("距離適性", "コース適性") and card.get(k) in ("○", "△", "－"):
                pack[k] = card[k]
    card.update(pack)
    return card
