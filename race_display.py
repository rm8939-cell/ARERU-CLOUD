"""レース画面の表示専用ヘルパー。

予想スコア・BUY判定・AI順位は一切触らない。
既存の展開予想・適性カード・過去走から、画面に出す情報だけを組み立てる。

条件変更は「検出できたか」と「効果を判定できるか」を分離する。
効果の根拠が無いものは データ不足 とし、勝手に加点しない。
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
HISTORY = DATA / "all_history.csv"

ZONE_ORDER = ("逃げ", "先行", "中団", "後方", "不明")
POS_TO_ZONE = {
    "逃げ": "逃げ",
    "先行": "先行",
    "差し": "中団",
    "追込": "後方",
}

# ブリンカー等は取得データに列が無い。検出不能を明示するための定数。
GEAR_UNAVAILABLE = {
    "検出": "データ不足",
    "効果": "データ不足",
    "ラベル": "馬具・ブリンカー",
    "詳細": "出走表・戦績に馬具列が無いため検出できない",
    "種別": "馬具",
}


def _blank(v) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s in ("", "—", "-", "None", "nan", "なし", "データなし", "データ不足")


def parse_surface_dist(text) -> tuple[str | None, int | None]:
    """『ダ1400』『芝 2000m』などから芝ダとメートルを取る。"""
    s = str(text or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return None, None
    surf = None
    if re.search(r"ダート|\bダ\b|^ダ|ダ\d", s):
        surf = "ダート"
    elif re.search(r"障害|\b障\b|^障", s):
        surf = "障害"
    elif re.search(r"芝", s):
        surf = "芝"
    m = re.search(r"(\d{3,4})", s)
    dist = int(m.group(1)) if m else None
    return surf, dist


def normalize_venue(text) -> str | None:
    s = str(text or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    # 『2福島6』『5中山8』のような開催表記から場名だけ抜く
    m = re.search(
        r"(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉|"
        r"大井|川崎|船橋|浦和|園田|姫路|名古屋|笠松|金沢|園田|"
        r"高知|佐賀|門別|盛岡|水沢|帯広)",
        s,
    )
    return m.group(1) if m else s


@lru_cache(maxsize=1)
def _history_frame() -> pd.DataFrame:
    if not HISTORY.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty or "馬名" not in df.columns:
        return df
    work = df.copy()
    if "年月日" in work.columns:
        work["_date"] = pd.to_datetime(
            work["年月日"].astype(str)
            .str.replace("年", "-", regex=False)
            .str.replace("月", "-", regex=False)
            .str.replace("日", "", regex=False),
            errors="coerce",
        )
        work = work.sort_values(["馬名", "_date"], ascending=[True, False])
    return work


def history_for_horse(name: str, before: str | None = None) -> list[dict]:
    """対象日より前の戦績。無ければ空。表示専用。"""
    df = _history_frame()
    if df.empty or not name:
        return []
    sub = df[df["馬名"].astype(str) == str(name)]
    if sub.empty:
        return []
    if before and "_date" in sub.columns:
        cut = pd.to_datetime(before, errors="coerce")
        if pd.notna(cut):
            sub = sub[sub["_date"].isna() | (sub["_date"] < cut)]
    return sub.head(12).to_dict("records")


def _mark_to_bar(mark) -> dict:
    """○/△/－ をバー表示用にする。数値を捏造しない。"""
    s = str(mark or "").strip()
    if s in ("○", "◎", "好転"):
        return {"表示": s, "バー": 84, "状態": "available"}
    if s in ("△",):
        return {"表示": s, "バー": 55, "状態": "available"}
    if s in ("－", "-", "×"):
        return {"表示": s, "バー": 28, "状態": "available"}
    if _blank(s):
        return {"表示": "データ不足", "バー": None, "状態": "insufficient"}
    # 既に文章ラベル（平均ペース適性など）
    return {"表示": s, "バー": None, "状態": "available"}


def aptitude_rows(horse: dict) -> list[dict]:
    """本命詳細用。実在フィールドだけ並べる。"""
    card = horse.get("カード") if isinstance(horse.get("カード"), dict) else {}
    rows = []
    # ラップは数値適合度があればそれを優先（既存 lap_aptitude の出力）
    fit = horse.get("ラップ適合度")
    lap_label = horse.get("ラップ適性表示") or horse.get("ラップ適性") or card.get("ラップ適性")
    if fit is not None and str(fit).strip() not in ("", "nan"):
        try:
            fv = float(fit)
            rows.append({
                "項目": "ラップ適性",
                "表示": f"{fv:.0f}" + (f" / {lap_label}" if not _blank(lap_label) else ""),
                "バー": max(0, min(100, fv)),
                "状態": "available",
                "AI評価に使用": True,
            })
        except (TypeError, ValueError):
            rows.append({"項目": "ラップ適性", **_mark_to_bar(lap_label), "AI評価に使用": True})
    else:
        rows.append({"項目": "ラップ適性", **_mark_to_bar(lap_label), "AI評価に使用": not _blank(lap_label)})

    for key, used in (
        ("展開適性", True),
        ("前走内容", False),  # 表示用転写。スコア加点そのものではない
        ("コース適性", True),
        ("距離適性", True),
        ("馬場適性", True),
    ):
        raw = horse.get(key)
        if _blank(raw) and key != "展開適性":
            raw = card.get(key)
        if key == "展開適性" and _blank(raw):
            raw = horse.get("展開相性") or card.get("展開相性")
        if key == "前走内容":
            if _blank(raw):
                rows.append({
                    "項目": key, "表示": "データ不足", "バー": None,
                    "状態": "insufficient", "AI評価に使用": False,
                })
            else:
                rows.append({
                    "項目": key, "表示": str(raw), "バー": None,
                    "状態": "available", "AI評価に使用": False,
                })
            continue
        bar = _mark_to_bar(raw)
        bar["項目"] = key
        bar["AI評価に使用"] = bool(used and bar["状態"] == "available")
        rows.append(bar)
    return rows


def race_type_label(anal: dict) -> str | None:
    """既存の展開傾向・ラップ傾向からレースタイプ文言を作る。新しい予測はしない。"""
    trend = anal.get("展開傾向")
    lap = anal.get("ラップ傾向")
    pace = anal.get("想定ペース")
    if _blank(trend) and _blank(lap) and _blank(pace):
        return None
    if lap == "後傾":
        return "後半の瞬発力勝負"
    if lap == "前傾":
        return "前半厳しい消耗戦"
    if trend == "差し有利":
        return "差し・追込が届く流れ"
    if trend == "前有利":
        return "前目有利の流れ"
    if pace == "スロー":
        return "スローからの粘り合い"
    if pace == "ハイ":
        return "ハイペースの差し競馬"
    if pace == "ミドル" or trend == "フラット":
        return "平均的な流れ"
    return None


def build_diagnosis(race: dict, anal: dict) -> dict:
    """AIレース診断カード用。値が無い項目は None のまま。"""
    conf = race.get("レース信頼度スコア")
    try:
        conf_v = float(conf) if conf not in (None, "") else None
    except (TypeError, ValueError):
        conf_v = None
    chaos = anal.get("荒れ指数")
    try:
        chaos_v = float(chaos) if chaos not in (None, "") else None
    except (TypeError, ValueError):
        chaos_v = None
    return {
        "信頼度": round(conf_v, 1) if conf_v is not None else None,
        "波乱度": round(chaos_v, 1) if chaos_v is not None else None,
        "想定ペース": anal.get("想定ペース表示") or anal.get("想定ペース"),
        "レースタイプ": race_type_label(anal),
        "展開傾向": anal.get("展開傾向"),
        "有利枠": anal.get("有利枠"),
    }


def style_float_to_pos(style_f) -> tuple[str | None, int | None]:
    """race_sim の脚質スコア(0逃げ〜1追込)を表示用ポジションへ。"""
    try:
        v = float(style_f)
    except (TypeError, ValueError):
        return None, None
    if v <= 0.28:
        return "逃げ", 1
    if v <= 0.45:
        return "先行", 2
    if v <= 0.65:
        return "差し", 3
    return "追込", 4


def assign_zones(horses: list[dict]) -> dict:
    """出走全馬を逃げ/先行/中団/後方へ振り分ける。1頭も落とさない。

    不明は本当に判断できない馬だけ。差し→中団、追込→後方。
    """
    zones = {z: [] for z in ZONE_ORDER}
    for p in horses or []:
        if not isinstance(p, dict):
            continue
        pos = p.get("想定ポジション") or p.get("脚質") or ""
        zone = POS_TO_ZONE.get(str(pos), "不明")
        if zone == "不明":
            txt = str(p.get("展開適性") or p.get("展開相性") or "")
            for k, z in POS_TO_ZONE.items():
                if k in txt:
                    zone = z
                    pos = k
                    break
        if zone == "不明" and p.get("脚質スコア") is not None:
            label, _n = style_float_to_pos(p.get("脚質スコア"))
            if label:
                zone = POS_TO_ZONE.get(label, "不明")
                pos = label
        entry = {
            "馬番": p.get("馬番表示") or p.get("馬番"),
            "馬番数字": p.get("馬番"),
            "枠番": p.get("枠番"),
            "馬名": p.get("馬名"),
            "想定位置": str(pos) if not _blank(pos) else (None if zone == "不明" else zone),
            "展開適性": p.get("展開適性") or p.get("展開相性"),
            "AI順位": p.get("AI順位"),
            "表示印": p.get("表示印"),
            "ゾーン": zone,
        }
        if _blank(entry["想定位置"]):
            entry["想定位置"] = "データ不足" if zone == "不明" else zone
        zones[zone].append(entry)
        p["展開ゾーン"] = zone
    return zones


def build_honmei_view(horse: dict | None, race: dict | None, anal: dict | None) -> str | None:
    """本命のAI見解。実データがある文だけつなぐ。捏造しない。"""
    if not isinstance(horse, dict):
        return None
    race = race or {}
    anal = anal or {}
    card = horse.get("カード") if isinstance(horse.get("カード"), dict) else {}
    bits: list[str] = []

    venue = str(race.get("開催地") or race.get("競馬場") or "").strip()
    dist = str(race.get("距離") or race.get("距離表示") or "").strip()
    course = horse.get("コース適性") or card.get("コース適性")
    distance = horse.get("距離適性") or card.get("距離適性")
    if not _blank(course) and course in ("○", "◎", "好転") and venue:
        bits.append(f"{venue}コース適性を評価")
    elif not _blank(course) and course not in ("－", "-", "×"):
        bits.append(f"コース適性は{course}")
    if not _blank(distance) and distance in ("○", "◎", "好転") and dist:
        bits.append(f"{dist}の距離適性も裏付け")
    elif not _blank(distance) and distance not in ("－", "-", "×"):
        bits.append(f"距離適性は{distance}")

    lap = horse.get("ラップ適性表示") or horse.get("ラップ適性") or card.get("ラップ適性")
    diag = race.get("AIレース診断") if isinstance(race.get("AIレース診断"), dict) else {}
    race_type = anal.get("レースタイプ") or diag.get("レースタイプ")
    pace = anal.get("想定ペース表示") or anal.get("想定ペース") or diag.get("想定ペース")
    if not _blank(lap):
        if not _blank(race_type):
            bits.append(f"想定「{race_type}」に対しラップ適性は{lap}")
        elif not _blank(pace):
            bits.append(f"想定ペース{pace}に対しラップ適性は{lap}")
        else:
            bits.append(f"ラップ適性は{lap}")

    tenkai = horse.get("展開適性") or horse.get("展開相性") or card.get("展開相性")
    if not _blank(tenkai):
        bits.append(f"展開面は{tenkai}")

    prev = horse.get("前走内容")
    if not _blank(prev):
        bits.append(f"前走は{prev}")

    # 重複除去しつつ最大3文
    out = []
    seen = set()
    for b in bits:
        b = str(b).strip().rstrip("。")
        if not b or b in seen:
            continue
        seen.add(b)
        out.append(b)
        if len(out) >= 3:
            break
    if not out:
        return None
    return "。".join(out) + "。"


def detect_condition_changes(
    horse: dict,
    race: dict,
    past: dict | None = None,
) -> list[dict]:
    """条件変更の検出。効果は原則データ不足（単純加点しない）。"""
    past = past or {}
    out: list[dict] = []
    name = str(horse.get("馬名") or "").strip()
    day = str(race.get("日付") or race.get("開催日") or past.get("日付") or "").strip() or None
    hist = history_for_horse(name, before=day)

    # 現在条件
    cur_surf, cur_dist = parse_surface_dist(
        race.get("距離") or race.get("距離表示") or race.get("芝ダ") or ""
    )
    if cur_surf is None:
        cur_surf, _ = parse_surface_dist(race.get("芝ダ") or race.get("馬場種別") or "")
    if cur_dist is None:
        try:
            cur_dist = int(re.search(r"(\d{3,4})", str(race.get("距離") or "")).group(1))
        except Exception:
            cur_dist = None
    cur_venue = normalize_venue(race.get("開催地") or race.get("競馬場") or "")
    cur_jockey = str(horse.get("騎手") or past.get("騎手") or "").strip() or None

    # 前走（history優先、なければ runners の場1）
    last = hist[0] if hist else None
    last_surf = last_dist = last_venue = last_jockey = None
    last_date = None
    if last:
        last_surf, last_dist = parse_surface_dist(last.get("距離"))
        last_venue = normalize_venue(last.get("場"))
        last_jockey = str(last.get("騎手") or "").strip() or None
        last_date = last.get("_date")
    else:
        last_venue = normalize_venue(past.get("場1"))
        # 馬場1 は天気のことが多く、芝ダ判定には使わない

    # 芝⇔ダート
    if cur_surf and last_surf:
        if cur_surf != last_surf and {cur_surf, last_surf} <= {"芝", "ダート"}:
            label = f"{last_surf}→{cur_surf}替わり"
            out.append({
                "検出": "YES", "効果": "データ不足", "ラベル": label,
                "詳細": f"前走は{last_surf}、今回は{cur_surf}",
                "種別": "SurfaceSwitch", "icon": "🔄",
            })
    elif not hist:
        out.append({
            "検出": "データ不足", "効果": "データ不足", "ラベル": "芝⇔ダート替わり",
            "詳細": "前走の芝ダ情報が無い", "種別": "SurfaceSwitch", "icon": "🔄",
        })

    # 距離
    if cur_dist and last_dist:
        delta = cur_dist - last_dist
        if delta <= -100:
            out.append({
                "検出": "YES", "効果": "データ不足", "ラベル": f"距離{abs(delta)}m短縮",
                "詳細": f"前走{last_dist}m → 今回{cur_dist}m",
                "種別": "DistanceShorten", "icon": "⬇️",
            })
        elif delta >= 100:
            out.append({
                "検出": "YES", "効果": "データ不足", "ラベル": f"距離{delta}m延長",
                "詳細": f"前走{last_dist}m → 今回{cur_dist}m",
                "種別": "DistanceExtend", "icon": "⬆️",
            })
    if cur_dist and hist:
        dists = []
        for h in hist:
            _, d = parse_surface_dist(h.get("距離"))
            if d:
                dists.append(d)
        if dists and all(abs(d - cur_dist) >= 200 for d in dists):
            out.append({
                "検出": "YES", "効果": "データ不足", "ラベル": "初距離",
                "詳細": f"近走に{cur_dist}m付近の実績なし",
                "種別": "FirstDistance", "icon": "📏",
            })
    elif not hist:
        out.append({
            "検出": "データ不足", "効果": "データ不足", "ラベル": "距離変化",
            "詳細": "前走距離が取れない", "種別": "Distance", "icon": "📏",
        })

    # コース
    if cur_venue and last_venue:
        if cur_venue != last_venue:
            out.append({
                "検出": "YES", "効果": "データ不足", "ラベル": "コース替わり",
                "詳細": f"前走{last_venue} → 今回{cur_venue}",
                "種別": "CourseChange", "icon": "🏁",
            })
    if cur_venue and hist:
        venues = {normalize_venue(h.get("場")) for h in hist}
        venues.discard(None)
        if venues and cur_venue not in venues:
            out.append({
                "検出": "YES", "効果": "データ不足", "ラベル": "初コース",
                "詳細": f"近走に{cur_venue}の実績なし",
                "種別": "FirstCourse", "icon": "🏁",
            })

    # 休み明け（履歴日付）
    if last_date is not None and day:
        try:
            cur_d = pd.to_datetime(day, errors="coerce")
            if pd.notna(cur_d) and pd.notna(last_date):
                gap = int((cur_d - last_date).days)
                if gap >= 84:
                    out.append({
                        "検出": "YES",
                        "効果": "既存ロジックで減点材料になる場合あり",
                        "ラベル": "長期休養明け",
                        "詳細": f"前走から{gap}日",
                        "種別": "LayoffLong", "icon": "💤",
                    })
                elif gap >= 42:
                    out.append({
                        "検出": "YES",
                        "効果": "既存ロジックで減点材料になる場合あり",
                        "ラベル": "休み明け",
                        "詳細": f"前走から{gap}日",
                        "種別": "Layoff", "icon": "💤",
                    })
        except Exception:
            pass
    else:
        # カードのマイナス材料に休み明けがあれば検出扱い
        blob = " ".join(str(x) for x in (horse.get("マイナス要因") or []))
        card = horse.get("カード") if isinstance(horse.get("カード"), dict) else {}
        blob += " " + str(card.get("不安材料") or "")
        if "長期休養" in blob:
            out.append({
                "検出": "YES", "効果": "既存ロジックで減点材料",
                "ラベル": "長期休養明け", "詳細": "不安材料に記録あり",
                "種別": "LayoffLong", "icon": "💤",
            })
        elif "休み明け" in blob:
            out.append({
                "検出": "YES", "効果": "既存ロジックで減点材料",
                "ラベル": "休み明け", "詳細": "不安材料に記録あり",
                "種別": "Layoff", "icon": "💤",
            })

    # 騎手変更
    if cur_jockey and last_jockey:
        # 空白差を無視
        a = re.sub(r"\s+", "", cur_jockey)
        b = re.sub(r"\s+", "", last_jockey)
        if a and b and a != b:
            out.append({
                "検出": "YES", "効果": "データ不足", "ラベル": "騎手変更",
                "詳細": f"前走{last_jockey} → 今回{cur_jockey}",
                "種別": "JockeyChange", "icon": "🧍",
            })
    elif not last_jockey:
        out.append({
            "検出": "データ不足", "効果": "データ不足", "ラベル": "騎手変更",
            "詳細": "前走騎手が取れない", "種別": "JockeyChange", "icon": "🧍",
        })

    # ブリンカー / 馬具 — データ列が無いので検出不能を明示
    out.append(dict(GEAR_UNAVAILABLE))
    out.append({
        "検出": "データ不足", "効果": "データ不足", "ラベル": "初ブリンカー",
        "詳細": "ブリンカー情報がデータに無い", "種別": "FirstBlinker", "icon": "🔥",
    })
    return out


def focus_condition_points(horses: list[dict], limit: int = 8) -> list[dict]:
    """レース全体の注目ポイント用に、検出YESだけを重複除去して返す。"""
    seen = set()
    pts = []
    for p in horses or []:
        for ch in p.get("条件変更") or []:
            if ch.get("検出") != "YES":
                continue
            key = (ch.get("ラベル"), p.get("馬名"))
            if key in seen:
                continue
            seen.add(key)
            pts.append({
                "馬名": p.get("馬名"),
                "馬番": p.get("馬番表示") or p.get("馬番"),
                "ラベル": ch.get("ラベル"),
                "詳細": ch.get("詳細"),
                "効果": ch.get("効果"),
                "icon": ch.get("icon") or "•",
            })
            if len(pts) >= limit:
                return pts
    return pts
