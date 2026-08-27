"""段階シミュレーションとレース展開・馬体補正の補助モジュール。

既存の AREru指数を能力ベースに、枠・騎手・休み明け・馬体重・脚質推定などを
加点減点したうえで、スタート→位置取り→ペース→4角→直線→ゴールを再現する。
上がり3F・血統・不利コメント等が未取得の場合は、着順×人気からの代理指標を使う。
"""
from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd

def _env_int(name: str, default: int) -> int:
    try:
        v = int(str(os.environ.get(name) or "").strip())
        return v if v > 0 else default
    except Exception:
        return default


# Render Free(512MB) では 10万回×全レースで親Web+子プロセスが合算OOMしやすい。
# ARERU_SIM_RUNS で上書き可。未設定時は Render なら 2万、それ以外は 10万。
_DEFAULT_SIM = 20_000 if str(os.environ.get("RENDER") or "").lower() in ("true", "1") else 100_000
SIM_RUNS = _env_int("ARERU_SIM_RUNS", _DEFAULT_SIM)
SIM_BATCH = _env_int("ARERU_SIM_BATCH", 2_500 if SIM_RUNS <= 25_000 else 5_000)
# 券種候補用に保持する着順サンプル上限（全量保持で一時メモリが倍増するのを防ぐ）
ORDERS_KEEP_MAX = _env_int("ARERU_ORDERS_KEEP_MAX", 25_000)

# 脚質: 0=逃げ寄り … 1=追込寄り
STYLE_NIGE = 0.15
STYLE_SENKO = 0.35
STYLE_SASHI = 0.65
STYLE_OIKOMI = 0.85


def num(x):
    return pd.to_numeric(x, errors="coerce")


def clamp(x, a=0.0, b=100.0) -> float:
    return float(max(a, min(b, x)))


def clean_name(x) -> str:
    return re.sub(r"[\s\u3000]+", "", str(x or "")).strip()


def parse_time_sec(v) -> float:
    s = str(v or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return float("nan")
    m = re.match(r"(\d+):(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(s)
    except Exception:
        return float("nan")


def parse_weight(v) -> float:
    s = str(v or "").strip()
    m = re.search(r"(\d{3,4})", s)
    return float(m.group(1)) if m else float("nan")


def dist_meters(dist) -> float:
    s = str(dist or "")
    m = re.search(r"(\d{3,4})", s)
    return float(m.group(1)) if m else float("nan")


def surface_of(dist) -> str:
    s = str(dist or "")
    if s.startswith("芝"):
        return "芝"
    if s.startswith("ダ"):
        return "ダ"
    return ""


def infer_style(finishes: np.ndarray, pops: np.ndarray) -> float:
    """着順と人気から脚質を推定（0=逃げ〜1=追込）。"""
    ok = ~np.isnan(finishes)
    if not ok.any():
        return 0.5
    f = finishes[ok]
    p = pops[ok] if pops is not None and len(pops) == len(finishes) else np.full_like(f, np.nan)
    # 人気上位で好走 → 先行寄り / 人気薄で好走 → 差し寄り
    frontish = []
    for fi, pi in zip(f, p):
        if not np.isnan(pi) and pi <= 3 and fi <= 3:
            frontish.append(0.25)
        elif not np.isnan(pi) and pi >= 8 and fi <= 3:
            frontish.append(0.75)
        elif fi <= 2:
            frontish.append(0.35)
        elif fi >= 10:
            frontish.append(0.6)
        else:
            frontish.append(0.5)
    return float(np.clip(np.mean(frontish), 0.05, 0.95))


def last3f_proxy(finishes: np.ndarray, pops: np.ndarray) -> float:
    """上がり代理: 人気以上に食い込むほど上がり評価を上げる。"""
    ok = ~np.isnan(finishes) & ~np.isnan(pops)
    if ok.sum() < 1:
        return 50.0
    gaps = (pops[ok] - finishes[ok]).astype(float)
    # 差し切り（人気薄→好着）を上がり性能とみなす
    score = 50 + float(np.nanmean(gaps)) * 6
    return clamp(score)


def trouble_proxy(finishes: np.ndarray, pops: np.ndarray) -> float:
    """前走不利代理: 人気に対して大きく負けた場合に不利とみなす（0〜1）。"""
    if np.isnan(finishes[0]) or np.isnan(pops[0]):
        return 0.0
    gap = float(finishes[0] - pops[0])
    if gap >= 6:
        return min(1.0, 0.35 + gap * 0.06)
    if gap >= 4:
        return 0.25
    return 0.0


def layoff_days(history: pd.DataFrame | None, horse: str, target) -> float:
    if history is None or getattr(history, "empty", True):
        return float("nan")
    h = history[(history["_horse"] == clean_name(horse)) & (history["_date"] < target)]
    if h.empty:
        return float("nan")
    last = h["_date"].max()
    try:
        return float((pd.Timestamp(target) - pd.Timestamp(last)).days)
    except Exception:
        return float("nan")


def weight_delta(history: pd.DataFrame | None, horse: str, target) -> float:
    """直近2走の馬体重差（kg）。増減が大きいと減点寄り。"""
    from areru_engine import ablation_enabled
    if not ablation_enabled('weight'):
        return float("nan")
    if history is None or getattr(history, "empty", True):
        return float("nan")
    h = history[(history["_horse"] == clean_name(horse)) & (history["_date"] < target)]
    h = h.sort_values("_date", ascending=False).head(2)
    if len(h) < 2:
        return float("nan")
    w = h["馬体重"].map(parse_weight).to_numpy(dtype=float)
    if np.isnan(w).any():
        return float("nan")
    return float(w[0] - w[1])


def jockey_bonus(history: pd.DataFrame | None, jockey: str, venue: str, target) -> float:
    from areru_engine import ablation_enabled
    if not ablation_enabled('jockey'):
        return 0.0
    if history is not None and not getattr(history, "empty", True) and str(jockey or "").strip():
        j = clean_name(jockey)
        h = history[(history["_date"] < target) & (history["騎手"].map(clean_name) == j)].tail(40)
        if len(h) >= 3:
            fin = num(h["着順"])
            place = float((fin <= 3).mean())
            bonus = (place - 0.22) * 18
            same = h[h["場"].astype(str) == str(venue)]
            if len(same) >= 2:
                bonus += (float((num(same["着順"]) <= 3).mean()) - 0.22) * 8
            return float(np.clip(bonus, -6, 8))
    try:
        from history_index import jockey_bonus_from_index
        return jockey_bonus_from_index(jockey, venue)
    except Exception:
        return 0.0


def calib_v2_enabled() -> bool:
    """PRESET=D: holdoutでも悪化が再現したSIM較正。"""
    return str(os.environ.get("ARERU_LOGIC_PRESET") or "").strip().upper() == "D"


def calib_flag(name: str) -> bool:
    """単体較正フラグ（BUY除外ではない）。ARERU_CALIB_<NAME>=1。"""
    key = f"ARERU_CALIB_{str(name or '').strip().upper()}"
    return str(os.environ.get(key) or "").strip().lower() in ("1", "true", "yes")


def is_sashi_style(style) -> bool:
    """診断スクリプトと同じ差し定義: 0.45超〜0.70。"""
    try:
        x = float(style)
    except (TypeError, ValueError):
        return False
    if x != x:
        return False
    return 0.45 < x <= 0.70


def gate_band(waku, n_horses: int) -> str:
    """診断スクリプトと同じ内/中/外。"""
    try:
        w = int(float(waku))
    except (TypeError, ValueError):
        return "不明"
    n = int(n_horses or 0)
    if n <= 0:
        return str(w)
    if w <= max(2, n // 4):
        return "内枠"
    if w >= max(n - 2, int(n * 3 / 4)):
        return "外枠"
    return "中枠"


def three_calib_adj(style, waku, n_horses: int, market_odds) -> tuple[float, list[str], list[str]]:
    """検証対象3条件だけの本命較正。フラグOFFならゼロ。

    1. SASHI_INNER: 差し×内枠を減点
    2. ODDS_INNER: 12.0-19.9倍×内枠を減点
    3. SASHI_SWEET: 差し×5.0-7.9倍×中枠を加点

    2026-08 検証で3つとも holdout 不採用確定。本番フラグは立てない。
    """
    delta = 0.0
    plus: list[str] = []
    minus: list[str] = []
    band = gate_band(waku, n_horses)
    try:
        odds = float(market_odds)
    except (TypeError, ValueError):
        odds = float("nan")
    sashi = is_sashi_style(style)

    if calib_flag("SASHI_INNER") and sashi and band == "内枠":
        delta -= 3.2
        minus.append("差し×内枠")
    if calib_flag("ODDS_INNER") and band == "内枠" and 12.0 <= odds < 20.0:
        delta -= 2.8
        minus.append("12-20倍×内枠")
    if calib_flag("SASHI_SWEET") and sashi and band == "中枠" and 5.0 <= odds < 8.0:
        delta += 2.0
        plus.append("差し×5-8倍×中枠")
    return delta, plus, minus


def gate_bias(venue: str, waku, n_horses: int, surface: str) -> float:
    """枠順補正（簡易）。ダート内枠・芝外枠をわずかに優遇。

    PRESET=D: NEW BUY の内枠は train/holdout とも大幅マイナス、外枠は相対的に良い。
    ダート内枠加点を外し、外枠をわずかに戻す（場名での除外フィルタではない）。
    """
    try:
        w = int(float(waku))
    except Exception:
        return 0.0
    if n_horses <= 0:
        return 0.0
    inner = w <= max(2, n_horses // 4)
    outer = w >= max(n_horses - 2, n_horses * 3 // 4)
    if surface == "ダ":
        if calib_v2_enabled():
            if inner:
                return -0.6
            if outer:
                return 0.8
            return 0.0
        if inner:
            return 2.2
        if outer:
            return -1.2
    if surface == "芝":
        if outer:
            return 1.4
        if inner:
            return -0.6
    return 0.0


def course_distance_fit(history: pd.DataFrame | None, horse: str, target, venue: str, dist_m: float, surface: str) -> tuple[float, list[str]]:
    from areru_engine import ablation_enabled
    reasons: list[str] = []
    if not ablation_enabled('course'):
        return 0.0, reasons
    if history is None or getattr(history, "empty", True) or np.isnan(dist_m):
        return 0.0, reasons
    h = history[(history["_horse"] == clean_name(horse)) & (history["_date"] < target)].sort_values("_date", ascending=False).head(12)
    if h.empty:
        return 0.0, reasons
    adj = 0.0
    hd = h["距離"].map(dist_meters)
    near = h[(hd - dist_m).abs() <= 200]
    if len(near) >= 2:
        rate = float((num(near["着順"]) <= 5).mean())
        adj += (rate - 0.35) * 14
        if rate >= 0.55:
            reasons.append("距離適性")
    if surface:
        same_s = h[h["距離"].astype(str).str.startswith(surface)]
        if len(same_s) >= 2:
            rate = float((num(same_s["着順"]) <= 5).mean())
            adj += (rate - 0.35) * 10
            if rate >= 0.55:
                reasons.append(f"{surface}適性")
    same_v = h[h["場"].astype(str) == str(venue)]
    if len(same_v) >= 2:
        rate = float((num(same_v["着順"]) <= 5).mean())
        adj += (rate - 0.35) * 8
        if rate >= 0.5:
            reasons.append("コース適性")
    return float(np.clip(adj, -8, 10)), reasons


def lap_aptitude(style: float, pace_label: str) -> tuple[float, str]:
    """ラップ適性: 前傾(ハイ)・平均・後傾(スロー)との相性。"""
    if pace_label == "ハイ":
        # 差し・追込有利
        fit = 50 + (style - 0.5) * 40
        label = "後傾寄り適性" if style >= 0.55 else "前傾不利気味"
    elif pace_label == "スロー":
        fit = 50 + (0.5 - style) * 40
        label = "前傾寄り適性" if style <= 0.45 else "後傾不利気味"
    else:
        fit = 52.0
        label = "平均ペース適性"
    return clamp(fit), label



def track_condition_bonus(history: pd.DataFrame | None, horse: str, target) -> tuple[float, list[str]]:
    """直近走の馬場状態適性（重・稍重での好走など）。"""
    from areru_engine import ablation_enabled
    if not ablation_enabled('track') or history is None or getattr(history, "empty", True):
        return 0.0, []
    h = history[(history["_horse"] == clean_name(horse)) & (history["_date"] < target)]
    h = h.sort_values("_date", ascending=False).head(6)
    if h.empty:
        return 0.0, []
    plus: list[str] = []
    adj = 0.0
    tracks = h["馬場"].astype(str).tolist() if "馬場" in h.columns else []
    fin = num(h["着順"]) if "着順" in h.columns else pd.Series(dtype=float)
    heavy_mask = [any(x in str(t) for x in ("重", "不良", "稍")) for t in tracks]
    if sum(heavy_mask) >= 2 and len(fin.dropna()) >= 2:
        heavy_fin = fin[heavy_mask[: len(fin)]]
        if len(heavy_fin) and float((heavy_fin <= 5).mean()) >= 0.5:
            adj += 1.5
            plus.append("重馬場適性")
    return adj, plus


def margin_bonus_from_row(row) -> tuple[float, list[str]]:
    """runners 列の着差1..5 から接戦ボーナス。"""
    from areru_engine import ablation_enabled
    if not ablation_enabled('margin'):
        return 0.0, []
    scores = []
    for i in range(1, 6):
        s = str(row.get(f"着差{i}") or "").strip()
        if not s:
            continue
        if any(x in s for x in ("ハナ", "クビ", "アタマ", "同着")):
            scores.append(2.5)
        elif "1/2" in s:
            scores.append(2.0)
        elif re.match(r"^[12](?:\.\d)?$", s):
            scores.append(1.5)
    if not scores:
        return 0.0, []
    avg = float(np.mean(scores[:3]))
    return min(2.5, avg * 0.85), (["前走接戦"] if avg >= 2.0 else [])


def time_trend_bonus(row, history: pd.DataFrame | None, horse: str, target) -> tuple[float, list[str]]:
    """タイム改善トレンド。"""
    from areru_engine import ablation_enabled
    if not ablation_enabled('time'):
        return 0.0, []
    times: list[float] = []
    for i in range(1, 6):
        ts = parse_time_sec(row.get(f"タイム{i}"))
        if not np.isnan(ts):
            times.append(ts)
    if len(times) < 2 and history is not None and not getattr(history, "empty", True):
        h = history[(history["_horse"] == clean_name(horse)) & (history["_date"] < target)]
        h = h.sort_values("_date", ascending=False).head(5)
        for _, xr in h.iterrows():
            ts = parse_time_sec(xr.get("タイム"))
            if not np.isnan(ts):
                times.append(ts)
    if len(times) < 2:
        return 0.0, []
    if times[0] + 0.25 < times[1]:
        delta = times[1] - times[0]
        if delta >= 0.8:
            return 2.0, ["タイム大幅改善"]
        return 1.0, ["タイム改善"]
    return 0.0, []


def build_profiles(g: pd.DataFrame, history: pd.DataFrame | None, target, venue: str, race_dist: str = "", field_kg_mean: float | None = None) -> list[dict[str, Any]]:
    n = len(g)
    surface = surface_of(race_dist) or _guess_surface(g, history, target)
    dist_m = dist_meters(race_dist)
    if np.isnan(dist_m) and history is not None and not history.empty:
        # 代表馬の直近距離
        sample = g.iloc[0]
        hh = history[(history["_horse"] == clean_name(sample.get("馬名"))) & (history["_date"] < target)]
        if not hh.empty:
            dist_m = dist_meters(hh.sort_values("_date", ascending=False).iloc[0].get("距離"))
            if not surface:
                surface = surface_of(hh.sort_values("_date", ascending=False).iloc[0].get("距離"))

    profiles = []
    for _, row in g.iterrows():
        finishes = np.array([num(row.get(f"着順{i}")) for i in range(1, 6)], dtype=float)
        pops = np.array([num(row.get(f"人気{i}")) for i in range(1, 6)], dtype=float)
        style = infer_style(finishes, pops)
        l3 = last3f_proxy(finishes, pops)
        trouble = trouble_proxy(finishes, pops)
        lay = layoff_days(history, row.get("馬名"), target)
        wdelta = weight_delta(history, row.get("馬名"), target)
        jbonus = jockey_bonus(history, row.get("騎手"), venue, target)
        tbonus, tplus = time_trend_bonus(row, history, row.get("馬名"), target)
        mbonus, mplus = margin_bonus_from_row(row)
        trbonus, trplus = track_condition_bonus(history, row.get("馬名"), target)
        gbias = gate_bias(venue, row.get("枠"), n, surface)
        cfit, c_reasons = course_distance_fit(history, row.get("馬名"), target, venue, dist_m, surface)

        adj = 0.0
        plus: list[str] = []
        minus: list[str] = []
        # 当日斤量（フィールド平均比）
        try:
            kg = float(num(row.get("斤量")))
        except (TypeError, ValueError):
            kg = float("nan")
        if field_kg_mean is not None and not np.isnan(kg):
            delta = kg - float(field_kg_mean)
            if delta >= 3.0:
                adj -= 2.0
                minus.append("斤量過多")
            elif delta >= 1.5:
                adj -= 1.0
                minus.append("斤量やや重い")
            elif delta <= -2.0:
                adj += 1.2
                plus.append("斤量軽量")
            elif delta <= -1.0:
                adj += 0.6
                plus.append("斤量やや軽い")
        if jbonus >= 2.5:
            adj += jbonus
            plus.append("騎手補正+")
        elif jbonus <= -2:
            adj += jbonus
            minus.append("騎手補正-")
        else:
            adj += jbonus
        if gbias >= 1.5:
            adj += gbias
            plus.append("枠順有利")
        elif gbias <= -1:
            adj += gbias
            minus.append("枠順不利")
        else:
            adj += gbias
        if not np.isnan(lay):
            if lay >= 84:
                adj -= 3.5
                minus.append("長期休養明け")
            elif lay >= 56:
                adj -= 1.5
                minus.append("休み明け")
            elif 14 <= lay <= 35:
                adj += 1.0
                plus.append("間隔良好")
        if not np.isnan(wdelta):
            if abs(wdelta) >= 12:
                adj -= 2.0
                minus.append("馬体重大幅増減")
            elif abs(wdelta) >= 8:
                adj -= 1.0
                minus.append("馬体重変動")
            elif calib_v2_enabled() and wdelta >= 3:
                # +2〜+8kg の BUY は train/holdout とも 0的中。増減不明は触らない。
                adj -= 1.2
                minus.append("馬体重増")
        if trouble >= 0.35:
            adj += 2.0 * trouble  # 不利明けの巻き返し余地
            plus.append("前走不利の可能性")
        if l3 >= 62:
            adj += 1.8
            plus.append("上がり評価高")
        elif l3 <= 40:
            adj -= 1.2
            minus.append("上がり物足りず")
        if tbonus >= 1.0:
            adj += tbonus
            plus.extend(tplus)
        if mbonus >= 1.0:
            adj += mbonus
            plus.extend(mplus)
        if trbonus >= 1.0:
            adj += trbonus
            plus.extend(trplus)
        adj += cfit
        for r in c_reasons:
            plus.append(r)
        tdelta, tplus, tminus = three_calib_adj(style, row.get("枠"), n, row.get("単勝オッズ"))
        adj += tdelta
        plus.extend(tplus)
        minus.extend(tminus)

        # 血統は未取得のため中立（将来拡張用）
        blood = 50.0
        # 厩舎は未取得 → 騎手連動の弱補正のみ
        stable = clamp(50 + jbonus * 0.4)

        profiles.append({
            "style": style,
            "last3f": l3,
            "trouble": trouble,
            "layoff": lay,
            "weight_delta": wdelta,
            "jockey": jbonus,
            "gate": gbias,
            "course_fit": cfit,
            "blood": blood,
            "stable": stable,
            "adj": float(np.clip(adj, -12, 12)),
            "plus": list(dict.fromkeys(plus))[:6],
            "minus": list(dict.fromkeys(minus))[:6],
            "surface": surface,
            "dist_m": dist_m,
        })
    return profiles


def _guess_surface(g, history, target) -> str:
    if history is None or getattr(history, "empty", True):
        return ""
    for _, row in g.iterrows():
        hh = history[(history["_horse"] == clean_name(row.get("馬名"))) & (history["_date"] < target)]
        if not hh.empty:
            return surface_of(hh.sort_values("_date", ascending=False).iloc[0].get("距離"))
    return ""


def predict_pace(profiles: list[dict]) -> dict[str, Any]:
    styles = np.array([p["style"] for p in profiles], dtype=float)
    nige = int((styles <= STYLE_SENKO).sum())
    senko = int(((styles > STYLE_SENKO) & (styles <= 0.5)).sum())
    sashi = int(((styles > 0.5) & (styles < STYLE_OIKOMI)).sum())
    oikomi = int((styles >= STYLE_OIKOMI).sum())
    front = nige + max(0, senko - 1)
    if front >= 4:
        pace = "ハイ"
    elif front <= 1:
        pace = "スロー"
    else:
        pace = "ミドル"

    # 有利度（0-100）
    if pace == "ハイ":
        adv = {"逃げ": 28, "先行": 42, "差し": 72, "追込": 68}
        summary = "先行争いが濃く、差し・追込が届きやすい展開。"
    elif pace == "スロー":
        adv = {"逃げ": 78, "先行": 70, "差し": 38, "追込": 30}
        summary = "ペース沈静が想定され、逃げ・先行残りが有利。"
    else:
        adv = {"逃げ": 52, "先行": 58, "差し": 55, "追込": 48}
        summary = "平均的な流れ。脚質の偏りは小さい。"

    # 有利枠（内/外）
    surface = next((p.get("surface") for p in profiles if p.get("surface")), "")
    if pace == "ハイ" and surface == "芝":
        favor_waku = "外枠"
    elif pace == "スロー" and surface == "ダ":
        favor_waku = "内枠"
    elif surface == "ダ":
        favor_waku = "内枠"
    else:
        favor_waku = "中〜外枠"

    chaos = clamp(35 + abs(front - 2.5) * 10 + (oikomi + sashi) * 2)
    return {
        "想定ペース": pace,
        "逃げ有利度": adv["逃げ"],
        "先行有利度": adv["先行"],
        "差し有利度": adv["差し"],
        "追込有利度": adv["追込"],
        "有利枠": favor_waku,
        "荒れ指数": round(chaos, 1),
        "AI総評": summary,
        "逃げ馬数": nige,
        "先行馬数": senko,
        "差し馬数": sashi,
        "追込馬数": oikomi,
    }


def style_label(style: float) -> str:
    if style <= STYLE_SENKO:
        return "逃げ・先行"
    if style <= 0.5:
        return "先行"
    if style < STYLE_OIKOMI:
        return "差し"
    return "追込"


def simulate_race_stages(g: pd.DataFrame, profiles: list[dict], pace: dict, runs: int = SIM_RUNS):
    """スタート〜ゴールの段階潜在変数で着順を再現するモンテカルロ。"""
    g = g.copy().reset_index(drop=True)
    n_h = len(g)
    base = g["AREru指数"].astype(float).to_numpy() + np.array([p["adj"] for p in profiles], dtype=float)
    cons = g["因子_consistency"].astype(float).to_numpy() if "因子_consistency" in g.columns else np.full(n_h, 50.0)
    sigma = np.clip(18 - (cons * 0.06), 8.0, 18.5)
    styles = np.array([p["style"] for p in profiles], dtype=float)
    last3 = np.array([p["last3f"] for p in profiles], dtype=float)
    trouble = np.array([p["trouble"] for p in profiles], dtype=float)
    gates = np.array([p["gate"] for p in profiles], dtype=float)

    pace_label = pace.get("想定ペース", "ミドル")
    # ペース係数: ハイで先行消耗、スローで先行温存
    if pace_label == "ハイ":
        front_fatigue = 1.15
        closer_boost = 1.12
    elif pace_label == "スロー":
        front_fatigue = 0.88
        closer_boost = 0.90
    else:
        front_fatigue = 1.0
        closer_boost = 1.0
    # PRESET=D: BUYの70%が差しで両期間とも全体より悪い。
    # 差し除外フィルタではなく、直線・道中の差し加点を下げて先行が本命に残れるようにする。
    closer_mid = 3.5 if calib_v2_enabled() else 8.0
    closer_stretch = 3.0 if calib_v2_enabled() else 6.5
    early_style = 22.0 if calib_v2_enabled() else 18.0

    seed = int(abs(hash(str(g.iloc[0]["race_id"]))) % (2**32 - 1))
    rng = np.random.default_rng(seed)

    finish_counts = np.zeros((n_h, n_h), dtype=np.int64)
    corner_sum = np.zeros(n_h, dtype=np.float64)
    # 券種用サンプルのみ保持（全 runs を積むと Render で親+子合算OOM）
    keep_max = min(int(ORDERS_KEEP_MAX), int(runs))
    kept = []
    kept_n = 0

    for start in range(0, runs, SIM_BATCH):
        n = min(SIM_BATCH, runs - start)
        # 能力ゆらぎ
        ability = rng.normal(base, sigma, size=(n, n_h))
        # スタート: 先行力 + 枠 + 不利回復 + ノイズ
        early = (1.0 - styles) * early_style + gates * 1.2 - trouble * 4.0
        start_pos = ability * 0.35 + early + rng.normal(0, 5.5, size=(n, n_h))
        # 位置取り（小さいほど前）
        position = -start_pos + rng.normal(0, 3.0, size=(n, n_h))
        # 道中ペース反応
        mid_score = (
            ability * 0.28
            - position * (0.55 * front_fatigue)
            + (styles[None, :] * closer_mid * closer_boost)
            + rng.normal(0, 4.0, size=(n, n_h))
        )
        # 4角順位（小さいほど前）
        corner_latent = -mid_score + rng.normal(0, 3.5, size=(n, n_h))
        corner_order = np.argsort(corner_latent, axis=1)
        corner_rank = np.empty_like(corner_order)
        rows = np.arange(n)[:, None]
        corner_rank[rows, corner_order] = np.arange(n_h)[None, :]
        corner_sum += corner_rank.sum(axis=0)

        # 直線の伸び: 上がり代理 + 差し脚 + 残り能力
        stretch = (
            ability * 0.45
            + (last3[None, :] - 50.0) * 0.22
            + styles[None, :] * closer_stretch * closer_boost
            - corner_rank * (1.1 if pace_label == "スロー" else 0.55)
            + rng.normal(0, 5.0, size=(n, n_h))
        )
        order = np.argsort(-stretch, axis=1)
        for pos in range(n_h):
            finish_counts[:, pos] += np.bincount(order[:, pos], minlength=n_h)
        # バッチ一時配列を早めに解放
        del ability, start_pos, position, mid_score, corner_latent, corner_order, corner_rank, stretch
        if kept_n < keep_max:
            take = min(n, keep_max - kept_n)
            kept.append(order[:take].copy())
            kept_n += take
        del order

    order_all = np.vstack(kept) if kept else np.zeros((0, n_h), dtype=np.int64)
    del kept
    return order_all, finish_counts, corner_sum / max(runs, 1)


def circle_ban(ban) -> str:
    circled = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
    try:
        n = int(float(ban))
        if 1 <= n <= len(circled):
            return circled[n - 1]
    except Exception:
        pass
    return str(ban or "－")


def stars_from_ev(ev: float | None, hit: float | None) -> str:
    if ev is None:
        return "★★☆☆☆"
    e = float(ev)
    h = float(hit or 0)
    score = 0
    if e >= 100:
        score += 1
    if e >= 115:
        score += 1
    if e >= 130:
        score += 1
    if e >= 150:
        score += 1
    if h >= 12 or e >= 170:
        score += 1
    score = max(1, min(5, score))
    return "★" * score + "☆" * (5 - score)
