"""キャッシュ済み過去走から馬ごとの特徴量を作る。

設計上の約束ごと。

* **未来情報を使わない。** 対象レースの日付 D に対し、`年月日 < D` の行しか見ない。
  スピード指数の基準値も D 未満の行だけから作る。
* **無いものは NaN。** 取得できなかった項目を 0 点に置き換えて減点することはしない。
  各特徴量には「何走ぶんのデータで計算したか」を示す件数列を必ず添える。
* 重み付けはここではしない。特徴量を独立に出すところまでが責務。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import netkeiba_client as nk

HORSE_CACHE = nk.CACHE
RACE_IDS_CACHE = Path("data/cache/race_horse_ids")

# 回り。帯広(ばんえい)は直線なのでどちらでもない。
RIGHT_HANDED = {
    "中山", "阪神", "京都", "札幌", "函館", "福島", "小倉",
    "大井", "船橋", "門別", "金沢", "笠松", "名古屋", "園田", "姫路",
    "高知", "佐賀",
}
LEFT_HANDED = {"東京", "中京", "新潟", "川崎", "浦和", "盛岡", "水沢"}

GOING_ORDER = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}

# クラス水準。数字が大きいほど上。地方と中央は体系が違うので別表にする。
JRA_CLASS = [
    (r"(G1|GI(?![IV])|Ｇ1|ジーワン)", 10.0),
    (r"(G2|GII(?!I)|Ｇ2)", 9.0),
    (r"(G3|GIII|Ｇ3)", 8.0),
    (r"(OP|オープン|\(L\)|リステッド)", 7.0),
    (r"(3勝|1600万)", 6.0),
    (r"(2勝|1000万)", 5.0),
    (r"(1勝|500万)", 4.0),
    (r"新馬", 2.0),
    (r"未勝利", 1.0),
]
NAR_CLASS = [
    (r"(JpnI(?![IV])|Jpn1)", 10.0),
    (r"(JpnII(?!I)|Jpn2)", 9.0),
    (r"(JpnIII|Jpn3)", 8.0),
    (r"A1", 7.0),
    (r"A2", 6.5),
    (r"A3", 6.0),
    (r"(^|[^A-Za-z])A[^0-9A-Za-z]", 6.5),
    (r"B1", 5.5),
    (r"B2", 5.0),
    (r"B3", 4.5),
    (r"(^|[^A-Za-z])B[^0-9A-Za-z]", 5.0),
    (r"C1", 4.0),
    (r"C2", 3.5),
    (r"C3", 3.0),
    (r"(^|[^A-Za-z])C[^0-9A-Za-z]", 3.5),
    (r"新馬", 2.0),
    (r"未勝利", 1.0),
    # 地方の2・3歳限定戦は名前にクラスが入らない（「3歳二」「○○賞(3歳)」など）。
    # 古馬のA/B/Cとは別体系なので、未勝利の上・C級の下に置く仮の序列とする。
    # この並びが妥当かどうかは特徴量の予測力テストで確かめる前提。
    (r"3歳", 2.8),
    (r"2歳", 2.4),
]
TROUBLE_WORDS = ("出遅", "不利", "躓", "つまず", "挟ま", "はさま", "詰ま", "ふらつ", "故障", "落馬")


def _to_float(x) -> float:
    try:
        v = float(str(x).strip())
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_time(s) -> float:
    """'1:30.9' / '58.4' を秒に。読めなければ NaN。"""
    t = str(s or "").strip()
    if not t:
        return np.nan
    m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", t)
    if m:
        return float(m.group(1)) * 60.0 + float(m.group(2))
    return _to_float(t)


def parse_distance(s) -> tuple[str, float]:
    """'芝2500' → ('芝', 2500.0)。"""
    t = str(s or "").strip()
    m = re.match(r"^(芝|ダ|障)\s*(\d{3,4})", t)
    if not m:
        return ("", np.nan)
    return (m.group(1), float(m.group(2)))


def parse_going(s) -> float:
    t = str(s or "").strip()
    if not t:
        return np.nan
    for k, v in GOING_ORDER.items():
        if t.startswith(k):
            return float(v)
    return np.nan


def parse_passing(s) -> list[float]:
    t = str(s or "").strip()
    if not t:
        return []
    out = []
    for part in re.split(r"[-ー－]", t):
        v = _to_float(part)
        if np.isfinite(v):
            out.append(v)
    return out


def parse_pace(s) -> tuple[float, float]:
    """'30.0-35.4' → (前半3F, 後半3F)。レース全体のペース。"""
    t = str(s or "").strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*[-ー－]\s*(\d+(?:\.\d+)?)$", t)
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))


def parse_weight(s) -> tuple[float, float]:
    """'438(+2)' → (438.0, 2.0)。"""
    t = str(s or "").strip()
    m = re.match(r"^(\d{3})\s*\(([-+]?\d+)\)", t)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    v = _to_float(re.sub(r"\D", "", t) or "")
    return (v, np.nan)


def parse_finish(s) -> float:
    """着順。中止/除外/取消は NaN。"""
    t = str(s or "").strip()
    m = re.match(r"^(\d+)", t)
    return float(m.group(1)) if m else np.nan


def class_level(race_name: str, source_hint: str = "") -> float:
    name = str(race_name or "")
    if not name:
        return np.nan
    table = NAR_CLASS if source_hint == "nar" else JRA_CLASS
    for pat, lv in table:
        if re.search(pat, name):
            return lv
    other = JRA_CLASS if source_hint == "nar" else NAR_CLASS
    for pat, lv in other:
        if re.search(pat, name):
            return lv
    return np.nan


def turn_of(venue: str) -> float:
    """右回り=1 / 左回り=-1 / 不明・直線=NaN。"""
    v = nk.normalize_venue_name(venue)
    if v in RIGHT_HANDED:
        return 1.0
    if v in LEFT_HANDED:
        return -1.0
    return np.nan


def _venue_from_place(s) -> str:
    """'5中山8' や '浦和' から会場名を取り出す。"""
    t = re.sub(r"\d", "", str(s or "")).strip()
    return nk.normalize_venue_name(t)


def load_history_frame(horse_ids=None) -> pd.DataFrame:
    """キャッシュ済み戦績を1枚の DataFrame にして型を付ける。"""
    if horse_ids is None:
        paths = sorted(HORSE_CACHE.glob("*.json"))
    else:
        paths = [HORSE_CACHE / f"{h}.json" for h in sorted(set(horse_ids))]
    recs: list[dict] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        hid = p.stem
        for r in rows:
            if isinstance(r, dict):
                r = dict(r)
                r["horse_id"] = hid
                recs.append(r)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    for c in nk.HISTORY_ROW_KEYS:
        if c not in df.columns:
            df[c] = ""

    df["date"] = pd.to_datetime(df["年月日"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["venue"] = df["場"].map(_venue_from_place)
    surf_dist = df["距離"].map(parse_distance)
    df["surface"] = [s for s, _ in surf_dist]
    df["dist"] = [d for _, d in surf_dist]
    df["going"] = df["馬場"].map(parse_going)
    df["n_runners"] = df["頭数"].map(_to_float)
    df["pop"] = df["人気"].map(_to_float)
    df["finish"] = df["着順"].map(parse_finish)
    df["time_sec"] = df["タイム"].map(parse_time)
    df["last3f"] = df["上り"].map(_to_float)
    pace = df["ペース"].map(parse_pace)
    df["race_first3f"] = [a for a, _ in pace]
    df["race_last3f"] = [b for _, b in pace]
    passing = df["通過"].map(parse_passing)
    df["pass_first"] = [p[0] if p else np.nan for p in passing]
    df["pass_last"] = [p[-1] if p else np.nan for p in passing]
    df["margin"] = df["着差"].map(_to_float)
    bw = df["馬体重"].map(parse_weight)
    df["body_weight"] = [a for a, _ in bw]
    df["body_weight_diff"] = [b for _, b in bw]
    df["carried"] = df["斤量"].map(_to_float)
    df["turn"] = df["venue"].map(turn_of)
    df["is_nar"] = (~df["venue"].isin(
        {"札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"}
    )).astype(float)
    df["class_lv"] = [
        class_level(nm, "nar" if nar else "jra")
        for nm, nar in zip(df["レース名"], df["is_nar"] > 0)
    ]
    df["trouble"] = [
        1.0 if any(w in str(b or "") for w in TROUBLE_WORDS) else
        (0.0 if str(b or "").strip() != "" else np.nan)
        for b in df["備考"]
    ]
    # 着順の相対位置。1着=0, 最下位=1。
    df["rel_finish"] = (df["finish"] - 1.0) / (df["n_runners"] - 1.0).replace(0, np.nan)
    df["rel_pop"] = (df["pop"] - 1.0) / (df["n_runners"] - 1.0).replace(0, np.nan)
    # 道中の位置取り。0=先頭、1=最後方。
    df["pos_early"] = (df["pass_first"] - 1.0) / (df["n_runners"] - 1.0).replace(0, np.nan)
    df["pos_late"] = (df["pass_last"] - 1.0) / (df["n_runners"] - 1.0).replace(0, np.nan)
    df["dist_bucket"] = (df["dist"] / 200.0).round() * 200.0
    return df


# レース前半（残り3Fまで）に要した時間。上がり3Fが98%取れるのでほぼ全場で計算できる。
def add_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["early_sec"] = df["time_sec"] - df["last3f"]
    # 前半区間の距離。600m以下のレースは前半が存在しないので除く。
    early_dist = df["dist"] - 600.0
    df.loc[early_dist <= 0, "early_sec"] = np.nan
    return df


BASELINE_COLUMNS = [
    "venue", "surface", "dist", "going",
    "t_med", "t_std", "t_n", "l_med", "l_std", "l_n", "e_med", "e_std", "e_n",
]


def course_baselines(hist: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """(会場, 芝ダ, 距離, 馬場) ごとの走破タイム・上がり・前半区間の基準。

    `before` より前の行だけで作るので、後の開催の情報が前に漏れることはない。
    """
    src = hist[(hist["date"] < before) & (hist["dist"] > 0)]
    if src.empty:
        return pd.DataFrame(columns=BASELINE_COLUMNS)
    keys = ["venue", "surface", "dist", "going"]
    g = src.groupby(keys)
    out = g.agg(
        t_med=("time_sec", "median"), t_std=("time_sec", "std"), t_n=("time_sec", "count"),
        l_med=("last3f", "median"), l_std=("last3f", "std"), l_n=("last3f", "count"),
        e_med=("early_sec", "median"), e_std=("early_sec", "std"), e_n=("early_sec", "count"),
    ).reset_index()
    # 標本が薄い区分の基準は信用しない。
    for med, std, n in (("t_med", "t_std", "t_n"), ("l_med", "l_std", "l_n"),
                        ("e_med", "e_std", "e_n")):
        thin = out[n] < 5
        out.loc[thin, [med, std]] = np.nan
    return out


def pace_baseline(hist: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """(会場, 芝ダ, 距離) ごとの前半3F基準。ペース列が公開される場だけで使える。"""
    src = hist[(hist["date"] < before) & hist["race_first3f"].notna() & (hist["dist"] > 0)]
    if src.empty:
        return pd.DataFrame(columns=["venue", "surface", "dist", "p_med", "p_std", "p_n"])
    g = src.groupby(["venue", "surface", "dist"])["race_first3f"]
    out = g.agg(p_med="median", p_std="std", p_n="size").reset_index()
    return out[out["p_n"] >= 5]


def race_pace_table(hist: pd.DataFrame, before: pd.Timestamp,
                    base: pd.DataFrame) -> pd.DataFrame:
    """各過去レースの「前半が速かったか」を、出走各馬の前半区間タイムから推定する。

    netkeiba のペース列は中央と南関東くらいしか公開されないので、
    それ以外の場でも使える代用値を作る。同一レースに属する行
    （同じ日・同じ場・同じR）の前半区間偏差を平均したもの。
    """
    src = hist[hist["date"] < before]
    if src.empty or base.empty:
        return pd.DataFrame(columns=["date", "venue", "race_no", "race_pace_fig", "race_pace_n"])
    m = src.merge(base[["venue", "surface", "dist", "going", "e_med", "e_std"]],
                  on=["venue", "surface", "dist", "going"], how="left")
    std = m["e_std"].replace(0, np.nan)
    # プラス＝基準より前半が速い＝ハイペース。
    m["e_fig"] = (-(m["early_sec"] - m["e_med"]) / std).clip(-4, 4)
    m["race_no"] = pd.to_numeric(m["レース"], errors="coerce")
    m = m[m["e_fig"].notna() & m["race_no"].notna()]
    if m.empty:
        return pd.DataFrame(columns=["date", "venue", "race_no", "race_pace_fig", "race_pace_n"])
    out = m.groupby(["date", "venue", "race_no"])["e_fig"].agg(
        race_pace_fig="mean", race_pace_n="size").reset_index()
    return out


def attach_baselines(past: pd.DataFrame, base: pd.DataFrame, pac: pd.DataFrame,
                     race_pace: pd.DataFrame | None = None) -> pd.DataFrame:
    """過去走に基準値を結合して、スピード指数・上がり指数・ペース偏差を出す。"""
    out = past
    if not base.empty:
        out = out.merge(base, on=["venue", "surface", "dist", "going"], how="left")
    else:
        for c in BASELINE_COLUMNS[4:]:
            out[c] = np.nan
    if not pac.empty:
        out = out.merge(pac, on=["venue", "surface", "dist"], how="left")
    else:
        out = out.assign(p_med=np.nan, p_std=np.nan, p_n=np.nan)

    # 速い＝プラスになるよう符号を反転。斤量差はここでは補正しない。
    out["speed_fig"] = (-(out["time_sec"] - out["t_med"])
                        / out["t_std"].replace(0, np.nan)).clip(-4, 4)
    # 上がり3Fがそのコース・馬場の標準よりどれだけ速かったか。ほぼ全場で取れる。
    out["last3f_fig"] = (-(out["last3f"] - out["l_med"])
                         / out["l_std"].replace(0, np.nan)).clip(-4, 4)
    # 自身の前半区間の速さ。脚質とペース耐性を見るのに使う。
    out["early_fig"] = (-(out["early_sec"] - out["e_med"])
                        / out["e_std"].replace(0, np.nan)).clip(-4, 4)
    # ペース列が公開される場での実測ペース偏差。
    out["pace_dev_true"] = (-(out["race_first3f"] - out["p_med"])
                            / out["p_std"].replace(0, np.nan)).clip(-4, 4)
    # 上がりがレース全体の後半3Fよりどれだけ速かったか（秒）。公開場のみ。
    out["last3f_edge"] = (out["race_last3f"] - out["last3f"]).clip(-5, 5)

    out["race_no"] = pd.to_numeric(out["レース"], errors="coerce")
    if race_pace is not None and not race_pace.empty:
        out = out.merge(race_pace, on=["date", "venue", "race_no"], how="left")
    else:
        out["race_pace_fig"] = np.nan
        out["race_pace_n"] = np.nan
    # 実測が取れるならそちらを優先し、取れない場では代用値を使う。
    out["pace_dev"] = out["pace_dev_true"].where(
        out["pace_dev_true"].notna(), out["race_pace_fig"])
    return out


def _wmean(values: pd.Series, weights: np.ndarray) -> float:
    v = values.to_numpy(dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan
    w = weights[: len(v)][ok]
    if w.sum() <= 0:
        return np.nan
    return float((v[ok] * w).sum() / w.sum())


def _content_score(row) -> float:
    """1走の「内容」。着順だけでなく、位置取りと上がりとペースを見る。

    着順そのものではなく「そのレースでどれだけ強い走りをしたか」を測る。
    材料が無い項目は平均扱いにせず、その項目を外して残りで平均する。
    """
    parts: list[float] = []
    rel = row.get("rel_finish")
    if pd.notna(rel):
        parts.append(1.0 - 2.0 * float(rel))  # 1着=+1, 最下位=-1
    spd = row.get("speed_fig")
    if pd.notna(spd):
        parts.append(float(np.clip(spd, -2, 2)) / 2.0)
    close = row.get("last3f_fig")
    if pd.notna(close):
        parts.append(float(np.clip(close, -2, 2)) / 2.0)
    # 後方から上がって着順をまとめた＝展開不利を克服した、と読む。
    pe, pl = row.get("pos_early"), row.get("rel_finish")
    if pd.notna(pe) and pd.notna(pl):
        parts.append(float(np.clip(pe - pl, -1, 1)))
    if not parts:
        return np.nan
    return float(np.mean(parts))


def build_features(
    past: pd.DataFrame,
    target: dict,
) -> dict:
    """1頭ぶんの特徴量。past は既に「対象日より前」に絞られていること。

    target は当日の条件 (surface, dist, venue, going, class_lv)。
    当日条件は発走前に判明している情報だけを使う。
    """
    out: dict[str, float] = {}
    out["過去走数"] = float(len(past))
    if past.empty:
        return out

    past = past.sort_values("date", ascending=False)
    w = np.array([1.0, 0.82, 0.65, 0.48, 0.34, 0.24, 0.17, 0.12, 0.08, 0.06])
    w = np.resize(w, max(len(past), len(w)))

    content = past.apply(_content_score, axis=1)
    prev = past.iloc[0]

    # 1. 前走内容
    out["前走内容"] = _content_score(prev)
    out["前走着順"] = float(prev.get("rel_finish")) if pd.notna(prev.get("rel_finish")) else np.nan
    out["前走上がり"] = float(prev["last3f_fig"]) if pd.notna(prev.get("last3f_fig")) else np.nan
    out["前走位置取り"] = float(prev["pos_early"]) if pd.notna(prev.get("pos_early")) else np.nan
    out["前走ペース"] = float(prev["pace_dev"]) if pd.notna(prev.get("pace_dev")) else np.nan
    out["前走不利"] = float(prev["trouble"]) if pd.notna(prev.get("trouble")) else np.nan
    out["前走スピード"] = float(prev["speed_fig"]) if pd.notna(prev.get("speed_fig")) else np.nan
    d0 = prev.get("date")
    out["休養日数"] = float((target["date"] - d0).days) if pd.notna(d0) else np.nan

    # 2. 近5走内容
    last5 = past.head(5)
    c5 = content.head(5)
    out["近5走内容"] = _wmean(c5, w)
    out["近5走内容件数"] = float(c5.notna().sum())
    out["近5走スピード"] = _wmean(last5["speed_fig"], w)
    out["近5走スピード最高"] = (
        float(last5["speed_fig"].max()) if last5["speed_fig"].notna().any() else np.nan
    )
    out["近5走安定度"] = (
        float(-last5["rel_finish"].std()) if last5["rel_finish"].notna().sum() >= 3 else np.nan
    )

    # 3. 上がり性能
    out["上がり性能"] = _wmean(past["last3f_fig"].head(5), w)
    out["上がり性能件数"] = float(past["last3f_fig"].head(5).notna().sum())
    out["上がり最速"] = (
        float(past["last3f_fig"].head(10).max())
        if past["last3f_fig"].head(10).notna().any() else np.nan
    )
    out["前半性能"] = _wmean(past["early_fig"].head(5), w)

    t_surf, t_dist = target.get("surface"), target.get("dist")
    t_venue, t_going = target.get("venue"), target.get("going")

    def _subset_score(mask: pd.Series, label: str, min_n: int = 1) -> None:
        sub = past[mask]
        out[f"{label}件数"] = float(len(sub))
        if len(sub) < min_n:
            out[label] = np.nan
            return
        sc = sub.apply(_content_score, axis=1)
        out[label] = _wmean(sc, w) if sc.notna().any() else np.nan

    # 4. 同距離適性（±100m を同距離とみなす）
    if pd.notna(t_dist):
        _subset_score((past["dist"] - float(t_dist)).abs() <= 100, "同距離適性")
    else:
        out["同距離適性"], out["同距離適性件数"] = np.nan, 0.0

    # 5. 同コース適性（同じ会場かつ同じ芝ダ）
    if t_venue and t_surf:
        _subset_score((past["venue"] == t_venue) & (past["surface"] == t_surf), "同コース適性")
    else:
        out["同コース適性"], out["同コース適性件数"] = np.nan, 0.0

    # 6. 芝/ダート適性
    if t_surf:
        _subset_score(past["surface"] == t_surf, "芝ダ適性")
    else:
        out["芝ダ適性"], out["芝ダ適性件数"] = np.nan, 0.0

    # 7. 馬場適性（良馬場かそれ以外か、で分ける）
    if pd.notna(t_going):
        wet = float(t_going) >= 1
        _subset_score((past["going"] >= 1) if wet else (past["going"] == 0), "馬場適性")
    else:
        out["馬場適性"], out["馬場適性件数"] = np.nan, 0.0

    # 8. ペース適性。ハイペース戦とスローペース戦で内容がどう変わるか。
    hi = past[past["pace_dev"] >= 0.5]
    lo = past[past["pace_dev"] <= -0.5]
    s_hi = hi.apply(_content_score, axis=1) if len(hi) else pd.Series(dtype=float)
    s_lo = lo.apply(_content_score, axis=1) if len(lo) else pd.Series(dtype=float)
    out["ハイペース適性"] = _wmean(s_hi, w) if len(s_hi) else np.nan
    out["スローペース適性"] = _wmean(s_lo, w) if len(s_lo) else np.nan
    out["ハイペース件数"] = float(len(hi))
    out["スローペース件数"] = float(len(lo))
    if pd.notna(out["ハイペース適性"]) and pd.notna(out["スローペース適性"]):
        out["ペース適性差"] = out["ハイペース適性"] - out["スローペース適性"]
    else:
        out["ペース適性差"] = np.nan
    out["平均位置取り"] = _wmean(past["pos_early"].head(5), w)

    # 9. 右回り/左回り適性
    t_turn = turn_of(t_venue) if t_venue else np.nan
    if pd.notna(t_turn):
        _subset_score(past["turn"] == t_turn, "回り適性")
    else:
        out["回り適性"], out["回り適性件数"] = np.nan, 0.0

    # 10. 距離延長/短縮
    if pd.notna(t_dist) and pd.notna(prev.get("dist")):
        delta = float(t_dist) - float(prev["dist"])
        out["距離変化"] = delta
        # 過去に同方向の距離変化を経験したときの内容。
        prev_dists = past["dist"].shift(-1)  # 日付降順なので shift(-1) が「その前走」
        chg = past["dist"] - prev_dists
        if abs(delta) >= 100:
            same_dir = (chg * delta) > 0
            _subset_score(same_dir & chg.abs().ge(100), "距離変化適性")
        else:
            out["距離変化適性"], out["距離変化適性件数"] = np.nan, 0.0
    else:
        out["距離変化"], out["距離変化適性"], out["距離変化適性件数"] = np.nan, np.nan, 0.0

    # 11. クラス変化
    t_class = target.get("class_lv")
    out["前走クラス"] = float(prev["class_lv"]) if pd.notna(prev.get("class_lv")) else np.nan
    out["最高経験クラス"] = (
        float(past["class_lv"].max()) if past["class_lv"].notna().any() else np.nan
    )
    if pd.notna(t_class) and pd.notna(out["前走クラス"]):
        out["クラス変化"] = float(t_class) - out["前走クラス"]
    else:
        out["クラス変化"] = np.nan
    if pd.notna(t_class):
        _subset_score((past["class_lv"] - float(t_class)).abs() <= 0.5, "同クラス適性")
        up = past[past["class_lv"] >= float(t_class)]
        out["格上経験"] = float(len(up))
    else:
        out["同クラス適性"], out["同クラス適性件数"], out["格上経験"] = np.nan, 0.0, np.nan

    return out


FEATURE_COLUMNS = [
    "過去走数", "前走内容", "前走着順", "前走上がり", "前走位置取り", "前走ペース",
    "前走不利", "前走スピード", "休養日数",
    "近5走内容", "近5走内容件数", "近5走スピード", "近5走スピード最高", "近5走安定度",
    "上がり性能", "上がり性能件数", "上がり最速", "前半性能",
    "同距離適性", "同距離適性件数", "同コース適性", "同コース適性件数",
    "芝ダ適性", "芝ダ適性件数", "馬場適性", "馬場適性件数",
    "ハイペース適性", "スローペース適性", "ハイペース件数", "スローペース件数",
    "ペース適性差", "平均位置取り",
    "回り適性", "回り適性件数",
    "距離変化", "距離変化適性", "距離変化適性件数",
    "前走クラス", "最高経験クラス", "クラス変化", "同クラス適性", "同クラス適性件数",
    "格上経験",
]
