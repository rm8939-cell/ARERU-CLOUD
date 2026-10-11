"""`data/cache/race_results` に貯めたレース結果を、特徴量計算に使える形に読む。

馬ごとの戦績ページから作る `past_features.load_history_frame` と違い、
こちらは出走全頭が必ず揃う。そのため

* 市場確率（控除率を抜いた 1/オッズ の正規化）が正しく計算できる
* レースの前半ペースを、全頭の前半区間タイムから推定できる

という2点が決定的に良い。列名と型は load_history_frame と揃えてあるので、
past_features の基準値計算・特徴量計算をそのまま使える。

注意: 単勝オッズは結果ページの確定オッズ。本番は発走前オッズなので、
ここでの市場ベースラインは本番よりわずかに鋭い側に寄る。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import past_features as pf

RACE_CACHE = Path("data/cache/race_results")

TURN_MAP = {"右": 1.0, "左": -1.0, "直": np.nan, "直線": np.nan}

JRA_VENUES = {"札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_runs(paths=None, *, limit: int | None = None) -> pd.DataFrame:
    """1行＝1頭の1走。past_features が期待する型付き列をそのまま持たせる。"""
    files = sorted(RACE_CACHE.glob("*.json")) if paths is None else list(paths)
    if limit:
        files = files[:limit]
    recs: list[dict] = []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        horses = d.get("horses") or []
        if not horses:
            continue
        head = {
            "race_id": d.get("race_id"), "年月日": d.get("date"),
            "venue": d.get("venue") or "", "レース": d.get("race_no"),
            "レース名": d.get("レース名") or "", "surface": d.get("芝ダ") or "",
            "dist": d.get("距離"), "going_raw": d.get("馬場") or "",
            "turn_raw": d.get("回り") or "", "天候": d.get("天候") or "",
            "n_runners": len(horses),
        }
        for h in horses:
            recs.append({**head, **h})
    if not recs:
        return pd.DataFrame()

    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["年月日"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["dist"] = _num(df["dist"])
    df["going"] = df["going_raw"].map(pf.parse_going)
    df["turn"] = df["turn_raw"].map(lambda s: TURN_MAP.get(str(s).strip(), np.nan))
    df["pop"] = _num(df["人気"])
    df["odds"] = _num(df["単勝オッズ"])
    df["finish"] = df["着順"].map(pf.parse_finish)
    df["time_sec"] = df["タイム"].map(pf.parse_time)
    df["last3f"] = _num(df["上り"])
    passing = df["通過"].map(pf.parse_passing)
    df["pass_first"] = [p[0] if p else np.nan for p in passing]
    df["pass_last"] = [p[-1] if p else np.nan for p in passing]
    df["margin"] = _num(df["着差"])
    bw = df["馬体重"].map(pf.parse_weight)
    df["body_weight"] = [a for a, _ in bw]
    df["body_weight_diff"] = [b for _, b in bw]
    df["carried"] = _num(df["斤量"])
    df["is_nar"] = (~df["venue"].isin(JRA_VENUES)).astype(float)
    df["class_lv"] = [
        pf.class_level(nm, "nar" if nar else "jra")
        for nm, nar in zip(df["レース名"], df["is_nar"] > 0)
    ]
    df["trouble"] = [
        1.0 if any(w in str(b or "") for w in pf.TROUBLE_WORDS) else
        (0.0 if str(b or "").strip() != "" else np.nan)
        for b in df.get("備考", pd.Series([""] * len(df)))
    ]
    denom = (df["n_runners"] - 1.0).replace(0, np.nan)
    df["rel_finish"] = (df["finish"] - 1.0) / denom
    df["rel_pop"] = (df["pop"] - 1.0) / denom
    df["pos_early"] = (df["pass_first"] - 1.0) / denom
    df["pos_late"] = (df["pass_last"] - 1.0) / denom
    df["dist_bucket"] = (df["dist"] / 200.0).round() * 200.0
    # ペース列はレース結果ページには無い。全頭の前半区間から推定する。
    df["race_first3f"] = np.nan
    df["race_last3f"] = np.nan
    df["馬場"] = df["going_raw"]
    df["場"] = df["venue"]
    df["頭数"] = df["n_runners"]
    return df


def market_prob(df: pd.DataFrame) -> pd.Series:
    """控除率を抜いた市場勝率。全頭のオッズが揃うレースだけ値が入る。"""
    o = df["odds"]
    ok = df.groupby("race_id")["odds"].transform(
        lambda s: s.notna().all() and bool((s > 1.0).all()))
    impl = 1.0 / o
    impl = impl.where(ok.fillna(False))
    total = impl.groupby(df["race_id"]).transform("sum")
    return impl / total


def usable_races(df: pd.DataFrame, *, min_runners: int = 5) -> pd.DataFrame:
    """確率モデルの評価に使えるレースだけ残す。

    * 全頭に単勝オッズがある（市場ベースラインが作れる）
    * 1着が1頭だけ確定している
    * 出走頭数が min_runners 以上
    """
    d = df.copy()
    d["market"] = market_prob(d)
    g = d.groupby("race_id")
    ok_market = g["market"].transform(lambda s: s.notna().all())
    ok_win = g["finish"].transform(lambda s: (s == 1).sum() == 1)
    ok_n = g["馬名"].transform("size") >= min_runners
    return d[ok_market.fillna(False) & ok_win.fillna(False) & ok_n].copy()
