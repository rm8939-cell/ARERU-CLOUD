#!/usr/bin/env python3
"""バックテスト対象レースについて、過去走特徴量のパネルを作る。

未来情報の遮断は2段構えにしている。

  1. 各馬の過去走は `年月日 < 対象日` のものだけを使う。
  2. スピード指数・ペース偏差の基準値も `年月日 < 対象日` の行だけから作り直す。
     対象日ごとに基準を作り直すので、後の開催の情報が前の開催に漏れない。

当日条件（芝ダ・距離・馬場・会場・クラス）は発走前に判明する情報なので使う。
バックテストでは出走各馬の戦績から「対象日・対象会場・対象R」の行を引いて復元する。
着順やタイムといった結果側の列は一切取り出さない。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netkeiba_client as nk  # noqa: E402
import past_features as pf  # noqa: E402

OUT_DIR = Path("data/past_features")


def venue_of_race_id(rid: str) -> str:
    code = str(rid)[4:6]
    return nk.VENUE_CODES.get(code, "")


def race_no_of(rid: str) -> float:
    try:
        return float(str(rid)[-2:])
    except (TypeError, ValueError):
        return np.nan


def load_runner_panel(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"入力が無い: {pattern}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if "date" not in d.columns:
            d["date"] = Path(f).stem.replace("runner_", "")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["race_id"] = out["race_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out[out["date"].notna()].copy()


def load_name_to_id() -> dict[str, dict[str, str]]:
    m: dict[str, dict[str, str]] = {}
    for p in pf.RACE_IDS_CACHE.glob("*.json"):
        try:
            m[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return m


def target_conditions(hist_rows: pd.DataFrame, rid: str, when: pd.Timestamp) -> dict:
    """対象レースの発走前条件を、出走各馬の戦績行から復元する。"""
    venue = venue_of_race_id(rid)
    rno = race_no_of(rid)
    sel = hist_rows[(hist_rows["date"] == when)]
    if venue:
        sel = sel[sel["venue"] == venue]
    if np.isfinite(rno):
        exact = sel[pd.to_numeric(sel["レース"], errors="coerce") == rno]
        if not exact.empty:
            sel = exact
    out = {"date": when, "venue": venue, "surface": "", "dist": np.nan,
           "going": np.nan, "class_lv": np.nan, "n_runners": np.nan}
    if sel.empty:
        return out
    for key, col in (("surface", "surface"), ("dist", "dist"),
                     ("going", "going"), ("n_runners", "n_runners")):
        vals = sel[col].dropna()
        vals = vals[vals != ""] if col == "surface" else vals
        if not vals.empty:
            out[key] = vals.mode().iloc[0]
    names = sel["レース名"].astype(str)
    names = names[names.str.strip() != ""]
    if not names.empty:
        nar = bool(sel["is_nar"].mean() > 0.5)
        out["class_lv"] = pf.class_level(names.mode().iloc[0], "nar" if nar else "jra")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner-glob",
                    default="data/quality_backtest_cache/prod_old/runner_*.csv")
    ap.add_argument("--out", default="data/past_features/panel.csv")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runners = load_runner_panel(args.runner_glob)
    print(f"対象 {runners['race_id'].nunique()}R / {len(runners)}頭")

    name_map = load_name_to_id()
    print(f"馬ID解決済みレース {len(name_map)}")

    def resolve(row) -> str:
        return (name_map.get(row["race_id"]) or {}).get(str(row["馬名"]), "")

    runners["horse_id"] = runners.apply(resolve, axis=1)
    hit = (runners["horse_id"] != "").mean()
    print(f"馬ID紐付け率 {hit*100:.1f}%")

    ids = sorted({h for h in runners["horse_id"] if h})
    print(f"読み込む戦績 {len(ids)} 頭ぶん")
    hist = pf.load_history_frame(ids)
    if hist.empty:
        raise SystemExit("戦績キャッシュが空")
    hist = pf.add_split_columns(hist)
    print(f"戦績行 {len(hist)} / 馬 {hist['horse_id'].nunique()}")

    by_horse = {h: g for h, g in hist.groupby("horse_id")}
    rows: list[dict] = []
    for when, day in runners.groupby("date"):
        base = pf.course_baselines(hist, when)
        pac = pf.pace_baseline(hist, when)
        rpace = pf.race_pace_table(hist, when, base)
        day_ids = {h for h in day["horse_id"] if h}
        day_hist = hist[hist["horse_id"].isin(day_ids)]
        cond_cache: dict[str, dict] = {}
        for rid, race in day.groupby("race_id"):
            r_ids = {h for h in race["horse_id"] if h}
            if rid not in cond_cache:
                cond_cache[rid] = target_conditions(
                    day_hist[day_hist["horse_id"].isin(r_ids)], rid, when
                )
            cond = cond_cache[rid]
            for _, r in race.iterrows():
                rec = {"race_id": rid, "馬名": r["馬名"], "date": when,
                        "horse_id": r["horse_id"],
                        "当日芝ダ": cond["surface"], "当日距離": cond["dist"],
                        "当日馬場": cond["going"], "当日会場": cond["venue"],
                        "当日クラス": cond["class_lv"]}
                hid = r["horse_id"]
                if not hid or hid not in by_horse:
                    rows.append(rec | {"過去走数": 0.0})
                    continue
                past = by_horse[hid]
                past = past[past["date"] < when]
                if past.empty:
                    rows.append(rec | {"過去走数": 0.0})
                    continue
                past = pf.attach_baselines(past.copy(), base, pac, rpace)
                feats = pf.build_features(past, cond)
                rows.append(rec | feats)
        print(f"  {when.date()} {day['race_id'].nunique()}R 完了", flush=True)

    out = pd.DataFrame(rows)
    for c in pf.FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out.to_csv(args.out, index=False)
    print(f"\n書き出し {args.out}  {len(out)}行")
    cov = {c: float(out[c].notna().mean() * 100) for c in pf.FEATURE_COLUMNS}
    print(json.dumps({k: round(v, 1) for k, v in cov.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
