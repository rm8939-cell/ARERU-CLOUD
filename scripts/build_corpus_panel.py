#!/usr/bin/env python3
"""大きいレースコーパスから、評価用の特徴量パネルを作る。

未来情報の遮断は前と同じ2段構え。

  1. 各馬の過去走は 対象レースの日付より前のものだけ
  2. スピード・上がり・前半区間の基準値も、対象日より前の走だけから作り直す

基準値は日ごとに作り直すと重いので、月の頭で区切って作り、その月の
レースにはその時点の基準を使う。月初の基準は前月末までの走しか見ないので
未来は入らない（基準が少し古くなる方向にしか誤らない）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import past_features as pf  # noqa: E402
import race_corpus as rc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-from", default="2026-04-01",
                    help="この日以降のレースを評価対象にする。それ以前は履歴の助走に使う")
    ap.add_argument("--eval-to", default="2026-09-30")
    ap.add_argument("--out", default="data/past_features/corpus_panel.csv")
    ap.add_argument("--min-runners", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    runs = rc.load_runs()
    print(f"読み込み {len(runs)}走 / {runs['race_id'].nunique()}R "
          f"/ {runs['date'].min().date()}〜{runs['date'].max().date()} "
          f"({time.time()-t0:.0f}s)")
    runs = pf.add_split_columns(runs)

    ev = rc.usable_races(runs, min_runners=args.min_runners)
    ev = ev[(ev["date"] >= args.eval_from) & (ev["date"] <= args.eval_to)]
    print(f"評価対象 {ev['race_id'].nunique()}R / {len(ev)}頭 "
          f"（全頭オッズあり・1着確定・{args.min_runners}頭以上）")
    if ev.empty:
        raise SystemExit("評価対象が0件")

    by_horse = {h: g.sort_values("date") for h, g in runs.groupby("horse_id") if h}
    rows: list[dict] = []
    ev = ev.copy()
    ev["month"] = ev["date"].values.astype("datetime64[M]")

    for month, chunk in ev.groupby("month"):
        cut = pd.Timestamp(month)
        base = pf.course_baselines(runs, cut)
        pace = pf.pace_baseline(runs, cut)
        rpace = pf.race_pace_table(runs, cut, base)
        n_done = 0
        for rid, race in chunk.groupby("race_id"):
            head = race.iloc[0]
            cond = {"date": head["date"], "venue": head["venue"],
                    "surface": head["surface"], "dist": head["dist"],
                    "going": head["going"], "class_lv": head["class_lv"]}
            when = head["date"]
            for _, r in race.iterrows():
                rec = {
                    "race_id": rid, "馬名": r["馬名"], "horse_id": r["horse_id"],
                    "date": when, "馬番": r.get("馬番"),
                    "頭数": r["n_runners"], "人気": r["pop"], "単勝オッズ": r["odds"],
                    "市場勝率": r["market"], "着順": r["finish"],
                    "当日芝ダ": cond["surface"], "当日距離": cond["dist"],
                    "当日馬場": cond["going"], "当日会場": cond["venue"],
                    "当日クラス": cond["class_lv"],
                }
                hid = r["horse_id"]
                past = by_horse.get(hid)
                if past is None:
                    rows.append(rec | {"過去走数": 0.0})
                    continue
                past = past[past["date"] < when]
                if past.empty:
                    rows.append(rec | {"過去走数": 0.0})
                    continue
                past = pf.attach_baselines(past.copy(), base, pace, rpace)
                rows.append(rec | pf.build_features(past, cond))
            n_done += 1
        print(f"  {str(month)[:7]} {n_done}R 完了 ({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    for c in pf.FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n書き出し {args.out}  {len(out)}行 / {out['race_id'].nunique()}R")
    print(f"過去走を1走以上持つ馬 {(out['過去走数'] > 0).mean()*100:.1f}%")
    cov = {c: round(float(out.loc[out['過去走数'] > 0, c].notna().mean() * 100), 1)
           for c in pf.FEATURE_COLUMNS if not c.endswith("件数") and c != "過去走数"}
    print(json.dumps(cov, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
