#!/usr/bin/env python3
"""本番モジュール race_grade が、検証したときと同じ判定を返すか確かめる。

検証（scripts/strategy_compare.py）は pandas 上でまとめて計算している。
本番は1レースずつ呼ぶ。実装が二本になるので、学習に使っていない期間の
実レースに両方を通して、区分と実績が一致することを確認する。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import race_grade as rg  # noqa: E402
from feature_stability import load_panel, split_periods  # noqa: E402
from market_edge_probe import fit_conditional_logit, predict_cl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/past_features/corpus_panel.csv")
    ap.add_argument("--out", default="data/quality_backtest/race_grade_verify.json")
    args = ap.parse_args()

    d = load_panel(args.panel)
    periods = split_periods(d, 4)
    model = rg.load_model()
    if model is None:
        print("data/race_grade_model.json が読めない")
        return 1
    print(f"係数の学習期間 {model['学習期間']} / {model['学習レース数']}R")

    y = d["win"].to_numpy(dtype=float)
    g = d["race_id"].to_numpy()
    X = d["log_impl"].to_numpy(dtype=float)[:, None]

    # 係数は助走+A+B で学習済み。評価はそこに入っていない期間Cだけ。
    prior = np.zeros(len(d), dtype=bool)
    for _, m in periods[:-1]:
        prior |= m
    test = periods[-1][1]
    beta = fit_conditional_logit(X[prior], y[prior], g[prior], l2=1e-4)
    p = predict_cl(X[test], beta, g[test])

    sub = d.loc[test].copy()
    sub["p"] = p
    rows = []
    for i, (rid, race) in enumerate(sub.groupby("race_id", sort=False)):
        pr = race["p"].to_numpy(dtype=float)
        od = race["単勝オッズ"].to_numpy(dtype=float)
        res = rg.grade(pr, od, seed=i)
        if res is None:
            continue
        fin = race["着順"].to_numpy(dtype=float)
        top = int(np.argmax(pr))
        rows.append({
            "race_id": rid, "区分": res["レース区分"],
            "予想しやすさ": res["予想しやすさ"], "荒れやすさ": res["荒れやすさ"],
            "本命勝ち": float(fin[top] == 1), "本命3着内": float(fin[top] <= 3),
            "本命オッズ": float(od[top]),
            "荒れた": float(od[fin == 1][0] >= rg.LONGSHOT_ODDS) if (fin == 1).any()
            else np.nan,
        })
    t = pd.DataFrame(rows).dropna(subset=["荒れた"])
    print(f"期間C {len(t)}R を本番モジュールで判定")

    out = {"学習期間": model["学習期間"], "評価レース数": len(t), "区分": []}
    print(f"\n{'区分':<16}{'レース数':>8}{'割合':>7}{'本命勝率':>9}{'本命複勝率':>10}"
          f"{'本命ROI':>9}{'荒れ率':>8}")
    for lab in ["買うべき", "少額なら買える", "見送る"]:
        s = t[t["区分"] == lab]
        if s.empty:
            continue
        w = s["本命勝ち"].to_numpy(dtype=float)
        row = {
            "区分": lab, "レース数": len(s),
            "割合": round(len(s) / len(t) * 100, 1),
            "本命勝率": round(float(w.mean() * 100), 2),
            "本命複勝率": round(float(s["本命3着内"].mean() * 100), 2),
            "本命ROI": round(float((w * s["本命オッズ"]).sum() / len(s) * 100), 2),
            "荒れ率": round(float(s["荒れた"].mean() * 100), 2),
        }
        out["区分"].append(row)
        print(f"{lab:<16}{row['レース数']:>8}{row['割合']:>7.1f}{row['本命勝率']:>9.2f}"
              f"{row['本命複勝率']:>10.2f}{row['本命ROI']:>9.2f}{row['荒れ率']:>8.2f}")

    allw = t["本命勝ち"].to_numpy(dtype=float)
    out["全レースで本命を買った場合"] = {
        "レース数": len(t),
        "本命勝率": round(float(allw.mean() * 100), 2),
        "本命複勝率": round(float(t["本命3着内"].mean() * 100), 2),
        "本命ROI": round(float((allw * t["本命オッズ"]).sum() / len(t) * 100), 2),
    }
    print(f"\n参考 全レース   {len(t):>8}  100.0{out['全レースで本命を買った場合']['本命勝率']:>9.2f}"
          f"{out['全レースで本命を買った場合']['本命複勝率']:>10.2f}"
          f"{out['全レースで本命を買った場合']['本命ROI']:>9.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n書き出し {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
