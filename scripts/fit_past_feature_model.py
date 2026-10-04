#!/usr/bin/env python3
"""採用候補の過去走特徴量について、確率モデルの係数を train だけで推定する。

手で重みを決めない。レース内 softmax の最尤推定で係数を出し、holdout で
改善が確認できたものだけを残す。前進選択なので、入れても holdout の
log loss が下がらない特徴量は入らない。
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

import past_features as pf  # noqa: E402
from market_edge_probe import (  # noqa: E402
    HOLDOUT_FROM,
    auc,
    fit_conditional_logit,
    load_panel,
    predict_cl,
    race_logloss,
)
from past_feature_signal import brier, neutral_z  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/past_features/panel.csv")
    ap.add_argument("--label", default="prod_old")
    ap.add_argument("--out", default="data/past_feature_model.json")
    ap.add_argument("--max-features", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=2e-4,
                    help="holdout log loss がこれ以上下がらない特徴量は採らない")
    args = ap.parse_args()

    base = load_panel(args.label)
    feats = pd.read_csv(args.panel, low_memory=False)
    feats["race_id"] = feats["race_id"].astype(str)
    feats["馬名"] = feats["馬名"].astype(str).str.strip()
    cand_cols = [c for c in pf.FEATURE_COLUMNS
                 if c in feats.columns and not c.endswith("件数")]
    df = base.merge(feats[["race_id", "馬名"] + cand_cols].drop_duplicates(["race_id", "馬名"]),
                    on=["race_id", "馬名"], how="left")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    tr = (df["date"] < HOLDOUT_FROM).to_numpy()
    ho = (df["date"] >= HOLDOUT_FROM).to_numpy()
    y = df["win"].to_numpy(dtype=float)
    g = df["race_id"].to_numpy()
    log_impl = df["log_impl"].to_numpy(dtype=float)
    print(f"train {int(df.loc[tr,'race_id'].nunique())}R / "
          f"holdout {int(df.loc[ho,'race_id'].nunique())}R")

    zcache: dict[str, np.ndarray] = {}
    for c in cand_cols:
        z, miss = neutral_z(df, c)
        if (1.0 - miss.mean()) >= 0.05 and not np.allclose(z, 0):
            zcache[c] = z

    def score(cols: list[str]) -> tuple[float, dict, np.ndarray]:
        X = np.column_stack([log_impl] + [zcache[c] for c in cols])
        beta = fit_conditional_logit(X[tr], y[tr], g[tr])
        res = {}
        for tag, m in (("train", tr), ("holdout", ho)):
            p = predict_cl(X[m], beta, g[m])
            res[tag] = {"logloss": round(race_logloss(p, y[m], g[m]), 5),
                        "brier": round(brier(p, y[m]), 6),
                        "auc": round(auc(p, y[m]), 4)}
        return res["holdout"]["logloss"], res, beta

    best_ll, base_res, base_beta = score([])
    print(f"基準（市場のみ）holdout logloss={best_ll:.5f} AUC={base_res['holdout']['auc']}")

    chosen: list[str] = []
    history = []
    while len(chosen) < args.max_features:
        cands = []
        for c in zcache:
            if c in chosen:
                continue
            ll, res, beta = score(chosen + [c])
            cands.append((ll, c, res, beta))
        if not cands:
            break
        cands.sort(key=lambda t: t[0])
        ll, c, res, beta = cands[0]
        gain = best_ll - ll
        if gain < args.min_gain:
            print(f"これ以上の改善なし（最良候補 {c} の改善 {gain:.5f}）")
            break
        chosen.append(c)
        best_ll = ll
        history.append({"追加": c, "holdout_logloss": ll,
                        "改善": round(gain, 5), **res})
        print(f"採用 {len(chosen)}: {c:<16} holdout logloss={ll:.5f} "
              f"(改善 {gain:.5f}) AUC={res['holdout']['auc']}")

    if not chosen:
        print("採用できる特徴量なし。モデルは書き出さない。")
        return 1

    ll, res, beta = score(chosen)
    coeffs = {c: round(float(b), 5) for c, b in zip(chosen, beta[1:])}
    out = {
        "学習期間": f"< {HOLDOUT_FROM}",
        "市場係数": round(float(beta[0]), 5),
        "係数": coeffs,
        "基準_市場のみ": base_res,
        "採用後": res,
        "前進選択の経過": history,
        "注記": "係数はレース内zスコアに対するもの。欠損はz=0（レース平均）扱い。",
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("\n" + json.dumps({"係数": coeffs, "基準": base_res["holdout"],
                             "採用後": res["holdout"]}, ensure_ascii=False, indent=1))
    print(f"書き出し {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
