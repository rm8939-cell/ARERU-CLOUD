#!/usr/bin/env python3
"""過去走特徴量が1個ずつ、本当に予測力を持つかを確かめる。

やり方はレース内 softmax（条件付きロジット）。

  M_market : log(市場implied) だけ
  M_feat   : log(市場implied) + 当該特徴量（レース内zスコア1本）

train で係数を推定し、holdout の log loss / Brier / AUC がどれだけ動くかを見る。
holdout が改善しない特徴量は採らない。

欠損の扱い。欠損馬を0点にすると「データが無い＝弱い」と減点することになるので、
レース内平均で埋めて z スコアを0にし、softmax 上で中立にする。
さらに「欠損だったかどうか」のフラグ自体に情報があるかも別途測る。
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
    top1_rate,
)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def neutral_z(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """レース内 z スコアと欠損フラグ。欠損はレース平均扱い（z=0）。"""
    v = pd.to_numeric(df[col], errors="coerce")
    miss = v.isna().to_numpy()
    g = v.groupby(df["race_id"])
    z = (v - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return z.fillna(0.0).to_numpy(dtype=float), miss.astype(float)


def evaluate(df: pd.DataFrame, cols: list[str], tr: np.ndarray, ho: np.ndarray) -> dict:
    X = np.column_stack(cols) if cols else np.zeros((len(df), 0))
    y = df["win"].to_numpy(dtype=float)
    g = df["race_id"].to_numpy()
    if X.shape[1] == 0:
        return {}
    beta = fit_conditional_logit(X[tr], y[tr], g[tr])
    out = {}
    for tag, m in (("train", tr), ("holdout", ho)):
        p = predict_cl(X[m], beta, g[m])
        out[tag] = {
            "logloss": round(race_logloss(p, y[m], g[m]), 5),
            "brier": round(brier(p, y[m]), 5),
            "auc": round(auc(p, y[m]), 4),
            "top1勝率": round(top1_rate(p, y[m], g[m]) * 100, 2),
        }
    out["係数"] = [round(float(b), 4) for b in beta]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/past_features/panel.csv")
    ap.add_argument("--label", default="prod_old")
    ap.add_argument("--out", default="data/quality_backtest/past_feature_signal.json")
    args = ap.parse_args()

    base = load_panel(args.label)
    feats = pd.read_csv(args.panel, low_memory=False)
    feats["race_id"] = feats["race_id"].astype(str)
    feats["馬名"] = feats["馬名"].astype(str).str.strip()
    keep = ["race_id", "馬名"] + [c for c in pf.FEATURE_COLUMNS if c in feats.columns]
    df = base.merge(feats[keep].drop_duplicates(["race_id", "馬名"]),
                    on=["race_id", "馬名"], how="left")
    print(f"評価対象 {df['race_id'].nunique()}R / {len(df)}頭")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    tr = (df["date"] < HOLDOUT_FROM).to_numpy()
    ho = (df["date"] >= HOLDOUT_FROM).to_numpy()
    print(f"train {int(df.loc[tr,'race_id'].nunique())}R / holdout {int(df.loc[ho,'race_id'].nunique())}R")

    log_impl = df["log_impl"].to_numpy(dtype=float)
    m0 = evaluate(df, [log_impl], tr, ho)
    print("\n== 市場のみ（基準） ==")
    print(json.dumps(m0, ensure_ascii=False))

    results = {"基準_市場のみ": m0, "特徴量": {}}
    rows = []
    for col in pf.FEATURE_COLUMNS:
        if col not in df.columns or col.endswith("件数") or col == "過去走数":
            continue
        z, miss = neutral_z(df, col)
        cov = float(1.0 - miss.mean())
        if cov < 0.02 or np.allclose(z, 0):
            results["特徴量"][col] = {"被覆率": round(cov * 100, 1), "判定": "データ不足で評価不能"}
            continue
        m1 = evaluate(df, [log_impl, z], tr, ho)
        d_ll = m1["holdout"]["logloss"] - m0["holdout"]["logloss"]
        d_br = m1["holdout"]["brier"] - m0["holdout"]["brier"]
        d_auc = m1["holdout"]["auc"] - m0["holdout"]["auc"]
        tr_coef = m1["係数"][1]
        results["特徴量"][col] = {
            "被覆率": round(cov * 100, 1),
            "係数": tr_coef,
            "holdout_logloss差": round(d_ll, 5),
            "holdout_brier差": round(d_br, 6),
            "holdout_AUC差": round(d_auc, 4),
            "train_logloss差": round(m1["train"]["logloss"] - m0["train"]["logloss"], 5),
        }
        rows.append((col, cov * 100, tr_coef, d_ll, d_auc,
                     m1["train"]["logloss"] - m0["train"]["logloss"]))

    rows.sort(key=lambda r: r[3])
    print("\n== 単独追加の効果（market + 当該特徴量、holdout）==")
    print(f"{'特徴量':<16}{'被覆%':>7}{'係数':>9}{'Δlogloss':>11}{'ΔAUC':>9}{'train Δll':>11}")
    for col, cov, coef, dll, dauc, dtr in rows:
        print(f"{col:<16}{cov:>7.1f}{coef:>9.4f}{dll:>11.5f}{dauc:>9.4f}{dtr:>11.5f}")

    # train と holdout の両方で改善した特徴量だけを候補にする。
    good = [r[0] for r in rows if r[3] < -1e-5 and r[5] < -1e-5]
    print(f"\ntrain/holdout 両方で logloss 改善: {good}")

    if good:
        cols = [log_impl] + [neutral_z(df, c)[0] for c in good]
        mall = evaluate(df, cols, tr, ho)
        results["合成_市場＋採用候補"] = {"採用候補": good, **mall}
        print("\n== 市場 + 採用候補をまとめて ==")
        print(json.dumps(mall, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
