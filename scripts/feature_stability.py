#!/usr/bin/env python3
"""特徴量の予測力が期間をまたいで維持されるかを調べる。

やり方はウォークフォワード。評価期間を日付順に A（古い）/ B（中間）/ C（直近）
へ3分割し、各期間について「その期間より前のデータだけ」で係数を推定して
その期間で測る。未来の情報は一切入らない。

各特徴量について見るのは次の3つ。

  * 期間ごとの log loss 改善（市場単独との差）
  * 係数の符号が3期間で一致しているか
  * 直近期間 C でも改善しているか

「全期間をまとめると良いが直近では悪い」ものを弾くのが目的なので、
採用優先度は C を重く見る。
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
    auc,
    fit_conditional_logit,
    predict_cl,
    race_logloss,
)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def neutral_z(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """レース内 z スコア。欠損はレース平均（z=0）にして減点しない。"""
    v = pd.to_numeric(df[col], errors="coerce")
    miss = v.isna().to_numpy()
    g = v.groupby(df["race_id"])
    z = (v - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return z.fillna(0.0).to_numpy(dtype=float), miss.astype(float)


def load_panel(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["race_id"] = d["race_id"].astype(str)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna() & d["市場勝率"].notna() & d["着順"].notna()].copy()
    d["win"] = (d["着順"] == 1).astype(float)
    d["top3"] = (d["着順"] <= 3).astype(float)
    d["log_impl"] = np.log(np.clip(d["市場勝率"].astype(float), 1e-6, None))
    return d.reset_index(drop=True)


def split_periods(d: pd.DataFrame, n: int = 3) -> list[tuple[str, np.ndarray]]:
    """期間をレース数が揃うように日付で n 分割する。

    先頭の塊は前に学習データが無いので評価できない。そこを「助走」と呼び、
    残りを古い順に A / B / C と並べる。どの期間も自分より前の期間だけで
    係数を推定して測るので、未来の情報は入らない。
    """
    races = d[["race_id", "date"]].drop_duplicates().sort_values("date")
    chunks = np.array_split(races["race_id"].to_numpy(), n)
    out = []
    for i, ids in enumerate(chunks):
        m = d["race_id"].isin(set(ids)).to_numpy()
        tag = "助走" if i == 0 else "ABCDEFG"[i - 1]
        sub = d.loc[m, "date"]
        out.append((f"{tag}({sub.min().date()}〜{sub.max().date()})", m))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/past_features/corpus_panel.csv")
    ap.add_argument("--out", default="data/quality_backtest/feature_stability.json")
    ap.add_argument("--periods", type=int, default=4)
    args = ap.parse_args()

    d = load_panel(args.panel)
    print(f"パネル {d['race_id'].nunique()}R / {len(d)}頭 "
          f"{d['date'].min().date()}〜{d['date'].max().date()}")

    periods = split_periods(d, args.periods)
    for tag, m in periods:
        print(f"  期間{tag} {int(d.loc[m,'race_id'].nunique())}R / {int(m.sum())}頭")

    y = d["win"].to_numpy(dtype=float)
    g = d["race_id"].to_numpy()
    log_impl = d["log_impl"].to_numpy(dtype=float)

    # 各期間について「その期間より前」を学習に使う。最古の期間は学習データが
    # 無いので評価しない（ここが未来情報を入れないための肝）。
    evals: list[tuple[str, np.ndarray, np.ndarray]] = []
    for i, (tag, m) in enumerate(periods):
        if i == 0:
            print(f"  {tag} は学習データが前に無いので係数推定の材料にだけ使う")
            continue
        prior = np.zeros(len(d), dtype=bool)
        for _, pm in periods[:i]:
            prior |= pm
        evals.append((tag, prior, m))

    def run(cols: list[str], prior: np.ndarray, test: np.ndarray, l2: float = 1e-4):
        X = np.column_stack([log_impl] + [zc[c] for c in cols]) if cols \
            else log_impl[:, None]
        beta = fit_conditional_logit(X[prior], y[prior], g[prior], l2=l2)
        p = predict_cl(X[test], beta, g[test])
        return {
            "logloss": race_logloss(p, y[test], g[test]),
            "brier": brier(p, y[test]),
            "auc": auc(p, y[test]),
        }, beta

    zc: dict[str, np.ndarray] = {}
    for c in pf.FEATURE_COLUMNS:
        if c not in d.columns or c.endswith("件数"):
            continue
        z, miss = neutral_z(d, c)
        if (1.0 - miss.mean()) >= 0.05 and not np.allclose(z, 0):
            zc[c] = z

    base: dict[str, dict] = {}
    for tag, prior, test in evals:
        base[tag], _ = run([], prior, test)
        print(f"\n基準（市場のみ）期間{tag}: logloss={base[tag]['logloss']:.5f} "
              f"brier={base[tag]['brier']:.6f} AUC={base[tag]['auc']:.4f}")

    results: dict[str, dict] = {}
    for c in sorted(zc):
        row = {"被覆率": round(float((d[c].notna()).mean() * 100), 1), "期間": {}}
        for tag, prior, test in evals:
            r, beta = run([c], prior, test)
            row["期間"][tag] = {
                "係数": round(float(beta[1]), 4),
                "Δlogloss": round(r["logloss"] - base[tag]["logloss"], 5),
                "Δbrier": round(r["brier"] - base[tag]["brier"], 7),
                "ΔAUC": round(r["auc"] - base[tag]["auc"], 4),
            }
        signs = {np.sign(v["係数"]) for v in row["期間"].values()}
        dl = [v["Δlogloss"] for v in row["期間"].values()]
        row["符号一致"] = bool(len(signs) == 1)
        row["全期間で改善"] = bool(all(x < 0 for x in dl))
        row["直近期間で改善"] = bool(dl[-1] < 0)
        row["平均Δlogloss"] = round(float(np.mean(dl)), 5)
        results[c] = row

    tags = [t for t, _, _ in evals]
    print(f"\n{'特徴量':<16}{'被覆%':>7}" + "".join(f"{t.split('(')[0]:>11}" for t in tags)
          + f"{'符号一致':>9}{'直近改善':>9}")
    order = sorted(results, key=lambda c: results[c]["平均Δlogloss"])
    for c in order:
        r = results[c]
        cells = "".join(f"{r['期間'][t]['Δlogloss']:>11.5f}" for t in tags)
        print(f"{c:<16}{r['被覆率']:>7.1f}{cells}"
              f"{'○' if r['符号一致'] else '×':>9}"
              f"{'○' if r['直近期間で改善'] else '×':>9}")

    stable = [c for c in order
              if results[c]["符号一致"] and results[c]["全期間で改善"]]
    recent_ok = [c for c in order if results[c]["直近期間で改善"]]
    downgrade = [c for c in order
                 if results[c]["平均Δlogloss"] < 0 and not results[c]["直近期間で改善"]]
    print(f"\n全期間で符号一致かつ改善: {stable}")
    print(f"直近期間で改善: {recent_ok}")
    print(f"平均では改善だが直近では悪化（優先度を下げる）: {downgrade}")

    out = {
        "パネル": {"レース数": int(d["race_id"].nunique()), "頭数": len(d),
                 "期間": f"{d['date'].min().date()}〜{d['date'].max().date()}"},
        "期間定義": [{"期間": t, "レース数": int(d.loc[m, "race_id"].nunique())}
                 for t, m in periods],
        "基準_市場のみ": {t: {k: round(v, 6) for k, v in base[t].items()} for t in tags},
        "特徴量": results,
        "安定して予測力あり": stable,
        "直近期間で改善": recent_ok,
        "優先度を下げる_直近で悪化": downgrade,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n書き出し {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
