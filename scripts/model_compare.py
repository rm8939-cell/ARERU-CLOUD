#!/usr/bin/env python3
"""モデルA（全特徴量）/ B（上位少数）/ C（既存＝市場のみ）を同じレースで比べる。

確率モデルの指標だけでなく、確率v2の選定ルールをそのまま当てはめた
運用指標（本命・穴馬・期待値帯・BUY）と、予測確率の較正も出す。

確率v2では最終勝率が AI の上乗せ重み 0 のとき市場確率そのものになる。
つまり「現在の既存モデル」の確率は市場確率と同一なので、モデルC＝市場のみ
が現行実装の忠実な再現になる。

選定ルールは areru_engine の確率v2分岐に合わせてある。
  本命   = 最終勝率が最大の馬
  対抗   = 残りで最終勝率が最大の馬
  穴馬   = 本命・対抗を除いた3着内率の上位3頭（注目馬＋穴馬）
  期待値 = 最終勝率 × 単勝オッズ
  BUY    = 本命の最終勝率が 35% 以上（レース信頼度70に相当）
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
from feature_stability import brier, load_panel, neutral_z, split_periods  # noqa: E402
from market_edge_probe import (  # noqa: E402
    auc,
    fit_conditional_logit,
    predict_cl,
    race_logloss,
)

BUY_WIN_FLOOR = 0.35


def place_probs_from_win(p: np.ndarray) -> np.ndarray:
    """勝率から3着内率を Plackett-Luce で導く（engine の実装と同じ考え方）。"""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, None)
    p = p / p.sum()
    n = len(p)
    if n <= 3:
        return np.ones(n)
    third = p.copy()
    for j in range(n):
        rest = 1.0 - p[j]
        if rest <= 1e-9:
            continue
        q = p / rest
        add = p[j] * q
        add[j] = 0.0
        third += add
        for k in range(n):
            if k == j:
                continue
            rest2 = rest - p[k]
            if rest2 <= 1e-9:
                continue
            r = p / rest2
            a2 = p[j] * q[k] * r
            a2[j] = 0.0
            a2[k] = 0.0
            third += a2
    return third


def calibration(p: np.ndarray, y: np.ndarray, bins=(0, .02, .05, .10, .15, .20,
                                                    .30, .40, .60, 1.01)) -> list[dict]:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 20:
            continue
        out.append({
            "予測確率帯": f"{lo*100:.0f}-{hi*100:.0f}%",
            "頭数": int(m.sum()),
            "予測平均": round(float(p[m].mean()) * 100, 2),
            "実績勝率": round(float(y[m].mean()) * 100, 2),
            "差": round(float(y[m].mean() - p[m].mean()) * 100, 2),
        })
    return out


def calib_error(p: np.ndarray, y: np.ndarray) -> float:
    """較正誤差（ECE）。予測確率帯ごとの |予測-実績| を頭数で重み付け。"""
    rows = calibration(p, y)
    if not rows:
        return float("nan")
    tot = sum(r["頭数"] for r in rows)
    return round(sum(abs(r["差"]) * r["頭数"] for r in rows) / tot, 3)


def operational(d: pd.DataFrame, p: np.ndarray, mask: np.ndarray) -> dict:
    """確率v2の選定ルールをそのまま当てはめた運用指標。"""
    sub = d.loc[mask].copy()
    sub["p"] = p
    recs_main, recs_hole, ev_rows = [], [], []
    for rid, race in sub.groupby("race_id"):
        pr = race["p"].to_numpy(dtype=float)
        if pr.sum() <= 0:
            continue
        pr = pr / pr.sum()
        order = np.argsort(-pr)
        third = place_probs_from_win(pr)
        rows = race.reset_index(drop=True)
        odds = rows["単勝オッズ"].to_numpy(dtype=float)
        win = rows["win"].to_numpy(dtype=float)
        top3 = rows["top3"].to_numpy(dtype=float)

        m = int(order[0])
        riv = int(order[1]) if len(order) > 1 else None
        recs_main.append({
            "race_id": rid, "勝": win[m], "複": top3[m], "odds": odds[m],
            "p": pr[m], "buy": pr[m] >= BUY_WIN_FLOOR,
        })
        excl = {m} | ({riv} if riv is not None else set())
        hole_order = [i for i in np.argsort(-third) if i not in excl][:3]
        for i in hole_order:
            recs_hole.append({"勝": win[i], "複": top3[i], "odds": odds[i]})
        for i in range(len(rows)):
            ev_rows.append({"ev": pr[i] * odds[i] * 100.0, "勝": win[i], "odds": odds[i]})

    M = pd.DataFrame(recs_main)
    H = pd.DataFrame(recs_hole)
    E = pd.DataFrame(ev_rows)

    def roi(df, col="勝"):
        if df.empty:
            return None
        return round(float((df[col] * df["odds"]).sum() / len(df) * 100), 2)

    out = {
        "本命": {
            "件数": len(M),
            "勝率": round(float(M["勝"].mean() * 100), 2) if len(M) else None,
            "複勝率": round(float(M["複"].mean() * 100), 2) if len(M) else None,
            "回収率": roi(M),
        },
        "穴馬": {
            "件数": len(H),
            "複勝率": round(float(H["複"].mean() * 100), 2) if len(H) else None,
            "回収率": roi(H),
        },
        "BUY": {},
        "期待値帯": [],
    }
    B = M[M["buy"]] if len(M) else M
    out["BUY"] = {
        "件数": len(B),
        "的中率": round(float(B["勝"].mean() * 100), 2) if len(B) else None,
        "回収率": roi(B),
    }
    if not E.empty:
        edges = [0, 60, 70, 75, 80, 85, 90, 95, 100, 110, 10000]
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = E[(E["ev"] >= lo) & (E["ev"] < hi)]
            if len(s) < 30:
                continue
            out["期待値帯"].append({
                "帯": f"{lo}-{hi}", "件数": len(s),
                "的中率": round(float(s["勝"].mean() * 100), 2),
                "回収率": roi(s),
            })
    return out


def topk_metrics(d: pd.DataFrame, p: np.ndarray, mask: np.ndarray) -> dict:
    sub = d.loc[mask].copy()
    sub["p"] = p
    first, top3 = [], []
    for _, race in sub.groupby("race_id"):
        o = np.argsort(-race["p"].to_numpy(dtype=float))
        w = race["win"].to_numpy(dtype=float)
        t = race["top3"].to_numpy(dtype=float)
        first.append(w[o[0]])
        top3.extend(t[o[:3]].tolist())
    return {
        "AI1位の勝率": round(float(np.mean(first)) * 100, 2),
        "AI1〜3位の複勝率": round(float(np.mean(top3)) * 100, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/past_features/corpus_panel.csv")
    ap.add_argument("--stability", default="data/quality_backtest/feature_stability.json")
    ap.add_argument("--out", default="data/quality_backtest/model_compare.json")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    d = load_panel(args.panel)
    periods = split_periods(d, 3)
    tags = [t for t, _ in periods]
    print(f"パネル {d['race_id'].nunique()}R / {len(d)}頭")
    for t, m in periods:
        print(f"  期間{t} {int(d.loc[m,'race_id'].nunique())}R")

    y = d["win"].to_numpy(dtype=float)
    g = d["race_id"].to_numpy()
    log_impl = d["log_impl"].to_numpy(dtype=float)

    zc: dict[str, np.ndarray] = {}
    for c in pf.FEATURE_COLUMNS:
        if c not in d.columns or c.endswith("件数"):
            continue
        z, miss = neutral_z(d, c)
        if (1.0 - miss.mean()) >= 0.05 and not np.allclose(z, 0):
            zc[c] = z
    all_cols = sorted(zc)
    print(f"候補特徴量 {len(all_cols)}")

    stab = json.loads(Path(args.stability).read_text(encoding="utf-8"))
    ranked = sorted(stab["特徴量"], key=lambda c: stab["特徴量"][c]["平均Δlogloss"])
    topk = [c for c in ranked if c in zc][: args.top_k]
    stable = [c for c in stab.get("安定して予測力あり", []) if c in zc]
    print(f"B: 上位{args.top_k} = {topk}")
    print(f"B2: 安定選択 = {stable}")

    models = {
        "A_全特徴量": (all_cols, 0.1),
        f"B_上位{args.top_k}": (topk, 1e-4),
        "C_既存(市場のみ)": ([], 1e-4),
    }
    if stable and set(stable) != set(topk):
        models["B2_安定選択"] = (stable, 1e-4)

    # 学習は「評価期間より前」だけ。直近期間Cを本番相当の評価に使う。
    prior_all = periods[0][1] | periods[1][1]
    test_c = periods[2][1]

    report: dict = {"パネル": {"レース数": int(d["race_id"].nunique()), "頭数": len(d)},
                    "期間": [{"期間": t, "レース数": int(d.loc[m, "race_id"].nunique())}
                           for t, m in periods],
                    "モデル": {}}

    for name, (cols, l2) in models.items():
        X = np.column_stack([log_impl] + [zc[c] for c in cols]) if cols \
            else log_impl[:, None]
        beta = fit_conditional_logit(X[prior_all], y[prior_all], g[prior_all], l2=l2)
        p = predict_cl(X[test_c], beta, g[test_c])
        yy = y[test_c]
        entry = {
            "特徴量数": len(cols),
            "確率指標": {
                "LogLoss": round(race_logloss(p, yy, g[test_c]), 5),
                "Brier": round(brier(p, yy), 6),
                "AUC": round(auc(p, yy), 4),
                "較正誤差ECE": calib_error(p, yy),
            },
            **topk_metrics(d, p, test_c),
            "運用指標": operational(d, p, test_c),
            "Calibration": calibration(p, yy),
        }
        # 期間ごとにも同じ学習・評価を繰り返す
        per = {}
        for i, (t, m) in enumerate(periods):
            if i == 0:
                continue
            prior = np.zeros(len(d), dtype=bool)
            for _, pm in periods[:i]:
                prior |= pm
            b = fit_conditional_logit(X[prior], y[prior], g[prior], l2=l2)
            pp = predict_cl(X[m], b, g[m])
            per[t] = {"LogLoss": round(race_logloss(pp, y[m], g[m]), 5),
                      "Brier": round(brier(pp, y[m]), 6),
                      "AUC": round(auc(pp, y[m]), 4),
                      "較正誤差ECE": calib_error(pp, y[m]),
                      **topk_metrics(d, pp, m)}
        entry["期間別"] = per
        if cols:
            entry["係数"] = {c: round(float(v), 5) for c, v in zip(cols, beta[1:])}
        report["モデル"][name] = entry
        print(f"\n== {name} ==")
        print(" ", json.dumps(entry["確率指標"], ensure_ascii=False))
        print("  AI1位勝率", entry["AI1位の勝率"], "/ AI1〜3位複勝率", entry["AI1〜3位の複勝率"])
        o = entry["運用指標"]
        print("  本命", o["本命"], "\n  穴馬", o["穴馬"], "\n  BUY", o["BUY"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n書き出し {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
