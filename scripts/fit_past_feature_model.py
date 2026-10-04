#!/usr/bin/env python3
"""採用候補の過去走特徴量について、確率モデルの係数を train だけで推定する。

手で重みを決めない。レース内 softmax の最尤推定で係数を出す。

どの特徴量を入れるかの判断も train の中だけで行う。train 期間を日付順に
K分割し、「前のブロックで学習して次のブロックで評価する」時系列CVの
log loss が下がる特徴量だけを前進選択で入れる。holdout は選択に一切
使わず、最後に一度だけ答え合わせに使う。
（holdout を見ながら特徴量を選ぶと、holdout の数字が選択バイアスで
良く見えてしまい、本番での再現性が測れなくなる。）
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
    ap.add_argument("--max-features", type=int, default=12)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--min-gain", type=float, default=2e-4,
                    help="時系列CVの log loss がこれ以上下がらない特徴量は採らない")
    ap.add_argument("--force", choices=["ridge", "greedy", "stable"], default="",
                    help="CVの判断を無視して方針を固定する（検証用）")
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

    # train 期間を日付順に K ブロックへ分け、「過去で学習→直後の期間で評価」を繰り返す。
    tr_dates = np.sort(df.loc[tr, "date"].unique())
    edges = np.array_split(tr_dates, args.folds)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(edges)):
        fit_upto = edges[i - 1][-1]
        va_from, va_to = edges[i][0], edges[i][-1]
        fit_m = tr & (df["date"] <= fit_upto).to_numpy()
        va_m = tr & (df["date"] >= va_from).to_numpy() & (df["date"] <= va_to).to_numpy()
        if fit_m.sum() and va_m.sum():
            folds.append((fit_m, va_m))
    print(f"時系列CV {len(folds)} 分割（train内のみ）")

    def cv_score(cols: list[str]) -> float:
        X = np.column_stack([log_impl] + [zcache[c] for c in cols])
        losses = []
        for fit_m, va_m in folds:
            beta = fit_conditional_logit(X[fit_m], y[fit_m], g[fit_m])
            p = predict_cl(X[va_m], beta, g[va_m])
            losses.append(race_logloss(p, y[va_m], g[va_m]))
        return float(np.mean(losses))

    def final_score(cols: list[str]) -> tuple[dict, np.ndarray]:
        X = np.column_stack([log_impl] + [zcache[c] for c in cols])
        beta = fit_conditional_logit(X[tr], y[tr], g[tr])
        res = {}
        for tag, m in (("train", tr), ("holdout", ho)):
            p = predict_cl(X[m], beta, g[m])
            res[tag] = {"logloss": round(race_logloss(p, y[m], g[m]), 5),
                        "brier": round(brier(p, y[m]), 6),
                        "auc": round(auc(p, y[m]), 4)}
        return res, beta

    base_cv = cv_score([])
    base_res, _ = final_score([])
    print(f"基準（市場のみ）CV logloss={base_cv:.5f} / "
          f"参考 holdout={base_res['holdout']['logloss']:.5f} AUC={base_res['holdout']['auc']}")

    all_cols = sorted(zcache)
    history = []

    # 方針A: 全特徴量を入れて L2 の強さだけ train内CV で決める。
    # 1000レース規模では貪欲選択より縮小推定のほうが安定する。
    def cv_score_l2(cols: list[str], l2: float) -> float:
        X = np.column_stack([log_impl] + [zcache[c] for c in cols])
        losses = []
        for fit_m, va_m in folds:
            beta = fit_conditional_logit(X[fit_m], y[fit_m], g[fit_m], l2=l2)
            p = predict_cl(X[va_m], beta, g[va_m])
            losses.append(race_logloss(p, y[va_m], g[va_m]))
        return float(np.mean(losses))

    grid = [3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 1e-4]
    ridge = [(cv_score_l2(all_cols, l2), l2) for l2 in grid]
    for cv, l2 in ridge:
        print(f"  L2={l2:<7} CV logloss={cv:.5f}")
    ridge.sort(key=lambda t: t[0])
    best_ridge_cv, best_l2 = ridge[0]
    print(f"L2 の最良 = {best_l2} (CV {best_ridge_cv:.5f} / 基準 {base_cv:.5f})")
    history.append({"方針": "全特徴量+L2", "最良L2": best_l2,
                    "CV_logloss": round(best_ridge_cv, 5),
                    "CV改善": round(base_cv - best_ridge_cv, 5)})

    # 方針B: 貪欲前進選択（CVのみで判断）。
    chosen: list[str] = []
    best_cv = base_cv
    while len(chosen) < args.max_features:
        cands = [(cv_score(chosen + [c]), c) for c in all_cols if c not in chosen]
        if not cands:
            break
        cands.sort(key=lambda t: t[0])
        cv, c = cands[0]
        if best_cv - cv < args.min_gain:
            print(f"前進選択はここで打ち切り（最良候補 {c} のCV改善 {best_cv - cv:.5f}）")
            break
        chosen.append(c)
        best_cv = cv
        print(f"  前進選択 {len(chosen)}: {c:<16} CV logloss={cv:.5f}")
    history.append({"方針": "貪欲前進選択", "採用": list(chosen),
                    "CV_logloss": round(best_cv, 5),
                    "CV改善": round(base_cv - best_cv, 5)})

    # 方針C: 安定性による選択。train の各フォールドで単独投入し、
    # 係数の符号が全フォールドで一致し、かつ平均CV改善がプラスの特徴量だけを残す。
    # 貪欲選択が少数のフォールドのノイズを拾うのを避けるため。
    stable: list[str] = []
    for c in all_cols:
        X = np.column_stack([log_impl, zcache[c]])
        signs, gains = [], []
        for fit_m, va_m in folds:
            b = fit_conditional_logit(X[fit_m], y[fit_m], g[fit_m])
            signs.append(np.sign(b[1]))
            p1 = predict_cl(X[va_m], b, g[va_m])
            b0 = fit_conditional_logit(log_impl[fit_m, None], y[fit_m], g[fit_m])
            p0 = predict_cl(log_impl[va_m, None], b0, g[va_m])
            gains.append(race_logloss(p0, y[va_m], g[va_m])
                         - race_logloss(p1, y[va_m], g[va_m]))
        if len(set(signs)) == 1 and float(np.mean(gains)) > 0:
            stable.append(c)
    stable_cv = cv_score(stable) if stable else float("inf")
    print(f"  安定選択 {len(stable)}個 {stable} CV logloss={stable_cv:.5f}")
    history.append({"方針": "安定性選択", "採用": list(stable),
                    "CV_logloss": round(stable_cv, 5) if stable else None,
                    "CV改善": round(base_cv - stable_cv, 5) if stable else None})

    # CV で勝った方針だけを holdout にかける。holdout は選択に使わない。
    if args.force:
        chosen = {"ridge": all_cols, "greedy": list(chosen), "stable": stable}[args.force]
        l2_used = best_l2 if args.force == "ridge" else 1e-4
        print(f"\n方針を {args.force} に固定（CVの判断を上書き）")
    elif stable and stable_cv < min(best_ridge_cv, best_cv):
        print(f"\nCVでは安定性選択が最良。これを holdout で検証する。")
        chosen = stable
        l2_used = 1e-4
    elif best_ridge_cv <= best_cv:
        print(f"\nCVでは「全特徴量+L2={best_l2}」が最良。これを holdout で検証する。")
        chosen = all_cols
        l2_used = best_l2
    else:
        print("\nCVでは貪欲前進選択が最良。これを holdout で検証する。")
        l2_used = 1e-4
    if not chosen:
        print("採用できる特徴量なし。モデルは書き出さない。")
        return 1

    def final_score_l2(cols: list[str], l2: float):
        X = np.column_stack([log_impl] + [zcache[c] for c in cols])
        beta = fit_conditional_logit(X[tr], y[tr], g[tr], l2=l2)
        res = {}
        for tag, m in (("train", tr), ("holdout", ho)):
            p = predict_cl(X[m], beta, g[m])
            res[tag] = {"logloss": round(race_logloss(p, y[m], g[m]), 5),
                        "brier": round(brier(p, y[m]), 6),
                        "auc": round(auc(p, y[m]), 4)}
        return res, beta

    final_score = lambda cols: final_score_l2(cols, l2_used)  # noqa: E731

    # 記録用に3方針すべてを holdout で測る。選択には使わない。
    audit = {}
    for tag, cols, l2 in (("全特徴量+L2", all_cols, best_l2),
                          ("貪欲前進選択", history[1]["採用"], 1e-4),
                          ("安定性選択", stable, 1e-4)):
        if not cols:
            continue
        r, _ = final_score_l2(list(cols), l2)
        audit[tag] = {"特徴量数": len(cols), **r["holdout"]}
        print(f"  [記録] {tag:<12} holdout logloss={r['holdout']['logloss']:.5f} "
              f"AUC={r['holdout']['auc']}")

    res, beta = final_score(chosen)
    coeffs = {c: round(float(b), 5) for c, b in zip(chosen, beta[1:])}
    improved = (res["holdout"]["logloss"] < base_res["holdout"]["logloss"]
                and res["holdout"]["auc"] >= base_res["holdout"]["auc"])
    print(f"\nholdout: logloss {base_res['holdout']['logloss']:.5f} → "
          f"{res['holdout']['logloss']:.5f} / AUC {base_res['holdout']['auc']} → "
          f"{res['holdout']['auc']}  判定={'改善' if improved else '改善せず'}")
    out = {
        "学習期間": f"< {HOLDOUT_FROM}",
        "特徴量選択": f"train内の時系列CV {len(folds)}分割（holdoutは未使用）",
        "L2": l2_used,
        "holdoutで改善したか": bool(improved),
        "市場係数": round(float(beta[0]), 5),
        "係数": coeffs,
        "基準_市場のみ": base_res,
        "採用後": res,
        "各方針の経過": history,
        "参考_3方針のholdout実測": audit,
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
