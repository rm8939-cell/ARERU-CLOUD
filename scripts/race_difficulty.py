#!/usr/bin/env python3
"""レース単位の「予想しやすさ」を作って、過去データで予測力を確かめる。

馬ごとの勝率が出たあと、そのレースがどれくらい当てやすいかはレース内の
確率の形で決まる。ここでは次の7つをレース特徴量にする。

  1. 1位と2位の勝率差
  2. 上位3頭の確率集中度
  3. 予測確率の分散（ばらつき）
  4. AI1位の信頼度（1位の勝率そのもの）
  5. 人気とAI評価の乖離
  6. 穴馬候補の存在（人気薄でAI評価が高い馬）
  7. シミュレーションの安定性（揺らしても1位が入れ替わらないか）

検証のしかたは馬の特徴量のときと同じウォークフォワード。期間を古い順に
分け、ある期間を評価するときは必ずその期間より前のデータだけで係数を
推定する。結果（着順・配当）は一切特徴量に入れない。

判定は2本のモデルで出す。
  H: AI1位が勝つ確率   → 高ければ「予想しやすい」
  U: 勝ち馬が人気薄になる確率 → 高ければ「荒れる可能性が高い」
  どちらでもなければ「混戦」。
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
from feature_stability import load_panel, neutral_z, split_periods  # noqa: E402
from market_edge_probe import auc, fit_conditional_logit, predict_cl  # noqa: E402

LONGSHOT_ODDS = 10.0
SIM_NOISE = 0.35
SIM_DRAWS = 400

RACE_FEATURES = [
    "1位2位差", "上位3集中度", "確率分散", "AI1位信頼度",
    "人気乖離", "穴馬候補", "SIM安定性",
]


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def sim_stability(p: np.ndarray, rng: np.random.Generator) -> float:
    """1位の勝率に不確実性を足しても1位が動かない割合。

    モデルの係数も過去走もノイズを含むので、log 能力値を少し揺らして
    何度も並べ替え、最初の1位が1位のままでいる頻度を数える。
    実機のモンテカルロ（race_sim）と同じ「揺らして安定性を見る」考え方。
    """
    top = int(np.argmax(p))
    lo = np.log(np.clip(p, 1e-9, None))
    noise = rng.normal(0.0, SIM_NOISE, size=(SIM_DRAWS, len(p)))
    return float((np.argmax(lo[None, :] + noise, axis=1) == top).mean())


def race_table(d: pd.DataFrame, p: np.ndarray, mask: np.ndarray,
               seed: int = 0) -> pd.DataFrame:
    """馬ごとの勝率からレース単位の特徴量と結果をまとめる。"""
    sub = d.loc[mask].copy()
    sub["p"] = p
    rng = np.random.default_rng(seed)
    rows = []
    for rid, race in sub.groupby("race_id"):
        pr = race["p"].to_numpy(dtype=float)
        if pr.sum() <= 0 or len(pr) < 4:
            continue
        pr = pr / pr.sum()
        odds = race["単勝オッズ"].to_numpy(dtype=float)
        win = race["着順"].to_numpy(dtype=float) == 1
        order = np.argsort(-pr)
        srt = pr[order]

        # 人気とAI評価の乖離。順位相関が低いほどAIが市場と違う形を見ている。
        rank_ai = pd.Series(-pr).rank().to_numpy()
        rank_mk = pd.Series(odds).rank().to_numpy()
        if np.std(rank_ai) < 1e-9 or np.std(rank_mk) < 1e-9:
            dev = 0.0
        else:
            dev = 1.0 - float(np.corrcoef(rank_ai, rank_mk)[0, 1])

        longshot = pr[odds >= LONGSHOT_ODDS]
        rows.append({
            "race_id": rid,
            "date": race["date"].iloc[0],
            "頭数": len(pr),
            "1位2位差": float(srt[0] - srt[1]),
            "上位3集中度": float(srt[:3].sum()),
            "確率分散": float(np.std(pr) * np.sqrt(len(pr))),
            "AI1位信頼度": float(srt[0]),
            "人気乖離": dev,
            "穴馬候補": float(longshot.max()) if longshot.size else 0.0,
            "SIM安定性": sim_stability(pr, rng),
            # 以下は結果。特徴量には使わない。
            "AI1位的中": float(win[order[0]]),
            "AI3位内的中": float(win[order[:3]].any()),
            "勝ち馬オッズ": float(odds[win][0]) if win.any() else np.nan,
            "勝ち馬人気": float(race["人気"].to_numpy(dtype=float)[win][0])
            if win.any() else np.nan,
            "レースlogloss": float(-np.log(max(pr[win][0], 1e-9))) if win.any()
            else np.nan,
        })
    out = pd.DataFrame(rows)
    out["荒れた"] = (out["勝ち馬オッズ"] >= LONGSHOT_ODDS).astype(float)
    return out


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                 iters: int = 4000, lr: float = 0.05) -> np.ndarray:
    """切片つきロジスティック回帰（Adam）。レース数が数千なのでこれで十分。"""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    m = np.zeros_like(b)
    v = np.zeros_like(b)
    for t in range(1, iters + 1):
        z = X @ b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        gr = X.T @ (p - y) / len(y) + l2 * b
        gr[0] -= l2 * b[0]
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr * gr
        mh = m / (1 - 0.9 ** t)
        vh = v / (1 - 0.999 ** t)
        b -= lr * mh / (np.sqrt(vh) + 1e-8)
    return b


def logistic_predict(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))


def zfit(tr: pd.DataFrame, cols: list[str]):
    mu = tr[cols].mean()
    sd = tr[cols].std().replace(0, 1.0)
    return lambda df: ((df[cols] - mu) / sd).to_numpy(dtype=float)


def single_signal_report(tab: pd.DataFrame) -> dict:
    """各signalが単独でどれだけ当たりやすさを説明するか。"""
    out = {}
    for c in RACE_FEATURES:
        v = tab[c].to_numpy(dtype=float)
        if np.std(v) < 1e-12:
            continue
        row = {
            "AI1位的中AUC": round(auc(v, tab["AI1位的中"].to_numpy()), 4),
            "荒れAUC": round(auc(v, tab["荒れた"].to_numpy()), 4),
            "logloss相関": round(float(np.corrcoef(v, tab["レースlogloss"])[0, 1]), 4),
            "五分位": [],
        }
        q = pd.qcut(tab[c], 5, labels=False, duplicates="drop")
        for b in sorted(pd.unique(q.dropna())):
            s = tab[q == b]
            row["五分位"].append({
                "帯": int(b) + 1, "件数": len(s),
                "AI1位勝率": round(float(s["AI1位的中"].mean() * 100), 2),
                "AI3位内率": round(float(s["AI3位内的中"].mean() * 100), 2),
                "荒れ率": round(float(s["荒れた"].mean() * 100), 2),
                "勝ち馬平均人気": round(float(s["勝ち馬人気"].mean()), 2),
            })
        out[c] = row
    return out


def class_report(tab: pd.DataFrame, label: pd.Series) -> list[dict]:
    rows = []
    for name in ["予想しやすい", "混戦", "荒れる可能性が高い"]:
        s = tab[label == name]
        if s.empty:
            continue
        rows.append({
            "区分": name, "件数": len(s),
            "AI1位勝率": round(float(s["AI1位的中"].mean() * 100), 2),
            "AI3位内率": round(float(s["AI3位内的中"].mean() * 100), 2),
            "荒れ率": round(float(s["荒れた"].mean() * 100), 2),
            "勝ち馬平均人気": round(float(s["勝ち馬人気"].mean()), 2),
            "勝ち馬平均オッズ": round(float(s["勝ち馬オッズ"].mean()), 2),
            "平均レースlogloss": round(float(s["レースlogloss"].mean()), 4),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/past_features/corpus_panel.csv")
    ap.add_argument("--stability", default="data/quality_backtest/feature_stability.json")
    ap.add_argument("--out", default="data/quality_backtest/race_difficulty.json")
    ap.add_argument("--periods", type=int, default=3)
    args = ap.parse_args()

    d = load_panel(args.panel)
    periods = split_periods(d, args.periods)
    print(f"パネル {d['race_id'].nunique()}R / {len(d)}頭")

    y = d["win"].to_numpy(dtype=float)
    g = d["race_id"].to_numpy()
    log_impl = d["log_impl"].to_numpy(dtype=float)

    cols: list[str] = []
    sp = Path(args.stability)
    if sp.exists():
        stab = json.loads(sp.read_text(encoding="utf-8"))
        cols = [c for c in stab.get("安定して予測力あり", [])
                if c in d.columns and c in pf.FEATURE_COLUMNS]
    zc = {c: neutral_z(d, c)[0] for c in cols}
    print(f"馬の特徴量 {len(cols)}本を使った勝率でレースを評価する: {cols}")

    X = np.column_stack([log_impl] + [zc[c] for c in cols]) if cols \
        else log_impl[:, None]

    # レース特徴量もウォークフォワードで作る。各期間の勝率は、その期間より
    # 前のデータだけで推定した係数から出す。
    tabs = []
    for i, (tag, m) in enumerate(periods):
        if i == 0:
            continue
        prior = np.zeros(len(d), dtype=bool)
        for _, pm in periods[:i]:
            prior |= pm
        beta = fit_conditional_logit(X[prior], y[prior], g[prior], l2=1e-4)
        p = predict_cl(X[m], beta, g[m])
        t = race_table(d, p, m, seed=i)
        t["期間"] = tag
        tabs.append(t)
        print(f"  期間{tag}: {len(t)}R")
    tab = pd.concat(tabs, ignore_index=True).sort_values("date")
    tab = tab[tab["勝ち馬オッズ"].notna()].reset_index(drop=True)
    print(f"レース表 {len(tab)}R  AI1位勝率 {tab['AI1位的中'].mean()*100:.2f}%  "
          f"荒れ率 {tab['荒れた'].mean()*100:.2f}%")

    signals = single_signal_report(tab)
    print(f"\n{'signal':<14}{'的中AUC':>9}{'荒れAUC':>9}{'LL相関':>9}")
    for c, r in sorted(signals.items(), key=lambda kv: -kv[1]["AI1位的中AUC"]):
        print(f"{c:<14}{r['AI1位的中AUC']:>9.4f}{r['荒れAUC']:>9.4f}"
              f"{r['logloss相関']:>9.4f}")

    # 合成モデルは前半で学習し後半で評価する。レース表は時系列順。
    cut = len(tab) // 2
    tr, ho = tab.iloc[:cut], tab.iloc[cut:]
    zx = zfit(tr, RACE_FEATURES)
    bh = logistic_fit(zx(tr), tr["AI1位的中"].to_numpy())
    bu = logistic_fit(zx(tr), tr["荒れた"].to_numpy())
    ph, pu = logistic_predict(zx(ho), bh), logistic_predict(zx(ho), bu)
    comb = {
        "学習R": len(tr), "評価R": len(ho),
        "予想しやすさモデルAUC": round(auc(ph, ho["AI1位的中"].to_numpy()), 4),
        "荒れモデルAUC": round(auc(pu, ho["荒れた"].to_numpy()), 4),
        "単一指標_AI1位信頼度AUC": round(
            auc(ho["AI1位信頼度"].to_numpy(), ho["AI1位的中"].to_numpy()), 4),
    }
    print(f"\n合成モデル（学習{len(tr)}R→評価{len(ho)}R）")
    print(" ", json.dumps(comb, ensure_ascii=False))

    hi_h = np.quantile(logistic_predict(zx(tr), bh), 2 / 3)
    hi_u = np.quantile(logistic_predict(zx(tr), bu), 2 / 3)
    label = pd.Series("混戦", index=ho.index)
    label[pu >= hi_u] = "荒れる可能性が高い"
    label[ph >= hi_h] = "予想しやすい"
    rows = class_report(ho, label)
    print(f"\n{'区分':<12}{'件数':>7}{'AI1位勝率':>10}{'AI3位内':>9}{'荒れ率':>8}"
          f"{'勝馬人気':>9}")
    for r in rows:
        print(f"{r['区分']:<12}{r['件数']:>7}{r['AI1位勝率']:>10.2f}"
              f"{r['AI3位内率']:>9.2f}{r['荒れ率']:>8.2f}{r['勝ち馬平均人気']:>9.2f}")

    out = {
        "レース数": len(tab),
        "期間別": [{"期間": t, "レース数": int((tab["期間"] == t).sum()),
                 "AI1位勝率": round(float(tab.loc[tab["期間"] == t, "AI1位的中"].mean()
                                      * 100), 2)}
                for t in tab["期間"].unique()],
        "使用した馬の特徴量": cols,
        "単独signal": signals,
        "合成モデル": comb,
        "係数_予想しやすさ": {c: round(float(v), 4)
                   for c, v in zip(RACE_FEATURES, bh[1:])},
        "係数_荒れ": {c: round(float(v), 4) for c, v in zip(RACE_FEATURES, bu[1:])},
        "区分別実績": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n書き出し {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
