#!/usr/bin/env python3
"""的中率重視 / EV重視 / 的中率+EV+信頼度 の3方針を同じ過去レースで比べる。

資金が少ない人が使う前提なので、見るのは「どれが一番儲かったか」ではない。
同じ点数を買ったときに

  * どれだけ当たるか（的中率・複勝率）
  * どれだけ減らさずに済むか（最大ドローダウン・最大連敗）
  * 確率が信用できるか（Brier・Log Loss・較正）

を分けて見る。3方針は買い目の点数を揃えて比べる。点数が違うと的中率も
ドローダウンも比べられないため、各方針とも「自分の基準で上位N点」を買う
形にし、N を揃える。しきい値は学習期間の分布から決め、評価期間には
触らせない。

払戻は単勝のみ。複勝配当はコーパスに無いので、複勝は的中率としてだけ
扱い、複勝ROIは出さない（無い数字を作らない）。
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
from market_edge_probe import (  # noqa: E402
    auc,
    fit_conditional_logit,
    predict_cl,
    race_logloss,
)
from model_compare import calibration, place_probs_from_win  # noqa: E402
from race_difficulty import (  # noqa: E402
    RACE_FEATURES,
    SIM_DRAWS,
    SIM_NOISE,
    logistic_fit,
    logistic_predict,
    zfit,
)

LONGSHOT_ODDS = 10.0
# 1レースあたり何点買うか。資金が少ない人向けなので少ない側を厚く見る。
VOLUMES = [0.05, 0.10, 0.20, 0.50]
MAIN_VOLUME = 0.10
# 対数成長率を測るときの1点あたり賭け金（資金に対する割合）。
BANKROLL_FRACTION = 0.02


def sim_top_rates(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """log能力値を揺らして並べ替えたとき、各馬が1位になる頻度。

    係数も過去走もノイズを含む。揺らしても1位が動かない馬は「確率の出方が
    安定している」ことになるので、これを馬ごとの信頼度として使う。
    """
    lo = np.log(np.clip(p, 1e-9, None))
    noise = rng.normal(0.0, SIM_NOISE, size=(SIM_DRAWS, len(p)))
    top = np.argmax(lo[None, :] + noise, axis=1)
    return np.bincount(top, minlength=len(p)) / SIM_DRAWS


def horse_table(d: pd.DataFrame, p: np.ndarray, mask: np.ndarray,
                seed: int = 0) -> pd.DataFrame:
    """馬ごとの勝率から、賭けの判断に要る列をすべて並べた表を作る。

    結果列（着順・配当）も入れるが、判断に使う列とは明確に分けてある。
    """
    sub = d.loc[mask].copy()
    sub["p_raw"] = p
    rng = np.random.default_rng(seed)
    out = []
    for rid, race in sub.groupby("race_id", sort=False):
        pr = race["p_raw"].to_numpy(dtype=float)
        if pr.sum() <= 0 or len(pr) < 4:
            continue
        pr = pr / pr.sum()
        odds = race["単勝オッズ"].to_numpy(dtype=float)
        third = place_probs_from_win(pr)
        sim = sim_top_rates(pr, rng)
        order = np.argsort(-pr)
        srt = pr[order]

        rank_ai = pd.Series(-pr).rank().to_numpy()
        rank_mk = pd.Series(odds).rank().to_numpy()
        dev = 0.0 if np.std(rank_ai) < 1e-9 or np.std(rank_mk) < 1e-9 \
            else 1.0 - float(np.corrcoef(rank_ai, rank_mk)[0, 1])
        longshot = pr[odds >= LONGSHOT_ODDS]

        t = pd.DataFrame({
            "race_id": rid,
            "date": race["date"].to_numpy(),
            "馬名": race["馬名"].to_numpy(),
            "p": pr,
            "3着内率": np.clip(third, 0.0, 1.0),
            "信頼度": sim,
            "odds": odds,
            "EV": pr * odds,
            "AI順位": np.argsort(np.argsort(-pr)) + 1,
            "頭数": len(pr),
            # レース単位の指標。同じレースの全行に同じ値が入る。
            "1位2位差": float(srt[0] - srt[1]),
            "上位3集中度": float(srt[:3].sum()),
            "確率分散": float(np.std(pr) * np.sqrt(len(pr))),
            "AI1位信頼度": float(srt[0]),
            "人気乖離": dev,
            "穴馬候補": float(longshot.max()) if longshot.size else 0.0,
            "SIM安定性": float(sim[order[0]]),
            # 結果。
            "win": (race["着順"].to_numpy(dtype=float) == 1).astype(float),
            "top3": (race["着順"].to_numpy(dtype=float) <= 3).astype(float),
        })
        for c in pf.FEATURE_COLUMNS:
            if c in race.columns:
                t[c] = race[c].to_numpy()
        out.append(t)
    return pd.concat(out, ignore_index=True)


def max_drawdown(profit: np.ndarray) -> float:
    """1点1単位で賭け続けたときの、山からの最大の落ち込み（単位）。"""
    if len(profit) == 0:
        return 0.0
    cum = np.cumsum(profit)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum))


def max_losing_streak(win: np.ndarray) -> int:
    best = cur = 0
    for w in win:
        cur = 0 if w > 0 else cur + 1
        best = max(best, cur)
    return best


def log_growth(profit: np.ndarray, f: float = BANKROLL_FRACTION) -> float:
    """資金の f を毎回賭けたときの、1点あたり平均対数成長率（%）。

    的中率が低く配当が大きい買い方は、同じROIでもここが低く出る。
    資金が少ない人にとっての「運用しやすさ」はROIよりこちらに近い。
    """
    if len(profit) == 0:
        return float("nan")
    return float(np.mean(np.log1p(np.clip(f * profit, -0.999999, None))) * 100)


def bet_metrics(b: pd.DataFrame, n_races: int) -> dict:
    """買い目の束に対する ①的中性能 ②収益性能 ③信頼性 をまとめて出す。"""
    if b.empty:
        return {"件数": 0}
    b = b.sort_values(["date", "race_id"])
    win = b["win"].to_numpy(dtype=float)
    odds = b["odds"].to_numpy(dtype=float)
    p = np.clip(b["p"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    profit = win * odds - 1.0
    hit_races = b.groupby("race_id")["win"].max()

    return {
        "件数": int(len(b)),
        "買ったレース数": int(b["race_id"].nunique()),
        "1レースあたりBET数": round(len(b) / max(n_races, 1), 3),
        "的中性能": {
            "勝率": round(float(win.mean() * 100), 2),
            "複勝率": round(float(b["top3"].mean() * 100), 2),
            "レース的中率": round(float(hit_races.mean() * 100), 2),
        },
        "収益性能": {
            "ROI": round(float((win * odds).sum() / len(b) * 100), 2),
            "平均オッズ": round(float(odds.mean()), 2),
            "的中時平均オッズ": round(float(odds[win > 0].mean()), 2)
            if win.any() else None,
            "平均EV": round(float(b["EV"].mean() * 100), 2),
            "1点あたり期待利益": round(float((b["EV"] - 1).mean() * 100), 2),
            "1点あたり実利益": round(float(profit.mean() * 100), 2),
            "最大ドローダウン": round(max_drawdown(profit), 1),
            "最大連敗": max_losing_streak(win),
            "対数成長率": round(log_growth(profit), 4),
        },
        "信頼性": {
            "Brier": round(float(np.mean((p - win) ** 2)), 6),
            "LogLoss": round(float(-np.mean(win * np.log(p)
                                            + (1 - win) * np.log(1 - p))), 5),
            "予測勝率平均": round(float(p.mean() * 100), 2),
            "実績勝率との差": round(float((win.mean() - p.mean()) * 100), 2),
            "Calibration": calibration(p, win),
        },
    }


def fit_race_grade(trr: pd.DataFrame) -> dict:
    """レース判定モデルの係数と区切りを、学習期間だけから作る。"""
    mu = trr[RACE_FEATURES].mean()
    sd = trr[RACE_FEATURES].std()
    sd = sd.where(sd > 1e-9, 1.0)
    z = ((trr[RACE_FEATURES] - mu) / sd).to_numpy(dtype=float)
    bh = logistic_fit(z, trr["AI1位的中"].to_numpy())
    bu = logistic_fit(z, trr["荒れた"].to_numpy())
    ph, pu = logistic_predict(z, bh), logistic_predict(z, bu)
    return {
        "特徴量": RACE_FEATURES,
        "平均": {c: float(mu[c]) for c in RACE_FEATURES},
        "標準偏差": {c: float(sd[c]) for c in RACE_FEATURES},
        "予想しやすさ係数": [float(v) for v in bh],
        "荒れやすさ係数": [float(v) for v in bu],
        "買うべき下限": float(np.quantile(ph, 2 / 3)),
        "見送り下限": float(np.quantile(ph, 1 / 3)),
        "荒れ上限": float(np.quantile(pu, 2 / 3)),
    }


def race_class(tab: pd.DataFrame, tr: pd.DataFrame) -> tuple[pd.Series, dict]:
    """レースを 買うべき / 少額なら買える / 見送る に分ける。

    予想しやすさ H と 荒れやすさ U の2本を学習期間だけで当てはめ、
    H が高く U が高くないレースを「買うべき」にする。混戦（H が低い）と
    荒れそうなレースは、期待値が高く見えても「見送る」に落とす。
    """
    races = tab.drop_duplicates("race_id")
    trr = tr.drop_duplicates("race_id")
    zx = zfit(trr, RACE_FEATURES)
    bh = logistic_fit(zx(trr), trr["AI1位的中"].to_numpy())
    bu = logistic_fit(zx(trr), trr["荒れた"].to_numpy())
    hi_h = float(np.quantile(logistic_predict(zx(trr), bh), 2 / 3))
    hi_u = float(np.quantile(logistic_predict(zx(trr), bu), 2 / 3))
    lo_h = float(np.quantile(logistic_predict(zx(trr), bh), 1 / 3))

    ph = logistic_predict(zx(races), bh)
    pu = logistic_predict(zx(races), bu)
    lab = pd.Series("少額なら買える", index=races["race_id"].to_numpy())
    lab[(ph >= hi_h) & (pu < hi_u)] = "買うべき"
    lab[(ph < lo_h) | (pu >= hi_u)] = "見送る"
    coef = {"予想しやすさ": {c: round(float(v), 4)
                        for c, v in zip(RACE_FEATURES, bh[1:])},
            "荒れやすさ": {c: round(float(v), 4)
                      for c, v in zip(RACE_FEATURES, bu[1:])}}
    return tab["race_id"].map(lab), coef


def pick_by_score(t: pd.DataFrame, score: np.ndarray, thr: float) -> pd.DataFrame:
    return t.loc[score >= thr]


def volume_threshold(score: np.ndarray, n_races: int, per_race: float) -> float:
    """学習期間で「1レースあたり per_race 点」になるしきい値を求める。"""
    want = int(round(n_races * per_race))
    valid = score[np.isfinite(score)]
    if want <= 0 or len(valid) == 0:
        return float("inf")
    if want >= len(valid):
        return float(np.min(valid))
    return float(np.partition(valid, -want)[-want])


# 勝率・期待値・信頼度の区切り。恣意的に動かさないよう固定値で置き、
# 意味の説明がつくところだけに置いた。
#   勝率30%  … 単勝で頭一つ抜けた馬の目安
#   EV100%   … 払戻の損益分岐そのもの
#   EV90%    … 分岐には届かないが、全体平均（控除率ぶん約79%）よりは明確に上
P_HIGH, P_MID = 0.30, 0.15
EV_HIGH, EV_MID = 1.00, 0.90


def axis_bands(t: pd.DataFrame, conf_cut: float) -> pd.DataFrame:
    """勝率・期待値・信頼度を別々の軸として帯に分ける。"""
    p = t["p"].to_numpy(dtype=float)
    ev = t["EV"].to_numpy(dtype=float)
    return pd.DataFrame({
        "勝率帯": np.where(p >= P_HIGH, "高勝率",
                        np.where(p >= P_MID, "中勝率", "低勝率")),
        "EV帯": np.where(ev >= EV_HIGH, "高EV",
                       np.where(ev >= EV_MID, "中EV", "低EV")),
        "信頼度帯": np.where(t["SIM安定性"].to_numpy(dtype=float) >= conf_cut,
                         "高信頼度", "低信頼度"),
    }, index=t.index)


def mix_gate(t: pd.DataFrame, cls: pd.Series, conf_cut: float) -> np.ndarray:
    """的中率とEVを別軸で見て、買ってよい馬だけを残す。

      高勝率 × 中EV以上 × 高信頼度 → 資金が少ない場合の優先候補
      中勝率 × 高EV     × 高信頼度 → 積極候補
      低勝率、または低EV、または低信頼度 → 見送り

    さらに、見送り判定のレース（混戦・荒れそう）の馬はすべて落とす。
    """
    b = axis_bands(t, conf_cut)
    conf = (b["信頼度帯"] == "高信頼度").to_numpy()
    prime = (b["勝率帯"] == "高勝率").to_numpy() & (b["EV帯"] != "低EV").to_numpy()
    aggressive = (b["勝率帯"] == "中勝率").to_numpy() & (b["EV帯"] == "高EV").to_numpy()
    return (prime | aggressive) & conf & (cls.to_numpy() != "見送る")


def strategy_scores(t: pd.DataFrame, cls: pd.Series,
                    conf_cut: float) -> dict[str, np.ndarray]:
    """3方針それぞれの「買いたい順」を作る。

    的中率重視 … 勝率が高い順。配当は見ない。
    EV重視     … 期待値が高い順。当たりやすさは見ない。
    複合       … 勝率・EV・信頼度の3軸を通った馬だけを、勝率順に。
    """
    p = t["p"].to_numpy(dtype=float)
    ev = t["EV"].to_numpy(dtype=float)
    mix = np.where(mix_gate(t, cls, conf_cut), p, -np.inf)
    return {"的中率重視": p, "EV重視": ev, "的中率+EV+信頼度": mix}


def grid_report(t: pd.DataFrame, conf_cut: float) -> list[dict]:
    """勝率×EV×信頼度のマス目ごとの実績。EVだけで買ってよいかを見る。"""
    b = axis_bands(t, conf_cut)
    s = t.copy()
    s[["勝率帯", "EV帯", "信頼度帯"]] = b
    out = []
    for key, grp in s.groupby(["勝率帯", "EV帯", "信頼度帯"], sort=False):
        if len(grp) < 100:
            continue
        win = grp["win"].to_numpy(dtype=float)
        out.append({
            "勝率帯": key[0], "EV帯": key[1], "信頼度帯": key[2],
            "件数": len(grp),
            "勝率": round(float(win.mean() * 100), 2),
            "複勝率": round(float(grp["top3"].mean() * 100), 2),
            "ROI": round(float((win * grp["odds"]).sum() / len(grp) * 100), 2),
            "平均オッズ": round(float(grp["odds"].mean()), 2),
            "平均EV": round(float(grp["EV"].mean() * 100), 2),
        })
    return sorted(out, key=lambda r: -r["ROI"])


def confidence_report(b: pd.DataFrame) -> list[dict]:
    """信頼度別の実績。レースのSIM安定性で区切る。"""
    out = []
    edges = [0, .2, .35, .5, .65, .8, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = b[(b["SIM安定性"] >= lo) & (b["SIM安定性"] < hi)]
        if len(s) < 30:
            continue
        win = s["win"].to_numpy(dtype=float)
        out.append({
            "信頼度帯": f"{lo*100:.0f}-{hi*100:.0f}%",
            "件数": len(s),
            "勝率": round(float(win.mean() * 100), 2),
            "複勝率": round(float(s["top3"].mean() * 100), 2),
            "ROI": round(float((win * s["odds"]).sum() / len(s) * 100), 2),
            "平均オッズ": round(float(s["odds"].mean()), 2),
        })
    return out


HONMEI_RULES = {
    "R1_勝率最大(現行)": lambda t: t["p"].to_numpy(dtype=float),
    "R2_3着内率最大": lambda t: t["3着内率"].to_numpy(dtype=float),
    "R3_勝率x信頼度": lambda t: (t["p"] * t["信頼度"]).to_numpy(dtype=float),
    "R4_勝率と3着内率の等分": lambda t: (
        t.groupby("race_id")["p"].transform(
            lambda s: (s - s.mean()) / (s.std() or 1))
        + t.groupby("race_id")["3着内率"].transform(
            lambda s: (s - s.mean()) / (s.std() or 1))).to_numpy(dtype=float),
}


def honmei_report(t: pd.DataFrame, extra: dict[str, np.ndarray]) -> dict:
    """本命の決め方を変えると何が変わるかを並べる。"""
    out = {}
    rules = {k: f(t) for k, f in HONMEI_RULES.items()} | extra
    for name, sc in rules.items():
        s = t.copy()
        s["_sc"] = sc
        idx = s.groupby("race_id")["_sc"].idxmax()
        h = s.loc[idx]
        rv = s.drop(index=idx)
        riv = rv.loc[rv.groupby("race_id")["p"].idxmax()]
        win = h["win"].to_numpy(dtype=float)
        out[name] = {
            "本命勝率": round(float(win.mean() * 100), 2),
            "本命複勝率": round(float(h["top3"].mean() * 100), 2),
            "本命ROI": round(float((win * h["odds"]).sum() / len(h) * 100), 2),
            "本命平均オッズ": round(float(h["odds"].mean()), 2),
            "対抗複勝率": round(float(riv["top3"].mean() * 100), 2),
            "AI1位と一致した割合": round(float((h["AI順位"] == 1).mean() * 100), 2),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/past_features/corpus_panel.csv")
    ap.add_argument("--stability", default="data/quality_backtest/feature_stability.json")
    ap.add_argument("--out", default="data/quality_backtest/strategy_compare.json")
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--model", default="market", choices=["market", "stable"],
                    help="勝率をどのモデルで出すか。market=市場のみ（現行相当）")
    ap.add_argument("--export-grade", default=None,
                    help="レース判定モデルの係数をこのパスに書き出す")
    args = ap.parse_args()

    d = load_panel(args.panel)
    periods = split_periods(d, args.periods)
    print(f"パネル {d['race_id'].nunique()}R / {len(d)}頭  勝率モデル={args.model}")
    for t, m in periods:
        print(f"  期間{t} {int(d.loc[m,'race_id'].nunique())}R")

    y = d["win"].to_numpy(dtype=float)
    g = d["race_id"].to_numpy()
    log_impl = d["log_impl"].to_numpy(dtype=float)

    cols: list[str] = []
    if args.model == "stable":
        stab = json.loads(Path(args.stability).read_text(encoding="utf-8"))
        cols = [c for c in stab.get("安定して予測力あり", [])
                if c in d.columns and c in pf.FEATURE_COLUMNS]
    zc = {c: neutral_z(d, c)[0] for c in cols}
    X = np.column_stack([log_impl] + [zc[c] for c in cols]) if cols \
        else log_impl[:, None]
    print(f"  追加特徴量 {len(cols)}本 {cols}")

    # 期間ごとにウォークフォワード。係数もしきい値もレース分類も、
    # その期間より前のデータだけから決める。
    tables: list[pd.DataFrame] = []
    for i, (tag, m) in enumerate(periods):
        prior = np.zeros(len(d), dtype=bool)
        for _, pm in periods[:i]:
            prior |= pm
        if i == 0:
            beta = fit_conditional_logit(X[m], y[m], g[m], l2=1e-4)
        else:
            beta = fit_conditional_logit(X[prior], y[prior], g[prior], l2=1e-4)
        p = predict_cl(X[m], beta, g[m])
        t = horse_table(d, p, m, seed=i)
        t["期間"] = tag
        t["評価対象"] = i > 0
        tables.append(t)
        print(f"  期間{tag}: {t['race_id'].nunique()}R / {len(t)}頭 を評価用に展開")
    T = pd.concat(tables, ignore_index=True).sort_values(["date", "race_id"])

    # レース単位の表（分類器の学習用）。
    R = T.sort_values(["race_id", "AI順位"]).groupby("race_id", sort=False).agg(
        date=("date", "first"), 期間=("期間", "first"),
        AI1位的中=("win", "first"),
        勝ち馬オッズ=("odds", lambda s: np.nan),
        **{c: (c, "first") for c in RACE_FEATURES}).reset_index()
    wo = T[T["win"] > 0].groupby("race_id")["odds"].first()
    R["勝ち馬オッズ"] = R["race_id"].map(wo)
    R = R[R["勝ち馬オッズ"].notna()].copy()
    R["荒れた"] = (R["勝ち馬オッズ"] >= LONGSHOT_ODDS).astype(float)
    T = T[T["race_id"].isin(set(R["race_id"]))].copy()

    tags = [t for t, _ in periods]
    report: dict = {
        "設定": {
            "勝率モデル": args.model,
            "追加特徴量": cols,
            "期間": [{"期間": t, "レース数": int((R["期間"] == t).sum())} for t in tags],
            "払戻": "単勝のみ。複勝配当はコーパスに無いため複勝は的中率としてのみ扱う",
            "賭け方": "1点1単位の定額。ドローダウンと連敗は時系列順に計算",
        },
        "方針比較": {}, "点数別": {}, "レース分類": {},
        "本命の決め方": {}, "信頼度別実績": {}, "軸別マス目": {},
        "軸の区切り": {"高勝率": P_HIGH, "中勝率": P_MID,
                  "高EV": EV_HIGH, "中EV": EV_MID,
                  "高信頼度": "学習期間のSIM安定性の中央値"},
    }

    # 評価は助走を除いた期間。各期間で、しきい値とレース分類は前の期間から。
    for ti in range(1, len(periods)):
        tag = tags[ti]
        prior_tags = tags[:ti]
        tr = T[T["期間"].isin(prior_tags)]
        te = T[T["期間"] == tag]
        trR = R[R["期間"].isin(prior_tags)]
        teR = R[R["期間"] == tag]
        n_tr, n_te = trR["race_id"].nunique(), teR["race_id"].nunique()

        cls_tr, coef = race_class(tr.merge(trR[["race_id", "AI1位的中", "荒れた"]],
                                           on="race_id", how="left"), trR)
        cls_te, _ = race_class(te.merge(teR[["race_id", "AI1位的中", "荒れた"]],
                                        on="race_id", how="left"), trR)
        cls_tr.index, cls_te.index = tr.index, te.index

        conf_cut = float(tr["SIM安定性"].median())
        sc_tr = strategy_scores(tr, cls_tr, conf_cut)
        sc_te = strategy_scores(te, cls_te, conf_cut)
        report["軸別マス目"][tag] = {"信頼度の区切り": round(conf_cut, 4),
                                "マス目": grid_report(te, conf_cut)}
        gated = te[mix_gate(te, cls_te, conf_cut)]
        report["方針比較"].setdefault("複合_ゲート通過すべて", {})[tag] = \
            bet_metrics(gated, n_te)

        for vol in VOLUMES:
            for name in sc_tr:
                thr = volume_threshold(sc_tr[name], n_tr, vol)
                bets = pick_by_score(te, sc_te[name], thr)
                key = f"{vol:.2f}点/R"
                report["点数別"].setdefault(key, {}).setdefault(name, {})[tag] = \
                    bet_metrics(bets, n_te)

        for name in sc_te:
            thr = volume_threshold(sc_tr[name], n_tr, MAIN_VOLUME)
            bets = pick_by_score(te, sc_te[name], thr)
            report["方針比較"].setdefault(name, {})[tag] = bet_metrics(bets, n_te)
            report["信頼度別実績"].setdefault(name, {})[tag] = confidence_report(bets)

        # 参考: 全レースで本命を1点だけ買う（現行のBUYに近い形）
        hon = te.loc[te.groupby("race_id")["p"].idxmax()]
        report["方針比較"].setdefault("参考_本命を全レース", {})[tag] = \
            bet_metrics(hon, n_te)
        report["方針比較"].setdefault("参考_本命かつ勝率35%以上", {})[tag] = \
            bet_metrics(hon[hon["p"] >= 0.35], n_te)
        hcls = hon["race_id"].map(
            pd.Series(cls_te.to_numpy(), index=te["race_id"].to_numpy())
            .groupby(level=0).first())
        report["方針比較"].setdefault("買うべきレースの本命のみ", {})[tag] = \
            bet_metrics(hon[hcls == "買うべき"], n_te)
        report["方針比較"].setdefault("買うべき+少額レースの本命のみ", {})[tag] = \
            bet_metrics(hon[hcls != "見送る"], n_te)
        for floor in (0.40, 0.50):
            report["方針比較"].setdefault(
                f"買うべきレースの本命かつ勝率{floor*100:.0f}%以上", {})[tag] = \
                bet_metrics(hon[(hcls == "買うべき") & (hon["p"] >= floor)], n_te)
        # 最終候補: 買うべきレースの本命に、3軸を通った馬を足す
        combo = pd.concat([hon[hcls == "買うべき"], gated]).drop_duplicates(
            subset=["race_id", "馬名"])
        report["方針比較"].setdefault("最終候補_買うべき本命+3軸通過", {})[tag] = \
            bet_metrics(combo, n_te)

        # レース分類ごとの実績
        cl = {}
        for lab in ["買うべき", "少額なら買える", "見送る"]:
            sub = te[cls_te == lab]
            if sub.empty:
                continue
            h = sub.loc[sub.groupby("race_id")["p"].idxmax()]
            win = h["win"].to_numpy(dtype=float)
            cl[lab] = {
                "レース数": int(sub["race_id"].nunique()),
                "本命勝率": round(float(win.mean() * 100), 2),
                "本命複勝率": round(float(h["top3"].mean() * 100), 2),
                "本命ROI": round(float((win * h["odds"]).sum() / len(h) * 100), 2),
                "1位2位差": round(float(h["1位2位差"].mean()), 4),
                "SIM安定性": round(float(h["SIM安定性"].mean()), 4),
                "荒れ率": round(float((teR.set_index("race_id")
                                    .loc[h["race_id"], "荒れた"]).mean() * 100), 2),
            }
        report["レース分類"][tag] = {"区分": cl, "係数": coef}

        # 本命の決め方。合成ルールは学習期間だけで係数を推定する。
        extra = {}
        feats = [c for c in cols if c in tr.columns] or \
            [c for c in pf.FEATURE_COLUMNS if c in tr.columns][:0]
        if feats:
            def zmat(df):
                base = [np.log(np.clip(df["p"].to_numpy(dtype=float), 1e-9, None))]
                for c in ["3着内率", "信頼度"] + feats:
                    v = pd.to_numeric(df[c], errors="coerce")
                    gg = v.groupby(df["race_id"])
                    z = ((v - gg.transform("mean"))
                         / gg.transform("std").replace(0, np.nan)).fillna(0.0)
                    base.append(z.to_numpy(dtype=float))
                return np.column_stack(base)
            bw = fit_conditional_logit(zmat(tr), tr["win"].to_numpy(dtype=float),
                                       tr["race_id"].to_numpy(), l2=1e-3)
            extra["R5_合成(係数は学習期間で推定)"] = zmat(te) @ bw
        report["本命の決め方"][tag] = honmei_report(te, extra)

    if args.export_grade:
        # 本番に載せる係数は、最後の期間を残して学習する。期間Cは答え合わせ用。
        trR = R[R["期間"].isin(tags[:-1])]
        gm = fit_race_grade(trR) | {
            "学習期間": tags[:-1], "学習レース数": int(trR["race_id"].nunique()),
            "勝率モデル": args.model,
        }
        Path(args.export_grade).write_text(
            json.dumps(gm, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"レース判定モデルを書き出し {args.export_grade}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    ev_tags = tags[1:]
    print(f"\n=== {MAIN_VOLUME}点/R で揃えた比較 ===")
    hdr = (f"{'方針':<22}{'期':>3}{'件数':>7}{'勝率':>7}{'複勝率':>7}{'ROI':>8}"
           f"{'平均Odds':>9}{'最大DD':>8}{'最大連敗':>8}{'対数成長':>9}{'Brier':>9}")
    print(hdr)
    for name, per in report["方針比較"].items():
        for tag, m in per.items():
            if not m.get("件数"):
                continue
            a, b2, c = m["的中性能"], m["収益性能"], m["信頼性"]
            print(f"{name:<22}{tag[:1]:>3}{m['件数']:>7}{a['勝率']:>7.2f}"
                  f"{a['複勝率']:>7.2f}{b2['ROI']:>8.2f}{b2['平均オッズ']:>9.2f}"
                  f"{b2['最大ドローダウン']:>8.1f}{b2['最大連敗']:>8d}"
                  f"{b2['対数成長率']:>9.4f}{c['Brier']:>9.5f}")
    print(f"\n書き出し {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
