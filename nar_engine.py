"""NAR地方専用予想エンジン。

JRAロジックを流用せず、ダート適性・開催場実績・近走の地方戦績を主軸にする。
モンテカルロ仮想レースは共通乱数エンジンを使うが、指数の作り方が異なる。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from areru_engine import (
    clamp,
    clean_name,
    num,
    parse_date,
    simulate_race,
    _optimize_ticket,
    _ticket_candidates,
)
from nar_client import NAR_VENUE_CODES

DATA = Path("data")
CONFIG = DATA / "areru_nar_config.json"
DEFAULT_WEIGHTS = {
    "dirt": 0.30,
    "venue": 0.22,
    "form": 0.20,
    "upset": 0.14,
    "consistency": 0.14,
}


def load_weights():
    if CONFIG.exists():
        try:
            return {**DEFAULT_WEIGHTS, **json.loads(CONFIG.read_text(encoding="utf-8")).get("weights", {})}
        except Exception:
            pass
    return DEFAULT_WEIGHTS.copy()


def venue_from_nar_race_id(race_id: str) -> str:
    rid = str(race_id)
    if re.fullmatch(r"\d{12}", rid):
        return NAR_VENUE_CODES.get(rid[4:6], f"地方{rid[4:6]}")
    return "地方"


def score_nar_runner(row, history, target, weights, venue: str):
    finishes = np.array([num(row.get(f"着順{i}")) for i in range(1, 6)], dtype=float)
    pops = np.array([num(row.get(f"人気{i}")) for i in range(1, 6)], dtype=float)
    valid = ~np.isnan(finishes)
    if not valid.any():
        form = upset = cons = 40.0
    else:
        q = np.clip(108 - finishes * 8, 0, 100)
        w = np.array([1.0, 0.82, 0.65, 0.48, 0.34])[: len(finishes)]
        ww = w[valid]
        form = float(np.sum(q[valid] * ww) / np.sum(ww))
        gaps = pops - finishes
        upset = clamp(50 + float(np.nanmean(gaps[valid])) * 7) if (~np.isnan(gaps[valid])).any() else 50
        cons = float(np.sum((finishes[valid] <= 5).astype(float) * 100 * ww) / np.sum(ww))

    dirt = 48.0
    venue_score = 45.0
    reasons = []
    if history is not None and not history.empty:
        h = history[
            (history["_horse"] == clean_name(row["馬名"])) & (history["_date"] < target)
        ].sort_values("_date", ascending=False).head(10)
        if len(h):
            dist = h["距離"].astype(str)
            dirt_mask = dist.str.contains("ダ", na=False)
            dirt_h = h[dirt_mask]
            if len(dirt_h) >= 2:
                df = num(dirt_h["着順"])
                rate = float((df <= 5).mean())
                dirt = clamp(40 + rate * 55)
                if rate >= 0.5:
                    reasons.append("ダート適性")
            same = h[h["場"].astype(str).str.contains(str(venue), na=False)]
            if len(same) >= 2:
                sf = num(same["着順"])
                vr = float((sf <= 5).mean())
                venue_score = clamp(42 + vr * 50)
                if vr >= 0.45:
                    reasons.append(f"{venue}実績")
            elif len(h) >= 2:
                reasons.append("他場転入")
        else:
            reasons.append("地方履歴少")
    else:
        reasons.append("地方履歴少")

    if form >= 65:
        reasons.append("近走地方好走")
    if upset >= 65:
        reasons.append("人気以上傾向")

    factors = {
        "dirt": dirt,
        "venue": venue_score,
        "form": form,
        "upset": upset,
        "consistency": cons,
    }
    score = sum(factors[k] * weights[k] for k in weights)
    return clamp(score), factors, reasons


def build_nar_predictions(target_str, runners, history=None, weights=None):
    target = pd.Timestamp(target_str)
    weights = weights or load_weights()
    r = runners.copy()
    r["_date"] = parse_date(r["日付"])
    r = r[r["_date"].dt.normalize() == target.normalize()].copy()
    if r.empty:
        raise ValueError(f"{target_str} のNAR出走データがありません")
    if history is not None:
        history = history.copy()
        history["_date"] = parse_date(history.get("日付", history.get("年月日")))
        history["_horse"] = history["馬名"].map(clean_name)

    scored = []
    for _, row in r.iterrows():
        venue = venue_from_nar_race_id(row["race_id"])
        s, f, why = score_nar_runner(row, history, target, weights, venue)
        x = row.to_dict()
        x.update(
            {
                "AREru指数": round(s, 2),
                **{f"因子_{k}": round(v, 1) for k, v in f.items()},
                "理由": " / ".join(dict.fromkeys(why[:4])) or "地方総合評価",
                "開催地": venue,
            }
        )
        scored.append(x)
    sd = pd.DataFrame(scored)
    out = []
    for race_id, g0 in sd.groupby("race_id", sort=False):
        # simulate_race expects 因子_consistency
        g0 = g0.copy()
        if "因子_consistency" not in g0.columns:
            g0["因子_consistency"] = g0.get("因子_consistency", 50)
        g, orders = simulate_race(g0.sort_values("AREru指数", ascending=False), 20000)
        n = len(g)
        top = g["AREru指数"].iloc[0]
        spread = top - g["AREru指数"].iloc[min(4, n - 1)]
        chaos = clamp(30 + (70 - top) * 0.7 + (16 - spread) * 1.2 + (g["因子_upset"] >= 65).mean() * 25)
        main = g.sort_values(["SIM3着内率", "AREru指数"], ascending=False).iloc[0]
        rest = g[g["馬名"] != main["馬名"]].copy()
        hole = rest["AREru指数"] * 0.45 + rest["因子_upset"] * 0.2 + rest["SIM3着内率"] * 0.35
        ranked = rest.assign(_h=hole).sort_values(["_h", "SIM3着内率"], ascending=False).head(4)
        marks = ["○", "▲", "△", "☆"]
        mark_rows = []
        for mark, (_, x) in zip(marks, ranked.iterrows()):
            mark_rows.append(
                {
                    "印": mark,
                    "馬名": str(x["馬名"]),
                    "3着内率": round(float(x["SIM3着内率"]), 1),
                    "理由": str(x["理由"]),
                }
            )
        main_place = float(main["SIM3着内率"])
        alt_place = float(ranked["SIM3着内率"].max()) if len(ranked) else 0
        clarity = max(0, main_place - float(g["SIM3着内率"].median()))
        bet = clamp(main_place * 0.4 + alt_place * 0.2 + clarity * 0.7 + chaos * 0.18)
        judge = "大荒れ警戒" if chaos >= 80 else ("波乱" if chaos >= 60 else ("注意" if chaos >= 40 else "平穏"))
        candidates = _ticket_candidates(g, orders)
        wide_plan = _optimize_ticket("ワイド", candidates["ワイド"], g, 3)
        quinella_plan = _optimize_ticket("馬連", candidates["馬連"], g, 2)
        trio_plan = _optimize_ticket("三連複", candidates["三連複"], g, 6)
        wide_score = clamp(wide_plan["的中期待"] * 2.0 + main_place * 0.45)
        quinella_score = clamp(quinella_plan["的中期待"] * 3.0 + float(main["SIM勝率"]) * 0.8)
        trio_score = clamp(trio_plan["的中期待"] * 5.0 + chaos * 0.25)
        plans = sorted(
            [("ワイド", wide_score, wide_plan), ("馬連", quinella_score, quinella_plan), ("三連複", trio_score, trio_plan)],
            key=lambda x: x[1],
            reverse=True,
        )
        best_kind = plans[0][0]

        def go_label(v):
            return "買い候補" if v >= 70 else ("条件付き" if v >= 55 else "見送り")

        def plan_text(plan):
            return (
                "｜".join(f"{x['馬名']}（仮想的中 {x['仮想的中率']}%）" for x in plan["買い目"])
                if plan["買い目"]
                else "見送り"
            )

        danger = g.sort_values("AREru指数").iloc[0] if len(g) else main
        out.append(
            {
                "race_id": race_id,
                "開催地": venue_from_nar_race_id(race_id),
                "レース": int(float(main["レース"])),
                "荒れ度": round(chaos, 1),
                "判定": judge,
                "荒れクラス": "storm" if chaos >= 80 else ("wave" if chaos >= 60 else ("caution" if chaos >= 40 else "calm")),
                "BET期待値": round(bet, 1),
                "BET判定": "",
                "BETクラス": "",
                "BET理由": "地方専用指数（ダート/開催場/近走）",
                "シミュレーション回数": 20000,
                "本命": str(main["馬名"]),
                "本命AREru指数": main["AREru指数"],
                "シミュレーション勝率": round(main["SIM勝率"], 1),
                "シミュレーション3着内率": round(main["SIM3着内率"], 1),
                "AI適正オッズ": round(main["AI適正オッズ"], 1),
                "本命理由": main["理由"],
                "人気馬危険": str(danger["馬名"]),
                "危険度": round(clamp(100 - float(danger["AREru指数"])), 1),
                "危険理由": "地方戦績と指数の乖離監視",
                "印データ": json.dumps(mark_rows, ensure_ascii=False),
                "推奨券種": best_kind,
                "馬券戦略理由": f"NAR専用。{best_kind}型。ダート/開催場適性を軸に20,000回仮想レースで圧縮。",
                "ワイド評価": round(wide_score, 1),
                "ワイド判定": go_label(wide_score),
                "ワイド買い目": plan_text(wide_plan),
                "ワイド圧縮": wide_plan["圧縮理由"],
                "馬連評価": round(quinella_score, 1),
                "馬連判定": go_label(quinella_score),
                "馬連買い目": plan_text(quinella_plan),
                "馬連圧縮": quinella_plan["圧縮理由"],
                "三連複評価": round(trio_score, 1),
                "三連複判定": go_label(trio_score),
                "三連複買い目": plan_text(trio_plan),
                "三連複圧縮": trio_plan["圧縮理由"],
                "合成オッズ": "券種別オッズ待ち",
                "期待回収率": "オッズ接続後に算出",
                "データ頭数": n,
                "source": "nar",
            }
        )

    result = pd.DataFrame(out).sort_values(["開催地", "レース"]).reset_index(drop=True)
    raw_bet = result["BET期待値"].astype(float).copy()
    order = raw_bet.rank(method="first", ascending=False).astype(int)
    total = len(result)
    pct = (total - order) / max(total - 1, 1)
    raw_min, raw_max = float(raw_bet.min()), float(raw_bet.max())
    raw_norm = (raw_bet - raw_min) / (raw_max - raw_min) if raw_max > raw_min else pd.Series(0.5, index=result.index)
    result["買い期待度基礎値"] = raw_bet.round(1)
    result["BET期待値"] = (38 + pct * 52 + raw_norm * 8).clip(0, 98.7).round(1)

    def grade(rank):
        if rank <= min(2, total):
            return "S"
        if rank <= min(5, total):
            return "A"
        if rank <= max(8, int(total * 0.35)):
            return "B"
        return "C"

    result["勝負ランク"] = order.map(grade)
    result["BET判定"] = result["勝負ランク"].map({"S": "今日の勝負", "A": "買い候補", "B": "オッズ次第", "C": "見送り"})
    result["BETクラス"] = result["勝負ランク"].map({"S": "battle", "A": "target", "B": "watch", "C": "skip"})
    return result, sd
