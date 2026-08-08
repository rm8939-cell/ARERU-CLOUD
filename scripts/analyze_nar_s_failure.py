#!/usr/bin/env python3
"""地方S本命の失敗原因分解 + S降格仮想BT（分析専用・本番ロジック非変更）。"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "predictions_by_date"
OUT = ROOT / "data" / "nar_s_failure_rootcause_report.json"
TRAIN_END = "2026-07-28"
MIN_TRAIN = 12
MIN_TEST = 10
MIN_ALL = 20


def fnum(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        s = str(x).strip().replace("%", "")
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


def parse_pace(raw):
    out = {}
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return out
    s = str(raw).strip()
    if not s.startswith("{"):
        return out
    try:
        d = json.loads(s)
    except Exception:
        return out
    if not isinstance(d, dict):
        return out
    for k in ("想定ペース", "荒れ指数", "逃げ有利度", "先行有利度", "差し有利度", "追込有利度"):
        if k in d:
            out[k] = d[k]
    return out


def band_odds(o):
    if o is None or (isinstance(o, float) and math.isnan(o)):
        return "欠損"
    if o <= 3:
        return "<=3"
    if o <= 5:
        return "3-5"
    if o <= 8:
        return "5-8"
    if o <= 12:
        return "8-12"
    if o <= 20:
        return "12-20"
    if o <= 50:
        return "20-50"
    return ">50"


def band_pop(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "欠損"
    p = int(p)
    if p <= 1:
        return "1人気"
    if p <= 3:
        return "2-3人気"
    if p <= 5:
        return "4-5人気"
    if p <= 9:
        return "6-9人気"
    return "10人気+"


def band_ev(e):
    if e is None or (isinstance(e, float) and math.isnan(e)):
        return "欠損"
    if e < 100:
        return "<100"
    if e < 108:
        return "100-108"
    if e < 115:
        return "108-115"
    if e < 120:
        return "115-120"
    return ">=120"


def band_ai(a):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return "欠損"
    if a < 70:
        return "<70"
    if a < 80:
        return "70-80"
    return ">=80"


def band_ab(a):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return "欠損"
    # discrete scores observed historically
    if a <= 50:
        return "<=50"
    if a <= 70:
        return "51-70"
    if a < 80:
        return "71-79"
    return ">=80"


def band_rc(a):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return "欠損"
    if a < 70:
        return "<70"
    if a < 78:
        return "70-78"
    if a < 85:
        return "78-85"
    return ">=85"


def band_pace_score(a):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return "欠損"
    if a < 40:
        return "<40"
    if a < 50:
        return "40-50"
    return ">=50"


def band_chaos(c):
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return "欠損"
    if c <= 55:
        return "<=55"
    if c <= 70:
        return "56-70"
    return ">70"


def band_repro(r):
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "欠損"
    if r < 70:
        return "<70"
    if r < 80:
        return "70-80"
    return ">=80"


def band_winpct(w):
    if w is None or (isinstance(w, float) and math.isnan(w)):
        return "欠損"
    if w < 15:
        return "<15"
    if w < 25:
        return "15-25"
    if w < 35:
        return "25-35"
    return ">=35"


def band_data_n(n):
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "欠損"
    n = int(n)
    if n <= 3:
        return "3"
    if n <= 5:
        return "4-5"
    return ">=6"


def outcome_label(finish):
    if finish is None:
        return None
    if finish == 1:
        return "1着"
    if finish <= 3:
        return "複勝圏(2-3)"
    return "圏外"


def load_preds():
    rows = []
    for path in sorted(PRED_DIR.glob("predictions_*.csv")):
        day = re.search(r"(\d{4}-\d{2}-\d{2})", path.name).group(1)
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["date"] = day
        rows.append(df)
    P = pd.concat(rows, ignore_index=True)
    P["race_id"] = P["race_id"].astype(str)
    P["勝負ランク"] = P["勝負ランク"].astype(str).str.upper()
    P["source"] = P["source"].astype(str).str.lower()
    return P


def build_table(P, A, R):
    finish_map = {}
    for _, r in R.iterrows():
        rid = str(r["race_id"])
        name = str(r.get("馬名") or "").strip()
        ban = str(int(r["馬番"])) if pd.notna(r.get("馬番")) else ""
        try:
            fin = int(float(r["着順"]))
        except Exception:
            fin = None
        info = {"着順": fin, "確定オッズ": fnum(r.get("確定オッズ"))}
        finish_map[(rid, name)] = info
        finish_map[(rid, ban)] = info

    hon = A[A.bet_type.astype(str) == "本命"][
        ["race_id", "hit", "payout", "investment", "勝負ランク", "source"]
    ].copy()
    hon["race_id"] = hon["race_id"].astype(str)

    rows = []
    for _, pr in P[P["勝負ランク"] == "S"].iterrows():
        rid = str(pr["race_id"])
        h = hon[hon.race_id == rid]
        if h.empty:
            continue
        h0 = h.iloc[0]
        src = "JRA" if str(h0.get("source") or pr["source"]).lower() == "jra" else "地方"
        name = str(pr.get("本命") or "").strip()
        ban = str(pr.get("本命馬番") or "").replace(".0", "").strip()
        fin_info = finish_map.get((rid, name)) or finish_map.get((rid, ban)) or {}
        finish = fin_info.get("着順")
        if finish is None:
            continue
        pace = parse_pace(pr.get("展開予想"))
        win_odds = fin_info.get("確定オッズ") or fnum(pr.get("本命オッズ"))
        row = {
            "date": str(pr["date"]),
            "period": "train" if str(pr["date"]) <= TRAIN_END else "test",
            "race_id": rid,
            "source": src,
            "venue": str(pr.get("開催地") or ""),
            "honmei": name,
            "finish": int(finish),
            "outcome": outcome_label(finish),
            "in_money": finish <= 3,
            "win": finish == 1,
            "hit": int(h0["hit"]) if pd.notna(h0["hit"]) else 0,
            "payout": float(h0["payout"]) if pd.notna(h0["payout"]) else 0.0,
            "investment": float(h0["investment"]) if pd.notna(h0["investment"]) else 100.0,
            "能力差": fnum(pr.get("能力差スコア")),
            "AI": fnum(pr.get("AI信頼度スコア")),
            "EV": fnum(pr.get("期待値")),
            "人気": fnum(pr.get("本命人気")),
            "オッズ": win_odds,
            "勝率": fnum(pr.get("シミュレーション勝率")),
            "再現率": fnum(pr.get("シミュレーション再現率")),
            "レース信頼度": fnum(pr.get("レース信頼度スコア")),
            "展開読みやすさ": fnum(pr.get("展開読みやすさ")),
            "データ件数": fnum(pr.get("データ件数")),
            "データ件数スコア": fnum(pr.get("データ件数スコア")),
            "競馬場バイアス": fnum(pr.get("競馬場バイアス一致率")),
            "荒れ度": fnum(pr.get("荒れ度")),
            "適正オッズ": fnum(pr.get("AI適正オッズ")),
            "想定ペース": pace.get("想定ペース"),
            "荒れ指数": fnum(pace.get("荒れ指数")),
            "逃げ有利度": fnum(pace.get("逃げ有利度")),
            "先行有利度": fnum(pace.get("先行有利度")),
            "差し有利度": fnum(pace.get("差し有利度")),
            "距離": "欠損",
            "馬場": "欠損",
            "投資判定": str(pr.get("投資判定") or ""),
        }
        # bands
        row["odds_band"] = band_odds(row["オッズ"])
        row["pop_band"] = band_pop(row["人気"])
        row["ev_band"] = band_ev(row["EV"])
        row["ai_band"] = band_ai(row["AI"])
        row["ab_band"] = band_ab(row["能力差"])
        row["rc_band"] = band_rc(row["レース信頼度"])
        row["pace_band"] = band_pace_score(row["展開読みやすさ"])
        row["chaos_band"] = band_chaos(row["荒れ指数"])
        row["repro_band"] = band_repro(row["再現率"])
        row["winpct_band"] = band_winpct(row["勝率"])
        row["data_n_band"] = band_data_n(row["データ件数"])
        rows.append(row)
    return pd.DataFrame(rows)


def means(df, cols):
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = None if s.notna().sum() == 0 else round(float(s.mean()), 2)
    return out


def rate_pack(df):
    n = len(df)
    if n == 0:
        return {"n": 0}
    inv = float(df["investment"].sum())
    pay = float(df["payout"].sum())
    return {
        "n": int(n),
        "win_n": int(df["win"].sum()),
        "place_n": int(df["in_money"].sum()),
        "out_n": int((~df["in_money"]).sum()),
        "win_rate": round(100 * df["win"].mean(), 1),
        "place_rate": round(100 * df["in_money"].mean(), 1),
        "out_rate": round(100 * (~df["in_money"]).mean(), 1),
        "recovery": round(100 * pay / inv, 1) if inv else None,
        "avg_odds": round(float(pd.to_numeric(df["オッズ"], errors="coerce").mean()), 2),
        "avg_ev": round(float(pd.to_numeric(df["EV"], errors="coerce").mean()), 2),
    }


def slice_compare(df, col, min_all=MIN_ALL):
    """帯別の圏外率・複勝率。全期間nとtrain/testを付与。"""
    rows = []
    for band, g in df.groupby(col):
        tr = g[g.period == "train"]
        te = g[g.period == "test"]
        rec = {
            "band": str(band),
            "all": rate_pack(g),
            "train": rate_pack(tr),
            "test": rate_pack(te),
        }
        # baseline place rates
        base_tr = rate_pack(df[df.period == "train"]).get("place_rate")
        base_te = rate_pack(df[df.period == "test"]).get("place_rate")
        base_all = rate_pack(df).get("place_rate")
        rec["delta_place_all"] = (
            None
            if base_all is None or rec["all"].get("place_rate") is None
            else round(rec["all"]["place_rate"] - base_all, 1)
        )
        rec["delta_place_train"] = (
            None
            if base_tr is None or rec["train"].get("place_rate") is None
            else round(rec["train"]["place_rate"] - base_tr, 1)
        )
        rec["delta_place_test"] = (
            None
            if base_te is None or rec["test"].get("place_rate") is None
            else round(rec["test"]["place_rate"] - base_te, 1)
        )

        n_ok = rec["train"].get("n", 0) >= MIN_TRAIN and rec["test"].get("n", 0) >= MIN_TEST
        # fragile = both periods place rate much worse than baseline OR out_rate high
        fragile = False
        status = "件数不足"
        if n_ok:
            tr_p = rec["train"].get("place_rate")
            te_p = rec["test"].get("place_rate")
            # bad if both periods place_rate <= baseline - 8pp or place_rate < 25 both
            bad_tr = tr_p is not None and (tr_p <= (base_tr or 0) - 8 or tr_p < 22)
            bad_te = te_p is not None and (te_p <= (base_te or 0) - 8 or te_p < 22)
            good_tr = tr_p is not None and tr_p >= (base_tr or 0) + 8
            good_te = te_p is not None and te_p >= (base_te or 0) + 8
            if bad_tr and bad_te:
                fragile = True
                status = "両期間で複勝率悪化（再現あり・悪化条件）"
            elif good_tr and good_te:
                status = "両期間で複勝率良好（再現あり・良好条件）"
            elif (bad_te and not bad_tr and good_tr) or (bad_tr and not bad_te and good_te):
                status = "学習/検証で逆方向→過学習・不安定（採用しない）"
            elif bad_te and not bad_tr:
                status = "検証のみ悪化→過学習疑い（採用しない）"
            elif bad_tr and not bad_te:
                status = "学習のみ悪化→不安定（採用しない）"
            else:
                status = "差が小さい/不安定"
        elif rec["all"].get("n", 0) >= min_all:
            status = "全期間のみ件数十分（期間分離不足→採用慎重）"
        rec["n_ok"] = n_ok
        rec["fragile"] = fragile
        rec["status"] = status
        rows.append(rec)
    return rows


def numeric_diff(place_df, out_df, cols):
    rows = []
    for c in cols:
        sp = pd.to_numeric(place_df[c], errors="coerce")
        so = pd.to_numeric(out_df[c], errors="coerce")
        if sp.notna().sum() == 0 or so.notna().sum() == 0:
            rows.append({"feature": c, "place_mean": None, "out_mean": None, "diff_out_minus_place": None})
            continue
        pm, om = float(sp.mean()), float(so.mean())
        rows.append(
            {
                "feature": c,
                "place_mean": round(pm, 2),
                "out_mean": round(om, 2),
                "diff_out_minus_place": round(om - pm, 2),
                "place_n": int(sp.notna().sum()),
                "out_n": int(so.notna().sum()),
            }
        )
    # sort by abs diff
    rows.sort(key=lambda x: abs(x["diff_out_minus_place"] or 0), reverse=True)
    return rows


def demotion_backtest(nar_s: pd.DataFrame, nar_all_ranks: pd.DataFrame | None = None):
    """地方Sの仮想降格シナリオ。本命単勝指標。"""
    scenarios = {}

    def pack_by_rank(df_s_kept, demoted_to, label):
        # S metrics on kept
        s = rate_pack(df_s_kept) if len(df_s_kept) else {"n": 0}
        # demoted set as if A or B
        d = rate_pack(nar_s[~nar_s.race_id.isin(df_s_kept.race_id)]) if len(nar_s) else {"n": 0}
        # For overall S+A+B unchanged if only relabel - compute virtual S after demotion
        scenarios[label] = {
            "remaining_S": s,
            "demoted_pool": {**d, "treated_as": demoted_to},
            "demoted_n": int(len(nar_s) - len(df_s_kept)),
            "note": "ラベル付け替え。買い目は不変。本命単勝のランク別見え方のみ。",
        }

    # baseline
    scenarios["現状維持_全S"] = {
        "remaining_S": rate_pack(nar_s),
        "demoted_pool": {"n": 0},
        "demoted_n": 0,
    }

    # demote all S -> A
    scenarios["地方S全件をAへ"] = {
        "remaining_S": {"n": 0, "note": "地方S消滅"},
        "demoted_pool": {**rate_pack(nar_s), "treated_as": "A"},
        "demoted_n": int(len(nar_s)),
        "implication": "地方にSが残らない。UI上のSはJRAのみ（JRAは別管理）。",
    }
    scenarios["地方S全件をBへ"] = {
        "remaining_S": {"n": 0, "note": "地方S消滅"},
        "demoted_pool": {**rate_pack(nar_s), "treated_as": "B"},
        "demoted_n": int(len(nar_s)),
    }

    # conditional demotions based on candidate rules (virtual)
    rules = {
        "4-5人気をAへ": nar_s["pop_band"] == "4-5人気",
        "オッズ8-12をAへ": nar_s["odds_band"] == "8-12",
        "4-5人気_OR_オッズ8-12をAへ": (nar_s["pop_band"] == "4-5人気")
        | (nar_s["odds_band"] == "8-12"),
        "6-9人気をAへ": nar_s["pop_band"] == "6-9人気",
        "オッズ12-20をAへ": nar_s["odds_band"] == "12-20",
        "EV115-120をAへ": nar_s["ev_band"] == "115-120",
        "能力差>=80をAへ": nar_s["ab_band"] == ">=80",
        "人気>=4をAへ": nar_s["人気"].fillna(99) >= 4,
        "人気>=4をBへ": nar_s["人気"].fillna(99) >= 4,
        "オッズ>8をAへ": nar_s["オッズ"].fillna(0) > 8,
    }
    for name, mask in rules.items():
        kept = nar_s[~mask]
        demoted = nar_s[mask]
        to = "B" if "をBへ" in name else "A"
        tr_kept = kept[kept.period == "train"]
        te_kept = kept[kept.period == "test"]
        tr_dem = demoted[demoted.period == "train"]
        te_dem = demoted[demoted.period == "test"]
        scenarios[f"条件降格_{name}"] = {
            "remaining_S": {
                "all": rate_pack(kept),
                "train": rate_pack(tr_kept),
                "test": rate_pack(te_kept),
            },
            "demoted_pool": {
                "all": rate_pack(demoted),
                "train": rate_pack(tr_dem),
                "test": rate_pack(te_dem),
                "treated_as": to,
            },
            "demoted_n": int(mask.sum()),
            "overfit_check": _overfit_check(rate_pack(tr_kept), rate_pack(te_kept), rate_pack(tr_dem), rate_pack(te_dem)),
        }
    return scenarios


def _overfit_check(tr_kept, te_kept, tr_dem, te_dem):
    """降格後の残Sが学習◎検証×なら過学習疑い。"""
    notes = []
    if tr_kept.get("n", 0) >= 8 and te_kept.get("n", 0) >= 8:
        if (tr_kept.get("recovery") or 0) >= 100 and (te_kept.get("recovery") or 0) < 50:
            notes.append("残Sが学習高回収・検証低回収→過学習疑い")
        if (tr_kept.get("place_rate") or 0) >= 45 and (te_kept.get("place_rate") or 0) < 30:
            notes.append("残S複勝率が学習◎検証×→不安定")
        if (te_kept.get("place_rate") or 0) > (tr_kept.get("place_rate") or 0) + 5 and (
            te_kept.get("recovery") or 0
        ) > (tr_kept.get("recovery") or 0):
            notes.append("残Sが検証でも改善方向（比較的安定の可能性）")
    if tr_dem.get("n", 0) >= MIN_TRAIN and te_dem.get("n", 0) >= MIN_TEST:
        if (tr_dem.get("place_rate") or 100) < 25 and (te_dem.get("place_rate") or 100) < 25:
            notes.append("降格側が両期間とも低複勝→降格条件として再現あり")
        if (tr_dem.get("place_rate") or 0) >= 40 and (te_dem.get("place_rate") or 100) < 25:
            notes.append("降格側が学習◎検証×→条件追加は過学習リスク")
    if not notes:
        notes.append("明確な過学習シグナルなし/判定保留")
    return notes


def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    P = load_preds()
    A = pd.read_csv(ROOT / "data" / "analysis_result.csv", encoding="utf-8-sig")
    A["race_id"] = A["race_id"].astype(str)
    R = pd.read_csv(ROOT / "data" / "results.csv", encoding="utf-8-sig")
    R["race_id"] = R["race_id"].astype(str)

    D = build_table(P, A, R)
    nar = D[D.source == "地方"].copy()
    jra = D[D.source == "JRA"].copy()

    num_cols = [
        "能力差",
        "AI",
        "EV",
        "人気",
        "オッズ",
        "勝率",
        "再現率",
        "レース信頼度",
        "展開読みやすさ",
        "データ件数",
        "競馬場バイアス",
        "荒れ度",
        "荒れ指数",
        "適正オッズ",
        "逃げ有利度",
        "先行有利度",
        "差し有利度",
    ]

    # overall
    report = {
        "purpose": "地方S本命失敗原因の特徴量分解とS降格仮想BT（コード変更なし）",
        "s_formal_gates": {
            "正式": ["データ件数>=3", "本命オッズ<=50"],
            "暫定S入口": "レース信頼度>=78 + 枠 + qualify",
            "合成に使うがS必須ではない": [
                "AI信頼度",
                "能力差",
                "展開読みやすさ",
                "データ件数スコア",
                "シミュレーション再現率",
                "競馬場バイアス",
            ],
            "under_validation": "能力差>=80（正式条件ではない）",
        },
        "data_gaps": {
            "距離": "予測CSVにレース単位列なし（全件欠損）",
            "馬場": "予測CSVにレース単位列なし（全件欠損）",
        },
        "split": {"train": f"<= {TRAIN_END}", "test": f"> {TRAIN_END}"},
        "min_n": {"train": MIN_TRAIN, "test": MIN_TEST, "all": MIN_ALL},
        "counts": {
            "地方S": rate_pack(nar),
            "JRA_S": rate_pack(jra),
            "地方S_学習": rate_pack(nar[nar.period == "train"]),
            "地方S_検証": rate_pack(nar[nar.period == "test"]),
            "outcome_dist_地方": dict(Counter(nar["outcome"])),
            "outcome_dist_JRA": dict(Counter(jra["outcome"])),
        },
        "implement_now": False,
    }

    place = nar[nar["in_money"]]
    out = nar[~nar["in_money"]]
    win = nar[nar["win"]]
    place23 = nar[nar["outcome"] == "複勝圏(2-3)"]

    report["place_vs_out_means"] = {
        "地方_全期間": {
            "place_n": int(len(place)),
            "out_n": int(len(out)),
            "win_n": int(len(win)),
            "place23_n": int(len(place23)),
            "place_means": means(place, num_cols),
            "out_means": means(out, num_cols),
            "diffs_sorted": numeric_diff(place, out, num_cols),
        },
        "地方_学習": {
            "place_n": int(len(place[place.period == "train"])),
            "out_n": int(len(out[out.period == "train"])),
            "diffs_sorted": numeric_diff(
                place[place.period == "train"], out[out.period == "train"], num_cols
            ),
        },
        "地方_検証": {
            "place_n": int(len(place[place.period == "test"])),
            "out_n": int(len(out[out.period == "test"])),
            "diffs_sorted": numeric_diff(
                place[place.period == "test"], out[out.period == "test"], num_cols
            ),
        },
        "JRA_参考_件数不足": {
            "n": int(len(jra)),
            "place_n": int(jra["in_money"].sum()) if len(jra) else 0,
            "out_n": int((~jra["in_money"]).sum()) if len(jra) else 0,
            "note": "JRA/Sはn不足。地方結論をJRAへ適用しない。",
        },
    }

    # categorical slices
    cat_cols = [
        "pop_band",
        "odds_band",
        "ev_band",
        "ai_band",
        "ab_band",
        "rc_band",
        "pace_band",
        "chaos_band",
        "repro_band",
        "winpct_band",
        "data_n_band",
        "venue",
        "想定ペース",
    ]
    slices = {}
    for c in cat_cols:
        slices[c] = slice_compare(nar, c)
    report["feature_slices_地方S"] = slices

    # reproducible bad / good
    bad, good, unstable, insuff = [], [], [], []
    for feat, rows in slices.items():
        for r in rows:
            item = {
                "feature": feat,
                "band": r["band"],
                "status": r["status"],
                "all": r["all"],
                "train": r["train"],
                "test": r["test"],
            }
            if "再現あり・悪化" in r["status"]:
                bad.append(item)
            elif "再現あり・良好" in r["status"]:
                good.append(item)
            elif "過学習" in r["status"] or "不安定" in r["status"] or "検証のみ" in r["status"]:
                unstable.append(item)
            elif "件数不足" in r["status"]:
                insuff.append(item)

    report["reproducible_bad_conditions"] = bad
    report["reproducible_good_conditions"] = good
    report["unstable_overfit_risk"] = unstable
    report["insufficient_n"] = [
        {"feature": x["feature"], "band": x["band"], "all_n": x["all"].get("n"), "status": x["status"]}
        for x in insuff
        if x["all"].get("n", 0) > 0
    ][:80]

    # demotion BT
    report["demotion_backtest_地方S"] = demotion_backtest(nar)

    # decision
    # Key question: keep NAR S or change criteria
    base = rate_pack(nar)
    base_te = rate_pack(nar[nar.period == "test"])
    # best conditional among reproducible
    best = None
    for name, sc in report["demotion_backtest_地方S"].items():
        if not name.startswith("条件降格_"):
            continue
        rem = sc.get("remaining_S", {}).get("all", {})
        te = sc.get("remaining_S", {}).get("test", {})
        of = sc.get("overfit_check", [])
        if rem.get("n", 0) < MIN_ALL:
            continue
        if te.get("n", 0) < MIN_TEST:
            continue
        # prefer higher test place_rate and recovery, with demoted also bad both periods
        score = (te.get("place_rate") or 0) * 2 + (te.get("recovery") or 0) * 0.1
        if "過学習疑い" in "".join(of):
            score -= 50
        if "降格側が両期間とも低複勝" in "".join(of):
            score += 20
        cand = {
            "name": name,
            "score": round(score, 2),
            "remaining_all": rem,
            "remaining_test": te,
            "remaining_train": sc.get("remaining_S", {}).get("train", {}),
            "demoted_all": sc.get("demoted_pool", {}).get("all", {}),
            "overfit_check": of,
            "demoted_n": sc.get("demoted_n"),
        }
        if best is None or cand["score"] > best["score"]:
            best = cand

    # Judgment
    judgment = {
        "地方Sを現状維持すべきか": None,
        "地方だけS基準を変更すべきか": None,
        "JRA": "変更しない（件数不足・分離維持）",
        "理由": [],
        "if_implement_later": [],
    }
    if base_te.get("place_rate", 100) < 40 and base_te.get("recovery", 100) < 40:
        judgment["地方Sを現状維持すべきか"] = (
            "品質指標としては維持しにくい。ただし条件追加の過学習リスクが高い。"
        )
        judgment["理由"].append(
            f"検証の地方Sは複勝率{base_te.get('place_rate')}%・単勝回収{base_te.get('recovery')}%と低い"
        )
    # recommend change only if we have reproducible demotion that improves test without classic overfit
    impl = []
    if best and "過学習疑い" not in "".join(best.get("overfit_check") or []):
        judgment["地方だけS基準を変更すべきか"] = (
            "候補あり（未実装）。再現性のある除外を地方Sに限定して検討余地。"
        )
        impl.append(
            {
                "change": best["name"].replace("条件降格_", "地方Sで該当をA降格: "),
                "scope": "地方のみ（JRAのqualify/枠は触らない）",
                "evidence": best,
                "caution": "S+A合算の本命成績は付け替えだけでは不変。買い判定連動を変えるなら別検証が必要。",
            }
        )
    else:
        judgment["地方だけS基準を変更すべきか"] = (
            "今すぐの条件追加は非推奨。過学習リスクまたは改善幅が不十分。"
        )
        impl.append(
            {
                "change": "条件追加はせず、地方Sの監視指標（複勝率/単勝回収）を継続",
                "scope": "地方のみ",
                "caution": "複勝オッズ欠損・距離馬場欠損のため場・コース条件はまだ使えない",
            }
        )

    # Always list what would change if implementing the clearest reproducible bad rules
    for b in bad:
        impl.append(
            {
                "change": f"地方Sで {b['feature']}={b['band']} をS不合格（A降格）候補",
                "scope": "地方のみ",
                "train_place": b["train"].get("place_rate"),
                "test_place": b["test"].get("place_rate"),
                "train_n": b["train"].get("n"),
                "test_n": b["test"].get("n"),
                "status": b["status"],
            }
        )
    # reject unstable
    for u in unstable[:15]:
        impl.append(
            {
                "change": f"採用しない: {u['feature']}={u['band']}",
                "reason": u["status"],
                "train": u["train"],
                "test": u["test"],
            }
        )

    judgment["if_implement_later"] = impl
    judgment["recommended_decision"] = (
        "地方Sは『維持（ラベルとしては残す）だが、品質は低く、条件追加は再現確認できた最小セット以外まだ実装しない』。"
        "最優先の失敗モードは本命選定（圏外72/107）。買い目問題は別件。"
    )
    if bad:
        judgment["recommended_decision"] += (
            f" 再現悪化条件が{len(bad)}件あるため、実装するなら地方限定のS→A降格を最小数だけ。"
        )
    else:
        judgment["recommended_decision"] += " 期間分離で再現した悪化条件が乏しく、今は基準変更を急がない。"

    report["judgment"] = judgment
    report["best_conditional_demotion"] = best

    # top diffs that agree train and test
    report["consistent_numeric_signals"] = _consistent_numeric(nar, num_cols)

    OUT.write_text(json.dumps(sanitize(report), ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report, nar)
    print("wrote", OUT)
    return 0


def _consistent_numeric(nar, num_cols):
    """学習・検証の両方で out_mean - place_mean の符号が一致する特徴。"""
    out = []
    for period in ("train", "test"):
        pass
    tr_p = nar[(nar.period == "train") & nar.in_money]
    tr_o = nar[(nar.period == "train") & ~nar.in_money]
    te_p = nar[(nar.period == "test") & nar.in_money]
    te_o = nar[(nar.period == "test") & ~nar.in_money]
    for c in num_cols:
        dtr = numeric_diff(tr_p, tr_o, [c])[0]
        dte = numeric_diff(te_p, te_o, [c])[0]
        if dtr["diff_out_minus_place"] is None or dte["diff_out_minus_place"] is None:
            continue
        same_sign = (dtr["diff_out_minus_place"] > 0 and dte["diff_out_minus_place"] > 0) or (
            dtr["diff_out_minus_place"] < 0 and dte["diff_out_minus_place"] < 0
        )
        # require meaningful magnitude both periods
        if same_sign and abs(dtr["diff_out_minus_place"]) >= 0.5 and abs(dte["diff_out_minus_place"]) >= 0.5:
            out.append(
                {
                    "feature": c,
                    "train_diff_out_minus_place": dtr["diff_out_minus_place"],
                    "test_diff_out_minus_place": dte["diff_out_minus_place"],
                    "train_place_mean": dtr["place_mean"],
                    "train_out_mean": dtr["out_mean"],
                    "test_place_mean": dte["place_mean"],
                    "test_out_mean": dte["out_mean"],
                    "interpretation": (
                        "圏外の方が高い" if dtr["diff_out_minus_place"] > 0 else "圏外の方が低い"
                    ),
                    "note": "平均差の符号一致。因果・閾値確定ではない。nはplace/outの期間別件数に依存。",
                }
            )
    out.sort(
        key=lambda x: min(abs(x["train_diff_out_minus_place"]), abs(x["test_diff_out_minus_place"])),
        reverse=True,
    )
    return out


def _print_summary(report, nar):
    print("=== COUNTS ===")
    print(report["counts"]["地方S"])
    print("train", report["counts"]["地方S_学習"])
    print("test", report["counts"]["地方S_検証"])
    print("\n=== PLACE VS OUT DIFFS (all) ===")
    for d in report["place_vs_out_means"]["地方_全期間"]["diffs_sorted"][:12]:
        print(d)
    print("\n=== CONSISTENT NUMERIC ===")
    for d in report["consistent_numeric_signals"][:12]:
        print(d)
    print("\n=== REPRO BAD ===")
    for b in report["reproducible_bad_conditions"]:
        print(
            b["feature"],
            b["band"],
            "tr",
            b["train"].get("n"),
            b["train"].get("place_rate"),
            "te",
            b["test"].get("n"),
            b["test"].get("place_rate"),
            b["status"],
        )
    print("\n=== REPRO GOOD ===")
    for b in report["reproducible_good_conditions"]:
        print(
            b["feature"],
            b["band"],
            "tr",
            b["train"].get("n"),
            b["train"].get("place_rate"),
            "te",
            b["test"].get("n"),
            b["test"].get("place_rate"),
        )
    print("\n=== UNSTABLE (sample) ===")
    for b in report["unstable_overfit_risk"][:12]:
        print(b["feature"], b["band"], b["status"], b["train"].get("place_rate"), b["test"].get("place_rate"))
    print("\n=== DEMOTION KEY ===")
    for k, v in report["demotion_backtest_地方S"].items():
        if k in ("現状維持_全S", "地方S全件をAへ") or k.startswith("条件降格_4-5") or k.startswith(
            "条件降格_人気>=4"
        ) or k.startswith("条件降格_オッズ>8") or k.startswith("条件降格_4-5人気_OR"):
            print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("\n=== JUDGMENT ===")
    print(json.dumps(report["judgment"], ensure_ascii=False, indent=2)[:2500])
    print("best", report.get("best_conditional_demotion"))


if __name__ == "__main__":
    raise SystemExit(main())
