#!/usr/bin/env python3
"""本命着順と買い目的中を分離し、仮想買い方別成績を出す（分析専用・本番ロジック非変更）。"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "predictions_by_date"
OUT = ROOT / "data" / "honmei_vs_ticket_fullperiod_report.json"
TRAIN_END = "2026-07-28"
MIN_N = 20
MIN_N_WEAK = 12


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


def est_place_odds(win_odds, field_n=None):
    """複勝オッズの参考近似。回収の採用判定には使わない。"""
    if win_odds is None or win_odds <= 1:
        return None
    k = 3.2
    if field_n and field_n >= 12:
        k = 3.6
    elif field_n and field_n <= 7:
        k = 2.6
    est = 1.0 + (win_odds - 1.0) / k
    return round(max(1.1, min(est, win_odds * 0.55)), 2)


def bet_metrics(hits, pays, invs, odds_list=None, label="", approx=False, min_n=MIN_N):
    n = len(hits)
    if n == 0:
        return {"label": label, "n": 0, "status": "件数0", "approx": approx, "adoptable": False}
    inv = float(sum(invs))
    pay = float(sum(pays))
    h = int(sum(hits))
    avg_odds = None
    if odds_list is not None:
        ol = [o for o in odds_list if o is not None]
        avg_odds = round(sum(ol) / len(ol), 2) if ol else None
    status = "集計可" if n >= min_n else "件数不足（採用候補にしない）"
    if approx:
        status = status + " / 回収は複勝オッズ近似（採用判定に使わない）"
    return {
        "label": label,
        "n": n,
        "hits": h,
        "hit_rate": round(100 * h / n, 1),
        "investment": int(inv),
        "payout": int(pay),
        "recovery": round(100 * pay / inv, 1) if inv else None,
        "avg_odds": avg_odds,
        "approx": approx,
        "status": status,
        "adoptable": (n >= min_n) and (not approx),
    }


def extract_ticket(series_of_dicts):
    hits, pays, invs = [], [], []
    for d in series_of_dicts:
        if d is None or (isinstance(d, float) and isinstance(d, float) and math.isnan(d)):
            continue
        if not isinstance(d, dict):
            continue
        hits.append(int(d.get("hit") or 0))
        pays.append(float(d.get("payout") or 0))
        invs.append(float(d.get("investment") or 0))
    return hits, pays, invs, len(hits)


def load_predictions() -> pd.DataFrame:
    rows = []
    for path in sorted(PRED_DIR.glob("predictions_*.csv")):
        day = re.search(r"(\d{4}-\d{2}-\d{2})", path.name).group(1)
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["date"] = day
        rows.append(df)
    P = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    P["race_id"] = P["race_id"].astype(str)
    P["勝負ランク"] = P["勝負ランク"].astype(str).str.upper()
    P["source"] = P["source"].astype(str).str.lower()
    return P


def build_race_table(P: pd.DataFrame, A: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    field_n = R.groupby("race_id").size().to_dict()
    finish_map = {}
    for _, r in R.iterrows():
        rid = str(r["race_id"])
        name = str(r.get("馬名") or "").strip()
        ban = str(int(r["馬番"])) if pd.notna(r.get("馬番")) else ""
        try:
            fin = int(float(r["着順"]))
        except Exception:
            fin = None
        info = {
            "着順": fin,
            "確定オッズ": fnum(r.get("確定オッズ")),
            "人気_結果": fnum(r.get("人気")),
        }
        finish_map[(rid, name)] = info
        finish_map[(rid, ban)] = info

    A_by = {rid: g for rid, g in A.groupby("race_id")}
    base = []
    for _, pr in P[P["勝負ランク"].isin(["S", "A", "B"])].iterrows():
        rid = str(pr["race_id"])
        if rid not in A_by:
            continue
        tickets = A_by[rid]
        hon_rows = tickets[tickets.bet_type == "本命"]
        if hon_rows.empty:
            continue
        h = hon_rows.iloc[0]
        rank = str(h.get("勝負ランク") or pr["勝負ランク"]).upper()
        if rank not in ("S", "A", "B"):
            continue
        src = "JRA" if str(h.get("source") or pr["source"]).lower() == "jra" else "地方"
        name = str(pr.get("本命") or h.get("prediction") or "").strip()
        ban = str(pr.get("本命馬番") or "").replace(".0", "").strip()
        fin_info = finish_map.get((rid, name)) or finish_map.get((rid, ban)) or {}
        finish = fin_info.get("着順")
        if finish is None:
            continue
        win_odds = (
            fin_info.get("確定オッズ")
            or fnum(pr.get("本命オッズ"))
            or fnum(h.get("本命オッズ"))
        )
        pop = fnum(pr.get("本命人気"))
        ev = fnum(pr.get("期待値")) or fnum(h.get("期待値"))
        ab = fnum(pr.get("能力差スコア")) or fnum(h.get("能力差スコア"))
        fn = int(field_n.get(rid, 0) or 0)
        place_odds = est_place_odds(win_odds, fn)

        def row_bt(bt):
            g = tickets[tickets.bet_type == bt]
            if g.empty:
                return None
            r = g.iloc[0]
            bought = str(r.get("購入対象")) in ("1", "1.0", "True", "true")
            return {
                "hit": int(r["hit"]) if pd.notna(r["hit"]) else 0,
                "payout": float(r["payout"]) if pd.notna(r["payout"]) else 0.0,
                "investment": float(r["investment"]) if pd.notna(r["investment"]) else 0.0,
                "購入対象": 1 if bought else 0,
            }

        wide = row_bt("ワイド")
        umaren = row_bt("馬連")
        sanrenpuku = row_bt("三連複")
        sanrentan = row_bt("三連単")
        bought = tickets[tickets["購入対象"].astype(str).isin(("1", "1.0", "True", "true"))]
        bought_inv = (
            float(pd.to_numeric(bought["investment"], errors="coerce").fillna(0).sum())
            if len(bought)
            else 0.0
        )
        bought_pay = (
            float(pd.to_numeric(bought["payout"], errors="coerce").fillna(0).sum())
            if len(bought)
            else 0.0
        )
        bought_hit = (
            int(pd.to_numeric(bought["hit"], errors="coerce").fillna(0).sum()) > 0
            if len(bought)
            else False
        )

        win_hit = 1 if finish == 1 else 0
        place_hit = 1 if finish <= 3 else 0
        v_win_pay = (win_odds * 100.0) if win_hit and win_odds else 0.0
        v_place_pay = (place_odds * 100.0) if place_hit and place_odds else 0.0

        base.append(
            {
                "date": str(pr["date"]),
                "period": "train" if str(pr["date"]) <= TRAIN_END else "test",
                "race_id": rid,
                "source": src,
                "rank": rank,
                "venue": str(pr.get("開催地") or ""),
                "honmei": name,
                "finish": int(finish),
                "in_money": finish <= 3,
                "win_odds": win_odds,
                "place_odds_est": place_odds,
                "pop": pop,
                "EV": ev,
                "能力差": ab,
                "field_n": fn,
                "投資判定": str(pr.get("投資判定") or h.get("投資判定") or ""),
                "推奨券種": str(h.get("推奨券種") or ""),
                "v_win_hit": win_hit,
                "v_win_pay": v_win_pay,
                "v_win_inv": 100.0,
                "v_place_hit": place_hit,
                "v_place_pay": v_place_pay,
                "v_place_inv": 100.0,
                "wide": wide,
                "umaren": umaren,
                "sanrenpuku": sanrenpuku,
                "sanrentan": sanrentan,
                "bought_n": int(len(bought)),
                "bought_hit": bool(bought_hit),
                "bought_pay": bought_pay,
                "bought_inv": bought_inv,
            }
        )
    return pd.DataFrame(base)


def separation_block(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    if n == 0:
        return {"label": label, "n": 0}

    finish_dist = {str(k): int(v) for k, v in sorted(Counter(df["finish"]).items())}
    in_m = int(df["in_money"].sum())
    win_n = int((df["finish"] == 1).sum())

    buckets = {
        "本命圏外": 0,
        "本命圏内_購入買い目的中": 0,
        "本命圏内_購入買い目外れ": 0,
        "本命圏内_購入なし_生成券的中": 0,
        "本命圏内_購入なし_生成券外れ": 0,
    }
    for _, r in df.iterrows():
        if not r["in_money"]:
            buckets["本命圏外"] += 1
            continue
        any_gen = any(
            [
                (r["wide"] or {}).get("hit"),
                (r["umaren"] or {}).get("hit"),
                (r["sanrenpuku"] or {}).get("hit"),
                (r["sanrentan"] or {}).get("hit"),
            ]
        )
        if r["bought_n"] > 0:
            if r["bought_hit"]:
                buckets["本命圏内_購入買い目的中"] += 1
            else:
                buckets["本命圏内_購入買い目外れ"] += 1
        else:
            if any_gen:
                buckets["本命圏内_購入なし_生成券的中"] += 1
            else:
                buckets["本命圏内_購入なし_生成券外れ"] += 1

    def means(mask):
        g = df[mask]
        if len(g) == 0:
            return {"n": 0}
        return {
            "n": int(len(g)),
            "avg_finish": round(float(g["finish"].mean()), 2),
            "avg_odds": round(float(pd.to_numeric(g["win_odds"], errors="coerce").mean()), 2),
            "avg_pop": round(float(pd.to_numeric(g["pop"], errors="coerce").mean()), 2)
            if g["pop"].notna().any()
            else None,
            "avg_ev": round(float(pd.to_numeric(g["EV"], errors="coerce").mean()), 2)
            if g["EV"].notna().any()
            else None,
            "avg_ability": round(float(pd.to_numeric(g["能力差"], errors="coerce").mean()), 2)
            if g["能力差"].notna().any()
            else None,
        }

    out_mask = ~df["in_money"]
    in_miss_mask = df["in_money"] & (df["bought_n"] > 0) & (~df["bought_hit"])

    inm = df[df["in_money"]]
    conv = {}
    if len(inm):
        for name, col in [("生成ワイド", "wide"), ("生成馬連", "umaren"), ("生成三連複", "sanrenpuku")]:
            hits = [(r[col] or {}).get("hit") for _, r in inm.iterrows() if r[col] is not None]
            if hits:
                conv[name] = {
                    "n_in_money_with_ticket": len(hits),
                    "hit_rate_given_honmei_in_money": round(
                        100 * sum(int(x or 0) for x in hits) / len(hits), 1
                    ),
                    "status": "集計可" if len(hits) >= MIN_N else "件数不足（採用候補にしない）",
                }
        inmb = inm[inm["bought_n"] > 0]
        if len(inmb):
            conv["購入対象"] = {
                "n_in_money_with_purchase": int(len(inmb)),
                "hit_rate_given_honmei_in_money": round(100 * inmb["bought_hit"].mean(), 1),
                "status": "集計可" if len(inmb) >= MIN_N else "件数不足（採用候補にしない）",
            }

    bets = {}
    bets["仮想単勝"] = bet_metrics(
        df["v_win_hit"], df["v_win_pay"], df["v_win_inv"], list(df["win_odds"]), "仮想単勝(本命100円)"
    )
    bets["仮想複勝"] = bet_metrics(
        df["v_place_hit"],
        df["v_place_pay"],
        df["v_place_inv"],
        list(df["place_odds_est"]),
        "仮想複勝(本命100円・オッズ近似)",
        approx=True,
    )
    for key, label in [
        ("wide", "生成ワイド"),
        ("umaren", "生成馬連"),
        ("sanrenpuku", "生成三連複"),
        ("sanrentan", "生成三連単"),
    ]:
        hits, pays, invs, present = extract_ticket(df[key])
        bets[label] = bet_metrics(hits, pays, invs, list(df["win_odds"]) if present else None, label)
        bets[label]["n_ticket_rows"] = present
        bets[label]["note"] = "analysis_result上の生成買い目（購入対象に限定しない）"

    pb = df[df["bought_n"] > 0]
    if len(pb):
        bets["現行購入対象"] = bet_metrics(
            pb["bought_hit"].astype(int),
            pb["bought_pay"],
            pb["bought_inv"],
            list(pb["win_odds"]),
            "現行購入対象(レース単位・いずれか的中)",
        )
        bets["現行購入対象"]["note"] = "レース単位: 購入対象券の合計投資/払戻"
    else:
        bets["現行購入対象"] = {"n": 0, "status": "購入対象なし", "adoptable": False}

    rec_hits, rec_pays, rec_invs, rec_odds = [], [], [], []
    mapping = {"ワイド": "wide", "馬連": "umaren", "三連複": "sanrenpuku", "三連単": "sanrentan"}
    for _, r in df.iterrows():
        k = mapping.get(str(r["推奨券種"]))
        if not k or r[k] is None:
            continue
        d = r[k]
        rec_hits.append(int(d["hit"]))
        rec_pays.append(float(d["payout"]))
        rec_invs.append(float(d["investment"]))
        rec_odds.append(r["win_odds"])
    bets["推奨券種のみ"] = (
        bet_metrics(rec_hits, rec_pays, rec_invs, rec_odds, "推奨券種のみ")
        if rec_hits
        else {"n": 0, "status": "なし", "adoptable": False}
    )

    n_out = buckets["本命圏外"]
    n_tm = buckets["本命圏内_購入買い目外れ"] + buckets["本命圏内_購入なし_生成券外れ"]
    if n < MIN_N_WEAK:
        verdict = "データ不足で判断不能"
    elif n_out >= max(n_tm, 1) * 1.3:
        verdict = "主因はランク/本命選定（圏外が多い）"
    elif n_tm >= max(n_out, 1) * 1.3:
        verdict = "主因は買い目生成（本命は圏内だが買い目が外れる）"
    else:
        verdict = "ランク選定と買い目生成の両方（混在）"

    return {
        "label": label,
        "n": n,
        "finish_dist": finish_dist,
        "win_n": win_n,
        "place_n": in_m,
        "win_rate": round(100 * win_n / n, 1),
        "place_rate": round(100 * in_m / n, 1),
        "buckets": buckets,
        "means_本命圏外": means(out_mask),
        "means_本命圏内_購入外れ": means(in_miss_mask),
        "conversion_given_in_money": conv,
        "bets": bets,
        "verdict": verdict,
        "conversion_summary": {
            "本命複勝率": round(100 * in_m / n, 1),
            "本命単勝率": round(100 * win_n / n, 1),
            "仮想単勝回収": bets["仮想単勝"].get("recovery"),
            "仮想複勝回収_近似": bets["仮想複勝"].get("recovery"),
            "生成ワイド回収": bets.get("生成ワイド", {}).get("recovery"),
            "現行購入回収": bets.get("現行購入対象", {}).get("recovery"),
        },
    }


def slim_bets(block):
    if not block or block.get("n", 0) == 0:
        return {}
    out = {}
    for k, v in block.get("bets", {}).items():
        out[k] = {
            kk: v.get(kk)
            for kk in [
                "n",
                "hits",
                "hit_rate",
                "recovery",
                "avg_odds",
                "investment",
                "payout",
                "status",
                "adoptable",
                "approx",
            ]
        }
    return out


def main() -> int:
    P = load_predictions()
    A = pd.read_csv(ROOT / "data" / "analysis_result.csv", encoding="utf-8-sig")
    A["race_id"] = A["race_id"].astype(str)
    A["source"] = A["source"].astype(str).str.lower()
    R = pd.read_csv(ROOT / "data" / "results.csv", encoding="utf-8-sig")
    R["race_id"] = R["race_id"].astype(str)

    D = build_race_table(P, A, R)

    report = {
        "purpose": "全期間S/A/Bについて本命着順と買い目的中を分離し、仮想買い方別成績で変換効率を検証（コード変更なし）",
        "period": {
            "all": f"{D['date'].min()} .. {D['date'].max()}",
            "train": f"<= {TRAIN_END}",
            "test": f"> {TRAIN_END}",
        },
        "definitions": {
            "馬券圏内": "着順1-3",
            "仮想単勝": "本命に100円。的中時払戻=確定オッズ×100",
            "仮想複勝": "本命に100円。的中=1-3着。払戻は複勝オッズ近似（公式オッズなし）。回収は採用判定に使わない",
            "生成ワイド/馬連/三連複/三連単": "analysis_resultの生成買い目（購入対象に限定しない）",
            "現行購入対象": "購入対象=1の券種合計（レース単位）",
            "min_n_adoptable": MIN_N,
            "馬単": "analysis_resultに評価行がなく比較対象外",
        },
        "n_total": int(len(D)),
        "counts": {
            str(k): int(v) for k, v in D.groupby(["source", "rank"]).size().items()
        },
        "scopes": {},
        "improvement_candidates": [],
        "rejected_insufficient": [],
        "implement_now": False,
    }

    for src in ["地方", "JRA"]:
        for rk in ["S", "A", "B"]:
            for period, pname in [(None, "全期間"), ("train", "学習"), ("test", "検証")]:
                d = D[(D.source == src) & (D["rank"] == rk)]
                if period:
                    d = d[d.period == period]
                key = f"{src}/{rk}/{pname}"
                report["scopes"][key] = separation_block(d, key)

    focus = {}
    for src in ["地方", "JRA"]:
        focus[src] = {}
        for pname in ["全期間", "学習", "検証"]:
            b = report["scopes"][f"{src}/S/{pname}"]
            focus[src][pname] = {
                "n": b.get("n"),
                "win_rate": b.get("win_rate"),
                "place_rate": b.get("place_rate"),
                "buckets": b.get("buckets"),
                "verdict": b.get("verdict"),
                "conversion_given_in_money": b.get("conversion_given_in_money"),
                "bets": slim_bets(b),
                "means_本命圏外": b.get("means_本命圏外"),
                "means_本命圏内_購入外れ": b.get("means_本命圏内_購入外れ"),
            }
    report["S_focus"] = focus

    ab_notes = {}
    for src in ["地方", "JRA"]:
        ab_notes[src] = {}
        for rk in ["A", "B"]:
            b = report["scopes"][f"{src}/{rk}/全期間"]
            ab_notes[src][rk] = {
                "n": b.get("n"),
                "place_rate": b.get("place_rate"),
                "win_rate": b.get("win_rate"),
                "verdict": b.get("verdict"),
                "buckets": b.get("buckets"),
                "仮想単勝": {
                    k: b.get("bets", {}).get("仮想単勝", {}).get(k)
                    for k in ["n", "hit_rate", "recovery", "status", "adoptable"]
                },
                "生成ワイド": {
                    k: b.get("bets", {}).get("生成ワイド", {}).get(k)
                    for k in ["n", "hit_rate", "recovery", "status", "adoptable"]
                },
                "現行購入": {
                    k: b.get("bets", {}).get("現行購入対象", {}).get(k)
                    for k in ["n", "hit_rate", "recovery", "status", "adoptable"]
                },
                "圏内時ワイド的中": b.get("conversion_given_in_money", {}).get("生成ワイド"),
            }
    report["AB_summary"] = ab_notes

    cands = []
    rej = []
    for src in ["地方", "JRA"]:
        b = report["scopes"][f"{src}/S/全期間"]
        if b.get("n", 0) < MIN_N:
            rej.append(
                {
                    "source": src,
                    "rank": "S",
                    "reason": f"全期間n={b.get('n')} < {MIN_N}",
                    "action": "採用候補にしない",
                }
            )
            continue
        bets = b.get("bets", {})
        win = bets.get("仮想単勝", {})
        wide = bets.get("生成ワイド", {})
        bought = bets.get("現行購入対象", {})
        conv = b.get("conversion_given_in_money", {})
        if b.get("place_rate") is not None and b["place_rate"] < 30:
            cands.append(
                {
                    "source": src,
                    "rank": "S",
                    "candidate": "本命複勝率が低く、買い目以前にS本命選定の見直しが必要",
                    "evidence": {
                        "n": b["n"],
                        "place_rate": b["place_rate"],
                        "win_rate": b["win_rate"],
                        "verdict": b["verdict"],
                    },
                    "status": "改善候補（未実装）",
                }
            )
        wide_conv = conv.get("生成ワイド", {})
        if (
            wide_conv.get("n_in_money_with_ticket", 0) >= MIN_N
            and wide_conv.get("hit_rate_given_honmei_in_money") is not None
            and wide_conv["hit_rate_given_honmei_in_money"] < 25
        ):
            cands.append(
                {
                    "source": src,
                    "rank": "S",
                    "candidate": "本命が圏内でもワイド相手が外れやすい。相手集合/券種配分の見直し",
                    "evidence": {
                        "n": b["n"],
                        "place_rate": b["place_rate"],
                        "圏内時ワイド的中": wide_conv,
                        "仮想単勝": {
                            k: win.get(k) for k in ["n", "hit_rate", "recovery", "status"]
                        },
                        "生成ワイド": {
                            k: wide.get(k) for k in ["n", "hit_rate", "recovery", "status"]
                        },
                        "現行購入": {
                            k: bought.get(k) for k in ["n", "hit_rate", "recovery", "status"]
                        },
                    },
                    "status": "改善候補（未実装）",
                }
            )
        if (
            win.get("adoptable")
            and wide.get("n", 0) >= MIN_N
            and win.get("recovery") is not None
            and wide.get("recovery") is not None
            and wide["recovery"] + 40 < win["recovery"]
        ):
            cands.append(
                {
                    "source": src,
                    "rank": "S",
                    "candidate": "Sでは本命単勝の仮想成績が生成ワイドを大きく上回る。単勝比重の検討",
                    "evidence": {
                        "仮想単勝回収": win.get("recovery"),
                        "生成ワイド回収": wide.get("recovery"),
                        "n": b["n"],
                    },
                    "status": "改善候補（未実装）",
                }
            )

        # train/test consistency check for place_rate low
        tr = report["scopes"][f"{src}/S/学習"]
        te = report["scopes"][f"{src}/S/検証"]
        if tr.get("n", 0) >= MIN_N_WEAK and te.get("n", 0) >= MIN_N_WEAK:
            if tr.get("place_rate", 100) < 35 and te.get("place_rate", 100) < 35:
                cands.append(
                    {
                        "source": src,
                        "rank": "S",
                        "candidate": "学習・検証とも本命複勝率が低い（選定問題の再現）",
                        "evidence": {
                            "train_n": tr["n"],
                            "train_place": tr["place_rate"],
                            "test_n": te["n"],
                            "test_place": te["place_rate"],
                        },
                        "status": "改善候補（未実装・再現あり）",
                    }
                )

    report["improvement_candidates"] = cands
    report["rejected_insufficient"] = rej
    report["conclusion"] = {
        "implement_now": False,
        "headline": (
            "本命着順と買い目を分離すると、地方Sは選定（圏外）と買い目変換の二重損失。"
            "仮想単勝と生成ワイド/購入のギャップが変換失敗を示す。JRA/Sは件数不足。"
        ),
    }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for src in ["地方", "JRA"]:
        print("=" * 60, src, "S")
        for pname in ["全期間", "学習", "検証"]:
            b = report["scopes"][f"{src}/S/{pname}"]
            print(
                f"\n--- {src}/S/{pname} n={b.get('n')} place={b.get('place_rate')} "
                f"win={b.get('win_rate')} verdict={b.get('verdict')}"
            )
            print("buckets", b.get("buckets"))
            print("conv", b.get("conversion_given_in_money"))
            for name, bet in (b.get("bets") or {}).items():
                print(
                    f"  {name}: n={bet.get('n')} hit={bet.get('hit_rate')} "
                    f"rec={bet.get('recovery')} avgOdds={bet.get('avg_odds')} "
                    f"status={bet.get('status')}"
                )
    for src in ["地方", "JRA"]:
        print("=" * 60, src, "A/B")
        for rk in ["A", "B"]:
            b = report["scopes"][f"{src}/{rk}/全期間"]
            print(
                f"{src}/{rk} n={b.get('n')} place={b.get('place_rate')} "
                f"win={b.get('win_rate')} verdict={b.get('verdict')}"
            )
            for name in [
                "仮想単勝",
                "仮想複勝",
                "生成ワイド",
                "生成馬連",
                "生成三連複",
                "現行購入対象",
            ]:
                bet = (b.get("bets") or {}).get(name, {})
                print(
                    f"  {name}: n={bet.get('n')} hit={bet.get('hit_rate')} "
                    f"rec={bet.get('recovery')} status={bet.get('status')}"
                )
    print("\nCANDIDATES:")
    for c in cands:
        print("-", c["source"], c["status"], c["candidate"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
