"""払戻ベースの実回収率・結果検証集計（P0-4）。

予想CSVの買い目と payouts.csv / results.csv を照合し、
実払戻があるレースだけ回収率を算出する。未取得は集計に入れない。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ticket_value import combo_key, name_to_umaban_map, parse_bet_line, _norm_umaban

DATA = Path("data")
ARCH = DATA / "predictions_by_date"


def load_payouts() -> pd.DataFrame:
    p = DATA / "payouts.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df
    df["race_id"] = df["race_id"].astype(str)
    df["券種"] = df["券種"].astype(str)
    df["組合せ"] = (
        df["組合せ"].astype(str).str.replace(r"[^0-9]", "-", regex=True).str.strip("-")
    )
    df["払戻"] = pd.to_numeric(df["払戻"], errors="coerce")
    return df


def payout_key(combo: str, kind: str) -> str:
    nums = [_norm_umaban(x) for x in re.split(r"[-ー－\s]+", str(combo)) if _norm_umaban(x)]
    return combo_key(nums, kind)


def score_map_for_race(scores: pd.DataFrame, race_id: str) -> dict[str, str]:
    if scores is None or scores.empty:
        return {}
    sub = scores[scores["race_id"].astype(str) == str(race_id)]
    return name_to_umaban_map(sub)


def analyze_date(target: str) -> dict:
    pred_path = ARCH / f"predictions_{target}.csv"
    score_path = ARCH / f"scores_{target}.csv"
    if not pred_path.exists():
        return {"date": target, "error": "predictions missing"}
    preds = pd.read_csv(pred_path)
    scores = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    payouts = load_payouts()
    if payouts.empty:
        return {
            "date": target,
            "races": int(len(preds)),
            "evaluated": 0,
            "note": "payouts.csv 未取得のため実回収率は算出しません",
        }

    stake_unit = 100  # 1点100円想定
    rows = []
    for _, pred in preds.iterrows():
        race_id = str(pred["race_id"])
        kind = str(pred.get("推奨券種", ""))
        if kind not in ("ワイド", "馬連", "三連複"):
            continue
        race_pays = payouts[(payouts["race_id"] == race_id) & (payouts["券種"] == kind)]
        if race_pays.empty:
            continue
        name_map = score_map_for_race(scores, race_id)
        lines = str(pred.get(f"{kind}買い目", "")).split("｜")
        hit_pay = 0
        points = 0
        for line in lines:
            names, _hit = parse_bet_line(line)
            if not names:
                continue
            umas = [name_map.get(n) or name_map.get(re.sub(r"\s+", "", n), "") for n in names]
            key = combo_key(umas, kind)
            if not key:
                continue
            points += 1
            # payout combo normalized to zero-padded joined compare
            for _, pay in race_pays.iterrows():
                pk = payout_key(pay["組合せ"], kind).replace("-", "")
                if pk == key.replace("-", "") or pk == key:
                    hit_pay += int(pay["払戻"]) if pd.notna(pay["払戻"]) else 0
        if points == 0:
            continue
        invested = points * stake_unit
        rows.append(
            {
                "race_id": race_id,
                "開催地": pred.get("開催地", ""),
                "レース": pred.get("レース", ""),
                "推奨券種": kind,
                "勝負ランク": pred.get("勝負ランク", ""),
                "点数": points,
                "投資": invested,
                "払戻合計": hit_pay,
                "回収率": round(hit_pay / invested * 100, 1) if invested else None,
                "的中": hit_pay > 0,
            }
        )

    detail = pd.DataFrame(rows)
    out = {
        "date": target,
        "races": int(len(preds)),
        "evaluated": int(len(detail)),
        "hits": int(detail["的中"].sum()) if len(detail) else 0,
        "invested": int(detail["投資"].sum()) if len(detail) else 0,
        "returned": int(detail["払戻合計"].sum()) if len(detail) else 0,
    }
    if out["invested"]:
        out["roi"] = round(out["returned"] / out["invested"] * 100, 1)
    else:
        out["roi"] = None
        out["note"] = "照合可能な払戻がありません"
    if len(detail):
        detail.to_csv(DATA / f"roi_{target}.csv", index=False, encoding="utf-8-sig")
    return out


def analyze_all() -> list[dict]:
    results = []
    for p in sorted(ARCH.glob("predictions_*.csv")):
        d = p.stem.replace("predictions_", "")
        results.append(analyze_date(d))
    (DATA / "roi_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", help="YYYY-MM-DD")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all or not args.date:
        rows = analyze_all()
        for r in rows:
            print(r, flush=True)
        return
    print(analyze_date(args.date), flush=True)


if __name__ == "__main__":
    main()
