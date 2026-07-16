"""NAR地方データ更新・予想生成（P0-5）。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from areru_engine import clean_name, parse_date
from nar_client import NarClient
from nar_engine import build_nar_predictions
from netkeiba_client import yyyymmdd
from refresh_data import build_runner_row, get_horse_history, history_rows_for_horse

DATA = Path("data")
ARCH = DATA / "nar_predictions_by_date"
SCORE = DATA / "nar_score_data.csv"
HIST = DATA / "nar_history.csv"
META = DATA / "nar_refresh_meta.json"
RESULTS = DATA / "nar_results.csv"


def discover_nar_dates(around: date | None = None, back: int = 3, forward: int = 2) -> list[str]:
    around = around or date.today()
    client = NarClient(sleep=0.15)
    found = []
    for i in range(-back, forward + 1):
        d = around + timedelta(days=i)
        key = d.strftime("%Y%m%d")
        try:
            ids = client.list_race_ids(key)
        except Exception as e:
            print(f"⚠️ NAR {key} 失敗: {e}", flush=True)
            continue
        if ids:
            found.append(d.isoformat())
            print(f"✅ NAR開催 {d.isoformat()} / {len(ids)}R", flush=True)
    return found


def refresh_nar_date(target: str, client: NarClient | None = None, max_races: int | None = None) -> dict:
    client = client or NarClient(sleep=0.18)
    race_ids = client.list_race_ids(yyyymmdd(target))
    if not race_ids:
        raise ValueError(f"NAR {target} にレースなし")
    if max_races:
        race_ids = race_ids[:max_races]
    print(f"\n📅 NAR {target} / {len(race_ids)}R", flush=True)

    score_rows, history_rows, result_rows = [], [], []
    for idx, race_id in enumerate(race_ids, start=1):
        meta = client.parse_race_id(race_id)
        print(f"🏇 {idx}/{len(race_ids)} {meta['venue']}{meta['race_number']}R", flush=True)
        result = client.fetch_result(race_id)
        race_date = result.get("日付") or target
        runners_by_name = {}
        if result.get("has_result"):
            for r in result["runners"]:
                runners_by_name[clean_name(r["馬名"])] = r
                result_rows.append(
                    {
                        "race_id": race_id,
                        "日付": race_date,
                        "レース": meta["race_number"],
                        "開催地": meta["venue"],
                        "馬名": r["馬名"],
                        "着順": r.get("着順", ""),
                        "馬番": r.get("馬番", ""),
                        "人気": r.get("人気", ""),
                        "horse_id": r.get("horse_id", ""),
                        "source": "nar",
                    }
                )
        shutuba = client.fetch_shutuba(race_id)
        horses = shutuba.get("horses") or []
        # 確定後は結果表の全頭を優先（取消除く実出走）
        if result.get("runners") and len(result["runners"]) > len(horses):
            horses = [
                {
                    "馬名": r["馬名"],
                    "horse_id": r.get("horse_id", ""),
                    "馬番": r.get("馬番", ""),
                }
                for r in result["runners"]
            ]
        if not horses:
            continue
        for horse in horses:
            hid = horse.get("horse_id", "")
            hist = get_horse_history(client, hid, use_cache=True)
            finish = None
            matched = runners_by_name.get(clean_name(horse["馬名"]))
            if matched:
                finish = matched.get("着順")
                if not horse.get("馬番"):
                    horse["馬番"] = matched.get("馬番", "")
            score_rows.append(
                build_runner_row(race_date, race_id, meta["race_number"], horse, hist, finish=finish)
            )
            history_rows.extend(
                history_rows_for_horse(
                    meta["race_number"], horse["馬名"], hid, race_id, hist, race_date
                )
            )

    score = pd.read_csv(SCORE) if SCORE.exists() else pd.DataFrame()
    if len(score):
        score = score[~parse_date(score["日付"]).dt.strftime("%Y-%m-%d").eq(target)]
    score = pd.concat([score, pd.DataFrame(score_rows)], ignore_index=True)
    score.to_csv(SCORE, index=False, encoding="utf-8-sig")

    if history_rows:
        hist_df = pd.read_csv(HIST) if HIST.exists() else pd.DataFrame()
        hist_df = pd.concat([hist_df, pd.DataFrame(history_rows)], ignore_index=True)
        hist_df = hist_df.drop_duplicates(subset=["馬名", "年月日", "レース名", "着順"], keep="last")
        hist_df.to_csv(HIST, index=False, encoding="utf-8-sig")

    if result_rows:
        old = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
        if len(old):
            old = old[~old["race_id"].astype(str).isin({r["race_id"] for r in result_rows})]
        pd.concat([old, pd.DataFrame(result_rows)], ignore_index=True).to_csv(
            RESULTS, index=False, encoding="utf-8-sig"
        )

    meta = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "date": target,
        "races": len(race_ids),
        "runners": len(score_rows),
        "results": len(result_rows),
        "source": "nar",
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ NAR {target} 保存 runners={len(score_rows)} results={len(result_rows)}", flush=True)
    return meta


def predict_nar(target: str):
    ARCH.mkdir(parents=True, exist_ok=True)
    runners = pd.read_csv(SCORE)
    history = pd.read_csv(HIST) if HIST.exists() else None
    result, scores = build_nar_predictions(target, runners, history)
    result.to_csv(ARCH / f"predictions_{target}.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(ARCH / f"scores_{target}.csv", index=False, encoding="utf-8-sig")
    print(f"✅ NAR予想 {target}: {len(result)}R", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--max-races", type=int, default=0, help="テスト用にレース数制限")
    ap.add_argument("--back", type=int, default=2)
    ap.add_argument("--forward", type=int, default=1)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    ARCH.mkdir(parents=True, exist_ok=True)

    if args.discover:
        print(discover_nar_dates(back=args.back, forward=args.forward))
        return

    targets = args.dates or []
    if args.auto or not targets:
        targets = discover_nar_dates(back=args.back, forward=args.forward)
    if not targets:
        print("NAR対象日なし", file=sys.stderr)
        sys.exit(1)

    client = NarClient(sleep=0.18)
    max_races = args.max_races or None
    for d in targets:
        try:
            refresh_nar_date(d, client=client, max_races=max_races)
            predict_nar(d)
        except Exception as e:
            print(f"❌ NAR {d}: {e}", flush=True)
    print("🔥 NAR更新完了", flush=True)


if __name__ == "__main__":
    main()
