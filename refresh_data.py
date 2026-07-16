"""最新開催日を自動取得し、runners.csv と predictions_by_date を更新する。

score_test_data.csv への依存を廃止し、canonical な出走データは data/runners.csv。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from areru_engine import parse_date
from netkeiba_client import NetkeibaClient

# パイプ実行時でも進捗が見えるようにする
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

DATA = Path("data")
RUNNERS = DATA / "runners.csv"
LEGACY = DATA / "score_test_data.csv"
PRED_DIR = DATA / "predictions_by_date"
PRED_DIR.mkdir(parents=True, exist_ok=True)

RUNNER_COLS = [
    "race_id", "日付", "レース", "馬名", "実着順",
    "着順1", "人気1", "着順2", "人気2", "着順3", "人気3",
    "着順4", "人気4", "着順5", "人気5",
]


def _normalize_runners(df: pd.DataFrame) -> pd.DataFrame:
    for c in RUNNER_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[RUNNER_COLS].copy()
    df["日付"] = parse_date(df["日付"]).dt.strftime("%Y-%m-%d")
    df["レース"] = pd.to_numeric(df["レース"], errors="coerce")
    return df.dropna(subset=["日付", "馬名"]).reset_index(drop=True)


def load_existing_runners() -> pd.DataFrame:
    if RUNNERS.exists():
        return _normalize_runners(pd.read_csv(RUNNERS, encoding="utf-8-sig"))
    if LEGACY.exists():
        print(f"↪️  初回移行: {LEGACY} → {RUNNERS}")
        return _normalize_runners(pd.read_csv(LEGACY, encoding="utf-8-sig"))
    return pd.DataFrame(columns=RUNNER_COLS)


def available_dates(runners: pd.DataFrame) -> list[str]:
    if runners.empty:
        return []
    d = parse_date(runners["日付"]).dropna().dt.strftime("%Y-%m-%d").unique().tolist()
    return sorted(d, reverse=True)


def build_date_runners(client: NetkeibaClient, target: str, include_results: bool = True) -> pd.DataFrame:
    ymd = target.replace("-", "")
    race_ids = client.list_race_ids(ymd)
    if not race_ids:
        print(f"⚠️  {target}: レースなし")
        return pd.DataFrame(columns=RUNNER_COLS)

    rows = []
    print(f"📥 {target}: {len(race_ids)}レース取得中...")
    for i, rid in enumerate(race_ids, 1):
        entries = client.fetch_entries(rid)
        results = client.fetch_results(rid) if include_results else {}
        print(f"  [{i}/{len(race_ids)}] {rid} 出走{len(entries)}頭")
        for e in entries:
            hist = client.fetch_horse_history(e["horse_id"]) if e.get("horse_id") else []
            score = client.past_five_for_score(hist, target)
            finish = results.get(e["馬名"], "")
            rows.append({
                "race_id": rid,
                "日付": e.get("日付") or target,
                "レース": e.get("レース"),
                "馬名": e["馬名"],
                "実着順": finish or score.get("実着順", ""),
                **{k: score[k] for k in score if k != "実着順"},
            })
            if finish:
                rows[-1]["実着順"] = finish
    return _normalize_runners(pd.DataFrame(rows))


def merge_runners(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new.empty:
        return base
    if base.empty:
        return new
    dates = set(parse_date(new["日付"]).dt.strftime("%Y-%m-%d"))
    keep = base[~parse_date(base["日付"]).dt.strftime("%Y-%m-%d").isin(dates)]
    return _normalize_runners(pd.concat([keep, new], ignore_index=True))


def save_runners(df: pd.DataFrame) -> None:
    df = _normalize_runners(df)
    df = df.sort_values(["日付", "レース", "馬名"]).reset_index(drop=True)
    df.to_csv(RUNNERS, index=False, encoding="utf-8-sig")
    print(f"✅ runners.csv 保存: {len(df)}行 / 開催日 {available_dates(df)}")


def generate_predictions(dates: list[str] | None = None) -> None:
    if dates:
        for d in dates:
            print(f"🔮 predictions 生成: {d}")
            subprocess.run([sys.executable, "replay_predict.py", d], check=True, timeout=600)
    else:
        print("🔮 predictions 全開催日生成 (--all)")
        subprocess.run([sys.executable, "replay_predict.py", "--all"], check=True, timeout=1800)


def refresh(
    dates: list[str] | None = None,
    discover: bool = True,
    latest_only: bool = False,
    lookback: int = 28,
    lookahead: int = 14,
    skip_predict: bool = False,
    migrate_only: bool = False,
) -> list[str]:
    runners = load_existing_runners()
    if migrate_only:
        save_runners(runners)
        if not skip_predict:
            generate_predictions()
        return available_dates(runners)

    client = NetkeibaClient()
    target_dates: list[str] = []
    if dates:
        target_dates = dates
    elif discover:
        found = client.discover_kaisai_dates(lookback=lookback, lookahead=lookahead)
        print(f"🗓️  自動検出開催日: {found}")
        if not found:
            raise SystemExit("開催日を検出できませんでした")
        if latest_only:
            # 最新開催日から差2日以内の開催日ブロック（通常は土日）を更新
            target_dates = [found[0]]
            for d in found[1:]:
                delta = abs((datetime.fromisoformat(found[0]) - datetime.fromisoformat(d)).days)
                if delta <= 2:
                    target_dates.append(d)
                else:
                    break
            target_dates = sorted(set(target_dates))
        else:
            # 既存に無い日 + 直近2開催日は必ず更新
            existing = set(available_dates(runners))
            recent = found[:4]
            target_dates = sorted(set(recent) | (set(found) - existing))
    else:
        target_dates = available_dates(runners)

    print(f"🎯 更新対象: {target_dates}")
    for d in target_dates:
        built = build_date_runners(client, d, include_results=True)
        runners = merge_runners(runners, built)

    save_runners(runners)
    av = available_dates(runners)
    if not skip_predict:
        # 更新した日を優先生成。未生成日もまとめて --all
        missing = [d for d in av if not (PRED_DIR / f"predictions_{d}.csv").exists()]
        to_gen = sorted(set(target_dates) | set(missing))
        if to_gen:
            generate_predictions(to_gen)
        else:
            generate_predictions(target_dates)
    return av


def main():
    ap = argparse.ArgumentParser(description="ARERU P0-2: 最新開催日取得 & runners/predictions 更新")
    ap.add_argument("--dates", nargs="*", help="YYYY-MM-DD を明示指定")
    ap.add_argument("--latest-only", action="store_true", help="最新開催週末だけ更新")
    ap.add_argument("--no-discover", action="store_true", help="開催日自動検出をしない")
    ap.add_argument("--skip-predict", action="store_true", help="predictions 生成をスキップ")
    ap.add_argument("--migrate-only", action="store_true", help="既存CSVの移行のみ")
    ap.add_argument("--lookback", type=int, default=28)
    ap.add_argument("--lookahead", type=int, default=14)
    ap.add_argument("--list", action="store_true", help="検出開催日を表示して終了")
    args = ap.parse_args()

    if args.list:
        client = NetkeibaClient()
        print("\n".join(client.discover_kaisai_dates(lookback=args.lookback, lookahead=args.lookahead)))
        return

    av = refresh(
        dates=args.dates,
        discover=not args.no_discover,
        latest_only=args.latest_only,
        lookback=args.lookback,
        lookahead=args.lookahead,
        skip_predict=args.skip_predict,
        migrate_only=args.migrate_only,
    )
    print()
    print("=" * 50)
    print("✅ P0-2 データ更新完了")
    print("開催日:", ", ".join(av))
    print("runners:", RUNNERS)
    print("predictions:", PRED_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()
