"""最新開催日を自動取得し、runners.csv と predictions_by_date を更新する。

PO-3/PO-7: JRA・NAR の単勝/券種オッズを取得して runners.csv へ統合し、
AIスコア再計算（predictions 再生成）まで一気通貫で行う。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from areru_engine import parse_date, source_from_race_id
from netkeiba_client import NetkeibaClient, infer_source

# パイプ実行時でも進捗が見えるようにする
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

DATA = Path("data")
RUNNERS = DATA / "runners.csv"
LEGACY = DATA / "score_test_data.csv"
PRED_DIR = DATA / "predictions_by_date"
ODDS_TICKETS = DATA / "odds_tickets"
PRED_DIR.mkdir(parents=True, exist_ok=True)
ODDS_TICKETS.mkdir(parents=True, exist_ok=True)

RUNNER_COLS = [
    "race_id", "日付", "レース", "馬名", "馬番", "枠", "騎手", "斤量", "実着順",
    "着順1", "人気1", "着順2", "人気2", "着順3", "人気3",
    "着順4", "人気4", "着順5", "人気5",
    "単勝オッズ", "人気", "オッズ更新日時", "source",
]


def _normalize_runners(df: pd.DataFrame) -> pd.DataFrame:
    for c in RUNNER_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[RUNNER_COLS].copy()
    df["日付"] = parse_date(df["日付"]).dt.strftime("%Y-%m-%d")
    df["レース"] = pd.to_numeric(df["レース"], errors="coerce")
    # 馬番は "4.0" にならないよう整数文字列へ
    ban = pd.to_numeric(df["馬番"], errors="coerce")
    df["馬番"] = ban.apply(lambda x: str(int(x)) if pd.notna(x) else "")
    waku = pd.to_numeric(df["枠"], errors="coerce")
    df["枠"] = waku.apply(lambda x: str(int(x)) if pd.notna(x) else "")
    pop = pd.to_numeric(df["人気"], errors="coerce")
    df["人気"] = pop.apply(lambda x: str(int(x)) if pd.notna(x) else "")
    odds = pd.to_numeric(df["単勝オッズ"], errors="coerce")
    df["単勝オッズ"] = odds.apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")
    # source 補完
    df["source"] = df.apply(
        lambda r: str(r["source"]).strip().lower()
        if str(r.get("source") or "").strip().lower() in ("jra", "nar")
        else source_from_race_id(r.get("race_id", "")),
        axis=1,
    )
    return df.dropna(subset=["日付", "馬名"]).reset_index(drop=True)


def load_existing_runners() -> pd.DataFrame:
    if RUNNERS.exists():
        return _normalize_runners(pd.read_csv(RUNNERS, encoding="utf-8-sig"))
    if LEGACY.exists():
        print(f"↪️  初回移行: {LEGACY} → {RUNNERS}")
        return _normalize_runners(pd.read_csv(LEGACY, encoding="utf-8-sig"))
    return pd.DataFrame(columns=RUNNER_COLS)


def available_dates(runners: pd.DataFrame, source: str | None = None) -> list[str]:
    if runners.empty:
        return []
    df = runners
    if source in ("jra", "nar"):
        df = df[df["source"].astype(str) == source]
    d = parse_date(df["日付"]).dropna().dt.strftime("%Y-%m-%d").unique().tolist()
    return sorted(d, reverse=True)


def _apply_win_odds(entries: list[dict], win_odds: dict[str, dict]) -> list[dict]:
    """馬番で単勝オッズを上書き。API未公開時は出馬表の値を残す。"""
    if not win_odds:
        return entries
    for e in entries:
        ban = str(e.get("馬番") or "").strip()
        info = win_odds.get(ban) or win_odds.get(ban.zfill(2))
        if not info:
            continue
        if info.get("単勝オッズ"):
            e["単勝オッズ"] = info["単勝オッズ"]
        if info.get("人気"):
            e["人気"] = info["人気"]
        if info.get("オッズ更新日時"):
            e["オッズ更新日時"] = info["オッズ更新日時"]
    return entries


def _save_ticket_odds(client: NetkeibaClient, race_id: str, source: str) -> None:
    """券種別オッズを data/odds_tickets/{race_id}.json に保存。"""
    rid = str(race_id)
    if not (rid.isdigit() and len(rid) == 12):
        return
    try:
        maps = client.fetch_ticket_odds_maps(rid, source=source)
        if not any(maps.get(k) for k in ("ワイド", "馬連", "三連複")):
            return
        (ODDS_TICKETS / f"{rid}.json").write_text(
            json.dumps(maps, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"  ⚠️ 券種オッズ保存失敗 {rid}: {e}")


def build_date_runners(
    client: NetkeibaClient,
    target: str,
    source: str = "jra",
    include_results: bool = True,
    include_odds: bool = True,
) -> pd.DataFrame:
    ymd = target.replace("-", "")
    race_ids = client.list_race_ids(ymd, source=source)
    if not race_ids:
        print(f"⚠️  {source.upper()} {target}: レースなし")
        return pd.DataFrame(columns=RUNNER_COLS)

    rows = []
    print(f"📥 {source.upper()} {target}: {len(race_ids)}レース取得中...")
    for i, rid in enumerate(race_ids, 1):
        entries = client.fetch_entries(rid, source=source)
        results = client.fetch_results(rid, source=source) if include_results else {}
        win_odds = client.fetch_win_odds(rid, source=source) if include_odds else {}
        entries = _apply_win_odds(entries, win_odds)
        odds_n = sum(1 for e in entries if e.get("単勝オッズ"))
        if include_odds and odds_n:
            _save_ticket_odds(client, rid, source)
        print(f"  [{i}/{len(race_ids)}] {rid} 出走{len(entries)}頭 オッズ{odds_n}頭")
        for e in entries:
            hist = client.fetch_horse_history(e["horse_id"]) if e.get("horse_id") else []
            score = client.past_five_for_score(hist, target)
            finish = results.get(e["馬名"], "")
            rows.append({
                "race_id": rid,
                "日付": e.get("日付") or target,
                "レース": e.get("レース"),
                "馬名": e["馬名"],
                "馬番": e.get("馬番", ""),
                "枠": e.get("枠", ""),
                "騎手": e.get("騎手", ""),
                "斤量": e.get("斤量", ""),
                "実着順": finish or score.get("実着順", ""),
                "単勝オッズ": e.get("単勝オッズ", ""),
                "人気": e.get("人気", ""),
                "オッズ更新日時": e.get("オッズ更新日時", ""),
                "source": source,
                **{k: score[k] for k in score if k != "実着順"},
            })
            if finish:
                rows[-1]["実着順"] = finish
    return _normalize_runners(pd.DataFrame(rows))


def refresh_odds_for_dates(
    client: NetkeibaClient,
    runners: pd.DataFrame,
    dates: list[str],
    source: str | None = None,
) -> pd.DataFrame:
    """既存 runners の対象日だけオッズ列を更新（履歴再取得なし）。"""
    if runners.empty or not dates:
        return runners
    df = runners.copy()
    date_mask = parse_date(df["日付"]).dt.strftime("%Y-%m-%d").isin(dates)
    if source in ("jra", "nar"):
        date_mask = date_mask & (df["source"].astype(str) == source)
    target = df[date_mask].copy()
    if target.empty:
        print(f"⚠️  オッズ更新対象なし: {dates} source={source}")
        return runners

    race_ids = sorted(set(target["race_id"].astype(str)))
    print(f"💰 オッズ更新: {len(race_ids)}レース / 日={dates} / source={source or 'all'}")
    odds_by_race: dict[str, dict] = {}
    for i, rid in enumerate(race_ids, 1):
        # 旧JRA URL 行はスキップ
        if not str(rid).isdigit() or len(str(rid)) != 12:
            print(f"  [{i}/{len(race_ids)}] {rid}: skip (非netkeiba ID)")
            continue
        src = source if source in ("jra", "nar") else infer_source(rid)
        win = client.fetch_win_odds(rid, source=src)
        odds_by_race[rid] = win
        n = len({k: v for k, v in win.items() if len(k) <= 2 and v.get("単勝オッズ")})
        if n:
            _save_ticket_odds(client, rid, src)
        print(f"  [{i}/{len(race_ids)}] {rid}: オッズ{n}頭")

    updated = 0
    for idx in target.index:
        rid = str(df.at[idx, "race_id"])
        ban = str(df.at[idx, "馬番"] or "").strip()
        info = (odds_by_race.get(rid) or {}).get(ban) or (odds_by_race.get(rid) or {}).get(ban.zfill(2))
        if not info or not info.get("単勝オッズ"):
            continue
        df.at[idx, "単勝オッズ"] = info["単勝オッズ"]
        if info.get("人気"):
            df.at[idx, "人気"] = info["人気"]
        if info.get("オッズ更新日時"):
            df.at[idx, "オッズ更新日時"] = info["オッズ更新日時"]
        updated += 1
    print(f"✅ オッズ反映: {updated}頭")
    return _normalize_runners(df)


def merge_runners(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """同一 race_id を差し替え（JRA/NAR 同日混在でも他ソースを消さない）。"""
    if new.empty:
        return base
    if base.empty:
        return new
    drop_ids = set(new["race_id"].astype(str))
    keep = base[~base["race_id"].astype(str).isin(drop_ids)]
    return _normalize_runners(pd.concat([keep, new], ignore_index=True))


def save_runners(df: pd.DataFrame) -> None:
    df = _normalize_runners(df)
    df = df.sort_values(["日付", "source", "レース", "馬名"]).reset_index(drop=True)
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


def _sources_list(source: str) -> list[str]:
    if source == "all":
        return ["jra", "nar"]
    return [source]


def refresh(
    dates: list[str] | None = None,
    discover: bool = True,
    latest_only: bool = False,
    lookback: int = 28,
    lookahead: int = 14,
    skip_predict: bool = False,
    migrate_only: bool = False,
    odds_only: bool = False,
    include_odds: bool = True,
    source: str = "jra",
) -> list[str]:
    runners = load_existing_runners()
    if migrate_only:
        save_runners(runners)
        if not skip_predict:
            generate_predictions()
        return available_dates(runners)

    client = NetkeibaClient()
    sources = _sources_list(source)
    all_target_dates: list[str] = []

    for src in sources:
        target_dates: list[str] = []
        if dates:
            target_dates = dates
        elif discover:
            found = client.discover_kaisai_dates(
                lookback=lookback, lookahead=lookahead, source=src
            )
            print(f"🗓️  {src.upper()} 自動検出開催日: {found}")
            if not found:
                print(f"⚠️  {src.upper()} 開催日なし")
                continue
            if latest_only:
                target_dates = [found[0]]
                for d in found[1:]:
                    delta = abs((datetime.fromisoformat(found[0]) - datetime.fromisoformat(d)).days)
                    if delta <= 2:
                        target_dates.append(d)
                    else:
                        break
                target_dates = sorted(set(target_dates))
            else:
                existing = set(available_dates(runners, source=src))
                recent = found[:4]
                target_dates = sorted(set(recent) | (set(found) - existing))
        else:
            target_dates = available_dates(runners, source=src)

        print(f"🎯 {src.upper()} 更新対象: {target_dates}")
        all_target_dates.extend(target_dates)

        if odds_only:
            runners = refresh_odds_for_dates(client, runners, target_dates, source=src)
        else:
            for d in target_dates:
                built = build_date_runners(
                    client, d, source=src, include_results=True, include_odds=include_odds
                )
                runners = merge_runners(runners, built)

    save_runners(runners)
    all_target_dates = sorted(set(all_target_dates))

    av = available_dates(runners)
    if not skip_predict:
        missing = [d for d in av if not (PRED_DIR / f"predictions_{d}.csv").exists()]
        to_gen = sorted(set(all_target_dates) | set(missing))
        if to_gen:
            generate_predictions(to_gen)
        elif all_target_dates:
            generate_predictions(all_target_dates)
    return av


def main():
    ap = argparse.ArgumentParser(
        description="ARERU PO-3/PO-7: JRA・NAR 開催日・オッズ取得 & runners/predictions 更新"
    )
    ap.add_argument("--dates", nargs="*", help="YYYY-MM-DD を明示指定")
    ap.add_argument("--latest-only", action="store_true", help="最新開催週末だけ更新")
    ap.add_argument("--no-discover", action="store_true", help="開催日自動検出をしない")
    ap.add_argument("--skip-predict", action="store_true", help="predictions 生成をスキップ")
    ap.add_argument("--migrate-only", action="store_true", help="既存CSVの移行のみ")
    ap.add_argument("--odds-only", action="store_true", help="オッズ列だけ再取得して再予想")
    ap.add_argument("--no-odds", action="store_true", help="オッズ取得をスキップ")
    ap.add_argument("--source", choices=["jra", "nar", "all"], default="jra")
    ap.add_argument("--lookback", type=int, default=28)
    ap.add_argument("--lookahead", type=int, default=14)
    ap.add_argument("--list", action="store_true", help="検出開催日を表示して終了")
    args = ap.parse_args()

    if args.list:
        client = NetkeibaClient()
        for src in _sources_list(args.source):
            print(f"# {src}")
            print("\n".join(client.discover_kaisai_dates(
                lookback=args.lookback, lookahead=args.lookahead, source=src
            )))
        return

    av = refresh(
        dates=args.dates,
        discover=not args.no_discover,
        latest_only=args.latest_only,
        lookback=args.lookback,
        lookahead=args.lookahead,
        skip_predict=args.skip_predict,
        migrate_only=args.migrate_only,
        odds_only=args.odds_only,
        include_odds=not args.no_odds,
        source=args.source,
    )
    print()
    print("=" * 50)
    print("✅ PO-7 データ更新完了")
    print("source:", args.source)
    print("開催日:", ", ".join(av))
    print("runners:", RUNNERS)
    print("predictions:", PRED_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()
