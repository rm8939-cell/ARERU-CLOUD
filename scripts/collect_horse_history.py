#!/usr/bin/env python3
"""出走馬の過去走データを netkeiba から収集して既存キャッシュに保存する。

2段階で動く。

  1. レースID → {馬名: horse_id} を解決し data/cache/race_horse_ids/ に保存
  2. 各 horse_id について fetch_horse_history を呼び data/cache/horse_results/ に保存

どちらも「ファイルが既にあればスキップ」なので、途中で止めて再実行できる。
値の捏造はしない。取れなかったものは取れなかったものとして失敗ログに残す。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import netkeiba_client as nk  # noqa: E402

RACE_IDS_CACHE = Path("data/cache/race_horse_ids")
HORSE_CACHE = nk.CACHE
FAIL_LOG = Path("data/cache/collect_failures.json")

_local = threading.local()
_lock = threading.Lock()


def client(sleep: float) -> nk.NetkeibaClient:
    """スレッドごとに Session を分ける。"""
    c = getattr(_local, "client", None)
    if c is None:
        c = nk.NetkeibaClient(sleep=sleep)
        _local.client = c
    return c


def race_ids_from_cache(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"レースIDの元データが見つからない: {pattern}")
    frames = [pd.read_csv(f, usecols=["race_id"]) for f in files]
    ids = pd.concat(frames)["race_id"].astype(str).unique().tolist()
    return sorted(ids)


def resolve_race_horses(rid: str, sleep: float) -> dict[str, str]:
    """レース結果ページから {馬名: horse_id} を取る。"""
    out_path = RACE_IDS_CACHE / f"{rid}.json"
    if out_path.exists():
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    soup = client(sleep)._get(f"{nk.DB}/race/{rid}/", encoding="euc-jp")
    mapping: dict[str, str] = {}
    for a in soup.select("a[href*='/horse/']"):
        m = re.search(r"/horse/(\d{6,12})", a.get("href", "") or "")
        name = a.get_text(strip=True)
        if m and name:
            mapping.setdefault(name, m.group(1))
    if not mapping:
        raise RuntimeError("馬リンク0件")
    out_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    time.sleep(sleep)
    return mapping


def fetch_one_horse(hid: str, sleep: float) -> int:
    rows = client(sleep).fetch_horse_history(hid, use_cache=True)
    time.sleep(sleep)
    return len(rows)


def run_pool(items, fn, workers: int, label: str):
    """失敗しても止めずに回し、(成功数, 失敗リスト) を返す。"""
    done = 0
    failures: list[tuple[str, str]] = []
    total = len(items)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append((str(it), f"{type(e).__name__}: {e}"))
            done += 1
            if done % 200 == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (total - done) / rate if rate else 0
                with _lock:
                    print(
                        f"[{label}] {done}/{total} 失敗{len(failures)} "
                        f"{rate:.1f}件/秒 残り{eta/60:.1f}分",
                        flush=True,
                    )
    return total - len(failures), failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runner-glob",
        default="data/quality_backtest_cache/prod_old/runner_*.csv",
        help="race_id 列を持つCSVのglob",
    )
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--limit-races", type=int, default=0)
    ap.add_argument("--limit-horses", type=int, default=0)
    ap.add_argument("--skip-races", action="store_true", help="馬ID解決を飛ばす")
    args = ap.parse_args()

    RACE_IDS_CACHE.mkdir(parents=True, exist_ok=True)
    HORSE_CACHE.mkdir(parents=True, exist_ok=True)

    race_fail: list[tuple[str, str]] = []
    rids = race_ids_from_cache(args.runner_glob)
    if args.limit_races:
        rids = rids[: args.limit_races]
    print(f"対象レース {len(rids)}")

    if not args.skip_races:
        todo = [r for r in rids if not (RACE_IDS_CACHE / f"{r}.json").exists()]
        print(f"馬ID未解決 {len(todo)} / {len(rids)}")
        if todo:
            _, race_fail = run_pool(
                todo, lambda r: resolve_race_horses(r, args.sleep), args.workers, "馬ID"
            )

    horse_ids: set[str] = set()
    for r in rids:
        p = RACE_IDS_CACHE / f"{r}.json"
        if not p.exists():
            continue
        try:
            horse_ids.update(json.loads(p.read_text(encoding="utf-8")).values())
        except Exception:
            continue
    horse_ids_list = sorted(horse_ids)
    print(f"ユニーク馬 {len(horse_ids_list)}")

    todo_h = [h for h in horse_ids_list if not (HORSE_CACHE / f"{h}.json").exists()]
    if args.limit_horses:
        todo_h = todo_h[: args.limit_horses]
    print(f"履歴未取得 {len(todo_h)}")
    horse_fail: list[tuple[str, str]] = []
    if todo_h:
        _, horse_fail = run_pool(
            todo_h, lambda h: fetch_one_horse(h, args.sleep), args.workers, "履歴"
        )

    have = sum(1 for h in horse_ids_list if (HORSE_CACHE / f"{h}.json").exists())
    summary = {
        "対象レース数": len(rids),
        "馬ID解決済みレース数": sum(
            1 for r in rids if (RACE_IDS_CACHE / f"{r}.json").exists()
        ),
        "ユニーク馬数": len(horse_ids_list),
        "履歴取得済み馬数": have,
        "レース失敗": race_fail[:50],
        "レース失敗数": len(race_fail),
        "履歴失敗": horse_fail[:50],
        "履歴失敗数": len(horse_fail),
    }
    FAIL_LOG.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if "失敗" not in k or k.endswith("数")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
