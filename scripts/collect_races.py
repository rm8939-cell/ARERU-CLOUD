#!/usr/bin/env python3
"""netkeiba のレース結果ページを日付範囲でまとめて取得する。

1レース1リクエストで、出走全頭の 着順・単勝オッズ・人気・タイム・通過・上り・
馬体重と、レース側の 距離・芝ダ・回り・馬場・天候・レース名 が揃う。

これを使うと「あるレースの予想に必要な過去走」も、同じ取得物の中の
より古い日付のレースから作れる。馬ごとの戦績ページを別に叩く必要がなく、
しかも出走全頭が必ず揃うので、一部の馬しか手元に無いことによる
出走表の歪みが起きない。

保存先:
  data/cache/race_lists/{YYYYMMDD}.json   その日のレースID一覧
  data/cache/race_results/{race_id}.json  1レース分の結果
既にあるファイルは取りに行かないので、途中で止めて再開できる。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netkeiba_client as nk  # noqa: E402

LIST_CACHE = Path("data/cache/race_lists")
RACE_CACHE = Path("data/cache/race_results")
FAIL_LOG = Path("data/cache/collect_races_failures.json")

_local = threading.local()
_lock = threading.Lock()

HORSE_COLS = {
    "着順": "着順", "枠番": "枠番", "馬番": "馬番", "馬名": "馬名", "性齢": "性齢",
    "斤量": "斤量", "騎手": "騎手", "タイム": "タイム", "着差": "着差",
    "通過": "通過", "上り": "上り", "単勝": "単勝オッズ", "人気": "人気",
    "馬体重": "馬体重", "備考": "備考", "調教師": "調教師",
}


def client(sleep: float) -> nk.NetkeibaClient:
    c = getattr(_local, "client", None)
    if c is None:
        c = nk.NetkeibaClient(sleep=sleep)
        _local.client = c
    return c


def with_retry(fn, attempts: int = 4, base_wait: float = 3.0):
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < attempts - 1:
                time.sleep(base_wait * (2 ** i))
    raise last if last else RuntimeError("retry failed")


def fetch_race_list(day: str, sleep: float) -> list[str]:
    p = LIST_CACHE / f"{day}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    soup = with_retry(
        lambda: client(sleep)._get(f"{nk.DB}/race/list/{day}/", encoding="euc-jp")
    )
    ids = sorted({
        m.group(1)
        for a in soup.select("a[href*='/race/']")
        for m in [re.search(r"/race/(\d{12})/?$", a.get("href", "") or "")]
        if m
    })
    p.write_text(json.dumps(ids), encoding="utf-8")
    time.sleep(sleep)
    return ids


def parse_race(soup, race_id: str) -> dict:
    title = soup.title.get_text(strip=True) if soup.title else ""
    dm = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
    race_date = (f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}"
                 if dm else "")
    race_name = title.split("｜")[0].strip() if "｜" in title else ""

    intro = soup.select_one("div.data_intro p span")
    info = intro.get_text(" ", strip=True) if intro else ""
    sm = re.search(r"(芝|ダ|障)\s*(左|右|直|直線)?\s*(\d{3,4})\s*m", info)
    surface = sm.group(1) if sm else ""
    turn = sm.group(2) or "" if sm else ""
    dist = int(sm.group(3)) if sm else None
    wm = re.search(r"天候\s*[:：]\s*(\S+)", info)
    weather = wm.group(1) if wm else ""
    gm = re.search(r"(?:ダート|芝|障)\s*[:：]\s*(\S+)", info)
    going = gm.group(1) if gm else ""

    table = soup.select_one("table.race_table_01")
    horses: list[dict] = []
    if table:
        rows = table.select("tr")
        hdr = [x.get_text(strip=True) for x in rows[0].find_all(["th", "td"])] if rows else []
        idx = {}
        for want, key in HORSE_COLS.items():
            if want in hdr:
                idx[key] = hdr.index(want)
        for tr in rows[1:]:
            tds = tr.find_all("td")
            if len(tds) < 8:
                continue
            rec: dict[str, str] = {}
            for key, i in idx.items():
                if 0 <= i < len(tds):
                    rec[key] = tds[i].get_text(strip=True)
            a = tr.select_one("a[href*='/horse/']")
            hm = re.search(r"/horse/([A-Za-z]?\d{6,12})", a.get("href", "") or "") if a else None
            rec["horse_id"] = hm.group(1) if hm else ""
            if rec.get("馬名"):
                horses.append(rec)

    return {
        "race_id": race_id,
        "date": race_date,
        "venue": nk.VENUE_CODES.get(str(race_id)[4:6], ""),
        "race_no": int(str(race_id)[-2:]) if str(race_id)[-2:].isdigit() else None,
        "レース名": race_name,
        "芝ダ": surface,
        "回り": turn,
        "距離": dist,
        "馬場": going,
        "天候": weather,
        "頭数": len(horses),
        "horses": horses,
    }


def fetch_race(rid: str, sleep: float) -> int:
    p = RACE_CACHE / f"{rid}.json"
    if p.exists():
        return 0
    soup = with_retry(
        lambda: client(sleep)._get(f"{nk.DB}/race/{rid}/", encoding="euc-jp")
    )
    rec = parse_race(soup, rid)
    if not rec["horses"]:
        raise RuntimeError("出走0頭")
    p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    time.sleep(sleep)
    return len(rec["horses"])


def run_pool(items, fn, workers: int, label: str):
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
            except Exception as e:  # noqa: BLE001
                failures.append((str(it), f"{type(e).__name__}: {str(e)[:120]}"))
            done += 1
            if done % 250 == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el else 0
                with _lock:
                    print(f"[{label}] {done}/{total} 失敗{len(failures)} "
                          f"{rate:.1f}件/秒 残り{(total-done)/rate/60 if rate else 0:.1f}分",
                          flush=True)
    return failures


def daterange(a: str, b: str):
    d0 = date.fromisoformat(a)
    d1 = date.fromisoformat(b)
    while d0 <= d1:
        yield d0.strftime("%Y%m%d")
        d0 += timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", required=True)
    ap.add_argument("--to", dest="d_to", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--lists-only", action="store_true")
    args = ap.parse_args()

    LIST_CACHE.mkdir(parents=True, exist_ok=True)
    RACE_CACHE.mkdir(parents=True, exist_ok=True)

    days = list(daterange(args.d_from, args.d_to))
    todo_days = [d for d in days if not (LIST_CACHE / f"{d}.json").exists()]
    print(f"日付 {len(days)} / 一覧未取得 {len(todo_days)}")
    list_fail = []
    if todo_days:
        list_fail = run_pool(todo_days, lambda d: fetch_race_list(d, args.sleep),
                             args.workers, "一覧")

    rids: list[str] = []
    for d in days:
        p = LIST_CACHE / f"{d}.json"
        if p.exists():
            try:
                rids.extend(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    rids = sorted(set(rids))
    print(f"レースID {len(rids)}")
    if args.lists_only:
        return

    todo = [r for r in rids if not (RACE_CACHE / f"{r}.json").exists()]
    print(f"結果未取得 {len(todo)}")
    race_fail = []
    if todo:
        race_fail = run_pool(todo, lambda r: fetch_race(r, args.sleep),
                             args.workers, "結果")

    have = sum(1 for r in rids if (RACE_CACHE / f"{r}.json").exists())
    summary = {
        "日付数": len(days), "レースID数": len(rids), "取得済みレース数": have,
        "一覧失敗数": len(list_fail), "結果失敗数": len(race_fail),
        "一覧失敗": list_fail[:30], "結果失敗": race_fail[:30],
    }
    FAIL_LOG.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
