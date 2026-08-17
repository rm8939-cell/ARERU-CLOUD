"""既存の取得関数だけで戦績・血統キャッシュを埋める補完スクリプト。

予想・BUY・EV は触らない。runners.csv / predictions_*.csv も書き換えない。
書き込むのは data/cache/ 配下（horse_ids.json / horse_results / horse_pedigree）のみ。

使い方:
    python3 scripts/backfill_horse_data.py --date 2026-08-01 --limit 40
    python3 scripts/backfill_horse_data.py --date 2026-08-01 --no-pedigree
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pandas as pd  # noqa: E402

import netkeiba_client as nk  # noqa: E402


def pick_horses(date_str: str, source: str = '') -> tuple[list[str], list[str]]:
    """予想CSVのピックカードから 表示対象の馬名 と race_id を取り出す。"""
    path = BASE / 'data' / 'predictions_by_date' / f'predictions_{date_str}.csv'
    if not path.exists():
        raise SystemExit(f'予想CSVがありません: {path}')
    df = pd.read_csv(path).fillna('')
    if source:
        if 'source' not in df.columns:
            raise SystemExit('source列がありません')
        df = df[df['source'].astype(str).str.lower() == source.lower()]
    names: list[str] = []
    race_ids: list[str] = []
    for _, row in df.iterrows():
        rid = str(row.get('race_id') or '').strip()
        if rid and rid not in race_ids:
            race_ids.append(rid)
        raw = row.get('ピックカード')
        try:
            cards = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith('[') else []
        except Exception:
            cards = []
        for card in cards:
            if not isinstance(card, dict):
                continue
            name = str(card.get('馬名') or '').strip()
            if name and name not in names:
                names.append(name)
    return names, race_ids


def build_id_map(client: nk.NetkeibaClient, race_ids: list[str]) -> dict:
    """既存 fetch_entries で 馬名→horse_id を集める（remember_horse_ids が保存）。"""
    for i, rid in enumerate(race_ids, 1):
        try:
            client.fetch_entries(rid)
        except Exception as e:
            print(f'  entries skip {rid}: {type(e).__name__}: {e}', flush=True)
        if i % 6 == 0:
            print(f'  entries {i}/{len(race_ids)}', flush=True)
    return nk.load_horse_ids()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)
    ap.add_argument('--source', default='')
    ap.add_argument('--limit', type=int, default=0, help='処理する馬数の上限（0=全件）')
    ap.add_argument('--no-history', action='store_true')
    ap.add_argument('--no-pedigree', action='store_true')
    a = ap.parse_args()

    names, race_ids = pick_horses(a.date, a.source)
    print(f'対象: 馬 {len(names)}頭 / レース {len(race_ids)}件', flush=True)

    client = nk.NetkeibaClient()
    id_map = build_id_map(client, race_ids)
    print(f'馬名→horse_id: {len(id_map)}件 -> {nk.HORSE_ID_MAP}', flush=True)

    targets = [(n, id_map.get(n, '')) for n in names]
    missing = [n for n, hid in targets if not hid]
    targets = [(n, hid) for n, hid in targets if hid]
    if a.limit:
        targets = targets[:a.limit]
    print(f'ID解決: {len(targets)}頭 / 未解決 {len(missing)}頭', flush=True)

    hist_ok = ped_ok = 0
    for i, (name, hid) in enumerate(targets, 1):
        if not a.no_history:
            try:
                rows = client.fetch_horse_history(hid, use_cache=False)
                if rows and nk.history_row_has_extras(rows[0]):
                    hist_ok += 1
            except Exception as e:
                print(f'  history skip {name}: {type(e).__name__}: {e}', flush=True)
        if not a.no_pedigree:
            try:
                ped = client.fetch_horse_pedigree(hid)
                if ped.get('父') or ped.get('母父'):
                    ped_ok += 1
            except Exception as e:
                print(f'  pedigree skip {name}: {type(e).__name__}: {e}', flush=True)
        if i % 10 == 0:
            print(f'  {i}/{len(targets)} 実測付き戦績={hist_ok} 血統={ped_ok}', flush=True)

    print(f'完了: 戦績(実測あり)={hist_ok} 血統={ped_ok} / 対象={len(targets)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
