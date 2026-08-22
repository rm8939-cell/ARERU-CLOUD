#!/usr/bin/env python3
"""runners.csv に タイム/着差/馬場 をスロットマッチで一括補完。"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pandas as pd
from history_index import backfill_runners_past_detail, build_master_history, feature_fill_report

RUNNERS = BASE / 'data' / 'runners.csv'


def main():
    if not RUNNERS.exists():
        raise SystemExit(f'missing {RUNNERS}')
    runners = pd.read_csv(RUNNERS, encoding='utf-8-sig', low_memory=False)
    history = build_master_history()
    before = feature_fill_report(runners, history)
    print('[before]', before.get('columns', {}).get('タイム1'), before.get('slot_matched_detail_any'))
    filled = backfill_runners_past_detail(runners, history)
    after = feature_fill_report(filled, history)
    print('[after]', after.get('columns', {}).get('タイム1'), after.get('slot_matched_detail_any'))
    filled.to_csv(RUNNERS, index=False, encoding='utf-8-sig')
    print(f'✅ wrote {RUNNERS} rows={len(filled)}')


if __name__ == '__main__':
    main()
