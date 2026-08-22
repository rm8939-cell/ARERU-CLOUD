#!/usr/bin/env python3
"""予想スコア特徴量の充足率・影響度監査。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
OUT = DATA / 'feature_coverage_report.json'


def _audit_runners():
    from history_index import build_master_history, feature_fill_report
    r = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    h = build_master_history()
    return feature_fill_report(r, h)


def _audit_bonus_activation(dates: list[str], sim_runs: int = 5000):
    """新旧で legacy-gated ボーナスが発火する割合。"""
    from areru_engine import history_detail_bonus, load_weights, score_runner
    from history_index import build_master_history, enrich_runner_history_fields
    from race_sim import margin_bonus_from_row, time_trend_bonus, track_condition_bonus, jockey_bonus

    history = build_master_history()
    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    from areru_engine import parse_date
    weights = load_weights()
    stats = {'legacy': {'n': 0, 'detail': 0, 'margin': 0, 'time': 0, 'track': 0, 'jockey': 0},
             'new': {'n': 0, 'detail': 0, 'margin': 0, 'time': 0, 'track': 0, 'jockey': 0}}

    for d in dates:
        day = runners[parse_date(runners['日付']).dt.strftime('%Y-%m-%d') == d]
        target = pd.Timestamp(d)
        for _, row in day.iterrows():
            venue = str(row.get('場1') or '')
            for legacy in (True, False):
                tag = 'legacy' if legacy else 'new'
                os.environ['ARERU_LEGACY_SCORE'] = '1' if legacy else '0'
                work = row.to_dict() if legacy else enrich_runner_history_fields(row, history, target)
                work = pd.Series(work)
                stats[tag]['n'] += 1
                b, _ = history_detail_bonus(work, history, target)
                if b > 0:
                    stats[tag]['detail'] += 1
                mb, _ = margin_bonus_from_row(work)
                if mb > 0:
                    stats[tag]['margin'] += 1
                tb, _ = time_trend_bonus(work, history, work.get('馬名'), target)
                if tb > 0:
                    stats[tag]['time'] += 1
                tr, _ = track_condition_bonus(history, work.get('馬名'), target)
                if tr > 0:
                    stats[tag]['track'] += 1
                jb = jockey_bonus(history, work.get('騎手'), venue, target)
                if abs(jb) >= 0.5:
                    stats[tag]['jockey'] += 1
    for tag in stats:
        n = max(stats[tag]['n'], 1)
        for k in list(stats[tag].keys()):
            if k == 'n':
                continue
            stats[tag][f'{k}_rate'] = round(stats[tag][k] / n * 100, 2)
    return stats


def main():
    results = pd.read_csv(DATA / 'results.csv', encoding='utf-8-sig')
    dates = sorted(results['date'].astype(str).unique())
    train = dates[: max(1, len(dates) * 7 // 10)]
    holdout = dates[len(train):]

    report = {
        'runners_fill': _audit_runners(),
        'bonus_activation': _audit_bonus_activation(dates, sim_runs=5000),
        'train_dates': train,
        'holdout_dates': holdout,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'\n📁 {OUT}')


if __name__ == '__main__':
    main()
