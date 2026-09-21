#!/usr/bin/env python3
"""レース画面の表示時間を実測する（予想ロジックは実行するだけで変更しない）。"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault('ARERU_SKIP_BOOT', '1')
os.environ.setdefault('ARERU_LEGACY_SCORE', '1')
os.environ.setdefault('ARERU_ENABLE_GENERATION', '0')
os.environ.setdefault('ARERU_PERF', '1')

WATCH = (
    'web_app.dates',
    'web_app._fs_sig',
    'web_app._nar_pred_ready',
    'web_app._pred_file_sources',
    'web_app._runners_need_source',
    'web_app.ensure_for_page',
    'web_app._ensure_pred_file_finalized',
    'web_app._read_predictions_for_venue_picker',
    'web_app._read_predictions_for_venue_detail',
    'web_app._main_ban_map',
    'web_app._horse_display_meta_for_records',
    'web_app.prep',
    'web_app.apply_display_ranks',
    'web_app.build_buy_candidates',
    'web_app.build_today_ai_board',
    'web_app.build_areru_pipeline_board',
    'web_app.analysis_data',
    'web_app.resolve_fetch_status',
    'ev_analysis.apply_expected_value',
    'ev_analysis.ensure_predictions_file_finalized',
    'pandas.read_csv',
)


def _install_timer():
    stats = defaultdict(lambda: {'n': 0, 'sec': 0.0})
    orig = {}

    def wrap(mod, name, qual):
        fn = getattr(mod, name)
        if getattr(fn, '_areru_timed', False):
            return

        def inner(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                rec = stats[qual]
                rec['n'] += 1
                rec['sec'] += time.perf_counter() - t0
        inner._areru_timed = True
        orig[qual] = fn
        setattr(mod, name, inner)

    import web_app
    import ev_analysis
    import pandas as pd
    wrap(web_app, 'dates', 'web_app.dates')
    wrap(web_app, '_fs_sig', 'web_app._fs_sig')
    wrap(web_app, '_nar_pred_ready', 'web_app._nar_pred_ready')
    wrap(web_app, '_pred_file_sources', 'web_app._pred_file_sources')
    wrap(web_app, '_runners_need_source', 'web_app._runners_need_source')
    wrap(web_app, 'ensure_for_page', 'web_app.ensure_for_page')
    wrap(web_app, '_ensure_pred_file_finalized', 'web_app._ensure_pred_file_finalized')
    wrap(web_app, '_read_predictions_for_venue_picker', 'web_app._read_predictions_for_venue_picker')
    wrap(web_app, '_read_predictions_for_venue_detail', 'web_app._read_predictions_for_venue_detail')
    wrap(web_app, '_main_ban_map', 'web_app._main_ban_map')
    wrap(web_app, '_horse_display_meta_for_records', 'web_app._horse_display_meta_for_records')
    wrap(web_app, 'prep', 'web_app.prep')
    wrap(web_app, 'apply_display_ranks', 'web_app.apply_display_ranks')
    wrap(web_app, 'build_buy_candidates', 'web_app.build_buy_candidates')
    wrap(web_app, 'build_today_ai_board', 'web_app.build_today_ai_board')
    wrap(web_app, 'build_areru_pipeline_board', 'web_app.build_areru_pipeline_board')
    wrap(web_app, 'analysis_data', 'web_app.analysis_data')
    wrap(web_app, 'resolve_fetch_status', 'web_app.resolve_fetch_status')
    wrap(web_app, 'apply_expected_value', 'web_app.apply_expected_value')
    wrap(ev_analysis, 'apply_expected_value', 'ev_analysis.apply_expected_value')
    wrap(ev_analysis, 'ensure_predictions_file_finalized', 'ev_analysis.ensure_predictions_file_finalized')
    wrap(pd, 'read_csv', 'pandas.read_csv')
    return stats


def run(url: str, label: str, stats) -> dict:
    from web_app import app
    client = app.test_client()
    for k in list(stats):
        stats[k]['n'] = 0
        stats[k]['sec'] = 0.0
    t0 = time.perf_counter()
    resp = client.get(url)
    total = time.perf_counter() - t0
    html = resp.get_data(as_text=True)
    buy = html.count('🔥 BUY') + html.count('class="fx-buy"') + html.count('data-race-judge="buy"')
    marks = {
        'honmei': html.count('◎本命') + html.count('本命'),
        'ui': 'areu-app-v20' in html or 'data-ui="areu-app-v20"' in html,
        'bytes': len(resp.data),
        'status': resp.status_code,
    }
    print(f'\n=== {label} {url} ===')
    print(f'total={total*1000:.0f}ms status={resp.status_code} bytes={len(resp.data)} buy_markers={buy} v20={marks["ui"]}')
    rows = sorted(stats.items(), key=lambda kv: -kv[1]['sec'])
    for name, rec in rows:
        if rec['n'] == 0:
            continue
        print(f'  {name:48s} n={rec["n"]:4d}  {rec["sec"]*1000:8.1f}ms')
    return {'total_ms': round(total * 1000, 1), 'status': resp.status_code, 'bytes': len(resp.data), 'buy': buy}


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else '/?date=2026-08-29&history=1&source=jra&mode=predict'
    tracemalloc.start()
    stats = _install_timer()
    from web_app import app  # noqa: F401  import after wraps
    first = run(url, '1st', stats)
    second = run(url, '2nd', stats)
    third = run(url, '3rd', stats)
    print('\nSUMMARY')
    print(f'  1st {first["total_ms"]}ms')
    print(f'  2nd {second["total_ms"]}ms')
    print(f'  3rd {third["total_ms"]}ms')


if __name__ == '__main__':
    main()
