#!/usr/bin/env python3
"""初回レース画面の処理内訳（予想ロジックは実行するだけ）。"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault('ARERU_SKIP_BOOT', '1')
os.environ.setdefault('ARERU_LEGACY_SCORE', '1')
os.environ.setdefault('ARERU_ENABLE_GENERATION', '0')

URL = '/?date=2026-08-29&history=1&source=jra&mode=predict'


def _wrap(mod, name, stats, qual=None):
    qual = qual or f'{mod.__name__}.{name}'
    fn = getattr(mod, name)

    def inner(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            rec = stats[qual]
            rec['n'] += 1
            rec['sec'] += time.perf_counter() - t0
    setattr(mod, name, inner)
    return fn


def main():
    stats = defaultdict(lambda: {'n': 0, 'sec': 0.0})
    import web_app
    import ev_analysis
    import pandas as pd
    from flask import render_template as _rt

    for mod, name in (
        (web_app, 'dates'),
        (web_app, 'dates_with_results'),
        (web_app, '_pred_file_sources'),
        (web_app, '_fs_sig'),
        (web_app, '_nar_pred_ready'),
        (web_app, '_latest_ready_pred_date'),
        (web_app, 'ensure_for_page'),
        (web_app, '_ensure_pred_file_finalized'),
        (web_app, '_read_predictions_for_venue_picker'),
        (web_app, '_read_predictions_for_venue_detail'),
        (web_app, '_main_ban_map'),
        (web_app, '_horse_display_meta_for_records'),
        (web_app, 'prep'),
        (web_app, '_prep_rank_cached'),
        (web_app, 'apply_display_ranks'),
        (web_app, 'build_buy_candidates'),
        (web_app, 'build_today_ai_board'),
        (web_app, 'build_areru_pipeline_board'),
        (web_app, 'analysis_data'),
        (web_app, 'apply_expected_value'),
        (ev_analysis, 'apply_expected_value'),
        (ev_analysis, 'ensure_predictions_file_finalized'),
        (ev_analysis, 'predictions_are_finalized'),
        (pd, 'read_csv'),
    ):
        _wrap(mod, name, stats)

    orig_render = web_app.render_template

    def timed_render(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_render(*a, **kw)
        finally:
            rec = stats['jinja.render_template']
            rec['n'] += 1
            rec['sec'] += time.perf_counter() - t0
    web_app.render_template = timed_render

    from web_app import app, _clear_runtime_caches
    client = app.test_client()
    _clear_runtime_caches()
    web_app._PRED_SOURCE_CACHE.clear()

    t0 = time.perf_counter()
    resp = client.get(URL)
    total = time.perf_counter() - t0
    html = resp.get_data(as_text=True)
    print(f'COLD MISS wall={total*1000:.1f}ms status={resp.status_code} bytes={len(resp.data)}')
    print(f'header={resp.headers.get("X-ARERU-Perf")}')
    print(f'buy={html.count("data-race-judge=") } ui_v20={"areu-app-v20" in html}')
    print()
    print(f'{"処理":48s} {"n":>4} {"ms":>10}  初回のみ?  キャッシュ可?')
    catalog = [
        ('DBアクセス', '(なし)', 0, '—', '—'),
        ('JRA/NAR外部取得', '(なし・CSV配信)', 0, '—', '—'),
        ('シミュレーション', '(なし・CSV焼き込み済み)', 0, '—', '—'),
    ]
    for name, rec in sorted(stats.items(), key=lambda kv: -kv[1]['sec']):
        if rec['n'] == 0:
            continue
        print(f'{name:48s} {rec["n"]:4d} {rec["sec"]*1000:10.1f}')
    print(f'{"TOTAL":48s} {"":4} {total*1000:10.1f}')

    # hit
    for k in list(stats):
        stats[k]['n'] = 0
        stats[k]['sec'] = 0.0
    t0 = time.perf_counter()
    resp2 = client.get(URL)
    total2 = time.perf_counter() - t0
    print(f'\nCACHE HIT wall={total2*1000:.1f}ms header={resp2.headers.get("X-ARERU-Perf")}')
    for name, rec in sorted(stats.items(), key=lambda kv: -kv[1]['sec']):
        if rec['n'] == 0:
            continue
        print(f'  {name:46s} n={rec["n"]:4d}  {rec["sec"]*1000:8.1f}ms')


if __name__ == '__main__':
    main()
