#!/usr/bin/env python3
"""特徴量アブレーションバックテスト（A〜H）。

同一レース群・同一オッズ・同一 SIM 設定で各特徴量 ON/OFF の ROI を比較する。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
OUT = DATA / 'ablation_report.json'
CACHE = DATA / 'ablation_cache'
LC_CACHE = DATA / 'logic_compare_cache'

# A=旧, B〜G=旧+単特徴, H=全新
MODES: dict[str, dict[str, str]] = {
    'A': {'ARERU_LEGACY_SCORE': '1'},
    'B': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_JOCKEY': '1'},
    'C': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_COURSE': '1'},
    'D': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_TIME': '1', 'ARERU_ABL_ENRICH': '1'},
    'E': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_MARGIN': '1', 'ARERU_ABL_ENRICH': '1'},
    'F': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_TRACK': '1', 'ARERU_ABL_ENRICH': '1'},
    'G': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_WEIGHT': '1'},
    'H': {'ARERU_LEGACY_SCORE': '0'},
}

MODE_LABELS = {
    'A': '旧ロジック',
    'B': '旧+騎手',
    'C': '旧+距離/コース',
    'D': '旧+タイム',
    'E': '旧+着差',
    'F': '旧+馬場',
    'G': '旧+馬体重',
    'H': '全特徴量',
}


def _apply_mode(mode: str) -> None:
    """環境変数をクリアしてモード設定を適用。"""
    for k in list(os.environ.keys()):
        if k.startswith('ARERU_ABL_') or k == 'ARERU_LEGACY_SCORE':
            os.environ.pop(k, None)
    for k, v in MODES[mode].items():
        os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-runs', type=int, default=5000)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--modes', default='A,B,C,D,E,F,G,H', help='カンマ区切り')
    args = ap.parse_args()

    from scripts.logic_compare_backtest import (
        _eligible_dates,
        _evaluate_bets,
        _load_history,
        _load_results,
        _predict_for_date,
        _summarize,
    )
    from replay_predict import available_dates, load_runners

    runners = load_runners()
    dates = _eligible_dates(available_dates(runners))
    history = _load_history()
    results = _load_results()

    print(f'[ablation] 検証日 {len(dates)}: {dates}', flush=True)

    report: dict = {
        '検証日': dates,
        'SIM_RUNS': args.sim_runs,
        'modes': {},
        'vs_A': {},
        'buy_diff_vs_A': {},
    }

    baseline_bets: list[dict] | None = None

    # ベースライン A を常に先に評価（キャッシュ再利用）
    for preload in ('A',):
        if preload not in args.modes.split(',') and preload not in report['modes']:
            _apply_mode(preload)
            bets_a: list[dict] = []
            races_a = 0
            for d in dates:
                if (LC_CACHE / f'predictions_legacy_{d}.csv').exists():
                    pred = pd.read_csv(LC_CACHE / f'predictions_legacy_{d}.csv', encoding='utf-8-sig')
                else:
                    pred, _ = _predict_for_date(d, legacy=True, history=history, sim_runs=args.sim_runs, use_cache=True)
                bets_a.extend(_evaluate_bets(pred, results, d))
                races_a += len(pred)
            sm_a = _summarize(bets_a, races_a)
            sm_a['label'] = MODE_LABELS['A']
            sm_a['env'] = dict(MODES['A'])
            report['modes']['A'] = sm_a
            baseline_bets = bets_a
            report['vs_A']['A'] = 0.0

    for mode in args.modes.split(','):
        mode = mode.strip().upper()
        if mode not in MODES:
            print(f'  skip unknown mode {mode}', flush=True)
            continue
        _apply_mode(mode)
        print(f'[ablation] mode {mode} ({MODE_LABELS[mode]}) ...', flush=True)

        CACHE.mkdir(parents=True, exist_ok=True)
        tag = mode.lower()
        bets: list[dict] = []
        races = 0
        preds_by_date = {}

        for d in dates:
            cache_pred = CACHE / f'pred_{tag}_{d}.csv'
            scores_path = CACHE / f'scores_{tag}_{d}.csv'
            if not args.no_cache and cache_pred.exists():
                pred = pd.read_csv(cache_pred, encoding='utf-8-sig')
                scores = pd.read_csv(scores_path, encoding='utf-8-sig') if scores_path.exists() else None
            elif not args.no_cache and mode == 'A' and (LC_CACHE / f'predictions_legacy_{d}.csv').exists():
                pred = pd.read_csv(LC_CACHE / f'predictions_legacy_{d}.csv', encoding='utf-8-sig')
                sp = LC_CACHE / f'scores_legacy_{d}.csv'
                scores = pd.read_csv(sp, encoding='utf-8-sig') if sp.exists() else None
            elif not args.no_cache and mode == 'H' and (LC_CACHE / f'predictions_new_{d}.csv').exists():
                pred = pd.read_csv(LC_CACHE / f'predictions_new_{d}.csv', encoding='utf-8-sig')
                sp = LC_CACHE / f'scores_new_{d}.csv'
                scores = pd.read_csv(sp, encoding='utf-8-sig') if sp.exists() else None
            else:
                pred, scores = _predict_for_date(
                    d, legacy=(mode != 'H'), history=history,
                    sim_runs=args.sim_runs, use_cache=False, respect_env=True,
                )
                pred.to_csv(cache_pred, index=False, encoding='utf-8-sig')
                if scores is not None:
                    scores.to_csv(CACHE / f'scores_{tag}_{d}.csv', index=False, encoding='utf-8-sig')

            preds_by_date[d] = pred
            ob = _evaluate_bets(pred, results, d, scores=None)
            bets.extend(ob)
            races += len(pred)

        sm = _summarize(bets, races)
        sm['label'] = MODE_LABELS[mode]
        sm['env'] = dict(MODES[mode])
        report['modes'][mode] = sm

        if mode == 'A':
            baseline_bets = bets
            report['vs_A'][mode] = 0.0
        elif 'A' in report['modes']:
            roi_a = report['modes']['A']['ROI']
            report['vs_A'][mode] = round(sm['ROI'] - roi_a, 2)
            # BUY 差分
            a_keys = {(b['date'], b['race_id']) for b in baseline_bets or []}
            m_keys = {(b['date'], b['race_id']) for b in bets}
            only_a = a_keys - m_keys
            only_m = m_keys - a_keys
            report['buy_diff_vs_A'][mode] = {
                'AのみBUY': len(only_a),
                '当モードのみBUY': len(only_m),
                'Aのみ的中': sum(1 for b in (baseline_bets or []) if (b['date'], b['race_id']) in only_a and b['的中']),
                '当モードのみ的中': sum(1 for b in bets if (b['date'], b['race_id']) in only_m and b['的中']),
            }
        else:
            report['vs_A'][mode] = None

    # 原因切り分けメタ
    fr = DATA / 'feature_coverage_report.json'
    if fr.exists():
        report['feature_coverage'] = json.loads(fr.read_text(encoding='utf-8'))

    # 最良/最悪特徴量
    deltas = [(m, v) for m, v in report['vs_A'].items() if m != 'A' and v is not None]
    if deltas:
        best = max(deltas, key=lambda x: x[1])
        worst = min(deltas, key=lambda x: x[1])
        report['best_feature_mode'] = {'mode': best[0], 'label': MODE_LABELS[best[0]], 'ROI_delta_vs_A': best[1]}
        report['worst_feature_mode'] = {'mode': worst[0], 'label': MODE_LABELS[worst[0]], 'ROI_delta_vs_A': worst[1]}

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'modes': {k: {'ROI': v['ROI'], 'BUY': v['BUY件数'], '的中率': v['的中率'], 'vs_A': report['vs_A'].get(k)} for k, v in report['modes'].items()},
        'best': report.get('best_feature_mode'),
        'worst': report.get('worst_feature_mode'),
    }, ensure_ascii=False, indent=2))
    print(f'\n📁 {OUT}')


if __name__ == '__main__':
    main()
