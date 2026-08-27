#!/usr/bin/env python3
"""差し×内枠 / 12-20×内枠 / 差し×5-8×中枠 を単体で train/holdout 検証する。

BUY除外フィルタではない。SIMプロファイル較正のみ。
本番は旧ロジック固定。全期間ROIだけでは採用しない。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
OUT_JSON = DATA / 'three_calib_report.json'
OUT_CSV = DATA / 'three_calib_table.csv'
OUT_VERDICT = DATA / 'three_calib_verdict.json'

from scripts.feature_search_backtest import (  # noqa: E402
    LOGIC_LABELS, MIN_BUY, STAKE, _adoption, _bootstrap_delta, _collect_logic,
    _pays, _summarize,
)

CANDIDATES = ('SASHI_INNER', 'ODDS_INNER', 'SASHI_SWEET')
KEYS = ('BUY件数', '的中率', '平均オッズ', 'ROI', '回収率', '的中件数', '払戻', '投資額')


def _scopes(all_rows: list[dict]) -> dict:
    buys = [r for r in all_rows if r.get('strict_buy')]
    return {
        'full': buys,
        'train': [r for r in buys if r.get('period') == 'train'],
        'holdout': [r for r in buys if r.get('period') == 'holdout'],
    }


def _pack(sm: dict) -> dict:
    return {k: sm.get(k) for k in KEYS}


def _judge(name: str, block: dict, newb: dict, oldb: dict, boot_vs_new: dict, boot_vs_old: dict) -> dict:
    """holdout点推定と再現性で 採用候補/保留/不採用。本番変更は別ゲート。"""
    reasons = []
    tr = block['train']['ROI']
    ho = block['holdout']['ROI']
    fu = block['full']['ROI']
    n_full = block['full']['BUY件数']
    n_new = newb['full']['BUY件数']
    d_tr_new = round(tr - newb['train']['ROI'], 2)
    d_ho_new = round(ho - newb['holdout']['ROI'], 2)
    d_fu_new = round(fu - newb['full']['ROI'], 2)
    d_tr_old = round(tr - oldb['train']['ROI'], 2)
    d_ho_old = round(ho - oldb['holdout']['ROI'], 2)
    d_fu_old = round(fu - oldb['full']['ROI'], 2)
    volume_ratio = (n_full / n_new) if n_new else 0.0
    ci_new = (boot_vs_new or {}).get('ci90') or [None, None]
    ci_old = (boot_vs_old or {}).get('ci90') or [None, None]

    holdout_vs_new = ho > newb['holdout']['ROI']
    train_vs_new = tr > newb['train']['ROI']
    holdout_vs_old = ho > oldb['holdout']['ROI']
    train_vs_old = tr > oldb['train']['ROI']
    volume_ok = n_full >= MIN_BUY and volume_ratio >= 0.80
    ci_new_clear = ci_new[0] is not None and ci_new[0] > 0
    ci_old_clear = ci_old[0] is not None and ci_old[0] > 0

    if n_full < MIN_BUY:
        reasons.append(f'BUY件数不足 {n_full}')
    if not volume_ok:
        reasons.append(f'BUY件数がNEWの80%未満 ratio={volume_ratio:.2f}（件数削減の見かけ改善を疑う）')
    if not train_vs_new:
        reasons.append(f'train が NEW 以下 {tr} vs {newb["train"]["ROI"]}')
    if not holdout_vs_new:
        reasons.append(f'holdout が NEW 以下 {ho} vs {newb["holdout"]["ROI"]}')
    if not train_vs_old:
        reasons.append(f'train が OLD 以下 {tr} vs {oldb["train"]["ROI"]}')
    if not holdout_vs_old:
        reasons.append(f'holdout が OLD 以下 {ho} vs {oldb["holdout"]["ROI"]}')
    if not ci_new_clear:
        reasons.append(f'vs NEW holdout 90%CI が0を跨ぐ {ci_new}')
    if not ci_old_clear:
        reasons.append(f'vs OLD holdout 90%CI が0を跨ぐ {ci_old}')
    if fu >= 0:
        reasons.append('全期間ROIがプラス')
    else:
        reasons.append(f'全期間ROIはマイナス {fu}%')

    if not holdout_vs_new or not train_vs_new or not volume_ok:
        status = '不採用'
        holdout_repro = 'なし'
    elif not ci_new_clear:
        status = '保留'
        holdout_repro = '点推定のみ（区間未確認）'
    else:
        status = '採用候補'
        holdout_repro = '点推定+90%CIでNEW超え'
        reasons.append('train/holdout とも NEW を上回り holdout差の区間も正')

    production_ok = (
        status == '採用候補'
        and train_vs_old and holdout_vs_old and ci_old_clear
        and fu > 0
    )
    return {
        '判定': status,
        'holdout再現性': holdout_repro,
        '本番変更してよいか': production_ok,
        'vs_NEW': {'train_pp': d_tr_new, 'holdout_pp': d_ho_new, 'full_pp': d_fu_new,
                   'holdout_ci90': ci_new, 'p_better': (boot_vs_new or {}).get('p_new_better')},
        'vs_OLD': {'train_pp': d_tr_old, 'holdout_pp': d_ho_old, 'full_pp': d_fu_old,
                   'holdout_ci90': ci_old, 'p_better': (boot_vs_old or {}).get('p_new_better')},
        'BUY件数比_vs_NEW': round(volume_ratio, 3),
        'reasons': reasons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-runs', type=int, default=2500)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    from scripts.logic_compare_backtest import _eligible_dates
    from replay_predict import available_dates, load_runners

    runners = load_runners()
    dates = _eligible_dates(available_dates(runners))
    split = max(1, int(len(dates) * 0.70))
    train_dates, holdout_dates = dates[:split], dates[split:]
    logics = ['OLD', 'NEW', *CANDIDATES]
    print(
        f'[three-calib] days={len(dates)} train={len(train_dates)} '
        f'holdout={len(holdout_dates)} logics={logics} workers={args.workers}',
        flush=True,
    )

    rows_by = {}
    for logic in logics:
        rows_by[logic] = _collect_logic(
            logic, dates, holdout_dates,
            sim_runs=args.sim_runs, no_cache=args.no_cache, workers=args.workers,
        )

    report = {
        '検証設計': {
            '開催日数': len(dates),
            'train': train_dates,
            'holdout': holdout_dates,
            'SIM_RUNS': args.sim_runs,
            '比較': 'BUYのみ・確定オッズ・1点100円',
            'BUY閾値': {'BUY_EV_FLOOR': 108, 'BUY_CONF_FLOOR': 58},
            '実装': 'SIMプロファイル較正のみ。BUY除外フィルタではない',
            '本番': 'OLD (ARERU_LEGACY_SCORE=1) 固定',
            '採用': 'trainとholdoutの双方でNEWを超え、holdout差の90%CIが正。全期間ROIだけでは判断しない',
            '本番変更': '上記に加え OLD も区間で超え、かつ全期間ROIがプラスのときのみ',
        },
        'strict_buy': {},
        'judgement': {},
    }
    for logic, rows in rows_by.items():
        sc = _scopes(rows)
        report['strict_buy'][logic] = {k: _summarize(v, f'{logic}/{k}') for k, v in sc.items()}

    oldb = report['strict_buy']['OLD']
    newb = report['strict_buy']['NEW']
    old_sc = _scopes(rows_by['OLD'])
    new_sc = _scopes(rows_by['NEW'])

    table_rows = []
    for logic in logics:
        block = report['strict_buy'][logic]
        row = {
            'logic': logic,
            'label': LOGIC_LABELS.get(logic, logic),
            'full': _pack(block['full']),
            'train': _pack(block['train']),
            'holdout': _pack(block['holdout']),
            'ROI差_vs_OLD_full': round(block['full']['ROI'] - oldb['full']['ROI'], 2),
            'ROI差_vs_OLD_train': round(block['train']['ROI'] - oldb['train']['ROI'], 2),
            'ROI差_vs_OLD_holdout': round(block['holdout']['ROI'] - oldb['holdout']['ROI'], 2),
            'ROI差_vs_NEW_full': round(block['full']['ROI'] - newb['full']['ROI'], 2),
            'ROI差_vs_NEW_train': round(block['train']['ROI'] - newb['train']['ROI'], 2),
            'ROI差_vs_NEW_holdout': round(block['holdout']['ROI'] - newb['holdout']['ROI'], 2),
        }
        if logic in CANDIDATES:
            boot_new = _bootstrap_delta(_pays(_scopes(rows_by[logic])['holdout']), _pays(new_sc['holdout']))
            boot_old = _bootstrap_delta(_pays(_scopes(rows_by[logic])['holdout']), _pays(old_sc['holdout']))
            # 既存 _adoption は OLD 基準の統計ゲート（本番用）
            row['adoption_vs_OLD'] = _adoption(
                block['full'], oldb['full'], block['train'], oldb['train'],
                block['holdout'], oldb['holdout'], boot_old,
            )
            row['judgement'] = _judge(logic, block, newb, oldb, boot_new, boot_old)
            report['judgement'][logic] = row['judgement']
        table_rows.append(row)

    adopt = [r['logic'] for r in table_rows if r.get('judgement', {}).get('判定') == '採用候補']
    hold = [r['logic'] for r in table_rows if r.get('judgement', {}).get('判定') == '保留']
    reject = [r['logic'] for r in table_rows if r.get('judgement', {}).get('判定') == '不採用']
    prod_ok = any(r.get('judgement', {}).get('本番変更してよいか') for r in table_rows)

    verdict = {
        '採用候補': adopt,
        '保留': hold,
        '不採用': reject,
        '本番ロジックを変更してよいか': False if not prod_ok else True,
        '本番': 'OLD (ARERU_LEGACY_SCORE=1)',
        'note': (
            'NEW自体がholdoutで統計確認できていない。'
            '候補が点推定でNEWを超えても、OLD超えの区間確認とプラスROIがなければ本番は変えない。'
        ),
    }
    report['comparison_table'] = {r['logic']: r for r in table_rows}
    report['verdict'] = verdict

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding='utf-8')

    with OUT_CSV.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'logic', 'scope', 'BUY件数', '的中率', '平均オッズ', 'ROI',
            'vs_OLD_pp', 'vs_NEW_pp', '判定', 'holdout再現性',
        ])
        for r in table_rows:
            j = r.get('judgement') or {}
            for scope, vs_old_k, vs_new_k in (
                ('full', 'ROI差_vs_OLD_full', 'ROI差_vs_NEW_full'),
                ('train', 'ROI差_vs_OLD_train', 'ROI差_vs_NEW_train'),
                ('holdout', 'ROI差_vs_OLD_holdout', 'ROI差_vs_NEW_holdout'),
            ):
                b = r[scope]
                w.writerow([
                    r['logic'], scope, b.get('BUY件数'), b.get('的中率'), b.get('平均オッズ'),
                    b.get('ROI'), r[vs_old_k], r[vs_new_k],
                    j.get('判定', ''), j.get('holdout再現性', ''),
                ])

    print(json.dumps({
        'verdict': verdict,
        'table': {
            k: {
                'full': v['full'], 'train': v['train'], 'holdout': v['holdout'],
                'vs_OLD_holdout': v['ROI差_vs_OLD_holdout'],
                'vs_NEW_holdout': v['ROI差_vs_NEW_holdout'],
                '判定': (v.get('judgement') or {}).get('判定'),
            }
            for k, v in report['comparison_table'].items()
        },
    }, ensure_ascii=False, indent=2))
    print(f'📁 {OUT_JSON}')
    print(f'📁 {OUT_CSV}')


if __name__ == '__main__':
    main()
