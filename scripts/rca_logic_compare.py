#!/usr/bin/env python3
"""旧 / 新 / 改善案A/B/C を同一条件で比較する RCA バックテスト。

OLD : ARERU_LEGACY_SCORE=1（ガウスSIM・追加特徴OFF）
NEW : ARERU_LEGACY_SCORE=0（現行新=全特徴）
A   : 段階SIM + 距離/コースのみ
B   : 段階SIM + 実データ特徴（タイム/着差/馬場）だがスコア詳細加点なし・騎手/馬体重なし
C   : B + スコア詳細加点（タイム+着差が揃うときのみ・低重み）

採用条件: train と holdout の双方で OLD より ROI 優位、かつ全期間 BUY>=100・ROI差>=5pp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
OUT = DATA / 'rca_logic_compare_report.json'
CACHE = DATA / 'rca_logic_cache'
STAKE = 100
MIN_BUY = 100

LOGICS = {
    'OLD': {'ARERU_LEGACY_SCORE': '1'},
    'NEW': {'ARERU_LEGACY_SCORE': '0'},
    'A': {'ARERU_LEGACY_SCORE': '0', 'ARERU_LOGIC_PRESET': 'A'},
    'B': {'ARERU_LEGACY_SCORE': '0', 'ARERU_LOGIC_PRESET': 'B'},
    'C': {'ARERU_LEGACY_SCORE': '0', 'ARERU_LOGIC_PRESET': 'C'},
    'D': {'ARERU_LEGACY_SCORE': '0', 'ARERU_LOGIC_PRESET': 'D'},
}

LOGIC_LABELS = {
    'OLD': '旧ロジック',
    'NEW': '現行新ロジック',
    'A': '改善案A:コースのみ',
    'B': '改善案B:SIM実データ(スコア加点なし)',
    'C': '改善案C:実データ+厳密detail',
    'D': '改善案D:差し/内枠/12-20妙味/馬体重増の較正',
}


def _clear_env():
    for k in list(os.environ.keys()):
        if k.startswith('ARERU_ABL_') or k in ('ARERU_LEGACY_SCORE', 'ARERU_LOGIC_PRESET'):
            os.environ.pop(k, None)


def _apply(logic: str):
    _clear_env()
    for k, v in LOGICS[logic].items():
        os.environ[k] = v


def _summarize(bets: list[dict], label: str) -> dict:
    n = len(bets)
    empty = {
        'label': label, 'BUY件数': 0, '的中件数': 0, '的中率': 0.0,
        '本命勝率': 0.0, '3着内率': 0.0, '投資額': 0, '払戻': 0.0,
        'ROI': 0.0, '回収率': 0.0, '平均オッズ': None, '平均期待値': None,
        '1レース期待値': None, '最大連敗': 0,
    }
    if n == 0:
        return empty
    hits = sum(1 for b in bets if b.get('的中'))
    top3 = sum(1 for b in bets if b.get('3着内'))
    invest = n * STAKE
    ret = float(sum(b.get('払戻') or 0 for b in bets))
    odds = [b['オッズ'] for b in bets if b.get('オッズ') is not None]
    evs = [b['期待値'] for b in bets if b.get('期待値') is not None]
    ordered = sorted(bets, key=lambda b: (str(b.get('date') or ''), str(b.get('race_id') or '')))
    max_lose = cur = 0
    for b in ordered:
        if b.get('的中'):
            cur = 0
        else:
            cur += 1
            max_lose = max(max_lose, cur)
    roi_pct = ret / invest * 100 if invest else 0.0
    hit_rate = hits / n * 100
    avg_ev = round(sum(evs) / len(evs), 2) if evs else None
    return {
        'label': label,
        'BUY件数': n,
        '的中件数': hits,
        '的中率': round(hit_rate, 2),
        '本命勝率': round(hit_rate, 2),
        '3着内率': round(top3 / n * 100, 2),
        '投資額': invest,
        '払戻': round(ret, 1),
        'ROI': round(roi_pct - 100, 2),
        '回収率': round(roi_pct, 2),
        '平均オッズ': round(sum(odds) / len(odds), 2) if odds else None,
        '平均期待値': avg_ev,
        '1レース期待値': avg_ev,
        '最大連敗': max_lose,
    }


def _adoption(candidate: dict, baseline: dict, train_c: dict, train_b: dict, hold_c: dict, hold_b: dict) -> dict:
    reasons = []
    ok = True
    if candidate['BUY件数'] < MIN_BUY or baseline['BUY件数'] < MIN_BUY:
        ok = False
        reasons.append(f"BUY件数不足 cand={candidate['BUY件数']} base={baseline['BUY件数']}")
    if hold_c['ROI'] <= hold_b['ROI']:
        ok = False
        reasons.append(f"holdout ROI 非優位 cand={hold_c['ROI']} base={hold_b['ROI']}")
    if train_c['ROI'] <= train_b['ROI']:
        ok = False
        reasons.append(f"train ROI 非優位 cand={train_c['ROI']} base={train_b['ROI']}")
    delta = round(candidate['ROI'] - baseline['ROI'], 2)
    if delta < 5.0:
        ok = False
        reasons.append(f"全期間ROI差 {delta}pp < 5")
    if ok:
        reasons.append('train/holdout 双方で旧を上回る')
    return {
        'adopt': ok,
        'deploy_allowed': False,  # 明示デプロイは別ゲート
        'ROI差_vs_OLD': delta,
        'reasons': reasons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-runs', type=int, default=3000)
    ap.add_argument('--logics', default='OLD,NEW,A,B,C')
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    from scripts.logic_compare_backtest import (
        _eligible_dates, _evaluate_bets, _load_history, _load_results, _predict_for_date,
    )
    from scripts.stable_holdout_compare import _race_rows
    from replay_predict import available_dates, load_runners

    CACHE.mkdir(parents=True, exist_ok=True)
    runners = load_runners()
    dates = _eligible_dates(available_dates(runners))
    # train/holdout: last 30% holdout
    split = max(1, int(len(dates) * 0.70))
    train_dates = dates[:split]
    holdout_dates = dates[split:]
    history = _load_history()
    results = _load_results()
    logics = [x.strip() for x in args.logics.split(',') if x.strip() in LOGICS]

    print(f'[rca-compare] days={len(dates)} train={len(train_dates)} holdout={len(holdout_dates)} logics={logics}', flush=True)

    rows_by_logic: dict[str, list] = {k: [] for k in logics}
    for logic in logics:
        _apply(logic)
        os.environ['ARERU_SIM_RUNS'] = str(args.sim_runs)
        for d in dates:
            period = 'holdout' if d in holdout_dates else 'train'
            cache_path = CACHE / f'pred_{logic}_{d}.csv'
            print(f'[{logic}] {d} ({period}) ...', flush=True)
            if cache_path.exists() and not args.no_cache:
                pred = pd.read_csv(cache_path, encoding='utf-8-sig')
            else:
                # respect_env=True so PRESET/LEGACY from env stick
                pred, _ = _predict_for_date(
                    d, legacy=(logic == 'OLD'), history=history,
                    sim_runs=args.sim_runs, use_cache=False, respect_env=True,
                )
                pred.to_csv(cache_path, index=False, encoding='utf-8-sig')
            day_rows = _race_rows(pred, results, d)
            for r in day_rows:
                r['period'] = period
                r['logic'] = logic
            rows_by_logic[logic].extend(day_rows)

    report: dict = {
        '検証日': dates,
        'train_dates': train_dates,
        'holdout_dates': holdout_dates,
        'SIM_RUNS': args.sim_runs,
        'logics': {k: LOGIC_LABELS[k] for k in logics},
        'strict_buy': {},
        'all_honmei': {},
        'adoption': {},
    }

    for logic in logics:
        all_rows = rows_by_logic[logic]
        buys = [r for r in all_rows if r.get('strict_buy')]
        for scope, subset in (
            ('full', buys),
            ('train', [r for r in buys if r.get('period') == 'train']),
            ('holdout', [r for r in buys if r.get('period') == 'holdout']),
        ):
            report['strict_buy'].setdefault(logic, {})[scope] = _summarize(subset, f'{logic}/{scope}')
        for scope, subset in (
            ('full', all_rows),
            ('train', [r for r in all_rows if r.get('period') == 'train']),
            ('holdout', [r for r in all_rows if r.get('period') == 'holdout']),
        ):
            report['all_honmei'].setdefault(logic, {})[scope] = _summarize(subset, f'{logic}/honmei/{scope}')

    base = report['strict_buy']['OLD']
    best_name = 'OLD'
    best_roi = base['full']['ROI']
    for logic in logics:
        if logic == 'OLD':
            continue
        cand = report['strict_buy'][logic]
        gate = _adoption(cand['full'], base['full'], cand['train'], base['train'], cand['holdout'], base['holdout'])
        report['adoption'][logic] = gate
        if gate['adopt'] and cand['full']['ROI'] > best_roi:
            best_roi = cand['full']['ROI']
            best_name = logic

    # also allow OLD to remain best
    report['best_candidate'] = {
        'logic': best_name,
        'label': LOGIC_LABELS[best_name],
        'adopt_for_production': best_name != 'OLD' and report['adoption'].get(best_name, {}).get('adopt', False),
        'note': '旧以下なら本番採用しない',
    }

    # metric table
    keys = ('BUY件数', '的中率', '本命勝率', '回収率', 'ROI', '平均オッズ', '平均期待値', '1レース期待値', '最大連敗')
    table = {}
    for logic in logics:
        table[logic] = {
            'label': LOGIC_LABELS[logic],
            'full': {k: report['strict_buy'][logic]['full'].get(k) for k in keys},
            'train': {k: report['strict_buy'][logic]['train'].get(k) for k in keys},
            'holdout': {k: report['strict_buy'][logic]['holdout'].get(k) for k in keys},
            'ROI差_vs_OLD_full': round(report['strict_buy'][logic]['full']['ROI'] - base['full']['ROI'], 2),
            'ROI差_vs_OLD_holdout': round(report['strict_buy'][logic]['holdout']['ROI'] - base['holdout']['ROI'], 2),
        }
    report['comparison_table'] = table

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'comparison_table': table, 'best': report['best_candidate'], 'adoption': report['adoption']}, ensure_ascii=False, indent=2))
    print(f'📁 {OUT}')


if __name__ == '__main__':
    main()
