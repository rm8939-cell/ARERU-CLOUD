#!/usr/bin/env python3
"""未使用特徴を旧ガウスSIMへ追加 → BUY ROI を train/holdout で旧比較する。

比較条件（固定）:
  - 同一開催日・同一レース・同一確定オッズ
  - BUY のみ（投資判定が「買い」）
  - BUY_EV_FLOOR=108 / BUY_CONF_FLOOR=58 / tanh表示 は変更しない
  - train = 日付の先頭 70%、holdout = 残り
  - 閾値グリッドサーチ禁止

採用: train と holdout の双方で OLD より ROI が高く、
      holdout のブートストラップ 90% 区間が 0 を跨がない場合のみ。
それ以外は「改善していない」として本番不採用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
OUT = DATA / 'feature_search_report.json'
TABLE = DATA / 'feature_search_table.json'
CACHE = DATA / 'rca_logic_cache'
EXTRA_CACHE = DATA / 'rca_extra_cache'
STAKE = 100
MIN_BUY = 100

# 旧は完全固定。X はガウスSIM維持 + 未使用特徴のみ。
LOGICS = {
    'OLD': {'ARERU_LEGACY_SCORE': '1'},
    'NEW': {'ARERU_LEGACY_SCORE': '0'},
    'X': {'ARERU_LEGACY_SCORE': '1', 'ARERU_LOGIC_PRESET': 'X'},
    'BURDEN': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_BURDEN': '1'},
    'GATE': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_SGATE': '1'},
    'WEIGHT': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_SWEIGHT': '1'},
    'JOCKEY': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_SJOCKEY': '1'},
    'LAYOFF': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_SLAYOFF': '1'},
    'STYLE': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_SSTYLE': '1'},
    'FIELD': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_SFIELD': '1'},
    'D': {'ARERU_LEGACY_SCORE': '0', 'ARERU_LOGIC_PRESET': 'D'},
    'SASHI_INNER': {'ARERU_LEGACY_SCORE': '0', 'ARERU_CALIB_SASHI_INNER': '1'},
    'ODDS_INNER': {'ARERU_LEGACY_SCORE': '0', 'ARERU_CALIB_ODDS_INNER': '1'},
    'SASHI_SWEET': {'ARERU_LEGACY_SCORE': '0', 'ARERU_CALIB_SASHI_SWEET': '1'},
}

LOGIC_LABELS = {
    'OLD': '本番旧（ガウスSIM・追加特徴OFF）',
    'NEW': '現行新（段階SIM+詳細特徴）',
    'X': '旧+未使用特徴すべて（斤量/枠/馬体重/騎手/休み/脚質/頭数）',
    'BURDEN': '旧+斤量',
    'GATE': '旧+枠×馬場',
    'WEIGHT': '旧+馬体重変化',
    'JOCKEY': '旧+騎手複勝',
    'LAYOFF': '旧+休み明け',
    'STYLE': '旧+脚質×頭数',
    'FIELD': '旧+頭数実績',
    'D': '新+holdout確認較正（差しSIM/内枠ダート/12-20妙味/馬体重増）',
    'SASHI_INNER': '新+差し×内枠減点',
    'ODDS_INNER': '新+12-20倍×内枠減点',
    'SASHI_SWEET': '新+差し×5-8倍×中枠加点',
    'XSEL': '旧+trainで優位だった特徴の合成',
}

INDIVIDUALS = ('BURDEN', 'GATE', 'WEIGHT', 'JOCKEY', 'LAYOFF', 'STYLE', 'FIELD')
FEAT_OF = {
    'BURDEN': 'burden',
    'GATE': 'sgate',
    'WEIGHT': 'sweight',
    'JOCKEY': 'sjockey',
    'LAYOFF': 'slayoff',
    'STYLE': 'sstyle',
    'FIELD': 'sfield',
}


def _clear_env():
    for k in list(os.environ.keys()):
        if k.startswith('ARERU_ABL_') or k.startswith('ARERU_CALIB_') or k in (
            'ARERU_LEGACY_SCORE', 'ARERU_LOGIC_PRESET', 'ARERU_XSEL_FEATURES',
        ):
            os.environ.pop(k, None)


def _apply(logic: str, xsel: str = ''):
    _clear_env()
    if logic == 'XSEL':
        os.environ['ARERU_LEGACY_SCORE'] = '1'
        os.environ['ARERU_LOGIC_PRESET'] = 'XSEL'
        os.environ['ARERU_XSEL_FEATURES'] = xsel
        return
    for k, v in LOGICS[logic].items():
        os.environ[k] = v


def _cache_path(logic: str, date: str) -> Path:
    if logic in ('OLD', 'NEW'):
        return CACHE / f'pred_{logic}_{date}.csv'
    EXTRA_CACHE.mkdir(parents=True, exist_ok=True)
    tag = logic.lower()
    return EXTRA_CACHE / f'pred_{tag}_{date}.csv'


def _summarize(bets: list[dict], label: str) -> dict:
    n = len(bets)
    empty = {
        'label': label, 'BUY件数': 0, '的中件数': 0, '的中率': 0.0,
        '投資額': 0, '払戻': 0.0, 'ROI': 0.0, '回収率': 0.0,
        '平均オッズ': None, 'サンプル数': 0,
    }
    if n == 0:
        return empty
    hits = sum(1 for b in bets if b.get('的中'))
    invest = n * STAKE
    ret = float(sum(b.get('払戻') or 0 for b in bets))
    odds = [b['オッズ'] for b in bets if b.get('オッズ') is not None]
    roi_pct = ret / invest * 100 if invest else 0.0
    return {
        'label': label,
        'BUY件数': n,
        'サンプル数': n,
        '的中件数': hits,
        '的中率': round(hits / n * 100, 2),
        '投資額': invest,
        '払戻': round(ret, 1),
        'ROI': round(roi_pct - 100, 2),
        '回収率': round(roi_pct, 2),
        '平均オッズ': round(sum(odds) / len(odds), 2) if odds else None,
    }


def _bootstrap_delta(new_pays: list[float], old_pays: list[float], n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(42)

    def roi(arr):
        if len(arr) == 0:
            return 0.0
        return float(np.mean(arr) / STAKE * 100 - 100)

    new = np.asarray(new_pays, dtype=float)
    old = np.asarray(old_pays, dtype=float)
    if len(new) == 0 or len(old) == 0:
        return {'mean_delta': None, 'p_new_better': None, 'ci90': None}
    deltas = []
    for _ in range(n_boot):
        ns = rng.choice(new, size=len(new), replace=True)
        os_ = rng.choice(old, size=len(old), replace=True)
        deltas.append(roi(ns) - roi(os_))
    d = np.asarray(deltas)
    return {
        'mean_delta': round(float(d.mean()), 2),
        'p_new_better': round(float((d > 0).mean()), 3),
        'ci90': [round(float(np.percentile(d, 5)), 2), round(float(np.percentile(d, 95)), 2)],
    }


def _adoption(cand_full, base_full, cand_tr, base_tr, cand_ho, base_ho, boot_ho) -> dict:
    reasons = []
    ok = True
    if cand_full['BUY件数'] < MIN_BUY or base_full['BUY件数'] < MIN_BUY:
        ok = False
        reasons.append(f"BUY件数不足 cand={cand_full['BUY件数']} base={base_full['BUY件数']}")
    if cand_tr['ROI'] <= base_tr['ROI']:
        ok = False
        reasons.append(f"train ROI 非優位 cand={cand_tr['ROI']} base={base_tr['ROI']}")
    if cand_ho['ROI'] <= base_ho['ROI']:
        ok = False
        reasons.append(f"holdout ROI 非優位 cand={cand_ho['ROI']} base={base_ho['ROI']}")
    ci = (boot_ho or {}).get('ci90') or [None, None]
    if ci[0] is None or ci[0] <= 0:
        ok = False
        reasons.append(f"holdout ROI差の90%区間が0を跨ぐ {ci}")
    if ok:
        reasons.append('train/holdout 双方で旧を上回り、holdout差は90%区間で正')
    return {
        'adopt': ok,
        'deploy_allowed': False,
        'ROI差_full': round(cand_full['ROI'] - base_full['ROI'], 2),
        'ROI差_train': round(cand_tr['ROI'] - base_tr['ROI'], 2),
        'ROI差_holdout': round(cand_ho['ROI'] - base_ho['ROI'], 2),
        'reasons': reasons,
        'holdout_bootstrap': boot_ho,
    }


def _pays(bets: list[dict]) -> list[float]:
    return [float(b.get('払戻') or 0) for b in bets]


def _one_day(payload: dict) -> list[dict]:
    """1日分を生成（プロセスプール用）。history はワーカー側で読む。"""
    logic = payload['logic']
    d = payload['date']
    sim_runs = payload['sim_runs']
    no_cache = payload['no_cache']
    xsel = payload.get('xsel') or ''
    hold_set = set(payload['holdout'])
    cache_path = Path(payload['cache_path'])
    _apply(logic, xsel)
    os.environ['ARERU_SIM_RUNS'] = str(sim_runs)
    os.environ['ARERU_FAST_GAUSS'] = '1'
    from scripts.logic_compare_backtest import _load_history, _load_results, _predict_for_date
    from scripts.stable_holdout_compare import _race_rows
    period = 'holdout' if d in hold_set else 'train'
    if cache_path.exists() and not no_cache:
        pred = pd.read_csv(cache_path, encoding='utf-8-sig')
    else:
        history = _load_history()
        pred, _ = _predict_for_date(
            d, legacy=str(os.environ.get('ARERU_LEGACY_SCORE') or '0') in ('1', 'true', 'yes'),
            history=history,
            sim_runs=sim_runs, use_cache=False, respect_env=True,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(cache_path, index=False, encoding='utf-8-sig')
    results = _load_results()
    day_rows = _race_rows(pred, results, d)
    for r in day_rows:
        r['period'] = period
        r['logic'] = logic
    return day_rows


def _collect_logic(logic: str, dates: list[str], holdout_dates: list[str], *,
                   sim_runs: int, no_cache: bool, xsel: str = '', workers: int = 4) -> list[dict]:
    payloads = []
    for d in dates:
        if logic == 'XSEL':
            tag = 'xsel_' + xsel.replace(',', '-')
            cache_path = EXTRA_CACHE / f'pred_{tag}_{d}.csv'
        else:
            cache_path = _cache_path(logic, d)
        payloads.append({
            'logic': logic, 'date': d, 'sim_runs': sim_runs, 'no_cache': no_cache,
            'xsel': xsel, 'holdout': holdout_dates, 'cache_path': str(cache_path),
        })
    # キャッシュ済みなら逐次の方が速い
    need_gen = [p for p in payloads if no_cache or not Path(p['cache_path']).exists()]
    use_pool = workers > 1 and len(need_gen) >= 2
    print(f'[{logic}] dates={len(dates)} generate={len(need_gen)} pool={use_pool}', flush=True)
    rows: list[dict] = []
    if not use_pool:
        for p in payloads:
            print(f'[{logic}] {p["date"]} ...', flush=True)
            rows.extend(_one_day(p))
        return rows
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_day, p): p['date'] for p in payloads}
        for fut in as_completed(futs):
            d = futs[fut]
            part = fut.result()
            print(f'[{logic}] done {d} races={len(part)}', flush=True)
            rows.extend(part)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-runs', type=int, default=2500)
    ap.add_argument('--logics', default='OLD,NEW,X,BURDEN,GATE,WEIGHT,JOCKEY,LAYOFF,STYLE,FIELD')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--skip-individuals', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    from scripts.logic_compare_backtest import _eligible_dates, _load_results
    from replay_predict import available_dates, load_runners

    runners = load_runners()
    dates = _eligible_dates(available_dates(runners))
    split = max(1, int(len(dates) * 0.70))
    train_dates = dates[:split]
    holdout_dates = dates[split:]
    results = _load_results()  # noqa: F841  # 存在確認
    logics = [x.strip() for x in args.logics.split(',') if x.strip()]
    if args.skip_individuals:
        logics = [x for x in logics if x not in INDIVIDUALS]

    print(
        f'[feat-search] days={len(dates)} train={len(train_dates)} '
        f'holdout={len(holdout_dates)} logics={logics} workers={args.workers}',
        flush=True,
    )

    rows_by: dict[str, list] = {}
    for logic in logics:
        if logic == 'XSEL':
            continue
        rows_by[logic] = _collect_logic(
            logic, dates, holdout_dates,
            sim_runs=args.sim_runs, no_cache=args.no_cache, workers=args.workers,
        )

    def scopes(all_rows):
        buys = [r for r in all_rows if r.get('strict_buy')]
        return {
            'full': buys,
            'train': [r for r in buys if r.get('period') == 'train'],
            'holdout': [r for r in buys if r.get('period') == 'holdout'],
        }

    report = {
        '検証設計': {
            '開催日数': len(dates),
            'train': train_dates,
            'holdout': holdout_dates,
            'SIM_RUNS': args.sim_runs,
            '比較対象': 'BUYのみ（投資判定が買い）',
            'BUY閾値': {'BUY_EV_FLOOR': 108, 'BUY_CONF_FLOOR': 58, '再現率': 42},
            '閾値探索': '禁止',
            '本番ロジック': 'OLD（ARERU_LEGACY_SCORE=1）固定',
            '表示EV': 'tanh圧縮 78-124 固定',
        },
        'strict_buy': {},
        'adoption': {},
        'train_winners': [],
    }

    for logic, all_rows in rows_by.items():
        sc = scopes(all_rows)
        report['strict_buy'][logic] = {
            k: _summarize(v, f'{logic}/{k}') for k, v in sc.items()
        }

    base = report['strict_buy']['OLD']
    old_sc = scopes(rows_by['OLD'])

    def eval_logic(name: str, all_rows: list):
        sc = scopes(all_rows)
        boot = _bootstrap_delta(_pays(sc['holdout']), _pays(old_sc['holdout']))
        gate = _adoption(
            _summarize(sc['full'], ''), base['full'],
            _summarize(sc['train'], ''), base['train'],
            _summarize(sc['holdout'], ''), base['holdout'],
            boot,
        )
        return sc, gate

    for logic, all_rows in rows_by.items():
        if logic == 'OLD':
            continue
        _, gate = eval_logic(logic, all_rows)
        report['adoption'][logic] = gate

    train_winners = []
    for logic in INDIVIDUALS:
        if logic not in report['strict_buy']:
            continue
        dtr = report['strict_buy'][logic]['train']['ROI'] - base['train']['ROI']
        if dtr > 0:
            train_winners.append(logic)
    report['train_winners'] = train_winners

    xsel_feats = ','.join(FEAT_OF[x] for x in train_winners)
    report['xsel_features'] = xsel_feats
    if xsel_feats:
        logic = 'XSEL'
        rows = _collect_logic(
            logic, dates, holdout_dates,
            sim_runs=args.sim_runs, no_cache=args.no_cache,
            xsel=xsel_feats, workers=args.workers,
        )
        rows_by[logic] = rows
        sc = scopes(rows)
        report['strict_buy'][logic] = {k: _summarize(v, f'{logic}/{k}') for k, v in sc.items()}
        _, gate = eval_logic(logic, rows)
        report['adoption'][logic] = gate

    # 条件別（新が旧を超えた subset）: 人気帯
    beats = []
    if 'NEW' in rows_by:
        from collections import defaultdict
        buckets = defaultdict(lambda: {'old': [], 'new': []})
        for r in old_sc['full']:
            buckets[r.get('人気帯') or '不明']['old'].append(r)
        new_buys = [r for r in rows_by['NEW'] if r.get('strict_buy')]
        for r in new_buys:
            buckets[r.get('人気帯') or '不明']['new'].append(r)
        for b, pair in buckets.items():
            so = _summarize(pair['old'], b)
            sn = _summarize(pair['new'], b)
            if sn['ROI'] > so['ROI'] and sn['BUY件数'] >= 20:
                beats.append({
                    '条件': f'人気帯={b}',
                    '旧ROI': so['ROI'], '新ROI': sn['ROI'],
                    '差': round(sn['ROI'] - so['ROI'], 2),
                    '旧n': so['BUY件数'], '新n': sn['BUY件数'],
                })
        report['new_beats_old_subsets'] = beats

    best_name = 'OLD'
    best_ok = False
    for logic, gate in report['adoption'].items():
        if gate.get('adopt') and report['strict_buy'][logic]['full']['ROI'] > (
            report['strict_buy'][best_name]['full']['ROI'] if best_name in report['strict_buy'] else -999
        ):
            best_name = logic
            best_ok = True

    report['best_candidate'] = {
        'logic': best_name if best_ok else 'OLD',
        'label': LOGIC_LABELS.get(best_name if best_ok else 'OLD'),
        'adopt_for_production': best_ok,
        'verdict': '改善を統計的に確認' if best_ok else '改善していない',
        'note': '未達なら本番ロジックとUI期待値表示は変更しない',
    }

    keys = ('BUY件数', '的中率', '回収率', 'ROI', '平均オッズ', 'サンプル数')
    table = {}
    for logic, block in report['strict_buy'].items():
        table[logic] = {
            'label': LOGIC_LABELS.get(logic, logic),
            'full': {k: block['full'].get(k) for k in keys},
            'train': {k: block['train'].get(k) for k in keys},
            'holdout': {k: block['holdout'].get(k) for k in keys},
            'ROI差_vs_OLD_full': round(block['full']['ROI'] - base['full']['ROI'], 2),
            'ROI差_vs_OLD_train': round(block['train']['ROI'] - base['train']['ROI'], 2),
            'ROI差_vs_OLD_holdout': round(block['holdout']['ROI'] - base['holdout']['ROI'], 2),
            '採用': bool(report['adoption'].get(logic, {}).get('adopt')),
        }
        if 'NEW' in report['strict_buy'] and logic not in ('OLD', 'NEW'):
            table[logic]['ROI差_vs_NEW_full'] = round(block['full']['ROI'] - report['strict_buy']['NEW']['full']['ROI'], 2)
            table[logic]['ROI差_vs_NEW_train'] = round(block['train']['ROI'] - report['strict_buy']['NEW']['train']['ROI'], 2)
            table[logic]['ROI差_vs_NEW_holdout'] = round(block['holdout']['ROI'] - report['strict_buy']['NEW']['holdout']['ROI'], 2)
            newb = report['strict_buy']['NEW']
            table[logic]['holdout_and_train_vs_NEW'] = (
                block['train']['ROI'] > newb['train']['ROI']
                and block['holdout']['ROI'] > newb['holdout']['ROI']
            )
    report['comparison_table'] = table
    TABLE.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'verdict': report['best_candidate'],
        'table': table,
        'train_winners': train_winners,
        'adoption': {k: {'adopt': v.get('adopt'), 'reasons': v.get('reasons')} for k, v in report['adoption'].items()},
    }, ensure_ascii=False, indent=2))
    print(f'📁 {OUT}')


if __name__ == '__main__':
    main()
