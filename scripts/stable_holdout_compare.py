#!/usr/bin/env python3
"""旧 vs 新ロジックの安定性検証（時系列ホールドアウト必須）。

採用条件（すべて）:
  - 厳格BUYが train / holdout の両方で旧を上回る
  - 厳格BUY合計が概ね100件以上（不足時は採用不可）
  - 人気偏り・単一日だけで差が説明されないこと

本番デプロイは常に禁止。ソフトBUY閾値はサンプル診断用。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
OUT = DATA / 'stable_holdout_report.json'
STAKE = 100
SOFT_EV_FLOORS = (100, 105, 108, 110, 115)
MIN_BUY_FOR_ADOPT = 100


def _json_safe(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _split_holdout(dates: list[str], holdout_ratio: float = 0.30) -> tuple[list[str], list[str]]:
    if len(dates) < 4:
        cut = max(1, len(dates) - 1)
    else:
        cut = max(2, int(round(len(dates) * (1 - holdout_ratio))))
        cut = min(cut, len(dates) - 1)
    return dates[:cut], dates[cut:]


def _pop_bucket(pop) -> str:
    try:
        p = float(pop)
    except (TypeError, ValueError):
        return '不明'
    if p <= 1:
        return '1番人気'
    if p <= 3:
        return '2-3番人気'
    if p <= 6:
        return '4-6番人気'
    return '7番人気以下'


def _race_rows(pred: pd.DataFrame, results: pd.DataFrame, date: str) -> list[dict]:
    from areru_engine import clean_name
    day = results[results['date'] == date]
    rows = []
    for _, row in pred.iterrows():
        rid = str(row.get('race_id', ''))
        horse = str(row.get('本命', '')).strip()
        venue = str(row.get('開催地') or row.get('場') or '')
        ev = pd.to_numeric(row.get('期待値'), errors='coerce')
        invest = str(row.get('投資判定') or '')
        pop = pd.to_numeric(row.get('本命人気'), errors='coerce')
        odds_pred = pd.to_numeric(row.get('本命オッズ'), errors='coerce')
        rr = day[(day['race_id'] == rid) & (day['馬名'].map(clean_name) == clean_name(horse))]
        if rr.empty:
            finish = pop_r = odds = float('nan')
            matched = False
        else:
            matched = True
            finish = float(pd.to_numeric(rr.iloc[0]['着順'], errors='coerce'))
            pop_r = float(pd.to_numeric(rr.iloc[0]['人気'], errors='coerce'))
            odds = float(pd.to_numeric(rr.iloc[0]['確定オッズ'], errors='coerce'))
        pop_v = float(pop) if pd.notna(pop) else (pop_r if pd.notna(pop_r) else float('nan'))
        odds_v = odds if pd.notna(odds) else (float(odds_pred) if pd.notna(odds_pred) else float('nan'))
        hit = bool(matched and pd.notna(finish) and finish == 1)
        top3 = bool(matched and pd.notna(finish) and finish <= 3)
        payout = float(odds_v) * STAKE if hit and pd.notna(odds_v) else 0.0
        rows.append({
            'date': date,
            'race_id': rid,
            'venue': venue,
            '本命': horse,
            '期待値': float(ev) if pd.notna(ev) else None,
            '投資判定': invest,
            'strict_buy': invest.startswith('買い'),
            '人気': pop_v if pd.notna(pop_v) else None,
            '人気帯': _pop_bucket(pop_v),
            'オッズ': odds_v if pd.notna(odds_v) else None,
            '着順': finish if pd.notna(finish) else None,
            '的中': hit,
            '3着内': top3,
            '照合': matched,
            '払戻': payout,
            'period': None,
        })
    return rows


def _filter_bets(rows: list[dict], mode: str, ev_floor: float | None = None) -> list[dict]:
    if mode == 'strict':
        return [r for r in rows if r.get('strict_buy')]
    if mode == 'soft':
        out = []
        for r in rows:
            ev = r.get('期待値')
            if ev is None:
                continue
            if float(ev) >= float(ev_floor):
                out.append(r)
        return out
    if mode == 'all_honmei':
        return list(rows)
    raise ValueError(mode)


def _summarize(bets: list[dict], label: str) -> dict:
    n = len(bets)
    if n == 0:
        return {
            'label': label, 'BUY件数': 0, '的中件数': 0, '的中率': 0.0,
            '3着内件数': 0, '3着内率': 0.0, '投資額': 0, '払戻': 0.0,
            'ROI': 0.0, '回収率': 0.0, '平均オッズ': None, '平均人気': None,
            '照合率': 0.0, '最大連敗': 0, '平均期待値': None, '1レース期待値': None,
        }
    hits = sum(1 for b in bets if b.get('的中'))
    top3 = sum(1 for b in bets if b.get('3着内'))
    invest = n * STAKE
    ret = float(sum(b.get('払戻') or 0 for b in bets))
    odds = [b['オッズ'] for b in bets if b.get('オッズ') is not None]
    pops = [b['人気'] for b in bets if b.get('人気') is not None]
    evs = [b['期待値'] for b in bets if b.get('期待値') is not None]
    matched = sum(1 for b in bets if b.get('照合'))
    roi_pct = (ret / invest * 100) if invest else 0.0
    # 最大連敗（日付→race_id 順）
    ordered = sorted(bets, key=lambda b: (str(b.get('date') or ''), str(b.get('race_id') or '')))
    max_lose = cur = 0
    for b in ordered:
        if b.get('的中'):
            cur = 0
        else:
            cur += 1
            max_lose = max(max_lose, cur)
    avg_ev = round(sum(evs) / len(evs), 2) if evs else None
    return {
        'label': label,
        'BUY件数': n,
        '的中件数': hits,
        '的中率': round(hits / n * 100, 2),
        '3着内件数': top3,
        '3着内率': round(top3 / n * 100, 2),
        '投資額': invest,
        '払戻': round(ret, 1),
        'ROI': round(roi_pct - 100, 2),
        '回収率': round(roi_pct, 2),
        '平均オッズ': round(sum(odds) / len(odds), 2) if odds else None,
        '平均人気': round(sum(pops) / len(pops), 2) if pops else None,
        '照合率': round(matched / n * 100, 1),
        '最大連敗': max_lose,
        '平均期待値': avg_ev,
        '1レース期待値': avg_ev,
    }


def _group_roi(bets: list[dict], key: str) -> dict:
    buckets: dict[str, list] = defaultdict(list)
    for b in bets:
        buckets[str(b.get(key) or '不明')].append(b)
    return {k: _summarize(v, k) for k, v in sorted(buckets.items(), key=lambda x: x[0])}


def _compare_pair(old_bets: list[dict], new_bets: list[dict], name: str) -> dict:
    o = _summarize(old_bets, f'旧/{name}')
    n = _summarize(new_bets, f'新/{name}')
    return {
        'name': name,
        '旧': o,
        '新': n,
        'BUY件数差': n['BUY件数'] - o['BUY件数'],
        'ROI差': round(n['ROI'] - o['ROI'], 2),
        '的中率差': round(n['的中率'] - o['的中率'], 2),
        '3着内率差': round(n['3着内率'] - o['3着内率'], 2),
        '人気別ROI_旧': _group_roi(old_bets, '人気帯'),
        '人気別ROI_新': _group_roi(new_bets, '人気帯'),
        '競馬場別ROI_旧': _group_roi(old_bets, 'venue'),
        '競馬場別ROI_新': _group_roi(new_bets, 'venue'),
        '期間別ROI_旧': _group_roi(old_bets, 'date'),
        '期間別ROI_新': _group_roi(new_bets, 'date'),
    }


def _adoption_gate(full: dict, train: dict, holdout: dict) -> dict:
    reasons = []
    ok = True
    fo, fn = full['旧'], full['新']
    to, tn = train['旧'], train['新']
    ho, hn = holdout['旧'], holdout['新']

    if fo['BUY件数'] < MIN_BUY_FOR_ADOPT or fn['BUY件数'] < MIN_BUY_FOR_ADOPT:
        ok = False
        reasons.append(
            f"厳格BUY件数不足（旧{fo['BUY件数']}/新{fn['BUY件数']} < {MIN_BUY_FOR_ADOPT}）"
        )
    if hn['ROI'] <= ho['ROI']:
        ok = False
        reasons.append(f"holdout ROI が旧以下（新{hn['ROI']} / 旧{ho['ROI']}）")
    if tn['ROI'] <= to['ROI']:
        ok = False
        reasons.append(f"train ROI が旧以下（新{tn['ROI']} / 旧{to['ROI']}）")
    # 全期間でも意味のある差（+5pp超）かつ holdout で非負の差を要求
    if full['ROI差'] < 5.0:
        ok = False
        reasons.append(f"全期間ROI差が小さい（{full['ROI差']}pp < 5）")
    if hn['的中率'] + 0.5 < ho['的中率'] and hn['ROI'] < ho['ROI'] + 15:
        ok = False
        reasons.append('holdout 的中率が旧より悪化しROI優位も弱い')

    new_fav = full['人気別ROI_新'].get('1番人気', {}).get('BUY件数', 0)
    if fn['BUY件数'] > 0 and new_fav / fn['BUY件数'] >= 0.70 and fn['平均人気'] is not None and fn['平均人気'] <= 1.5:
        if full['ROI差'] > 0:
            ok = False
            reasons.append('新ロジックBUYが1番人気に偏り、見かけの改善の可能性')

    if ok and full['ROI差'] <= 0:
        ok = False
        reasons.append('全期間ROI差が非正')

    if ok:
        reasons.append('train/holdout 双方で厳格BUYが旧を上回る')

    return {
        'adopt': ok,
        'deploy_allowed': False,
        'reasons': reasons,
        'note': 'ユーザー指示により Render デプロイは禁止。採用フラグは検証記録のみ。',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-runs', type=int, default=10000)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--holdout-ratio', type=float, default=0.30)
    args = ap.parse_args()

    from replay_predict import available_dates, load_runners
    from scripts.logic_compare_backtest import (
        _eligible_dates,
        _load_history,
        _load_results,
        _predict_for_date,
    )

    all_dates = available_dates(load_runners())
    dates = _eligible_dates(all_dates)
    train_dates, holdout_dates = _split_holdout(dates, args.holdout_ratio)
    history = _load_history()
    results = _load_results()

    print(f'[stable] eligible={len(dates)} train={train_dates} holdout={holdout_dates}', flush=True)

    old_rows: list[dict] = []
    new_rows: list[dict] = []
    for d in dates:
        period = 'train' if d in train_dates else 'holdout'
        print(f'[stable] {d} ({period}) ...', flush=True)
        try:
            old_pred, _ = _predict_for_date(
                d, legacy=True, history=history, sim_runs=args.sim_runs, use_cache=not args.no_cache,
            )
            new_pred, _ = _predict_for_date(
                d, legacy=False, history=history, sim_runs=args.sim_runs, use_cache=not args.no_cache,
            )
        except Exception as e:
            print(f'  skip {d}: {e}', flush=True)
            continue
        for r in _race_rows(old_pred, results, d):
            r['period'] = period
            old_rows.append(r)
        for r in _race_rows(new_pred, results, d):
            r['period'] = period
            new_rows.append(r)

    def slice_period(rows, period=None):
        if period is None:
            return rows
        return [r for r in rows if r.get('period') == period]

    report = {
        '検証日': dates,
        'train_dates': train_dates,
        'holdout_dates': holdout_dates,
        'SIM_RUNS': args.sim_runs,
        'MIN_BUY_FOR_ADOPT': MIN_BUY_FOR_ADOPT,
        'strict_buy': {},
        'soft_buy': {},
        'all_honmei': {},
    }

    for period_name, period in (('full', None), ('train', 'train'), ('holdout', 'holdout')):
        o = slice_period(old_rows, period)
        n = slice_period(new_rows, period)
        report['strict_buy'][period_name] = _compare_pair(
            _filter_bets(o, 'strict'), _filter_bets(n, 'strict'), f'strict/{period_name}'
        )
        report['all_honmei'][period_name] = _compare_pair(
            _filter_bets(o, 'all_honmei'), _filter_bets(n, 'all_honmei'), f'honmei/{period_name}'
        )
        soft = {}
        for floor in SOFT_EV_FLOORS:
            soft[str(floor)] = _compare_pair(
                _filter_bets(o, 'soft', floor), _filter_bets(n, 'soft', floor),
                f'soft>={floor}/{period_name}',
            )
        report['soft_buy'][period_name] = soft

    gate = _adoption_gate(
        report['strict_buy']['full'],
        report['strict_buy']['train'],
        report['strict_buy']['holdout'],
    )
    report['adoption'] = gate

    sf = report['strict_buy']['full']
    sh = report['strict_buy']['holdout']
    st = report['strict_buy']['train']
    ah = report['all_honmei']['full']
    metric_keys = (
        'BUY件数', '的中率', '3着内率', 'ROI', '回収率',
        '平均オッズ', '平均人気', '最大連敗', '平均期待値', '1レース期待値',
    )
    summary = {
        '厳格BUY_全期間': {
            '旧': {k: sf['旧'][k] for k in metric_keys},
            '新': {k: sf['新'][k] for k in metric_keys},
            'ROI差': sf['ROI差'],
            '人気別ROI_旧': {k: v.get('ROI') for k, v in sf['人気別ROI_旧'].items()},
            '人気別ROI_新': {k: v.get('ROI') for k, v in sf['人気別ROI_新'].items()},
            '競馬場別ROI_旧': {k: v.get('ROI') for k, v in sf['競馬場別ROI_旧'].items()},
            '競馬場別ROI_新': {k: v.get('ROI') for k, v in sf['競馬場別ROI_新'].items()},
            '期間別ROI_旧': {k: v.get('ROI') for k, v in sf['期間別ROI_旧'].items()},
            '期間別ROI_新': {k: v.get('ROI') for k, v in sf['期間別ROI_新'].items()},
        },
        '厳格BUY_train': {
            '旧': {k: st['旧'][k] for k in ('BUY件数', '的中率', '3着内率', 'ROI', '回収率')},
            '新': {k: st['新'][k] for k in ('BUY件数', '的中率', '3着内率', 'ROI', '回収率')},
            'ROI差': st['ROI差'],
        },
        '厳格BUY_holdout': {
            '旧': {k: sh['旧'][k] for k in ('BUY件数', '的中率', '3着内率', 'ROI', '回収率')},
            '新': {k: sh['新'][k] for k in ('BUY件数', '的中率', '3着内率', 'ROI', '回収率')},
            'ROI差': sh['ROI差'],
        },
        '全レース本命_全期間': {
            '旧': {k: ah['旧'][k] for k in metric_keys},
            '新': {k: ah['新'][k] for k in metric_keys},
            'ROI差': ah['ROI差'],
        },
        'soft_buy_counts_full': {
            str(fl): {
                '旧BUY': report['soft_buy']['full'][str(fl)]['旧']['BUY件数'],
                '新BUY': report['soft_buy']['full'][str(fl)]['新']['BUY件数'],
                'ROI差': report['soft_buy']['full'][str(fl)]['ROI差'],
            }
            for fl in SOFT_EV_FLOORS
        },
        'adoption': gate,
    }
    OUT.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\n📁 {OUT}')


if __name__ == '__main__':
    main()
