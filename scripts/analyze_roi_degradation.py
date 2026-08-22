#!/usr/bin/env python3
"""旧 vs 新 BUY 差分の原因分析（-27.2pp 切り分け）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
CACHE = DATA / 'logic_compare_cache'


def main():
    report_path = DATA / 'logic_compare_report.json'
    if not report_path.exists():
        print('logic_compare_report.json がありません', file=sys.stderr)
        sys.exit(1)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    fair = report.get('fair_backtest') or {}
    old_bets = fair.get('旧BUY詳細') or []
    new_bets = fair.get('新BUY詳細') or []

    old_keys = {(b['date'], b['race_id']): b for b in old_bets}
    new_keys = {(b['date'], b['race_id']): b for b in new_bets}

    only_old = set(old_keys) - set(new_keys)
    only_new = set(new_keys) - set(old_keys)
    both = set(old_keys) & set(new_keys)

    analysis = {
        '旧BUY件数': len(old_bets),
        '新BUY件数': len(new_bets),
        '旧のみBUY': len(only_old),
        '新のみBUY': len(only_new),
        '共通BUY': len(both),
        '旧のみ詳細': [old_keys[k] for k in sorted(only_old)],
        '新のみ詳細': [new_keys[k] for k in sorted(only_new)],
        '共通で的中差': [],
    }

    for k in both:
        o, n = old_keys[k], new_keys[k]
        if o.get('本命') != n.get('本命') or o.get('的中') != n.get('的中'):
            analysis['共通で的中差'].append({'key': k, '旧': o, '新': n})

    # レース単位: 本命変更
    dates = fair.get('検証日') or []
    honmei_changes = []
    for d in dates:
        op = CACHE / f'predictions_legacy_{d}.csv'
        np_ = CACHE / f'predictions_new_{d}.csv'
        if not op.exists() or not np_.exists():
            continue
        odf = pd.read_csv(op, encoding='utf-8-sig')
        ndf = pd.read_csv(np_, encoding='utf-8-sig')
        for rid in set(odf['race_id'].astype(str)) | set(ndf['race_id'].astype(str)):
            o = odf[odf['race_id'].astype(str) == rid]
            n = ndf[ndf['race_id'].astype(str) == rid]
            if o.empty or n.empty:
                continue
            oh, nh = str(o.iloc[0].get('本命', '')), str(n.iloc[0].get('本命', ''))
            ob, nb = str(o.iloc[0].get('投資判定', '')), str(n.iloc[0].get('投資判定', ''))
            if oh != nh or ob != nb:
                honmei_changes.append({
                    'date': d, 'race_id': rid,
                    '旧本命': oh, '新本命': nh,
                    '旧判定': ob, '新判定': nb,
                    '旧EV': o.iloc[0].get('期待値'),
                    '新EV': n.iloc[0].get('期待値'),
                })

    analysis['本命/判定変更レース数'] = len(honmei_changes)
    analysis['本命変更サンプル'] = honmei_changes[:30]

    # ROI 寄与分解（旧のみ/新のみ/共通）
    def _roi(bets):
        if not bets:
            return {'n': 0, 'hits': 0, 'roi': 0.0, 'payout': 0}
        invest = len(bets) * 100
        payout = sum(b.get('払戻', 0) for b in bets)
        hits = sum(1 for b in bets if b.get('的中'))
        return {'n': len(bets), 'hits': hits, 'roi': round(payout / invest * 100 - 100, 2), 'payout': payout}

    analysis['ROI分解'] = {
        '旧のみ': _roi([old_keys[k] for k in only_old]),
        '新のみ': _roi([new_keys[k] for k in only_new]),
        '共通': _roi([old_keys[k] for k in both]),
    }

    out = DATA / 'roi_degradation_analysis.json'
    out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        '旧BUY': analysis['旧BUY件数'],
        '新BUY': analysis['新BUY件数'],
        '旧のみ': analysis['旧のみBUY'],
        '新のみ': analysis['新のみBUY'],
        'ROI分解': analysis['ROI分解'],
        '本命変更': analysis['本命/判定変更レース数'],
    }, ensure_ascii=False, indent=2))
    print(f'\n📁 {out}')


if __name__ == '__main__':
    main()
