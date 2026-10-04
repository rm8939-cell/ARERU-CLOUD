#!/usr/bin/env python3
"""quality_backtest.py の結果を新旧で並べ、採否を機械的に判定する。

採用条件（ROI だけで決めない）
  - 確率モデルの質（holdout の対数損失・AUC）が悪化しないこと
  - 本命の勝率・複勝率が holdout で改善すること
  - ROI は 90% 信頼区間つきで併記し、点推定だけで採否を決めない
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data' / 'quality_backtest'


def g(d: dict, *path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return default
        if cur is None:
            return default
    return cur


def model_row(sec: dict, name: str) -> dict:
    for m in g(sec, '確率モデル', '勝率モデル', default=[]) or []:
        if m['モデル'] == name:
            return m
    return {}


def fmt(v, suffix='') -> str:
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.2f}{suffix}'
    return f'{v}{suffix}'


def table(rows: list[list[str]]) -> str:
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        out.append('  '.join(str(c).ljust(w[j]) for j, c in enumerate(r)).rstrip())
        if i == 0:
            out.append('  '.join('-' * w[j] for j in range(len(r))))
    return '\n'.join(out)


def compare(old: dict, new: dict, tag: str) -> dict:
    o, n = old[tag], new[tag]
    out = {'区間': tag, 'レース数': o['レース数']}
    pairs = [
        ('本命 勝率', ('運用成績', '本命', '勝率')),
        ('本命 複勝率', ('運用成績', '本命', '複勝率')),
        ('本命 ROI', ('運用成績', '本命', 'ROI')),
        ('本命 平均オッズ', ('運用成績', '本命', '平均オッズ')),
        ('穴馬 複勝率', ('運用成績', '穴馬', '複勝率')),
        ('穴馬 ROI', ('運用成績', '穴馬', 'ROI')),
        ('BUY 件数', ('運用成績', 'BUY', '件数')),
        ('BUY 的中率', ('運用成績', 'BUY', '勝率')),
        ('BUY 回収率', ('運用成績', 'BUY', '回収率')),
        ('危険人気 3着外率', ('運用成績', '危険人気馬', '3着外に飛んだ率')),
        ('AI1位の勝率', ('確率モデル', 'AI1位の勝率(SIM勝率)')),
        ('AI1〜3位の複勝率', ('確率モデル', 'AI1〜3位の複勝率(SIM勝率)')),
    ]
    for label, path in pairs:
        out[label] = {'旧': g(o, *path), '新': g(n, *path)}
    for label, mname in (('勝率モデル logloss', 'AI(SIM勝率)'),):
        om, nm = model_row(o, mname), model_row(n, mname)
        out[label] = {'旧': om.get('logloss'), '新': nm.get('logloss')}
        out['勝率モデル AUC'] = {'旧': om.get('AUC'), '新': nm.get('AUC')}
    mm = model_row(o, '市場(1/オッズ)')
    out['参考 市場 logloss'] = mm.get('logloss')
    out['参考 市場 AUC'] = mm.get('AUC')
    out['参考 1番人気 ROI'] = g(o, '市場ベースライン', '1番人気', 'ROI')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', default='prod_old')
    ap.add_argument('--new', default='prob_v2')
    a = ap.parse_args()
    old = json.loads((DATA / f'{a.old}.json').read_text(encoding='utf-8'))
    new = json.loads((DATA / f'{a.new}.json').read_text(encoding='utf-8'))

    report = {'旧': a.old, '新': a.new, '比較': []}
    for tag in ('full', 'train', 'holdout'):
        if not old[tag]['レース数'] or not new[tag]['レース数']:
            continue
        c = compare(old, new, tag)
        report['比較'].append(c)
        rows = [['指標', '旧(本番)', '新(確率v2)', '差']]
        for k, v in c.items():
            if not isinstance(v, dict):
                continue
            d = (v['新'] - v['旧']) if (isinstance(v['旧'], (int, float))
                                       and isinstance(v['新'], (int, float))) else None
            rows.append([k, fmt(v['旧']), fmt(v['新']),
                         (f'{d:+.2f}' if isinstance(d, float) else fmt(d))])
        print(f"\n===== {tag}  {c['レース数']}R =====")
        print(table(rows))
        print(f"  参考: 市場logloss={c['参考 市場 logloss']} 市場AUC={c['参考 市場 AUC']} "
              f"1番人気ROI={c['参考 1番人気 ROI']}")

    # 採否判定は holdout のみで行う
    ho = next((c for c in report['比較'] if c['区間'] == 'holdout'), None)
    if ho:
        checks = {
            '本命勝率が改善': ho['本命 勝率']['新'] > ho['本命 勝率']['旧'],
            '本命複勝率が改善': ho['本命 複勝率']['新'] > ho['本命 複勝率']['旧'],
            '確率モデルのloglossが悪化しない':
                (ho['勝率モデル logloss']['新'] or 9) <= (ho['勝率モデル logloss']['旧'] or 9) + 1e-6,
            'AUCが悪化しない':
                (ho['勝率モデル AUC']['新'] or 0) >= (ho['勝率モデル AUC']['旧'] or 0) - 1e-6,
        }
        report['採否'] = {'条件': checks, '採用': all(checks.values())}
        print('\n===== 採否（holdout のみで判定）=====')
        for k, v in checks.items():
            print(f'  {"OK " if v else "NG "} {k}')
        print(f"  => {'採用' if all(checks.values()) else '不採用'}")

    p = DATA / f'compare_{a.old}_vs_{a.new}.json'
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\nwrote', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
