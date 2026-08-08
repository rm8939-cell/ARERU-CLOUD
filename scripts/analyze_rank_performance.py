#!/usr/bin/env python3
"""過去予想 × 実績から S/A/B ランクと買い判定の成績を出す。

使い方:
  python3 scripts/analyze_rank_performance.py
  python3 scripts/analyze_rank_performance.py --write data/rank_analysis_report.json

見ている指標:
  - 本命単勝の的中率・回収率・平均オッズ（ランク品質の主指標）
  - 投資判定=買い に絞った本命成績
  - 能力差・レース信頼度・EV・オッズ帯などの条件別本命成績
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / 'data' / 'predictions_by_date'
ANALYSIS = ROOT / 'data' / 'analysis_result.csv'
RESULTS = ROOT / 'data' / 'results.csv'


def _load_predictions() -> pd.DataFrame:
    rows = []
    for path in sorted(PRED_DIR.glob('predictions_*.csv')):
        day = re.search(r'(\d{4}-\d{2}-\d{2})', path.name)
        df = pd.read_csv(path, encoding='utf-8-sig')
        df['date'] = day.group(1) if day else ''
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _rank_block(df: pd.DataFrame) -> dict:
    out = {}
    for rk in ('S', 'A', 'B', 'C', 'D'):
        g = df[df['勝負ランク'].astype(str).str.upper() == rk]
        if g.empty:
            out[rk] = {'bets': 0}
            continue
        inv = float(pd.to_numeric(g['investment'], errors='coerce').fillna(0).sum())
        pay = float(pd.to_numeric(g['payout'], errors='coerce').fillna(0).sum())
        hits = int(pd.to_numeric(g['hit'], errors='coerce').fillna(0).sum())
        odds = pd.to_numeric(g.get('本命オッズ'), errors='coerce')
        out[rk] = {
            'bets': int(len(g)),
            'hits': hits,
            'hit_rate': round(hits / len(g) * 100.0, 1),
            'investment': int(inv),
            'payout': int(pay),
            'recovery': round(pay / inv * 100.0, 1) if inv else None,
            'profit': int(pay - inv),
            'avg_odds': round(float(odds.mean()), 2) if odds.notna().any() else None,
        }
    return out


def _condition_block(honmei: pd.DataFrame) -> dict:
    """本命行に対する条件別成績。"""
    h = honmei.copy()
    h['odds'] = pd.to_numeric(h.get('本命オッズ'), errors='coerce')
    h['ab'] = pd.to_numeric(h.get('能力差スコア'), errors='coerce')
    h['rc'] = pd.to_numeric(h.get('レース信頼度スコア'), errors='coerce')
    h['ev'] = pd.to_numeric(h.get('期待値'), errors='coerce')
    h['ai'] = pd.to_numeric(h.get('AI信頼度スコア'), errors='coerce')

    def one(mask, label):
        s = h[mask]
        if len(s) < 8:
            return None
        inv = float(s['investment'].sum())
        pay = float(s['payout'].sum())
        return {
            'label': label,
            'bets': int(len(s)),
            'hit_rate': round(float(s['hit'].mean()) * 100.0, 1),
            'avg_odds': round(float(s['odds'].mean()), 2) if s['odds'].notna().any() else None,
            'recovery': round(pay / inv * 100.0, 1) if inv else None,
            'profit': int(pay - inv),
        }

    specs = [
        (h['勝負ランク'].isin(['S', 'A']), 'rank S/A'),
        (h['投資判定'].astype(str) == '買い', 'invest=買い'),
        (h['ab'] >= 80, 'ability>=80'),
        (h['ab'] >= 70, 'ability>=70'),
        (h['rc'] >= 68, 'race_conf>=68'),
        ((h['rc'] >= 48) & (h['rc'] < 58), 'race_conf 48-58'),
        (h['ev'] >= 108, 'EV>=108 (旧買い閾値)'),
        ((h['ev'] >= 100) & (h['ev'] < 108), 'EV 100-108'),
        (h['odds'] <= 10, 'odds<=10'),
        (h['odds'] > 50, 'odds>50'),
        (h['source'].astype(str).str.lower() == 'jra', 'JRA'),
        (h['source'].astype(str).str.lower() == 'nar', 'NAR'),
    ]
    out = []
    for mask, label in specs:
        row = one(mask, label)
        if row:
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', default='', help='JSON レポート出力先')
    args = ap.parse_args()

    if not ANALYSIS.exists():
        print('analysis_result.csv がありません。先に results.py を走らせてください。')
        return 1

    analysis = pd.read_csv(ANALYSIS, encoding='utf-8-sig')
    preds = _load_predictions()
    keep = [c for c in (
        'race_id', '投資判定', '本命オッズ', '期待値', '能力差スコア',
        'レース信頼度スコア', 'AI信頼度スコア',
    ) if c in preds.columns]
    preds = preds[keep].copy()
    preds['race_id'] = preds['race_id'].astype(str)
    analysis['race_id'] = analysis['race_id'].astype(str)
    merged = analysis.merge(preds, on='race_id', how='left', suffixes=('', '_pred'))

    honmei = merged[merged['bet_type'].astype(str) == '本命'].copy()
    buy_honmei = honmei[honmei['投資判定'].astype(str) == '買い'].copy()

    report = {
        'pred_files': len(list(PRED_DIR.glob('predictions_*.csv'))),
        'analysis_rows': int(len(analysis)),
        'honmei_rows': int(len(honmei)),
        'by_rank_honmei': _rank_block(honmei),
        'by_rank_buy_honmei': _rank_block(buy_honmei),
        'conditions_honmei': _condition_block(honmei),
        'notes': [
            '主指標は本命チケット。ワイド等は買い目生成の別問題。',
            'S は能力差≥80 を主軸にした厳格条件。展開安定≥65 は到達不能だったため廃止。',
            '買い判定の EV 下限は 108→100。能力差下限とオッズ上限を追加。',
        ],
    }

    print('=== 本命 × 勝負ランク ===')
    for rk, v in report['by_rank_honmei'].items():
        if not v.get('bets'):
            continue
        print(
            f"  {rk}: n={v['bets']} hit={v['hit_rate']}% "
            f"avg_odds={v['avg_odds']} rec={v['recovery']}% profit={v['profit']:+d}"
        )
    print('=== 本命 × 投資判定=買い × ランク ===')
    for rk, v in report['by_rank_buy_honmei'].items():
        if not v.get('bets'):
            continue
        print(
            f"  {rk}: n={v['bets']} hit={v['hit_rate']}% "
            f"avg_odds={v['avg_odds']} rec={v['recovery']}% profit={v['profit']:+d}"
        )
    print('=== 条件別（本命） ===')
    for row in report['conditions_honmei']:
        print(
            f"  {row['label']:22} n={row['bets']:4} hit={row['hit_rate']:5.1f}% "
            f"odds={row['avg_odds']} rec={row['recovery']}% profit={row['profit']:+d}"
        )

    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
