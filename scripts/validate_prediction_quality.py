#!/usr/bin/env python3
"""予想品質の変更前後比較（scores / predictions CSV）。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ev_analysis import _buy_quality_score, _edge_pp, finalize_predictions_df


def _top3_hit(row, scores: pd.DataFrame) -> bool | None:
    fin = row.get('実着順')
    if fin is None or (isinstance(fin, float) and pd.isna(fin)):
        return None
    try:
        f = int(float(fin))
    except (TypeError, ValueError):
        return None
    rid = row.get('race_id')
    race = scores[scores['race_id'] == rid].sort_values('AREru指数', ascending=False)
    if race.empty or 'AREru指数' not in race.columns:
        return None
    top3 = set(race.head(3)['馬名'].astype(str))
    return str(row.get('馬名')) in top3


def analyze(date: str, baseline_path: Path | None = None) -> dict:
    pred_path = ROOT / 'data' / 'predictions_by_date' / f'predictions_{date}.csv'
    score_path = ROOT / 'data' / 'predictions_by_date' / f'scores_{date}.csv'
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    df = pd.read_csv(pred_path, encoding='utf-8-sig')
    finalized = finalize_predictions_df(df)

    buy = finalized[finalized['投資判定'].astype(str).str.startswith('買')]
    metrics = {
        'date': date,
        'races': len(finalized),
        'buy_count': len(buy),
        'buy_avg_ev': float(buy['期待値'].mean()) if len(buy) else None,
        'buy_avg_pop': float(buy['本命人気'].mean()) if len(buy) and '本命人気' in buy.columns else None,
        'buy_avg_edge': None,
        'buy_avg_quality': None,
        'buy_horses': [],
    }

    edges = []
    qs = []
    for _, r in buy.iterrows():
        rec = r.to_dict()
        edge = _edge_pp(rec.get('補正勝率'), rec.get('市場暗示勝率'))
        q = _buy_quality_score(rec)
        edges.append(edge)
        qs.append(q)
        metrics['buy_horses'].append({
            'race': f"{r.get('開催地')} R{r.get('レース')}",
            'horse': r.get('本命'),
            'ev': r.get('期待値'),
            'edge_pp': round(edge, 1),
            'quality': q,
            'pop': r.get('本命人気'),
        })
    if edges:
        metrics['buy_avg_edge'] = round(sum(edges) / len(edges), 2)
    if qs:
        metrics['buy_avg_quality'] = round(sum(qs) / len(qs), 1)

    if score_path.exists():
        scores = pd.read_csv(score_path, encoding='utf-8-sig')
        if 'AREru指数' not in scores.columns and 'race_id' in scores.columns:
            pass
        else:
            hits = []
            for _, row in scores.iterrows():
                h = _top3_hit(row, scores)
                if h is not None:
                    hits.append(h)
            if hits:
                metrics['index_top3_rate'] = round(sum(hits) / len(hits) * 100, 1)

    if baseline_path and baseline_path.exists():
        with open(baseline_path, encoding='utf-8') as f:
            base = json.load(f)
        base_buy = [x for x in base if str(x.get('buy', '')).startswith('買')]
        metrics['baseline_buy_count'] = len(base_buy)
        base_edges = []
        for x in base_buy:
            aw = x.get('adj_win')
            mi = x.get('market_impl')
            if aw and mi:
                base_edges.append(float(aw) - float(mi))
        if base_edges:
            metrics['baseline_avg_edge'] = round(sum(base_edges) / len(base_edges), 2)

    return metrics


if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else '2026-08-22'
    baseline = Path('/tmp/baseline_2026-08-22.json')
    m = analyze(date, baseline)
    print(json.dumps(m, ensure_ascii=False, indent=2))
