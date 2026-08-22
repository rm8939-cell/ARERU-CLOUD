#!/usr/bin/env python3
"""新旧予想ロジックの比較とバックテスト。

旧: ARERU_LEGACY_SCORE=1（タイム/着差/馬場/騎手/馬体重補正なし + ガウスSIM）
新: 既定（past_five + 詳細履歴 + race_sim プロファイル）
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
OUT = DATA / 'logic_compare_report.json'


def _load_history():
    p = DATA / 'all_history.csv'
    if not p.exists():
        return None
    h = pd.read_csv(p, encoding='utf-8-sig')
    from areru_engine import parse_date, clean_name
    h['_date'] = parse_date(h['年月日'])
    h['_horse'] = h['馬名'].map(clean_name)
    return h


def _run_for_date(target: str, legacy: bool):
    os.environ['ARERU_LEGACY_SCORE'] = '1' if legacy else '0'
    os.environ.setdefault('ARERU_SIM_RUNS', '5000')
    from replay_predict import load_runners, run_date
    runners = load_runners()
    history = pd.read_csv(DATA / 'all_history.csv', encoding='utf-8-sig')
    from areru_engine import parse_date, clean_name
    history['_date'] = parse_date(history['年月日'])
    history['_horse'] = history['馬名'].map(clean_name)
    out_path = run_date(target, runners, history)
    df = pd.read_csv(out_path, encoding='utf-8-sig')
    tag = 'legacy' if legacy else 'new'
    scores_path = DATA / 'predictions_by_date' / f'scores_{target}.csv'
    scores = pd.read_csv(scores_path, encoding='utf-8-sig') if scores_path.exists() else pd.DataFrame()
    return df, scores, tag


def compare_race(target: str, race_id: str | None = None) -> dict:
    """1日または1レースの新旧比較。"""
    old_pred, old_scores, _ = _run_for_date(target, legacy=True)
    new_pred, new_scores, _ = _run_for_date(target, legacy=False)

    if race_id:
        old_pred = old_pred[old_pred['race_id'].astype(str) == str(race_id)]
        new_pred = new_pred[new_pred['race_id'].astype(str) == str(race_id)]
        old_scores = old_scores[old_scores['race_id'].astype(str) == str(race_id)] if not old_scores.empty else old_scores
        new_scores = new_scores[new_scores['race_id'].astype(str) == str(race_id)] if not new_scores.empty else new_scores

    horses = []
    if not old_scores.empty and not new_scores.empty:
        merged = old_scores.merge(
            new_scores,
            on=['race_id', '馬名'],
            suffixes=('_旧', '_新'),
            how='outer',
        )
        for _, r in merged.iterrows():
            old_s = float(r.get('AREru指数_旧') or 0)
            new_s = float(r.get('AREru指数_新') or 0)
            old_rank = int(r.get('順位_旧') or 99) if '順位_旧' in r else None
            new_rank = int(r.get('順位_新') or 99) if '順位_新' in r else None
            horses.append({
                'race_id': str(r.get('race_id')),
                '馬名': r.get('馬名'),
                '旧スコア': round(old_s, 2),
                '新スコア': round(new_s, 2),
                '旧順位': old_rank,
                '新順位': new_rank,
                '順位変化': (old_rank - new_rank) if old_rank and new_rank else None,
            })

    race_rows = []
    for rid in sorted(set(old_pred['race_id'].astype(str)) | set(new_pred['race_id'].astype(str))):
        o = old_pred[old_pred['race_id'].astype(str) == rid]
        n = new_pred[new_pred['race_id'].astype(str) == rid]
        if o.empty or n.empty:
            continue
        o0, n0 = o.iloc[0], n.iloc[0]
        race_rows.append({
            'race_id': rid,
            'レース': n0.get('レース'),
            '開催地': n0.get('開催地'),
            '旧本命': o0.get('本命'),
            '新本命': n0.get('本命'),
            '旧期待値': o0.get('期待値'),
            '新期待値': n0.get('期待値'),
            '旧投資判定': o0.get('投資判定'),
            '新投資判定': n0.get('投資判定'),
            'EV変化': round(float(n0.get('期待値') or 0) - float(o0.get('期待値') or 0), 1),
            'BUY変化': f"{o0.get('投資判定')}→{n0.get('投資判定')}",
        })

    return {
        'date': target,
        'race_id': race_id,
        '馬別比較': horses[:200],
        'レース別比較': race_rows,
        '旧BUY件数': int((old_pred['投資判定'].astype(str).str.startswith('買い')).sum()),
        '新BUY件数': int((new_pred['投資判定'].astype(str).str.startswith('買い')).sum()),
    }


def backtest_buy_performance(dates: list[str]) -> dict:
    """過去日のBUY成績（新ロジック再生成）。"""
    from areru_engine import source_from_race_id
    results_path = DATA / 'results.csv'
    if not results_path.exists():
        return {'error': 'results.csv がありません'}

    results = pd.read_csv(results_path, encoding='utf-8-sig')
    bets = []
    for d in dates:
        try:
            pred, _, _ = _run_for_date(d, legacy=False)
        except Exception as e:
            print(f'skip {d}: {e}', flush=True)
            continue
        buys = pred[pred['投資判定'].astype(str).str.startswith('買い')]
        for _, row in buys.iterrows():
            rid = str(row.get('race_id', ''))
            horse = str(row.get('本命', '')).strip()
            odds = pd.to_numeric(row.get('本命オッズ'), errors='coerce')
            ev = pd.to_numeric(row.get('期待値'), errors='coerce')
            src = source_from_race_id(rid)
            race_no = row.get('レース')
            hit = False
            payout = 0.0
            rr = results[
                (results['race_id'].astype(str) == rid)
                & (results['馬名'].astype(str).str.strip() == horse)
            ] if 'race_id' in results.columns else pd.DataFrame()
            if rr.empty and 'レース' in results.columns:
                rr = results[
                    (results['レース'] == race_no)
                    & (results['馬名'].astype(str).str.strip() == horse)
                ]
            if not rr.empty:
                fin = pd.to_numeric(rr.iloc[0].get('着順'), errors='coerce')
                hit = pd.notna(fin) and fin == 1
                if hit and pd.notna(odds):
                    payout = float(odds) * 100
            bets.append({
                'date': d,
                'race_id': rid,
                'source': src,
                '本命': horse,
                '期待値': float(ev) if pd.notna(ev) else None,
                'オッズ': float(odds) if pd.notna(odds) else None,
                '的中': hit,
                '払戻': payout,
            })

    if not bets:
        return {'error': 'BUYデータなし', 'dates': dates}

    df = pd.DataFrame(bets)
    n = len(df)
    hits = int(df['的中'].sum())
    invest = n * 100
    ret = float(df['払戻'].sum())
    roi = (ret / invest * 100) if invest else 0.0
    return {
        '検証日数': len(dates),
        'BUY件数': n,
        '的中数': hits,
        '的中率': round(hits / n * 100, 2) if n else 0,
        '投資': invest,
        '払戻': round(ret, 0),
        '回収率': round(roi, 2),
        'ROI': round(roi - 100, 2),
        '平均期待値': round(float(df['期待値'].mean()), 1) if df['期待値'].notna().any() else None,
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='2026-08-22', help='比較対象日 YYYY-MM-DD')
    ap.add_argument('--race-id', default='', help='特定レースID')
    ap.add_argument('--backtest-days', type=int, default=14, help='バックテスト日数')
    args = ap.parse_args()

    from replay_predict import load_runners, available_dates
    runners = load_runners()
    dates = available_dates(runners)
    bt_dates = dates[-args.backtest_days:] if dates else []

    report = _json_safe({
        'compare': compare_race(args.date, args.race_id or None),
        'backtest_new': backtest_buy_performance(bt_dates),
        'backtest_dates': bt_dates,
    })
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'\n📁 {OUT}')


if __name__ == '__main__':
    main()
