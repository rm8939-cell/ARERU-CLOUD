"""AIランク閾値のROI学習フック（PO-5）。

現状は各ランクの実回収率を集計し、候補閾値を探索して
data/areru_v2_config.json の rank_thresholds を更新できる。
将来は強化学習やベイズ最適化に差し替え可能な入口として使う。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from areru_engine import (
    CONFIG_FILE,
    DEFAULT_RANK_THRESHOLDS,
    RANK_LABELS,
    assign_ai_ranks,
    load_rank_thresholds,
)

DATA = Path('data')
ARCH = DATA / 'predictions_by_date'
ANALYSIS_CSV = DATA / 'analysis_result.csv'


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'version': '2.0', 'weights': {}, 'rank_thresholds': dict(DEFAULT_RANK_THRESHOLDS)}


def _save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')


def _prediction_frames() -> list[tuple[str, pd.DataFrame]]:
    frames = []
    if not ARCH.exists():
        return frames
    for path in sorted(ARCH.glob('predictions_*.csv')):
        try:
            df = pd.read_csv(path, encoding='utf-8-sig').fillna('')
        except Exception:
            continue
        if df.empty or 'race_id' not in df.columns:
            continue
        date = path.stem.replace('predictions_', '')
        frames.append((date, df))
    return frames


def _attach_rank_to_analysis(analysis: pd.DataFrame, pred_rank: dict[str, str]) -> pd.DataFrame:
    out = analysis.copy()
    out['勝負ランク'] = out['race_id'].astype(str).map(lambda r: pred_rank.get(r, ''))
    return out


def rank_roi_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """ランク別 件数・的中率・回収率・ROI・収支。"""
    rows = []
    bands = [('S', ['S']), ('A', ['A']), ('B', ['B']), ('C', ['C']), ('S+A', ['S', 'A']), ('全部', ['S', 'A', 'B', 'C', ''])]
    for label, ranks in bands:
        if label == '全部':
            g = analysis
        else:
            g = analysis[analysis['勝負ランク'].isin(ranks)]
        n = len(g)
        inv = float(g['investment'].sum()) if n else 0.0
        pay = float(g['payout'].sum()) if n else 0.0
        hits = int(g['hit'].sum()) if n else 0
        profit = pay - inv
        recovery = round(pay / inv * 100, 1) if inv else 0.0
        roi = round(profit / inv * 100, 1) if inv else 0.0
        rows.append({
            'rank': label,
            'label': RANK_LABELS.get(label, label),
            'count': n,
            'hits': hits,
            'hit_rate': round(hits / n * 100, 1) if n else 0.0,
            'recovery': recovery,
            'roi': roi,
            'profit': int(profit),
            'investment': int(inv),
            'payout': int(pay),
        })
    return pd.DataFrame(rows)


def current_rank_map() -> dict[str, str]:
    """保存済み predictions から race_id → 勝負ランク。"""
    meta = {}
    for _, df in _prediction_frames():
        if '勝負ランク' not in df.columns:
            continue
        for _, row in df.iterrows():
            rid = str(row.get('race_id', '')).strip()
            if rid and rid not in meta:
                meta[rid] = str(row.get('勝負ランク', '') or '').upper()
    return meta


def evaluate_thresholds(thresholds: dict) -> dict:
    """候補閾値で predictions を再グレードし、analysis と突合して S+A 回収率を返す。"""
    if not ANALYSIS_CSV.exists():
        return {'ok': False, 'error': 'analysis_result.csv がありません'}
    analysis = pd.read_csv(ANALYSIS_CSV, encoding='utf-8-sig').fillna('')
    for c in ['hit', 'payout', 'investment', 'profit', 'roi']:
        if c in analysis.columns:
            analysis[c] = pd.to_numeric(analysis[c], errors='coerce').fillna(0)

    pred_rank = {}
    for _, df in _prediction_frames():
        # 再付与には BET期待値（表示値）ではなく基礎値を優先
        work = df.copy()
        if '買い期待度基礎値' in work.columns and work['買い期待度基礎値'].astype(str).str.len().gt(0).any():
            work['BET期待値'] = pd.to_numeric(work['買い期待度基礎値'], errors='coerce')
        elif 'BET期待値' in work.columns:
            work['BET期待値'] = pd.to_numeric(work['BET期待値'], errors='coerce')
        else:
            continue
        work = work.dropna(subset=['BET期待値'])
        if work.empty:
            continue
        graded = assign_ai_ranks(work, thresholds)
        for _, row in graded.iterrows():
            rid = str(row.get('race_id', '')).strip()
            if rid:
                pred_rank[rid] = str(row.get('勝負ランク', '')).upper()

    joined = _attach_rank_to_analysis(analysis, pred_rank)
    summary = rank_roi_summary(joined)
    sa = summary[summary['rank'] == 'S+A']
    s_only = summary[summary['rank'] == 'S']
    metric = float(sa['recovery'].iloc[0]) if len(sa) else 0.0
    s_recovery = float(s_only['recovery'].iloc[0]) if len(s_only) else 0.0
    # Sが極端に少ない閾値は避ける: 件数ペナルティ
    sa_count = int(sa['count'].iloc[0]) if len(sa) else 0
    score = metric + min(sa_count, 40) * 0.05 + max(0.0, s_recovery - 90) * 0.1
    return {
        'ok': True,
        'score': score,
        's_plus_a_recovery': metric,
        's_recovery': s_recovery,
        'summary': summary,
        'thresholds': thresholds,
    }


def search_best_thresholds(write: bool = False) -> dict:
    """簡易グリッド探索で S+A 回収率を最大化する閾値を探す。"""
    base = load_rank_thresholds()
    candidates = [dict(base)]
    for s_n in (1, 2, 3):
        for a_n in (3, 4, 5, 6, 8):
            if a_n < s_n:
                continue
            for b_pct in (0.25, 0.35, 0.45):
                candidates.append({
                    **base,
                    'mode': 'percentile',
                    's_top_n': s_n,
                    'a_top_n': a_n,
                    'b_percentile': b_pct,
                    'b_min_n': max(6, a_n + 2),
                })

    best = None
    reports = []
    seen = set()
    for th in candidates:
        key = (th['mode'], th['s_top_n'], th['a_top_n'], round(th['b_percentile'], 3))
        if key in seen:
            continue
        seen.add(key)
        ev = evaluate_thresholds(th)
        if not ev.get('ok'):
            return ev
        reports.append(ev)
        if best is None or ev['score'] > best['score']:
            best = ev

    assert best is not None
    cfg = _load_config()
    result = {
        'ok': True,
        'best_thresholds': best['thresholds'],
        'best_score': best['score'],
        's_plus_a_recovery': best['s_plus_a_recovery'],
        'summary': best['summary'].to_dict('records'),
        'tried': len(reports),
        'written': False,
    }
    if write:
        cfg['rank_thresholds'] = {
            **DEFAULT_RANK_THRESHOLDS,
            **best['thresholds'],
            'updated_by': 'rank_optimizer',
            'last_s_plus_a_recovery': best['s_plus_a_recovery'],
        }
        _save_config(cfg)
        result['written'] = True
        # 学習ログ
        log_path = DATA / 'rank_optimizer_log.csv'
        best['summary'].to_csv(log_path, index=False, encoding='utf-8-sig')
    return result


def report_current() -> pd.DataFrame:
    if not ANALYSIS_CSV.exists():
        return pd.DataFrame()
    analysis = pd.read_csv(ANALYSIS_CSV, encoding='utf-8-sig').fillna('')
    for c in ['hit', 'payout', 'investment', 'profit', 'roi']:
        if c in analysis.columns:
            analysis[c] = pd.to_numeric(analysis[c], errors='coerce').fillna(0)
    joined = _attach_rank_to_analysis(analysis, current_rank_map())
    return rank_roi_summary(joined)


def main():
    parser = argparse.ArgumentParser(description='AIランク閾値のROI学習')
    parser.add_argument('--report', action='store_true', help='現行ランクのROIを表示')
    parser.add_argument('--optimize', action='store_true', help='閾値を探索')
    parser.add_argument('--write', action='store_true', help='最適閾値をconfigへ書き込み')
    args = parser.parse_args()

    if args.report or not (args.optimize or args.write):
        summary = report_current()
        if summary.empty:
            print('analysis_result.csv がありません。先に python3 results.py を実行してください。')
            return
        print('\n==== AIランク別 ROI ====')
        print(summary.to_string(index=False))
        print('\n閾値:', json.dumps(load_rank_thresholds(), ensure_ascii=False))

    if args.optimize or args.write:
        out = search_best_thresholds(write=args.write)
        if not out.get('ok'):
            print('❌', out.get('error'))
            return
        print('\n==== 最適化結果 ====')
        print(json.dumps({
            'best_thresholds': out['best_thresholds'],
            's_plus_a_recovery': out['s_plus_a_recovery'],
            'best_score': out['best_score'],
            'tried': out['tried'],
            'written': out['written'],
        }, ensure_ascii=False, indent=2))
        print(pd.DataFrame(out['summary']).to_string(index=False))


if __name__ == '__main__':
    main()
