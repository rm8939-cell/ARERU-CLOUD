#!/usr/bin/env python3
"""新旧予想ロジックの公平比較バックテスト。

旧: ARERU_LEGACY_SCORE=1（タイム/着差/馬場/騎手/馬体重補正なし）
新: 既定（past_five + 詳細履歴 + race_sim プロファイル）

同一 SIM エンジン・同一日・同一レース・同一オッズ（results.csv 確定オッズ）で BUY 成績を比較する。
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
CACHE = DATA / 'logic_compare_cache'
STAKE = 100  # 1点100円


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


def _load_history() -> pd.DataFrame:
    h = pd.read_csv(DATA / 'all_history.csv', encoding='utf-8-sig')
    from areru_engine import parse_date, clean_name, expand_scoring_history
    h['_date'] = parse_date(h['年月日'])
    h['_horse'] = h['馬名'].map(clean_name)
    return expand_scoring_history(h)


def _load_results() -> pd.DataFrame:
    p = DATA / 'results.csv'
    if not p.exists():
        raise FileNotFoundError('data/results.csv がありません')
    r = pd.read_csv(p, encoding='utf-8-sig')
    r['race_id'] = r['race_id'].astype(str)
    r['date'] = r['date'].astype(str)
    r['馬名'] = r['馬名'].astype(str).str.strip()
    r['着順'] = pd.to_numeric(r['着順'], errors='coerce')
    r['人気'] = pd.to_numeric(r['人気'], errors='coerce')
    r['確定オッズ'] = pd.to_numeric(r['確定オッズ'], errors='coerce')
    return r


def _result_dates() -> set[str]:
    return set(_load_results()['date'].astype(str).unique())


def _eligible_dates(all_dates: list[str]) -> list[str]:
    """結果照合可能な日のみ（未来日・未確定日を除外）。"""
    have = _result_dates()
    return [d for d in all_dates if d in have]


def _predict_for_date(target: str, legacy: bool, history: pd.DataFrame, *, sim_runs: int, use_cache: bool) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """1日分の predictions + scores を生成（キャッシュ可）。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = 'legacy' if legacy else 'new'
    cache_pred = CACHE / f'predictions_{tag}_{target}.csv'
    cache_scores = CACHE / f'scores_{tag}_{target}.csv'
    if use_cache and cache_pred.exists():
        scores = pd.read_csv(cache_scores, encoding='utf-8-sig') if cache_scores.exists() else None
        return pd.read_csv(cache_pred, encoding='utf-8-sig'), scores

    os.environ['ARERU_LEGACY_SCORE'] = '1' if legacy else '0'
    os.environ['ARERU_SIM_RUNS'] = str(sim_runs)
    from replay_predict import load_runners, run_date
    runners = load_runners()
    run_date(target, runners, history)
    src = DATA / 'predictions_by_date' / f'predictions_{target}.csv'
    scores_src = DATA / 'predictions_by_date' / f'scores_{target}.csv'
    df = pd.read_csv(src, encoding='utf-8-sig')
    df.to_csv(cache_pred, index=False, encoding='utf-8-sig')
    scores = None
    if scores_src.exists():
        scores = pd.read_csv(scores_src, encoding='utf-8-sig')
        scores.to_csv(cache_scores, index=False, encoding='utf-8-sig')
    return df, scores


def _ai_rank_for_honmei(row, scores: pd.DataFrame | None) -> float | None:
    """本命馬のAI指数順位（scores CSV）。"""
    if scores is None or scores.empty:
        return None
    rid = str(row.get('race_id', ''))
    horse = str(row.get('本命', '')).strip()
    g = scores[scores['race_id'].astype(str) == rid]
    if g.empty or 'AREru指数' not in g.columns:
        return None
    g = g.copy()
    g['_rank'] = g['AREru指数'].rank(ascending=False, method='first')
    hit = g[g['馬名'].astype(str).str.strip() == horse]
    if hit.empty:
        return None
    try:
        return float(hit.iloc[0]['_rank'])
    except (TypeError, ValueError):
        return None


def _evaluate_bets(pred: pd.DataFrame, results: pd.DataFrame, date: str, scores: pd.DataFrame | None = None) -> list[dict]:
    """BUY レースごとに結果照合。"""
    from areru_engine import clean_name
    day_res = results[results['date'] == date]
    rows = []
    buys = pred[pred['投資判定'].astype(str).str.startswith('買い')]
    for _, row in buys.iterrows():
        rid = str(row.get('race_id', ''))
        horse = str(row.get('本命', '')).strip()
        ev = pd.to_numeric(row.get('期待値'), errors='coerce')
        pop = pd.to_numeric(row.get('本命人気'), errors='coerce')
        pred_odds = pd.to_numeric(row.get('本命オッズ'), errors='coerce')

        rr = day_res[(day_res['race_id'] == rid) & (day_res['馬名'] == horse)]
        if rr.empty:
            # 馬名の空白差異フォールバック
            rr = day_res[(day_res['race_id'] == rid) & (day_res['馬名'].map(clean_name) == clean_name(horse))]
        matched = not rr.empty
        finish = pd.to_numeric(rr.iloc[0]['着順'], errors='coerce') if matched else float('nan')
        res_pop = pd.to_numeric(rr.iloc[0]['人気'], errors='coerce') if matched else float('nan')
        res_odds = pd.to_numeric(rr.iloc[0]['確定オッズ'], errors='coerce') if matched else float('nan')
        odds = res_odds if pd.notna(res_odds) else pred_odds
        hit = bool(matched and pd.notna(finish) and finish == 1)
        payout = float(odds) * STAKE if hit and pd.notna(odds) else 0.0

        ai_rank = _ai_rank_for_honmei(row, scores)
        pop_val = pop if pd.notna(pop) else res_pop
        upgraded = False
        if ai_rank is not None and pd.notna(pop_val):
            upgraded = float(ai_rank) < float(pop_val)  # AI順位の方が上位 = 人気より評価↑

        rows.append({
            'date': date,
            'race_id': rid,
            '本命': horse,
            '期待値': float(ev) if pd.notna(ev) else None,
            '本命人気': float(pop_val) if pd.notna(pop_val) else None,
            'AI順位': ai_rank,
            '人気より評価UP': upgraded,
            '的中': hit,
            '払戻': payout,
            '照合': matched,
            '着順': float(finish) if pd.notna(finish) else None,
            'オッズ': float(odds) if pd.notna(odds) else None,
        })
    return rows


def _summarize(bets: list[dict], pred_total_races: int) -> dict:
    if not bets:
        return {
            '予想件数': pred_total_races,
            'BUY件数': 0,
            '的中件数': 0,
            '的中率': 0.0,
            '投資額': 0,
            '払戻': 0.0,
            'ROI': 0.0,
            '回収率': 0.0,
            '平均EV': None,
            'BUY馬平均人気': None,
            '人気より評価UP件数': 0,
            '人気より評価UP的中率': None,
            '照合率': 0.0,
        }
    df = pd.DataFrame(bets)
    n = len(df)
    hits = int(df['的中'].sum())
    invest = n * STAKE
    ret = float(df['払戻'].sum())
    roi_pct = (ret / invest * 100) if invest else 0.0
    matched = int(df['照合'].sum()) if '照合' in df.columns else n
    up = df[df['人気より評価UP'] == True]  # noqa: E712
    up_hits = int(up['的中'].sum()) if len(up) else 0
    return {
        '予想件数': pred_total_races,
        'BUY件数': n,
        '的中件数': hits,
        '的中率': round(hits / n * 100, 2) if n else 0.0,
        '投資額': invest,
        '払戻': round(ret, 0),
        'ROI': round(roi_pct - 100, 2),
        '回収率': round(roi_pct, 2),
        '平均EV': round(float(df['期待値'].mean()), 2) if df['期待値'].notna().any() else None,
        'BUY馬平均人気': round(float(df['本命人気'].mean()), 2) if df['本命人気'].notna().any() else None,
        '人気より評価UP件数': int(len(up)),
        '人気より評価UP的中率': round(up_hits / len(up) * 100, 2) if len(up) else None,
        '照合率': round(matched / n * 100, 1) if n else 0.0,
    }


def run_fair_backtest(dates: list[str], *, sim_runs: int = 5000, use_cache: bool = True) -> dict:
    history = _load_history()
    results = _load_results()
    old_bets: list[dict] = []
    new_bets: list[dict] = []
    old_races = 0
    new_races = 0
    daily = []

    for d in dates:
        print(f'[backtest] {d} ...', flush=True)
        try:
            old_pred, old_scores = _predict_for_date(d, legacy=True, history=history, sim_runs=sim_runs, use_cache=use_cache)
            new_pred, new_scores = _predict_for_date(d, legacy=False, history=history, sim_runs=sim_runs, use_cache=use_cache)
        except Exception as e:
            print(f'  skip {d}: {e}', flush=True)
            daily.append({'date': d, 'error': str(e)[:120]})
            continue

        ob = _evaluate_bets(old_pred, results, d, scores=old_scores)
        nb = _evaluate_bets(new_pred, results, d, scores=new_scores)
        old_bets.extend(ob)
        new_bets.extend(nb)
        old_races += len(old_pred)
        new_races += len(new_pred)
        daily.append({
            'date': d,
            '旧': _summarize(ob, len(old_pred)),
            '新': _summarize(nb, len(new_pred)),
        })

    old_sum = _summarize(old_bets, old_races)
    new_sum = _summarize(new_bets, new_races)
    return {
        '検証日': dates,
        '検証日数': len(dates),
        'SIM_RUNS': sim_runs,
        '旧ロジック': old_sum,
        '新ロジック': new_sum,
        'ROI改善幅': round((new_sum.get('ROI') or 0) - (old_sum.get('ROI') or 0), 2),
        '回収率改善幅': round((new_sum.get('回収率') or 0) - (old_sum.get('回収率') or 0), 2),
        '的中率改善幅': round((new_sum.get('的中率') or 0) - (old_sum.get('的中率') or 0), 2),
        '日別': daily,
        '旧BUY詳細': old_bets,
        '新BUY詳細': new_bets,
    }


def compare_race(target: str, race_id: str | None = None) -> dict:
    """1日のレース単位新旧比較（概要）。"""
    history = _load_history()
    old_pred, _ = _predict_for_date(target, legacy=True, history=history, sim_runs=5000, use_cache=True)
    new_pred, _ = _predict_for_date(target, legacy=False, history=history, sim_runs=5000, use_cache=True)
    if race_id:
        old_pred = old_pred[old_pred['race_id'].astype(str) == str(race_id)]
        new_pred = new_pred[new_pred['race_id'].astype(str) == str(race_id)]
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
        'レース別比較': race_rows,
        '旧BUY件数': int((old_pred['投資判定'].astype(str).str.startswith('買い')).sum()),
        '新BUY件数': int((new_pred['投資判定'].astype(str).str.startswith('買い')).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='2026-08-22', help='比較対象日')
    ap.add_argument('--race-id', default='')
    ap.add_argument('--backtest-days', type=int, default=0, help='0=結果照合可能な全期間')
    ap.add_argument('--sim-runs', type=int, default=5000)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--fair-only', action='store_true', help='公平バックテストのみ')
    args = ap.parse_args()

    from replay_predict import load_runners, available_dates
    runners = load_runners()
    all_dates = available_dates(runners)
    eligible = _eligible_dates(all_dates)
    if args.backtest_days > 0:
        bt_dates = eligible[-args.backtest_days:]
    else:
        bt_dates = eligible

    print(f'[info] 全開催日 {len(all_dates)} / 結果照合可能 {len(eligible)} / 検証対象 {len(bt_dates)}', flush=True)
    print(f'[info] 検証日: {bt_dates}', flush=True)

    fair = run_fair_backtest(bt_dates, sim_runs=args.sim_runs, use_cache=not args.no_cache)
    report = _json_safe({
        'fair_backtest': fair,
        'compare': None if args.fair_only else compare_race(args.date, args.race_id or None),
    })

    # サマリー出力（詳細BUYリストはファイルのみ）
    summary = {
        '検証日': bt_dates,
        '旧ロジック': fair['旧ロジック'],
        '新ロジック': fair['新ロジック'],
        'ROI改善幅': fair['ROI改善幅'],
        '回収率改善幅': fair['回収率改善幅'],
        '的中率改善幅': fair['的中率改善幅'],
        '日別': fair['日別'],
    }
    OUT.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'\n📁 {OUT}')


if __name__ == '__main__':
    main()
