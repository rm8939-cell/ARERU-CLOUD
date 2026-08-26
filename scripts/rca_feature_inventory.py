#!/usr/bin/env python3
"""予想エンジン RCA: 特徴量インベントリ / リーク検査 / スコア差分診断。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
OUT = DATA / 'rca_feature_inventory.json'


def _fill_rate(s: pd.Series) -> float:
    x = s.astype(str).str.strip()
    ok = x.notna() & (x != '') & (x.str.lower() != 'nan') & (x != '--') & (x != 'None')
    return float(ok.mean() * 100) if len(s) else 0.0


def inventory() -> dict:
    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    history = pd.read_csv(DATA / 'all_history.csv', encoding='utf-8-sig', low_memory=False)

    used_in_score = {
        '着順1-5': 'score_runner performance/upset/consistency/trend',
        '人気1-5': 'score_runner upset/value',
        '場1-5 / レース名1-5': 'NAR→JRA scale + context',
        '単勝オッズ / 人気': 'value 市場補正 + BUY/EV',
        'タイム1-5': 'history_detail_bonus + race_sim time (新のみ)',
        '着差1-5': 'history_detail_bonus + race_sim margin (新のみ)',
        '馬場1-5': 'history_detail_bonus + race_sim track (新のみ)',
        '枠': 'race_sim gate_bias',
        '騎手': 'race_sim jockey_bonus (新のみ)',
        'source / race_id': '会場・JRA/NAR判定',
    }
    used_in_history = {
        '年月日/馬名': '日付フィルタ (_date < target)',
        '距離/場': 'course_distance_fit + context',
        '頭数': 'context field-size',
        '着順/人気': 'context + jockey stats',
        '騎手': 'jockey_bonus',
        '馬体重': 'weight_delta (新のみ)',
        'タイム/着差/馬場': 'detail bonus + SIM',
        '斤量': '未使用（取得のみ）',
        '通過/ペース/上り': '馬キャッシュにあれば未使用',
    }
    unused_acquired = [
        'runners.斤量（PRESET=X の burden でのみ指数加点。本番旧では未使用）',
        'runners.枠（段階SIMの gate_bias のみ。旧ガウスでは PRESET=X の sgate）',
        'runners.実着順（結果検証用・スコア未使用・リーク禁止）',
        'runners.オッズ更新日時',
        'all_history.斤量',
        'all_history.馬体重（段階SIMの weight のみ。旧では PRESET=X の sweight）',
        'all_history.騎手（段階SIMの jockey のみ。旧では PRESET=X の sjockey）',
        'all_history.頭数（context で弱使用。PRESET=X の sfield で強化）',
        'all_history.今回レース',
        '脚質（infer_style は段階SIMのみ。旧では PRESET=X の sstyle）',
        '休み明け（段階SIMの layoff のみ。旧では PRESET=X の slayoff）',
        'horse_cache.通過 / ペース / 上り（キャッシュがある場合は未使用）',
    ]

    runner_fill = {}
    for c in runners.columns:
        runner_fill[c] = round(_fill_rate(runners[c]), 2)
    hist_fill = {c: round(_fill_rate(history[c]), 2) for c in history.columns}

    detail_cols = [c for c in runners.columns if any(x in c for x in ('タイム', '着差', '馬場'))]
    return {
        'runners_rows': int(len(runners)),
        'history_rows': int(len(history)),
        'runner_detail_cols_present': detail_cols,
        'runner_fill_pct': runner_fill,
        'history_fill_pct': hist_fill,
        'used_in_score_runner_or_sim': used_in_score,
        'used_in_all_history': used_in_history,
        'acquired_but_unused': unused_acquired,
        'old_vs_new_score_diff': {
            '旧': '6因子のみ + ガウスSIM（追加特徴OFF）',
            '新': '6因子 + history_detail_bonus*0.12 + 段階SIM(騎手/馬体重/タイム/着差/馬場/コース/枠/休み)',
            '共通': 'context_features / NAR換算 / 市場オッズvalue補正',
        },
    }


def leak_checks() -> dict:
    from areru_engine import parse_date, clean_name
    history = pd.read_csv(DATA / 'all_history.csv', encoding='utf-8-sig', low_memory=False)
    history['_date'] = parse_date(history['年月日'])
    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    runners['_date'] = parse_date(runners['日付'])

    # same-day history rows relative to any runner date
    issues = []
    sample_dates = sorted(runners['_date'].dropna().dt.normalize().unique())[-5:]
    same_day = 0
    for d in sample_dates:
        n = int(((history['_date'].dt.normalize() == d)).sum())
        same_day += n
    # 実着順 fill = post-race contamination risk if ever used
    finish_fill = _fill_rate(runners['実着順']) if '実着順' in runners.columns else 0.0

    return {
        'history_filter_contract': '_date < target（score_runner / history_detail / race_sim）',
        'same_day_history_rows_on_last5_runner_dates': same_day,
        '実着順_fill_pct': round(finish_fill, 2),
        '実着順_used_in_score': False,
        'market_odds_note': '当日単勝オッズは意図的特徴。確定オッズ混入時は軽微リークになり得る',
        'issues': issues,
    }


def score_delta_sample(date: str, sim_runs: int = 500) -> dict:
    """1日分の旧新スコア差分サンプル。"""
    os.environ['ARERU_SIM_RUNS'] = str(sim_runs)
    from scripts.logic_compare_backtest import _load_history, _predict_for_date
    history = _load_history()
    old_pred, old_scores = _predict_for_date(date, legacy=True, history=history, sim_runs=sim_runs, use_cache=False)
    new_pred, new_scores = _predict_for_date(date, legacy=False, history=history, sim_runs=sim_runs, use_cache=False)
    out = {
        'date': date,
        '旧BUY': int(old_pred['投資判定'].astype(str).str.startswith('買い').sum()),
        '新BUY': int(new_pred['投資判定'].astype(str).str.startswith('買い').sum()),
    }
    if old_scores is not None and new_scores is not None and not old_scores.empty and not new_scores.empty:
        m = old_scores.merge(new_scores, on=['race_id', '馬名'], suffixes=('_旧', '_新'))
        if 'AREru指数_旧' in m.columns and 'AREru指数_新' in m.columns:
            d = m['AREru指数_新'].astype(float) - m['AREru指数_旧'].astype(float)
            out['score_delta'] = {
                'mean': round(float(d.mean()), 3),
                'std': round(float(d.std()), 3),
                'pct_changed_gt_1': round(float((d.abs() > 1).mean() * 100), 2),
                'pct_changed_gt_3': round(float((d.abs() > 3).mean() * 100), 2),
            }
        # 本命差分
        op = old_pred.set_index('race_id')['本命'].astype(str)
        np_ = new_pred.set_index('race_id')['本命'].astype(str)
        common = op.index.intersection(np_.index)
        from areru_engine import clean_name
        diff = sum(1 for rid in common if clean_name(op.loc[rid]) != clean_name(np_.loc[rid]))
        out['本命不一致レース'] = diff
        out['本命不一致率'] = round(diff / max(len(common), 1) * 100, 2)
    return out


def main():
    report = {
        'inventory': inventory(),
        'leak_checks': leak_checks(),
    }
    # pick a mid date with races
    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    dates = sorted(runners['日付'].astype(str).unique())
    sample = dates[len(dates) // 2] if dates else None
    if sample:
        print(f'[rca] score delta sample date={sample}', flush=True)
        try:
            report['score_delta_sample'] = score_delta_sample(sample, sim_runs=800)
        except Exception as e:
            report['score_delta_sample'] = {'error': str(e)}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print(f'\n📁 {OUT}')


if __name__ == '__main__':
    main()
