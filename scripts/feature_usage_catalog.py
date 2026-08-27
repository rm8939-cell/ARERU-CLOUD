#!/usr/bin/env python3
"""特徴量利用状況の洗い出し + 全本命の勝率キャリブレーション。

本番ロジックは変更しない。BUY閾値も触らない。
目的は「同じBUY条件のまま、推定勝率・期待値・ランキング精度を改善できるか」の調査。

出力:
  data/feature_usage_catalog.json / .csv
  data/calibration_report.json
  data/unused_feature_signal.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
CACHE = DATA / 'rca_logic_cache'
OUT_CATALOG = DATA / 'feature_usage_catalog.json'
OUT_CSV = DATA / 'feature_usage_catalog.csv'
OUT_CALIB = DATA / 'calibration_report.json'
OUT_SIGNAL = DATA / 'unused_feature_signal.csv'

SPARSE_FILL_MAX = 25.0  # % 未満は現状使えない
USABLE_FILL_MIN = 70.0
RESIDUAL_MIN_N = 80
RESIDUAL_MIN_ABS_RHO = 0.04


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (obj != obj) else float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _fill_rate(s: pd.Series) -> float:
    x = s.astype(str).str.strip()
    ok = x.notna() & (x != '') & (x.str.lower() != 'nan') & (x != '--') & (x != 'None')
    return float(ok.mean() * 100) if len(s) else 0.0


def _num(v):
    return pd.to_numeric(v, errors='coerce')


def _dates(logic: str) -> list[str]:
    return sorted({p.name.split('_')[-1].replace('.csv', '') for p in CACHE.glob(f'pred_{logic}_*.csv')})


def _split(dates: list[str]):
    cut = max(1, int(len(dates) * 0.70))
    return dates[:cut], dates[cut:]


def _odds_band(o) -> str:
    if o is None or (isinstance(o, float) and (o != o)):
        return '不明'
    x = float(o)
    if x < 3:
        return '1.0-2.9'
    if x < 5:
        return '3.0-4.9'
    if x < 8:
        return '5.0-7.9'
    if x < 12:
        return '8.0-11.9'
    if x < 20:
        return '12.0-19.9'
    if x < 40:
        return '20.0-39.9'
    return '40.0+'


def _pop_band(p) -> str:
    if p is None or (isinstance(p, float) and (p != p)):
        return '不明'
    x = float(p)
    if x <= 1:
        return '1番人気'
    if x <= 3:
        return '2-3番人気'
    if x <= 6:
        return '4-6番人気'
    if x <= 9:
        return '7-9番人気'
    return '10番人気以下'


def _win_band(w) -> str:
    if w is None or (isinstance(w, float) and (w != w)):
        return '不明'
    x = float(w)
    if x < 6:
        return '<6%'
    if x < 10:
        return '6-10%'
    if x < 15:
        return '10-15%'
    if x < 22:
        return '15-22%'
    return '22%+'


def _ece_brier(p: np.ndarray, y: np.ndarray, n_bins: int = 8) -> dict:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    n = int(len(p))
    if n == 0:
        return {'n': 0, 'brier': None, 'ece': None, 'mean_p': None, 'mean_y': None, 'bins': []}
    brier = float(np.mean((p - y) ** 2))
    # quantile bins (empty-safe)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(p, qs))
    if len(edges) < 3:
        edges = np.linspace(float(p.min()), float(p.max()) + 1e-9, min(n_bins, n) + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    bins = []
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        mp = float(p[m].mean())
        my = float(y[m].mean())
        w = float(m.mean())
        ece += w * abs(mp - my)
        bins.append({
            'bin': b,
            'n': int(m.sum()),
            'mean_p': round(mp, 4),
            'mean_y': round(my, 4),
            'gap': round(my - mp, 4),
        })
    return {
        'n': n,
        'brier': round(brier, 6),
        'ece': round(ece, 6),
        'mean_p': round(float(p.mean()), 4),
        'mean_y': round(float(y.mean()), 4),
        'overconfidence': round(float(p.mean() - y.mean()), 4),
        'bins': bins,
    }


def _spearman(a, b) -> float | None:
    s = np.asarray(pd.to_numeric(a, errors='coerce'), dtype=float)
    t = np.asarray(pd.to_numeric(b, errors='coerce'), dtype=float)
    m = np.isfinite(s) & np.isfinite(t)
    if int(m.sum()) < 20:
        return None
    rs = pd.Series(s[m]).rank().to_numpy(dtype=float)
    rt = pd.Series(t[m]).rank().to_numpy(dtype=float)
    rs = rs - rs.mean()
    rt = rt - rt.mean()
    den = float(np.sqrt((rs ** 2).sum() * (rt ** 2).sum()))
    if den < 1e-12:
        return None
    return float((rs * rt).sum() / den)


def _code_catalog() -> list[dict]:
    """コード経路に基づく静的インベントリ。測定値は後で付与。"""
    return [
        {
            'id': 'past_finish',
            'name': '着順1-5',
            'source': 'runners.csv',
            'old': 'used',
            'new': 'used',
            'path': 'score_runner performance/consistency/trend',
        },
        {
            'id': 'past_pop',
            'name': '人気1-5',
            'source': 'runners.csv',
            'old': 'used',
            'new': 'used',
            'path': 'score_runner upset/value + last3f/style proxy',
        },
        {
            'id': 'past_venue',
            'name': '場1-5 / レース名1-5',
            'source': 'runners.csv',
            'old': 'used',
            'new': 'used',
            'path': 'NAR→JRA scale + context_features',
        },
        {
            'id': 'market_odds',
            'name': '単勝オッズ / 人気',
            'source': 'runners.csv',
            'old': 'used',
            'new': 'used',
            'path': 'value市場補正 + SIM市場収縮 + EV/BUY',
        },
        {
            'id': 'source_venue',
            'name': 'source / race_id / 開催地',
            'source': 'runners.csv',
            'old': 'used',
            'new': 'used',
            'path': 'JRA/NAR判定・会場名',
        },
        {
            'id': 'hist_date_horse',
            'name': '年月日 / 馬名',
            'source': 'all_history.csv',
            'old': 'used',
            'new': 'used',
            'path': '_date < target フィルタ',
        },
        {
            'id': 'hist_dist_venue',
            'name': '距離 / 場',
            'source': 'all_history.csv',
            'old': 'used',
            'new': 'used',
            'path': 'context 距離・会場適性。NEWは course_distance_fit も',
        },
        {
            'id': 'hist_heads',
            'name': '頭数',
            'source': 'all_history.csv',
            'old': 'used_weak',
            'new': 'used_weak',
            'path': 'context 頭数減経験のみ。PRESET=X の sfield で強化（holdout失敗）',
        },
        {
            'id': 'hist_finish_pop',
            'name': '着順 / 人気（履歴）',
            'source': 'all_history.csv',
            'old': 'used',
            'new': 'used',
            'path': 'context + NEW jockey_bonus',
        },
        {
            'id': 'consistency_sigma',
            'name': '因子_consistency → ガウスσ',
            'source': 'derived',
            'old': 'used',
            'new': 'unused_in_stage',
            'path': 'OLD simulate_race の sigma。NEW段階SIMでは使わない',
        },
        {
            'id': 'runners_time',
            'name': 'タイム1-5',
            'source': 'runners.csv',
            'old': 'unused',
            'new': 'used_if_filled',
            'path': 'history_detail_bonus / time_trend_bonus。本番OLDは ablation OFF',
        },
        {
            'id': 'runners_margin',
            'name': '着差1-5',
            'source': 'runners.csv',
            'old': 'unused',
            'new': 'used_if_filled',
            'path': 'history_detail_bonus / margin_bonus_from_row。本番OLD OFF',
        },
        {
            'id': 'runners_track',
            'name': '馬場1-5',
            'source': 'runners.csv',
            'old': 'unused',
            'new': 'used_if_filled',
            'path': 'history_detail_bonus / track_condition_bonus。本番OLD OFF',
        },
        {
            'id': 'hist_time',
            'name': 'タイム（履歴）',
            'source': 'all_history.csv',
            'old': 'unused',
            'new': 'used_fallback',
            'path': 'runners欠損時の NEW フォールバック。OLD未使用',
        },
        {
            'id': 'hist_margin',
            'name': '着差（履歴）',
            'source': 'all_history.csv',
            'old': 'unused',
            'new': 'used_fallback',
            'path': 'NEW 着差ボーナス。OLD未使用',
        },
        {
            'id': 'hist_track',
            'name': '馬場（履歴）',
            'source': 'all_history.csv',
            'old': 'unused',
            'new': 'used',
            'path': 'NEW track_condition_bonus。OLD未使用',
        },
        {
            'id': 'runners_jockey',
            'name': '騎手',
            'source': 'runners.csv',
            'old': 'unused',
            'new': 'used',
            'path': 'NEW jockey_bonus。OLDは PRESET=X sjockey のみ（holdout失敗）',
        },
        {
            'id': 'hist_jockey',
            'name': '騎手（履歴）',
            'source': 'all_history.csv',
            'old': 'unused',
            'new': 'used',
            'path': 'NEW 騎手複勝率。OLD未使用',
        },
        {
            'id': 'hist_weight',
            'name': '馬体重',
            'source': 'all_history.csv',
            'old': 'unused',
            'new': 'used',
            'path': 'NEW weight_delta。OLDは PRESET=X sweight（holdout失敗）',
        },
        {
            'id': 'runners_waku',
            'name': '枠',
            'source': 'runners.csv',
            'old': 'unused',
            'new': 'used',
            'path': 'NEW gate_bias。OLDガウスは無視。PRESET=X sgate はholdout失敗',
        },
        {
            'id': 'runners_kg',
            'name': '斤量（当日）',
            'source': 'runners.csv',
            'old': 'unused',
            'new': 'used',
            'path': 'NEW フィールド平均比。OLDは PRESET=X burden（holdout失敗）',
        },
        {
            'id': 'hist_kg',
            'name': '斤量（履歴）',
            'source': 'all_history.csv',
            'old': 'unused',
            'new': 'unused',
            'path': '取得済み。予測ロジック未配線',
        },
        {
            'id': 'layoff',
            'name': '休み明け日数',
            'source': 'derived from all_history',
            'old': 'unused',
            'new': 'used',
            'path': 'NEW layoff。OLDは PRESET=X slayoff（holdout失敗）',
        },
        {
            'id': 'style_proxy',
            'name': '脚質（着順×人気代理）',
            'source': 'derived',
            'old': 'unused',
            'new': 'used',
            'path': 'NEW infer_style。実通過/上りではない。OLDは PRESET=X sstyle',
        },
        {
            'id': 'last3f_proxy',
            'name': '上がり3F代理（着順×人気）',
            'source': 'derived',
            'old': 'unused',
            'new': 'used_proxy',
            'path': 'NEW last3f_proxy。実上がり3Fは未取得',
        },
        {
            'id': 'blood_stable',
            'name': '血統 / 厩舎',
            'source': 'hardcoded',
            'old': 'unused',
            'new': 'placeholder',
            'path': 'build_profiles で blood=50 / stable≈騎手連動。データ未取得',
        },
        {
            'id': 'pass_pace_last3f',
            'name': '通過 / ペース / 上り',
            'source': 'netkeiba extra headers',
            'old': 'missing',
            'new': 'missing',
            'path': '取得コードはあるが all_history に列なし。horse_results キャッシュ空',
        },
        {
            'id': 'hist_this_race',
            'name': '今回レース',
            'source': 'all_history.csv',
            'old': 'unused',
            'new': 'unused',
            'path': 'フラグ列。スコア未使用',
        },
        {
            'id': 'runners_finish_today',
            'name': '実着順',
            'source': 'runners.csv',
            'old': 'leak_forbidden',
            'new': 'leak_forbidden',
            'path': '事後結果。スコア投入禁止',
        },
        {
            'id': 'odds_updated_at',
            'name': 'オッズ更新日時',
            'source': 'runners.csv',
            'old': 'unused',
            'new': 'unused',
            'path': 'メタデータ。予測未使用',
        },
        {
            'id': 'runners_umaban',
            'name': '馬番',
            'source': 'runners.csv',
            'old': 'display',
            'new': 'display',
            'path': '表示用。枠と別。予測未使用',
        },
        {
            'id': 'claimable_prob',
            'name': '補正勝率（EV）',
            'source': 'derived',
            'old': 'used',
            'new': 'used',
            'path': '_claimable_ai_prob + _edge_take_rate。表示EVの入力',
        },
        {
            'id': 'place_shrink',
            'name': '連対/複勝 shrink',
            'source': 'derived',
            'old': 'used',
            'new': 'used',
            'path': 'simulate_race でフィールド平均へ縮小。単勝は別途市場収縮',
        },
    ]


def _load_honmei(logic: str, results: pd.DataFrame, dates: list[str],
                 train_set: set[str], holdout_set: set[str]) -> pd.DataFrame:
    from areru_engine import clean_name
    from ev_analysis import score_horse_ev

    rows = []
    for d in dates:
        p = CACHE / f'pred_{logic}_{d}.csv'
        if not p.exists():
            continue
        pred = pd.read_csv(p, encoding='utf-8-sig')
        day_res = results[results['date'] == d]
        period = 'holdout' if d in holdout_set else 'train'
        for _, row in pred.iterrows():
            rid = str(row.get('race_id', ''))
            horse = str(row.get('本命') or '').strip()
            hn = clean_name(horse)
            rr = day_res[(day_res['race_id'].astype(str) == rid) & (day_res['馬名'].map(clean_name) == hn)]
            pred_odds = _num(row.get('本命オッズ'))
            odds = float(pred_odds) if pd.notna(pred_odds) else None
            hit = None
            finish = None
            fav_win = None
            winner_pop = None
            if not rr.empty:
                finish = float(_num(rr.iloc[0]['着順'])) if pd.notna(_num(rr.iloc[0]['着順'])) else None
                o = _num(rr.iloc[0]['確定オッズ'])
                if pd.notna(o):
                    odds = float(o)
                hit = 1.0 if finish == 1.0 else 0.0
            race_res = day_res[day_res['race_id'].astype(str) == rid]
            if not race_res.empty:
                win_row = race_res[_num(race_res['着順']) == 1]
                if not win_row.empty:
                    winner_pop = float(_num(win_row.iloc[0]['人気'])) if pd.notna(_num(win_row.iloc[0]['人気'])) else None
                    fav = race_res.loc[_num(race_res['人気']).idxmin()] if _num(race_res['人気']).notna().any() else None
                    if fav is not None:
                        fav_win = 1.0 if float(_num(fav['着順']) or 99) == 1.0 else 0.0
            sim = _num(row.get('シミュレーション勝率'))
            fair = _num(row.get('AI適正オッズ'))
            conf = _num(row.get('AI信頼度スコア'))
            repro = _num(row.get('シミュレーション再現率'))
            n_data = int(_num(row.get('本命データ件数')) or _num(row.get('データ件数')) or 0)
            apt = _num(row.get('能力差スコア'))
            reasons = str(row.get('本命理由') or '')
            market = float(pred_odds) if pd.notna(pred_odds) else odds
            scored = {}
            if market and pd.notna(sim):
                scored = score_horse_ev(
                    market, float(sim),
                    float(fair) if pd.notna(fair) else None,
                    float(conf) if pd.notna(conf) else 50.0,
                    float(repro) if pd.notna(repro) else 50.0,
                    n_data,
                    float(apt) if pd.notna(apt) else 50.0,
                    reasons,
                )
            adj = scored.get('補正勝率')
            implied = scored.get('市場暗示勝率')
            buy = str(row.get('投資判定') or '').startswith('買い')
            pop = _num(row.get('本命人気'))
            rows.append({
                'logic': logic,
                'date': d,
                'period': period,
                'race_id': rid,
                '本命': horse,
                'hit': hit,
                'finish': finish,
                'odds': odds,
                '人気': float(pop) if pd.notna(pop) else None,
                'sim_pct': float(sim) if pd.notna(sim) else None,
                'adj_pct': float(adj) if adj is not None else None,
                'implied_pct': float(implied) if implied is not None else None,
                'buy': buy,
                'fav_win': fav_win,
                'winner_pop': winner_pop,
                '指数': _num(row.get('本命AREru指数')),
                'sim3': _num(row.get('シミュレーション3着内率')),
                'source': str(row.get('source') or ''),
                '開催地': str(row.get('開催地') or ''),
            })
    return pd.DataFrame(rows)


def _attach_unused_features(honmei: pd.DataFrame) -> pd.DataFrame:
    from areru_engine import _margin_tightness, clean_name, parse_date
    from race_sim import infer_style, parse_time_sec, parse_weight, dist_meters

    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    runners['race_id'] = runners['race_id'].astype(str)
    runners['_horse'] = runners['馬名'].map(clean_name)
    hist = pd.read_csv(DATA / 'all_history.csv', encoding='utf-8-sig', low_memory=False)
    hist['_date'] = parse_date(hist['年月日'])
    hist['_horse'] = hist['馬名'].map(clean_name)
    hist = hist.dropna(subset=['_date', '_horse']).sort_values('_date', ascending=False)

    field_kg = runners.groupby('race_id')['斤量'].apply(lambda s: pd.to_numeric(s, errors='coerce').mean())
    field_n = runners.groupby('race_id').size()

    last_by_horse: dict[str, pd.DataFrame] = {}
    for horse, g in hist.groupby('_horse', sort=False):
        last_by_horse[horse] = g

    extra = []
    for _, row in honmei.iterrows():
        rid = str(row['race_id'])
        hn = clean_name(row['本命'])
        target = pd.Timestamp(row['date'])
        rg = runners[(runners['race_id'] == rid) & (runners['_horse'] == hn)]
        kg = waku = umaban = None
        finishes = pops = None
        jockey = ''
        t1 = m1 = tr1 = None
        if not rg.empty:
            g = rg.iloc[0]
            kg = float(_num(g.get('斤量'))) if pd.notna(_num(g.get('斤量'))) else None
            waku = float(_num(g.get('枠'))) if pd.notna(_num(g.get('枠'))) else None
            umaban = float(_num(g.get('馬番'))) if pd.notna(_num(g.get('馬番'))) else None
            jockey = str(g.get('騎手') or '').strip()
            finishes = np.array([_num(g.get(f'着順{i}')) for i in range(1, 6)], dtype=float)
            pops = np.array([_num(g.get(f'人気{i}')) for i in range(1, 6)], dtype=float)
            t1 = parse_time_sec(g.get('タイム1'))
            m1 = str(g.get('着差1') or '')
            tr1 = str(g.get('馬場1') or '')
        fkg = float(field_kg.get(rid)) if rid in field_kg.index and pd.notna(field_kg.get(rid)) else None
        kg_delta = (kg - fkg) if kg is not None and fkg is not None else None
        n_field = int(field_n.get(rid, 0))
        style = float(infer_style(finishes, pops)) if finishes is not None else float('nan')

        last_w = last_w2 = last_time = last_kg = last_heads = last_dist = last_track = None
        last_finish = last_pop = last_margin = layoff = None
        prev = last_by_horse.get(hn)
        if prev is not None:
            past = prev[prev['_date'] < target]
            if not past.empty:
                a = past.iloc[0]
                last_w = parse_weight(a.get('馬体重'))
                last_time = parse_time_sec(a.get('タイム'))
                last_kg = float(_num(a.get('斤量'))) if pd.notna(_num(a.get('斤量'))) else None
                last_heads = float(_num(a.get('頭数'))) if pd.notna(_num(a.get('頭数'))) else None
                last_dist = dist_meters(a.get('距離'))
                last_track = str(a.get('馬場') or '')
                last_finish = float(_num(a.get('着順'))) if pd.notna(_num(a.get('着順'))) else None
                last_pop = float(_num(a.get('人気'))) if pd.notna(_num(a.get('人気'))) else None
                last_margin = str(a.get('着差') or '')
                try:
                    layoff = float((target - pd.Timestamp(a['_date'])).days)
                except Exception:
                    layoff = None
                if len(past) >= 2:
                    last_w2 = parse_weight(past.iloc[1].get('馬体重'))
        wdelta = None
        if last_w is not None and last_w == last_w:
            if last_w2 is not None and last_w2 == last_w2:
                wdelta = float(last_w - last_w2)
        margin_score = _margin_tightness(last_margin or m1)
        extra.append({
            '斤量': kg,
            '斤量差': kg_delta,
            '枠': waku,
            '馬番': umaban,
            '騎手': jockey,
            '頭数': n_field,
            'style': None if style != style else style,
            'last_weight': None if last_w is None or last_w != last_w else float(last_w),
            'weight_delta': wdelta,
            'last_time': None if last_time is None or last_time != last_time else float(last_time),
            'runners_time1': None if t1 != t1 else float(t1),
            'last_kg': last_kg,
            'last_heads': last_heads,
            'last_dist': None if last_dist is None or last_dist != last_dist else float(last_dist),
            'last_track': last_track,
            'last_finish': last_finish,
            'last_pop': last_pop,
            'margin_tight': margin_score,
            'layoff': layoff,
            'runners_track1': tr1,
        })
    return pd.concat([honmei.reset_index(drop=True), pd.DataFrame(extra)], axis=1)


def _jockey_wr_train_only(df: pd.DataFrame, train_dates: set[str]) -> pd.Series:
    """train期間の結果から騎手勝率。holdoutへリークしない。"""
    from areru_engine import clean_name
    results = pd.read_csv(DATA / 'results.csv', encoding='utf-8-sig', low_memory=False)
    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    runners['race_id'] = runners['race_id'].astype(str)
    runners['_horse'] = runners['馬名'].map(clean_name)
    results['race_id'] = results['race_id'].astype(str)
    results['_horse'] = results['馬名'].map(clean_name)
    tr = results[results['date'].astype(str).isin(train_dates)]
    m = tr.merge(runners[['race_id', '_horse', '騎手']], on=['race_id', '_horse'], how='left')
    m['_j'] = m['騎手'].map(clean_name)
    m['win'] = (_num(m['着順']) == 1).astype(float)
    stats = m.groupby('_j').agg(n=('win', 'size'), wr=('win', 'mean'))
    stats = stats[stats['n'] >= 8]
    wr_map = stats['wr'].to_dict()
    field_avg = float(m['win'].mean()) if len(m) else 0.1

    out = []
    for _, row in df.iterrows():
        j = clean_name(row.get('騎手'))
        out.append(wr_map.get(j, field_avg) - field_avg)
    return pd.Series(out, index=df.index, name='jockey_wr_excess')


def _slice_calib(df: pd.DataFrame, pcol: str, label: str) -> dict:
    sub = df[df['hit'].notna() & df[pcol].notna()].copy()
    if sub.empty:
        return {'label': label, 'n': 0}
    p = sub[pcol].to_numpy(dtype=float) / 100.0
    y = sub['hit'].to_numpy(dtype=float)
    out = _ece_brier(p, y)
    out['label'] = label
    out['hit_rate'] = round(float(y.mean()) * 100, 2)
    out['mean_claimed_pct'] = round(float(sub[pcol].mean()), 2)
    return out


def _band_table(df: pd.DataFrame, band_col: str, pcol: str) -> list[dict]:
    rows = []
    sub = df[df['hit'].notna()].copy()
    for band, g in sub.groupby(band_col, dropna=False):
        n = len(g)
        if n == 0:
            continue
        hits = float(g['hit'].mean()) * 100
        mp = float(g[pcol].mean()) if g[pcol].notna().any() else None
        rows.append({
            'band': str(band),
            'n': int(n),
            'hit_rate': round(hits, 2),
            'mean_p': None if mp is None else round(mp, 2),
            'gap_pp': None if mp is None else round(hits - mp, 2),
        })
    return rows


def _residual_table(df: pd.DataFrame, features: dict[str, str], train_dates: set[str]) -> list[dict]:
    """feature vs (hit - adj_p)。train/holdout 同符号なら検証候補。"""
    work = df[df['hit'].notna() & df['adj_pct'].notna()].copy()
    work['resid'] = work['hit'] - work['adj_pct'] / 100.0
    work['jockey_wr_excess'] = _jockey_wr_train_only(work, train_dates)
    out = []
    for col, label in features.items():
        if col not in work.columns:
            continue
        x = pd.to_numeric(work[col], errors='coerce')
        fill = float(x.notna().mean() * 100)
        rec = {
            'feature': col,
            'label': label,
            'fill_pct': round(fill, 2),
            'n': int(x.notna().sum()),
        }
        for period in ('train', 'holdout'):
            g = work[work['period'] == period]
            xx = pd.to_numeric(g[col], errors='coerce')
            rho_res = _spearman(xx, g['resid'])
            rho_hit = _spearman(xx, g['hit'])
            rec[f'{period}_n'] = int(xx.notna().sum())
            rec[f'{period}_rho_resid'] = None if rho_res is None else round(rho_res, 4)
            rec[f'{period}_rho_hit'] = None if rho_hit is None else round(rho_hit, 4)
        tr = rec.get('train_rho_resid')
        ho = rec.get('holdout_rho_resid')
        same = (
            tr is not None and ho is not None
            and rec['train_n'] >= RESIDUAL_MIN_N and rec['holdout_n'] >= RESIDUAL_MIN_N
            and abs(tr) >= RESIDUAL_MIN_ABS_RHO and abs(ho) >= RESIDUAL_MIN_ABS_RHO
            and (tr > 0) == (ho > 0)
        )
        rec['same_sign_both_splits'] = bool(same)
        rec['verifiable_for_prob'] = bool(same and fill >= USABLE_FILL_MIN)
        out.append(rec)
    return out


def _classify(item: dict, fill: float, residual_hit: bool, prior_roi_fail: bool) -> str:
    old = item['old']
    if old in ('leak_forbidden',):
        return 'sparse_or_unusable'
    if item['id'] in ('pass_pace_last3f', 'blood_stable'):
        return 'sparse_or_unusable'
    if fill is not None and fill < SPARSE_FILL_MAX and old in ('unused', 'used_if_filled', 'missing', 'placeholder'):
        return 'sparse_or_unusable'
    if old in ('used', 'used_weak', 'display') or item['id'] in ('claimable_prob', 'place_shrink', 'consistency_sigma'):
        return 'currently_used'
    if residual_hit and not prior_roi_fail:
        return 'verifiable'
    if residual_hit and prior_roi_fail:
        return 'verifiable'
    if old in ('unused', 'used_if_filled', 'used_fallback', 'used_proxy'):
        return 'acquired_unused'
    return 'acquired_unused'


def main():
    os.environ.setdefault('ARERU_LEGACY_SCORE', '1')
    results = pd.read_csv(DATA / 'results.csv', encoding='utf-8-sig', low_memory=False)
    results['date'] = results['date'].astype(str)
    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    hist = pd.read_csv(DATA / 'all_history.csv', encoding='utf-8-sig', low_memory=False)

    dates = _dates('OLD')
    train_dates, holdout_dates = _split(dates)
    train_set, holdout_set = set(train_dates), set(holdout_dates)

    runner_fill = {c: round(_fill_rate(runners[c]), 2) for c in runners.columns}
    hist_fill = {c: round(_fill_rate(hist[c]), 2) for c in hist.columns}
    horse_cache_n = 0
    cache_dir = DATA / 'cache' / 'horse_results'
    if cache_dir.exists():
        horse_cache_n = sum(1 for _ in cache_dir.iterdir() if _.is_file())

    print(f'[catalog] dates={len(dates)} train={len(train_dates)} holdout={len(holdout_dates)}', flush=True)
    old = _load_honmei('OLD', results, dates, train_set, holdout_set)
    new = _load_honmei('NEW', results, dates, train_set, holdout_set)
    print(f'[catalog] honmei OLD={len(old)} NEW={len(new)}', flush=True)

    old_x = _attach_unused_features(old)
    print('[catalog] unused features attached', flush=True)

    residual_feats = {
        '斤量': '当日斤量',
        '斤量差': '当日斤量−フィールド平均',
        '枠': '枠番',
        '馬番': '馬番',
        '頭数': '出走頭数',
        'style': '脚質代理',
        'last_weight': '前走馬体重',
        'weight_delta': '馬体重増減',
        'last_time': '前走タイム秒',
        'last_kg': '前走斤量',
        'last_heads': '前走頭数',
        'last_dist': '前走距離',
        'last_finish': '前走着順',
        'last_pop': '前走人気',
        'margin_tight': '前走着差タイトさ',
        'layoff': '休み明け日数',
        'jockey_wr_excess': '騎手勝率超過（train推定）',
        '人気': '当日人気（対照・既使用）',
        'odds': '当日オッズ（対照・既使用）',
        'sim_pct': 'SIM勝率（対照・既使用）',
    }
    residual = _residual_table(old_x, residual_feats, train_set)

    calib = {}
    for logic, df in (('OLD', old), ('NEW', new)):
        calib[logic] = {}
        for period, g in (('full', df), ('train', df[df['period'] == 'train']),
                          ('holdout', df[df['period'] == 'holdout'])):
            gg = g[g['hit'].notna()]
            block = {
                'n_honmei': int(len(gg)),
                'n_buy': int(gg['buy'].sum()) if 'buy' in gg.columns else 0,
                'honmei_hit_rate': round(float(gg['hit'].mean()) * 100, 2) if len(gg) else None,
                'fav_hit_rate': round(float(gg['fav_win'].mean()) * 100, 2) if gg['fav_win'].notna().any() else None,
                'sim': _slice_calib(gg, 'sim_pct', 'SIM勝率'),
                'adj': _slice_calib(gg, 'adj_pct', '補正勝率'),
                'market': _slice_calib(gg, 'implied_pct', '市場1/odds'),
                'by_odds_adj': _band_table(gg.assign(band=gg['odds'].map(_odds_band)), 'band', 'adj_pct'),
                'by_pop_adj': _band_table(gg.assign(band=gg['人気'].map(_pop_band)), 'band', 'adj_pct'),
                'by_sim_band': _band_table(gg.assign(band=gg['sim_pct'].map(_win_band)), 'band', 'sim_pct'),
            }
            # ranking among 本命: higher claimed p should hit more
            block['rank_corr'] = {
                'sim_vs_hit': None if _spearman(gg['sim_pct'], gg['hit']) is None else round(_spearman(gg['sim_pct'], gg['hit']), 4),
                'adj_vs_hit': None if _spearman(gg['adj_pct'], gg['hit']) is None else round(_spearman(gg['adj_pct'], gg['hit']), 4),
                'market_vs_hit': None if _spearman(gg['implied_pct'], gg['hit']) is None else round(_spearman(gg['implied_pct'], gg['hit']), 4),
                'pop_vs_hit': None if _spearman(-gg['人気'], gg['hit']) is None else round(_spearman(-gg['人気'], gg['hit']), 4),
            }
            # BUY subset calibration (same gates, diagnostic only)
            buy = gg[gg['buy'] == True]
            block['buy'] = {
                'n': int(len(buy)),
                'hit_rate': round(float(buy['hit'].mean()) * 100, 2) if len(buy) else None,
                'mean_adj': round(float(buy['adj_pct'].mean()), 2) if len(buy) and buy['adj_pct'].notna().any() else None,
                'mean_sim': round(float(buy['sim_pct'].mean()), 2) if len(buy) and buy['sim_pct'].notna().any() else None,
                'mean_market': round(float(buy['implied_pct'].mean()), 2) if len(buy) and buy['implied_pct'].notna().any() else None,
            }
            calib[logic][period] = block

    fill_map = {
        'past_finish': runner_fill.get('着順1'),
        'past_pop': runner_fill.get('人気1'),
        'past_venue': runner_fill.get('場1'),
        'market_odds': runner_fill.get('単勝オッズ'),
        'source_venue': 100.0,
        'hist_date_horse': 100.0,
        'hist_dist_venue': hist_fill.get('距離'),
        'hist_heads': hist_fill.get('頭数'),
        'hist_finish_pop': hist_fill.get('着順'),
        'consistency_sigma': 100.0,
        'runners_time': runner_fill.get('タイム1'),
        'runners_margin': runner_fill.get('着差1'),
        'runners_track': runner_fill.get('馬場1'),
        'hist_time': hist_fill.get('タイム'),
        'hist_margin': hist_fill.get('着差'),
        'hist_track': hist_fill.get('馬場'),
        'runners_jockey': runner_fill.get('騎手'),
        'hist_jockey': hist_fill.get('騎手'),
        'hist_weight': hist_fill.get('馬体重'),
        'runners_waku': runner_fill.get('枠'),
        'runners_kg': runner_fill.get('斤量'),
        'hist_kg': hist_fill.get('斤量'),
        'layoff': 100.0,
        'style_proxy': runner_fill.get('着順1'),
        'last3f_proxy': runner_fill.get('着順1'),
        'blood_stable': 0.0,
        'pass_pace_last3f': 0.0,
        'hist_this_race': hist_fill.get('今回レース'),
        'runners_finish_today': runner_fill.get('実着順'),
        'odds_updated_at': runner_fill.get('オッズ更新日時'),
        'runners_umaban': runner_fill.get('馬番'),
        'claimable_prob': 100.0,
        'place_shrink': 100.0,
    }

    residual_by_id = {
        'runners_kg': '斤量差',
        'hist_kg': 'last_kg',
        'runners_waku': '枠',
        'hist_weight': 'weight_delta',
        'layoff': 'layoff',
        'style_proxy': 'style',
        'hist_time': 'last_time',
        'hist_margin': 'margin_tight',
        'hist_heads': 'last_heads',
        'runners_jockey': 'jockey_wr_excess',
        'runners_umaban': '馬番',
    }
    resid_lookup = {r['feature']: r for r in residual}
    prior_roi_fail = {
        'runners_kg', 'runners_waku', 'hist_weight', 'runners_jockey',
        'layoff', 'style_proxy', 'hist_heads',
    }

    lists = {
        'currently_used': [],
        'acquired_unused': [],
        'sparse_or_unusable': [],
        'verifiable': [],
    }
    rows_out = []
    for item in _code_catalog():
        fid = item['id']
        fill = fill_map.get(fid)
        rkey = residual_by_id.get(fid)
        rinfo = resid_lookup.get(rkey) if rkey else None
        residual_hit = bool(rinfo and rinfo.get('verifiable_for_prob'))
        cat = _classify(item, fill, residual_hit, fid in prior_roi_fail)
        # production OLD で used なものは currently_used を優先
        if item['old'] in ('used', 'used_weak') and cat != 'sparse_or_unusable':
            cat = 'currently_used'
        if item['old'] == 'display':
            cat = 'acquired_unused'
        join_pct = None
        lw = resid_lookup.get('last_weight') or {}
        if lw.get('fill_pct') is not None:
            join_pct = lw['fill_pct']
        if fid in ('hist_time', 'hist_margin', 'hist_track', 'hist_weight', 'hist_kg', 'layoff'):
            cat = 'sparse_or_unusable'
        note = ''
        if fid in prior_roi_fail:
            note = 'PRESET=X の指数加点はholdout ROI失敗。残差が両split再現しない限り確率補正にも使わない'
        if fid == 'runners_time':
            note = 'runners列は約10%しか埋まらない。履歴フォールバックも本命馬名接合が約15%'
        if fid in ('hist_time', 'hist_margin', 'hist_track', 'hist_weight', 'hist_kg', 'layoff'):
            note = (
                f'all_history の行充足は高いが、本命馬名との接合は {join_pct}% 。'
                '履歴マスタのカバレッジ拡張が先。現状の本命予測には実質欠損'
            )
        if fid == 'pass_pace_last3f':
            note = 'netkeiba_client は取得できるが all_history に列がなくキャッシュも空'
        if fid == 'blood_stable':
            note = 'コード上は中立プレースホルダ。取得パイプラインなし'
        if fid == 'claimable_prob':
            note = '補正勝率は市場よりまだ過大。isotonic/人気帯キャリブが次の本線'
        if fid == 'runners_umaban':
            note = '表示用。予測ロジック未使用'
        if fid == 'style_proxy':
            note = (
                '本命全体では差し寄りで hit は低いが p がさらに下がり残差が正。'
                'BUY層の差し×内枠は0的中のため単純加点は禁止。層別キャリブのみ検証可'
            )
        rec = {
            **item,
            'category': cat,
            'fill_pct': fill,
            'production_old_uses': item['old'] in ('used', 'used_weak'),
            'residual': None if rinfo is None else {
                'train_rho_resid': rinfo.get('train_rho_resid'),
                'holdout_rho_resid': rinfo.get('holdout_rho_resid'),
                'same_sign_both_splits': rinfo.get('same_sign_both_splits'),
            },
            'prior_roi_addon_failed_holdout': fid in prior_roi_fail,
            'note': note,
        }
        # verifiable 上書き: キャリブそのもの
        lists[cat].append(rec)
        rows_out.append(rec)

    # 明示的な検証候補（残差 or キャリブギャップ）。BUYフィルタ禁止
    old_full = calib['OLD']['full']
    old_ho = calib['OLD']['holdout']
    old_tr = calib['OLD']['train']
    verify_extra = []
    # 補正勝率の過信が両splitで再現
    if (
        old_tr['adj']['overconfidence'] and old_ho['adj']['overconfidence']
        and old_tr['adj']['overconfidence'] > 0 and old_ho['adj']['overconfidence'] > 0
    ):
        verify_extra.append({
            'id': 'winrate_isotonic',
            'name': '補正勝率のisotonic/人気帯キャリブ',
            'source': 'derived (SIM + オッズ)',
            'old': 'used_miscalibrated',
            'new': 'used_miscalibrated',
            'path': '同じBUYゲートのまま p を実測に合わせる。件数削減フィルタではない',
            'category': 'verifiable',
            'fill_pct': 100.0,
            'production_old_uses': True,
            'prior_roi_addon_failed_holdout': False,
            'note': (
                f"本番OLD 全本命: 補正勝率 {old_full['adj']['mean_claimed_pct']}% vs 的中 {old_full['honmei_hit_rate']}% "
                f"(train overconf {old_tr['adj']['overconfidence']}, holdout {old_ho['adj']['overconfidence']})。"
                f"BUY holdout: 補正 {old_ho['buy']['mean_adj']}% vs 的中 {old_ho['buy']['hit_rate']}% "
                f"(市場暗示 {old_ho['buy']['mean_market']}%)。"
                f"12-20倍帯は全本命でも補正8.7% vs 的中3.4%。"
                f"1番人気的中 {old_full['fav_hit_rate']}% に対し本命的中 {old_full['honmei_hit_rate']}%。"
                f"Brierは補正がSIMより良く、市場とほぼ同等。次はオッズ/人気帯でpを実測に合わせる"
                f"（BUY_EV_FLOOR=108は固定。追加条件フィルタ禁止）。"
            ),
            'metric': {
                'train_brier_sim': old_tr['sim']['brier'],
                'train_brier_adj': old_tr['adj']['brier'],
                'train_brier_market': old_tr['market']['brier'],
                'holdout_brier_sim': old_ho['sim']['brier'],
                'holdout_brier_adj': old_ho['adj']['brier'],
                'holdout_brier_market': old_ho['market']['brier'],
            },
        })
    for r in residual:
        if not r.get('verifiable_for_prob'):
            continue
        if r['feature'] in ('人気', 'odds', 'sim_pct', 'last_finish', 'last_pop', 'style'):
            continue  # 既使用・対照、または style_proxy と重複
        verify_extra.append({
            'id': f"resid_{r['feature']}",
            'name': r['label'],
            'source': 'acquired',
            'old': 'unused',
            'new': 'see_catalog',
            'path': '残差(hit-p)がtrain/holdout同符号。確率・ランキング補正の検証対象。BUY除外禁止',
            'category': 'verifiable',
            'fill_pct': r['fill_pct'],
            'production_old_uses': False,
            'prior_roi_addon_failed_holdout': r['feature'] in (
                '斤量', '斤量差', '枠', 'weight_delta', 'layoff', 'style', 'jockey_wr_excess', 'last_heads',
            ),
            'note': '指数へ雑に足すのではなく、勝率キャリブの共変量としてholdout Brier/順位相関で検証する',
            'residual': r,
        })
    # de-dup into verifiable list
    seen = {x['id'] for x in lists['verifiable']}
    for v in verify_extra:
        if v['id'] not in seen:
            lists['verifiable'].append(v)
            seen.add(v['id'])

    catalog = {
        'frozen': {
            'production': 'OLD (ARERU_LEGACY_SCORE=1)',
            'three_candidates': ['SASHI_INNER', 'ODDS_INNER', 'SASHI_SWEET'],
            'three_candidates_verdict': '不採用確定',
            'buy_filters_forbidden': True,
            'simple_condition_filters_forbidden': True,
            'production_change_allowed': False,
            'note': '次の改善は同じBUY条件のまま、馬ごとの推定勝率・期待値・ランキング精度。',
        },
        'split': {
            'n_dates': len(dates),
            'train': train_dates,
            'holdout': holdout_dates,
        },
        'fill': {
            'runners_rows': int(len(runners)),
            'history_rows': int(len(hist)),
            'runner_fill_pct': runner_fill,
            'history_fill_pct': hist_fill,
            'horse_results_cache_files': horse_cache_n,
            'history_has_通過': '通過' in hist.columns,
            'history_has_ペース': 'ペース' in hist.columns,
            'history_has_上り': '上り' in hist.columns,
            'honmei_history_join_pct': (resid_lookup.get('last_weight') or {}).get('fill_pct'),
            'honmei_history_join_note': 'all_history は行充足が高いが、本命馬名との接合は約15%。タイム/着差/馬体重を本命予測へ入れるには履歴マスタ拡充が先',
        },
        'currently_used': lists['currently_used'],
        'acquired_unused': lists['acquired_unused'],
        'sparse_or_unusable': lists['sparse_or_unusable'],
        'verifiable': lists['verifiable'],
        'residual_all': residual,
    }

    cal_report = {
        'scope': '全本命（BUYに限らない）。確定オッズ。train=先頭70%',
        'production': 'OLD',
        'calibration': calib,
        'reading': {
            'overconfidence': 'mean_p > mean_y なら推定勝率が的中を上回る（過信）',
            'brier': '小さいほど良い。市場1/oddsがSIMより小さければ人気補正が先',
            'rank_corr': '本命集団内で高いpほど的中しやすいか（1頭/レースなので弱い）',
            'next': 'BUYを減らすフィルタではなく、pのキャリブと未使用高充足特徴の確率補正',
        },
    }

    OUT_CATALOG.write_text(json.dumps(_json_safe(catalog), ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_CALIB.write_text(json.dumps(_json_safe(cal_report), ensure_ascii=False, indent=2), encoding='utf-8')

    with OUT_CSV.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'category', 'id', 'name', 'source', 'old', 'new', 'fill_pct',
            'production_old_uses', 'prior_roi_fail', 'note', 'path',
        ])
        for cat in ('currently_used', 'acquired_unused', 'sparse_or_unusable', 'verifiable'):
            for rec in lists[cat]:
                w.writerow([
                    cat, rec.get('id'), rec.get('name'), rec.get('source'),
                    rec.get('old'), rec.get('new'), rec.get('fill_pct'),
                    rec.get('production_old_uses'), rec.get('prior_roi_addon_failed_holdout'),
                    rec.get('note'), rec.get('path'),
                ])

    with OUT_SIGNAL.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(residual[0].keys()) if residual else ['feature'])
        w.writeheader()
        for r in residual:
            w.writerow(r)

    # 短いサマリを標準出力
    print('\n=== 4分類 ===')
    for cat in ('currently_used', 'acquired_unused', 'sparse_or_unusable', 'verifiable'):
        print(f'{cat}: {len(lists[cat])}')
        for rec in lists[cat]:
            print(f'  - {rec.get("id")}: {rec.get("name")}')
    print('\n=== OLD 本命キャリブ ===')
    for period in ('train', 'holdout'):
        b = calib['OLD'][period]
        print(
            f"{period} n={b['n_honmei']} hit={b['honmei_hit_rate']}% fav={b['fav_hit_rate']}% "
            f"SIM p={b['sim']['mean_p']} y={b['sim']['mean_y']} brier={b['sim']['brier']} | "
            f"補正 p={b['adj']['mean_p']} y={b['adj']['mean_y']} brier={b['adj']['brier']} | "
            f"市場 p={b['market']['mean_p']} y={b['market']['mean_y']} brier={b['market']['brier']}"
        )
    print(f'\n📁 {OUT_CATALOG}\n📁 {OUT_CSV}\n📁 {OUT_CALIB}\n📁 {OUT_SIGNAL}')


if __name__ == '__main__':
    main()
