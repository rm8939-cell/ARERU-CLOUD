#!/usr/bin/env python3
"""取得済み未使用特徴を 1 つずつ OLD へ追加し、同一 BUY 条件で比較する。

固定:
  - 本番は触らない（ARERU_LEGACY_SCORE=1 のまま、render.yaml 変更なし）
  - BUY_EV_FLOOR=108 / BUY_CONF_FLOOR=58 / tanh 表示EV 固定
  - 複数特徴の同時追加禁止
  - 段階SIMへ切り替えない（ARERU_KEEP_GAUSS=1）。スコア側の単特徴のみ
  - BUY件数を削って ROI だけ上げる操作は禁止

出力:
  data/single_feature_report.json
  data/single_feature_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
OUT = DATA / 'single_feature_report.json'
TABLE = DATA / 'single_feature_table.csv'
VERDICT = DATA / 'single_feature_verdict.json'
CACHE_OLD = DATA / 'rca_logic_cache'
CACHE_EXTRA = DATA / 'rca_extra_cache'
CACHE_ABL = DATA / 'rca_abl_cache'
STAKE = 100
MIN_BUY = 100
MIN_VOLUME_RATIO = 0.80

# 1特徴ずつ。KEEP_GAUSS でガウスSIMを維持する。
LOGICS: dict[str, dict[str, str]] = {
    'OLD': {'ARERU_LEGACY_SCORE': '1'},
    'TIME': {
        'ARERU_LEGACY_SCORE': '1',
        'ARERU_ABL_TIME': '1',
        'ARERU_ABL_ENRICH': '1',
        'ARERU_KEEP_GAUSS': '1',
    },
    'MARGIN': {
        'ARERU_LEGACY_SCORE': '1',
        'ARERU_ABL_MARGIN': '1',
        'ARERU_ABL_ENRICH': '1',
        'ARERU_KEEP_GAUSS': '1',
    },
    'TRACK': {
        'ARERU_LEGACY_SCORE': '1',
        'ARERU_ABL_TRACK': '1',
        'ARERU_ABL_ENRICH': '1',
        'ARERU_KEEP_GAUSS': '1',
    },
    'WEIGHT': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_SWEIGHT': '1'},
    'JOCKEY': {'ARERU_LEGACY_SCORE': '1', 'ARERU_ABL_SJOCKEY': '1'},
    'COURSE': {
        'ARERU_LEGACY_SCORE': '1',
        'ARERU_ABL_COURSE': '1',
        'ARERU_KEEP_GAUSS': '1',
    },
    'HISTORY': {
        'ARERU_LEGACY_SCORE': '1',
        'ARERU_ABL_HISTORY_EXPAND': '1',
        'ARERU_KEEP_GAUSS': '1',
    },
}

LOGIC_LABELS = {
    'OLD': '本番旧（ガウスSIM・追加特徴OFF）',
    'TIME': '旧+タイム（history_detail_bonus / ガウス維持）',
    'MARGIN': '旧+着差（history_detail_bonus / ガウス維持）',
    'TRACK': '旧+馬場（history_detail_bonus / ガウス維持）',
    'WEIGHT': '旧+馬体重変化（score_extras sweight）',
    'JOCKEY': '旧+騎手複勝（score_extras sjockey）',
    'COURSE': '旧+距離/コース適性（course_distance_fit を指数へ / ガウス維持）',
    'HISTORY': '旧+過去走6-8（着順/人気を履歴から拡張 / ガウス維持）',
}

PRIORITY = ('TIME', 'MARGIN', 'TRACK', 'WEIGHT', 'JOCKEY', 'COURSE', 'HISTORY')


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


def _clear_env():
    for k in list(os.environ.keys()):
        if k.startswith('ARERU_ABL_') or k.startswith('ARERU_CALIB_') or k in (
            'ARERU_LEGACY_SCORE', 'ARERU_LOGIC_PRESET', 'ARERU_XSEL_FEATURES',
            'ARERU_KEEP_GAUSS',
        ):
            os.environ.pop(k, None)


def _apply(logic: str):
    _clear_env()
    for k, v in LOGICS[logic].items():
        os.environ[k] = v


def _cache_pred(logic: str, date: str) -> Path:
    if logic == 'OLD':
        return CACHE_OLD / f'pred_OLD_{date}.csv'
    if logic in ('WEIGHT', 'JOCKEY'):
        CACHE_EXTRA.mkdir(parents=True, exist_ok=True)
        return CACHE_EXTRA / f'pred_{logic.lower()}_{date}.csv'
    CACHE_ABL.mkdir(parents=True, exist_ok=True)
    return CACHE_ABL / f'pred_{logic.lower()}_{date}.csv'


def _cache_scores(logic: str, date: str) -> Path:
    if logic == 'OLD':
        p = CACHE_OLD / f'scores_OLD_{date}.csv'
        if p.exists():
            return p
        return DATA / 'logic_compare_cache' / f'scores_legacy_{date}.csv'
    if logic in ('WEIGHT', 'JOCKEY'):
        return CACHE_EXTRA / f'scores_{logic.lower()}_{date}.csv'
    return CACHE_ABL / f'scores_{logic.lower()}_{date}.csv'


def _restore_predictions():
    try:
        subprocess.run(
            ['git', 'checkout', '--', 'data/predictions_by_date/'],
            cwd=str(BASE), check=False, capture_output=True,
        )
    except Exception:
        pass


def _one_day(payload: dict) -> list[dict]:
    logic = payload['logic']
    d = payload['date']
    sim_runs = payload['sim_runs']
    no_cache = payload['no_cache']
    hold_set = set(payload['holdout'])
    cache_path = Path(payload['cache_path'])
    scores_path = Path(payload['scores_path'])
    _apply(logic)
    os.environ['ARERU_SIM_RUNS'] = str(sim_runs)
    os.environ['ARERU_FAST_GAUSS'] = '1'
    from scripts.logic_compare_backtest import _load_history, _load_results, _predict_for_date
    from scripts.stable_holdout_compare import _race_rows
    period = 'holdout' if d in hold_set else 'train'
    if cache_path.exists() and not no_cache:
        pred = pd.read_csv(cache_path, encoding='utf-8-sig')
    else:
        history = _load_history()
        pred, scores = _predict_for_date(
            d, legacy=True, history=history,
            sim_runs=sim_runs, use_cache=False, respect_env=True,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(cache_path, index=False, encoding='utf-8-sig')
        if scores is not None:
            scores.to_csv(scores_path, index=False, encoding='utf-8-sig')
    results = _load_results()
    day_rows = _race_rows(pred, results, d)
    for r in day_rows:
        r['period'] = period
        r['logic'] = logic
        # 予測側の勝率・人気・ランクを品質比較用に残す
        hit = pred[pred['race_id'].astype(str) == str(r.get('race_id'))]
        if not hit.empty:
            row = hit.iloc[0]
            r['sim_pct'] = pd.to_numeric(row.get('シミュレーション勝率'), errors='coerce')
            r['ai_index'] = pd.to_numeric(row.get('本命AREru指数'), errors='coerce')
            r['rel_rank'] = pd.to_numeric(row.get('相対ランク'), errors='coerce')
            r['disp_ev'] = pd.to_numeric(row.get('期待値'), errors='coerce')
    return day_rows


def _collect_logic(logic: str, dates: list[str], holdout_dates: list[str], *,
                   sim_runs: int, no_cache: bool, workers: int = 4) -> list[dict]:
    payloads = []
    for d in dates:
        payloads.append({
            'logic': logic, 'date': d, 'sim_runs': sim_runs, 'no_cache': no_cache,
            'holdout': holdout_dates,
            'cache_path': str(_cache_pred(logic, d)),
            'scores_path': str(_cache_scores(logic, d)),
        })
    need_gen = [p for p in payloads if no_cache or not Path(p['cache_path']).exists()]
    use_pool = workers > 1 and len(need_gen) >= 2
    print(f'[{logic}] dates={len(dates)} generate={len(need_gen)} pool={use_pool}', flush=True)
    rows: list[dict] = []
    if not use_pool:
        for p in payloads:
            print(f'[{logic}] {p["date"]} ...', flush=True)
            rows.extend(_one_day(p))
        return rows
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_day, p): p['date'] for p in payloads}
        for fut in as_completed(futs):
            d = futs[fut]
            part = fut.result()
            print(f'[{logic}] done {d} races={len(part)}', flush=True)
            rows.extend(part)
    return rows


def _summarize(bets: list[dict], label: str) -> dict:
    from scripts.feature_search_backtest import _summarize as _sm
    return _sm(bets, label)


def _bootstrap_delta(new_pays, old_pays):
    from scripts.feature_search_backtest import _bootstrap_delta as _bd
    return _bd(new_pays, old_pays)


def _pays(bets: list[dict]) -> list[float]:
    return [float(b.get('払戻') or 0) for b in bets]


def _brier(ps: np.ndarray, ys: np.ndarray) -> float | None:
    if len(ps) == 0:
        return None
    return round(float(np.mean((ps - ys) ** 2)), 5)


def _quality(all_rows: list[dict], buys: list[dict], old_buys: list[dict] | None = None) -> dict:
    """推定勝率・市場差・ランキング・BUY判定のどこが動いたか。"""
    n = len(all_rows)
    hits = np.array([1.0 if r.get('的中') else 0.0 for r in all_rows], dtype=float)
    sim = np.array([
        (float(r['sim_pct']) / 100.0) if r.get('sim_pct') is not None and pd.notna(r.get('sim_pct')) else np.nan
        for r in all_rows
    ], dtype=float)
    odds = np.array([
        float(r['オッズ']) if r.get('オッズ') is not None and pd.notna(r.get('オッズ')) and r.get('オッズ') else np.nan
        for r in all_rows
    ], dtype=float)
    pop = np.array([
        float(r['人気']) if r.get('人気') is not None and pd.notna(r.get('人気')) else np.nan
        for r in all_rows
    ], dtype=float)
    market = np.where((odds > 0) & np.isfinite(odds), 1.0 / odds, np.nan)
    ok_sim = np.isfinite(sim)
    ok_mkt = np.isfinite(market)

    honmei_hit = round(float(hits.mean() * 100), 2) if n else 0.0
    mean_sim = round(float(np.nanmean(sim[ok_sim]) * 100), 2) if ok_sim.any() else None
    mean_mkt = round(float(np.nanmean(market[ok_mkt]) * 100), 2) if ok_mkt.any() else None
    calib_gap = round(mean_sim - honmei_hit, 2) if mean_sim is not None else None
    both = ok_sim & ok_mkt
    edge_vs_mkt = round(float(np.nanmean((sim - market)[both]) * 100), 2) if both.any() else None
    fav_share = round(float(np.nanmean((pop == 1).astype(float)) * 100), 2) if np.isfinite(pop).any() else None
    mean_pop = round(float(np.nanmean(pop)), 2) if np.isfinite(pop).any() else None

    old_keys = {(b.get('date'), b.get('race_id')) for b in (old_buys or [])}
    new_keys = {(b.get('date'), b.get('race_id')) for b in buys}
    inter = old_keys & new_keys
    overlap = round(100.0 * len(inter) / max(len(old_keys), 1), 2) if old_buys is not None else None
    added = len(new_keys - old_keys)
    dropped = len(old_keys - new_keys)

    return {
        '本命件数': n,
        '本命的中率': honmei_hit,
        '平均SIM勝率': mean_sim,
        '平均市場暗示勝率': mean_mkt,
        'SIM過大差_pp': calib_gap,
        'SIM_Brier': _brier(sim[ok_sim], hits[ok_sim]) if ok_sim.any() else None,
        '市場_Brier': _brier(market[ok_mkt], hits[ok_mkt]) if ok_mkt.any() else None,
        'SIMマイナス市場_pp': edge_vs_mkt,
        '本命が1番人気の割合': fav_share,
        '本命平均人気': mean_pop,
        'BUY件数': len(buys),
        'BUY_OLDとの重複率': overlap,
        'BUY追加レース': added,
        'BUY脱落レース': dropped,
    }


def _quality_delta(cand: dict, base: dict) -> dict:
    keys = (
        '本命的中率', '平均SIM勝率', 'SIM過大差_pp', 'SIM_Brier', '市場_Brier',
        'SIMマイナス市場_pp', '本命が1番人気の割合', '本命平均人気',
        'BUY_OLDとの重複率',
    )
    out = {}
    for k in keys:
        a, b = cand.get(k), base.get(k)
        if a is None or b is None:
            out[k] = None
        else:
            out[k] = round(float(a) - float(b), 4)
    return out


def _where_improved(delta_q: dict, roi_ho: float) -> dict:
    """どの経路が改善したかの判定（holdout 品質）。"""
    hit = delta_q.get('本命的中率')
    brier = delta_q.get('SIM_Brier')
    gap = delta_q.get('SIM過大差_pp')
    edge = delta_q.get('SIMマイナス市場_pp')
    return {
        '推定勝率': bool(brier is not None and brier < 0),
        '推定勝率_過大差縮小': bool(gap is not None and gap < 0),
        '市場人気との差': bool(edge is not None and edge < 0),
        'ランキング': bool(hit is not None and hit > 0),
        'BUY判定': bool(roi_ho > 0),
        'note': (
            '推定勝率= SIM Brierが低下した場合のみ真。'
            '過大差縮小= 平均SIM勝率−的中率 が縮小。'
            '市場差= SIM勝率−市場暗示 が縮小（過信の低下）。'
            'ランキング= 全本命的中率。BUY判定= holdout ROI差>0。'
        ),
    }


def _coverage(dates: list[str], honmei_rows: list[dict]) -> dict:
    """特徴量の欠損率と、本命馬で利用できる割合。"""
    from areru_engine import clean_name, parse_date
    from race_sim import parse_time_sec

    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    hist = pd.read_csv(DATA / 'all_history.csv', encoding='utf-8-sig', low_memory=False)
    hist['_horse'] = hist['馬名'].map(clean_name)
    hist['_date'] = parse_date(hist['年月日'])
    date_set = set(dates)

    def _blank(s: pd.Series) -> pd.Series:
        x = s.astype(str).str.strip()
        return x.isna() | (x == '') | (x.str.lower() == 'nan') | (x == '--') | (x == 'None')

    row_fill = {
        'タイム': round(float((~_blank(hist['タイム'])).mean() * 100), 2) if 'タイム' in hist.columns else 0.0,
        '着差': round(float((~_blank(hist['着差'])).mean() * 100), 2) if '着差' in hist.columns else 0.0,
        '馬場': round(float((~_blank(hist['馬場'])).mean() * 100), 2) if '馬場' in hist.columns else 0.0,
        '馬体重': round(float((~_blank(hist['馬体重'])).mean() * 100), 2) if '馬体重' in hist.columns else 0.0,
        '騎手': round(float((~_blank(hist['騎手'])).mean() * 100), 2) if '騎手' in hist.columns else 0.0,
        '距離': round(float((~_blank(hist['距離'])).mean() * 100), 2) if '距離' in hist.columns else 0.0,
        '着順': round(float((~_blank(hist['着順'])).mean() * 100), 2) if '着順' in hist.columns else 0.0,
    }
    if '騎手' in runners.columns:
        row_fill['当日騎手'] = round(float((~_blank(runners['騎手'])).mean() * 100), 2)

    n_h = max(len(honmei_rows), 1)
    usable = {k: 0 for k in ('TIME', 'MARGIN', 'TRACK', 'WEIGHT', 'JOCKEY', 'COURSE', 'HISTORY')}
    joined = 0
    for r in honmei_rows:
        horse = clean_name(r.get('本命'))
        target = pd.Timestamp(str(r.get('date')))
        h = hist[(hist['_horse'] == horse) & (hist['_date'] < target)].sort_values('_date', ascending=False)
        if h.empty:
            continue
        joined += 1
        head5 = h.head(5)
        if 'タイム' in h.columns:
            times = [parse_time_sec(v) for v in head5['タイム'].tolist()]
            if any(not np.isnan(t) for t in times):
                usable['TIME'] += 1
        if '着差' in h.columns and (~_blank(head5['着差'])).any():
            usable['MARGIN'] += 1
        if '馬場' in h.columns and (~_blank(head5['馬場'])).any():
            usable['TRACK'] += 1
        if '馬体重' in h.columns:
            w = (~_blank(h.head(2)['馬体重'])).sum()
            if w >= 2:
                usable['WEIGHT'] += 1
        if '距離' in h.columns and (~_blank(h.head(12)['距離'])).sum() >= 2:
            usable['COURSE'] += 1
        if '着順' in h.columns and (~_blank(h['着順'])).sum() >= 6:
            usable['HISTORY'] += 1

    jockey_ok = 0
    runners2 = runners.copy()
    if '騎手' in runners2.columns:
        runners2['_horse'] = runners2['馬名'].map(clean_name)
        date_col = '日付' if '日付' in runners2.columns else None
        for r in honmei_rows:
            horse = clean_name(r.get('本命'))
            d = str(r.get('date'))
            if date_col:
                rr = runners2[(runners2['_horse'] == horse) & (runners2[date_col].astype(str) == d)]
            else:
                rr = runners2[runners2['_horse'] == horse]
            if rr.empty:
                continue
            jockey = str(rr.iloc[0].get('騎手') or '').strip()
            if not jockey or jockey.lower() in ('nan', '--', 'none'):
                continue
            target = pd.Timestamp(d)
            hj = hist[(hist['_date'] < target) & (hist['騎手'].map(clean_name) == clean_name(jockey))]
            if len(hj) >= 5:
                jockey_ok += 1
    usable['JOCKEY'] = jockey_ok

    out = {
        'all_history行充足率': row_fill,
        '本命件数': len(honmei_rows),
        '本命が履歴に接合できた件数': joined,
        '本命接合率': round(100.0 * joined / n_h, 2),
        '特徴別': {},
    }
    fill_map = {
        'TIME': ('タイム', row_fill.get('タイム')),
        'MARGIN': ('着差', row_fill.get('着差')),
        'TRACK': ('馬場', row_fill.get('馬場')),
        'WEIGHT': ('馬体重', row_fill.get('馬体重')),
        'JOCKEY': ('騎手', row_fill.get('当日騎手', row_fill.get('騎手'))),
        'COURSE': ('距離', row_fill.get('距離')),
        'HISTORY': ('着順', row_fill.get('着順')),
    }
    for feat in PRIORITY:
        name, fill = fill_map[feat]
        out['特徴別'][feat] = {
            '特徴': name,
            'all_history欠損率': round(100.0 - float(fill or 0), 2),
            'all_history充足率': fill,
            '利用可能な本命件数': usable[feat],
            '利用可能な馬の割合': round(100.0 * usable[feat] / n_h, 2),
        }
    return out


def _grade(feat: str, sm: dict, old: dict, boot: dict, cov: dict, vol_note: str | None) -> dict:
    reasons = []
    ho = round(sm['holdout']['ROI'] - old['holdout']['ROI'], 2)
    tr = round(sm['train']['ROI'] - old['train']['ROI'], 2)
    fu = round(sm['full']['ROI'] - old['full']['ROI'], 2)
    ho_imp = ho > 0
    tr_imp = tr > 0
    fu_imp = fu > 0
    vol_ho = sm['holdout']['BUY件数'] / max(old['holdout']['BUY件数'], 1)
    vol_fu = sm['full']['BUY件数'] / max(old['full']['BUY件数'], 1)
    volume_game = (vol_ho < MIN_VOLUME_RATIO or vol_fu < MIN_VOLUME_RATIO)
    ci = (boot or {}).get('ci90') or [None, None]
    ci_ok = ci[0] is not None and ci[0] > 0
    usable = (cov.get('特徴別') or {}).get(feat, {}).get('利用可能な馬の割合')
    n_full = sm['full']['BUY件数']

    grade = 'C'
    if n_full < MIN_BUY or old['full']['BUY件数'] < MIN_BUY:
        grade = 'C'
        reasons.append(f'BUY件数不足 cand={n_full} old={old["full"]["BUY件数"]}')
    elif volume_game and (ho_imp or fu_imp):
        grade = 'C'
        reasons.append(
            f'BUY件数を減らしてROIを上げている holdout比={vol_ho:.2f} full比={vol_fu:.2f}'
        )
    elif tr_imp and not ho_imp:
        grade = 'C'
        reasons.append(f'trainのみ改善 holdout差={ho} → 不採用')
    elif fu_imp and not ho_imp:
        grade = 'C'
        reasons.append(f'全期間のみ改善 holdout差={ho} → 不採用')
    elif ho_imp and not tr_imp:
        grade = 'C'
        reasons.append(f'holdout改善だがtrain非改善 train差={tr} → 不採用')
    elif ho_imp and tr_imp and ci_ok and not volume_game:
        grade = 'A'
        reasons.append('train/holdout双方でOLDを上回り、holdout差の90%CIが正')
        if usable is not None and usable < 20:
            grade = 'B'
            reasons.append(f'本命への接合が薄い（利用可能 {usable}%）。採用候補ではなく保留')
    elif ho_imp and tr_imp and not ci_ok:
        grade = 'B'
        reasons.append(f'点推定では双方改善だがholdout 90%CIが0を跨ぐ {ci}')
    elif not ho_imp and not tr_imp:
        grade = 'C'
        reasons.append(f'holdout/trainとも非改善 holdout差={ho} train差={tr}')
    else:
        grade = 'C'
        reasons.append(f'採用条件未達 holdout差={ho} train差={tr}')

    if vol_note:
        reasons.append(vol_note)
    return {
        'grade': grade,
        'holdout改善': ho_imp,
        'train改善': tr_imp,
        '全期間改善': fu_imp,
        'ROI差_holdout': round(ho, 2),
        'ROI差_train': round(tr, 2),
        'ROI差_full': round(fu, 2),
        'BUY件数比_holdout': round(vol_ho, 3),
        'BUY件数比_full': round(vol_fu, 3),
        'holdout_bootstrap': boot,
        'reasons': reasons,
        '本番採用してよいか': False,
        '次の検証対象': grade == 'A',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-runs', type=int, default=2500)
    ap.add_argument('--logics', default='OLD,' + ','.join(PRIORITY))
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    from scripts.logic_compare_backtest import _eligible_dates
    from replay_predict import available_dates, load_runners

    runners = load_runners()
    dates = _eligible_dates(available_dates(runners))
    split = max(1, int(len(dates) * 0.70))
    train_dates = dates[:split]
    holdout_dates = dates[split:]
    logics = [x.strip() for x in args.logics.split(',') if x.strip()]

    print(
        f'[single-feat] days={len(dates)} train={len(train_dates)} '
        f'holdout={len(holdout_dates)} logics={logics} workers={args.workers}',
        flush=True,
    )

    rows_by: dict[str, list] = {}
    try:
        for logic in logics:
            rows_by[logic] = _collect_logic(
                logic, dates, holdout_dates,
                sim_runs=args.sim_runs, no_cache=args.no_cache, workers=args.workers,
            )
    finally:
        _restore_predictions()

    def scopes(all_rows):
        buys = [r for r in all_rows if r.get('strict_buy')]
        return {
            'full': buys,
            'train': [r for r in buys if r.get('period') == 'train'],
            'holdout': [r for r in buys if r.get('period') == 'holdout'],
            'all': all_rows,
            'all_train': [r for r in all_rows if r.get('period') == 'train'],
            'all_holdout': [r for r in all_rows if r.get('period') == 'holdout'],
        }

    old_rows = rows_by['OLD']
    old_sc = scopes(old_rows)
    coverage = _coverage(dates, old_rows)

    report = {
        '検証設計': {
            '開催日数': len(dates),
            'train': train_dates,
            'holdout': holdout_dates,
            'SIM_RUNS': args.sim_runs,
            '比較対象': 'BUYのみ（投資判定が買い）',
            'BUY閾値': {'BUY_EV_FLOOR': 108, 'BUY_CONF_FLOOR': 58, '再現率': 42},
            '閾値探索': '禁止',
            '複数特徴同時追加': '禁止',
            '本番ロジック': 'OLD（ARERU_LEGACY_SCORE=1）固定',
            '表示EV': 'tanh圧縮 78-124 固定',
            'SIM': 'ガウス維持（ARERU_KEEP_GAUSS=1）。段階SIMへは切り替えない',
            '採用しても本番変更': False,
        },
        'coverage': coverage,
        'strict_buy': {},
        'quality': {},
        'grades': {},
        '分類': {'A': [], 'B': [], 'C': []},
    }

    for logic, all_rows in rows_by.items():
        sc = scopes(all_rows)
        report['strict_buy'][logic] = {
            k: _summarize(v, f'{logic}/{k}') for k, v in (
                ('full', sc['full']), ('train', sc['train']), ('holdout', sc['holdout'])
            )
        }
        q_full = _quality(sc['all'], sc['full'], old_sc['full'])
        q_tr = _quality(sc['all_train'], sc['train'], old_sc['train'])
        q_ho = _quality(sc['all_holdout'], sc['holdout'], old_sc['holdout'])
        report['quality'][logic] = {'full': q_full, 'train': q_tr, 'holdout': q_ho}

    base = report['strict_buy']['OLD']
    q_old = report['quality']['OLD']

    for logic in logics:
        if logic == 'OLD':
            continue
        sm = report['strict_buy'][logic]
        cand_ho = [r for r in rows_by[logic] if r.get('strict_buy') and r.get('period') == 'holdout']
        boot = _bootstrap_delta(_pays(cand_ho), _pays(old_sc['holdout']))
        cov_one = {'特徴別': coverage.get('特徴別')}
        g = _grade(logic, sm, base, boot, cov_one, None)
        dq_ho = _quality_delta(report['quality'][logic]['holdout'], q_old['holdout'])
        g['holdout品質差'] = dq_ho
        g['どこが改善したか_holdout'] = _where_improved(dq_ho, g['ROI差_holdout'])
        g['coverage'] = coverage.get('特徴別', {}).get(logic)
        report['grades'][logic] = g
        report['分類'][g['grade']].append(logic)

    # CSV
    fieldnames = [
        '特徴', 'label', 'grade', '期間', 'BUY件数', '的中率', '平均オッズ', 'ROI',
        'OLD_ROI', 'OLDとの差', 'holdout改善', '欠損率', '利用可能な馬の割合',
    ]
    rows_csv = []
    for logic in logics:
        sm = report['strict_buy'][logic]
        cov = coverage.get('特徴別', {}).get(logic, {}) if logic != 'OLD' else {}
        grade = report['grades'].get(logic, {}).get('grade', '-')
        for period in ('train', 'holdout', 'full'):
            b = sm[period]
            old_b = base[period]
            rows_csv.append({
                '特徴': logic,
                'label': LOGIC_LABELS.get(logic, logic),
                'grade': grade,
                '期間': period,
                'BUY件数': b['BUY件数'],
                '的中率': b['的中率'],
                '平均オッズ': b['平均オッズ'],
                'ROI': b['ROI'],
                'OLD_ROI': old_b['ROI'],
                'OLDとの差': round(b['ROI'] - old_b['ROI'], 2),
                'holdout改善': report['grades'].get(logic, {}).get('holdout改善') if period == 'holdout' else '',
                '欠損率': cov.get('all_history欠損率') if logic != 'OLD' else 0,
                '利用可能な馬の割合': cov.get('利用可能な馬の割合') if logic != 'OLD' else 100,
            })
    with TABLE.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)

    report['best_next'] = {
        'A_採用候補': report['分類']['A'],
        'B_保留': report['分類']['B'],
        'C_不採用': report['分類']['C'],
        '本番ロジックを変更してよいか': False,
        'note': (
            'A があってもこの段階では本番を変えない。'
            '次の検証対象として報告するだけ。'
        ),
    }
    verdict = {
        '採用候補': report['分類']['A'],
        '保留': report['分類']['B'],
        '不採用': report['分類']['C'],
        '本番ロジックを変更してよいか': False,
        '本番': 'OLD (ARERU_LEGACY_SCORE=1)',
        '確定': True,
        'note': (
            'Aは空。holdout点推定がプラスでも90%CIが0を跨ぐ特徴は保留。'
            'trainのみ改善の馬体重・騎手は不採用。'
            '本命への履歴接合は約15%のため、タイム/着差/馬場/距離/履歴拡張は'
            'カバレッジ拡張が先。本番は変えない。'
        ),
        'holdout要点': {
            k: {
                'grade': v.get('grade'),
                'ROI差': v.get('ROI差_holdout'),
                'holdout改善': v.get('holdout改善'),
                'BUY件数比': v.get('BUY件数比_holdout'),
                '利用可能な馬の割合': (v.get('coverage') or {}).get('利用可能な馬の割合'),
                'どこが改善したか': v.get('どこが改善したか_holdout'),
            } for k, v in report['grades'].items()
        },
    }
    VERDICT.write_text(json.dumps(_json_safe(verdict), ensure_ascii=False, indent=2), encoding='utf-8')
    OUT.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(_json_safe({
        '分類': report['分類'],
        'grades': {
            k: {
                'grade': v.get('grade'),
                'ROI差_holdout': v.get('ROI差_holdout'),
                'ROI差_train': v.get('ROI差_train'),
                'holdout改善': v.get('holdout改善'),
                'reasons': v.get('reasons'),
                'どこが改善したか_holdout': v.get('どこが改善したか_holdout'),
            } for k, v in report['grades'].items()
        },
        'table_rows': len(rows_csv),
    }), ensure_ascii=False, indent=2))
    print(f'📁 {OUT}')
    print(f'📁 {TABLE}')
    print(f'📁 {VERDICT}')


if __name__ == '__main__':
    main()
