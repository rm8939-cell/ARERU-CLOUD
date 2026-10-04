#!/usr/bin/env python3
"""予想品質バックテスト基盤。

ROI だけを見ると BUY 件数が数百しかなく、標準誤差が ±40pp 規模になる。
そこで「確率モデルとしての良し悪し（全馬・全レース）」と「運用成績（BUY）」を
分けて測る。市場（単勝オッズ）を同じ土俵のベースラインとして必ず並べる。

出力する指標（ユーザー要求の一覧を含む）
  - AI1位の勝率 / AI1〜3位の複勝率
  - 本命勝率 / 本命複勝率 / 穴馬複勝率 / 危険人気馬の的中
  - BUY的中率 / BUY回収率
  - 期待値別の実回収率 / 信頼度別の実績
  - log loss・Brier・AUC・較正（AI vs 市場）
  - 市場ベースライン（1番人気、単勝全賭け）

使い方:
    python3 scripts/quality_backtest.py --label base
    python3 scripts/quality_backtest.py --label cand --no-cache
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
CACHE = DATA / 'quality_backtest_cache'
OUT_DIR = DATA / 'quality_backtest'
STAKE = 100

# 学習/検証の分割は stable_holdout_compare と揃える（最後の 13 開催日を holdout）。
HOLDOUT_FROM = '2026-08-10'


# ---------------------------------------------------------------- データ読込

def load_results() -> pd.DataFrame:
    r = pd.read_csv(DATA / 'results.csv', encoding='utf-8-sig', low_memory=False)
    r['race_id'] = r['race_id'].astype(str)
    r['date'] = r['date'].astype(str)
    r['馬名'] = r['馬名'].astype(str).str.strip()
    for c in ('着順', '人気', '確定オッズ'):
        r[c] = pd.to_numeric(r[c], errors='coerce')
    return r


def load_history() -> pd.DataFrame:
    from history_index import build_master_history
    return build_master_history()


def result_dates(results: pd.DataFrame) -> list[str]:
    return sorted(results['date'].unique())


# ------------------------------------------------------- 1日分の予想を再生成

def replay_date(date: str, history: pd.DataFrame, *, sim_runs: int, label: str,
                use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """本番と同じ build_predictions を回し、レース単位+馬単位の表を返す。"""
    cdir = CACHE / label
    cdir.mkdir(parents=True, exist_ok=True)
    race_p = cdir / f'race_{date}.csv'
    runner_p = cdir / f'runner_{date}.csv'
    if use_cache and race_p.exists() and runner_p.exists():
        return (pd.read_csv(race_p, encoding='utf-8-sig', low_memory=False),
                pd.read_csv(runner_p, encoding='utf-8-sig', low_memory=False))

    import areru_engine
    from areru_engine import build_predictions, parse_date

    os.environ['ARERU_SIM_RUNS'] = str(sim_runs)
    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    mask = parse_date(runners['日付']).dt.strftime('%Y-%m-%d') == date
    day = runners.loc[mask].copy()
    if day.empty:
        return None

    sink: list[pd.DataFrame] = []
    areru_engine.RUNNER_PROB_SINK = sink
    try:
        race_df, _scores = build_predictions(date, day, history)
    finally:
        areru_engine.RUNNER_PROB_SINK = None
    runner_df = pd.concat(sink, ignore_index=True) if sink else pd.DataFrame()
    runner_df['date'] = date
    race_df['date'] = date
    race_df.to_csv(race_p, index=False, encoding='utf-8-sig')
    runner_df.to_csv(runner_p, index=False, encoding='utf-8-sig')
    return race_df, runner_df


# ------------------------------------------------------------------ 指標計算

def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U による AUC。同値は 0.5 で数える。"""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order), dtype=float)
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _race_softmax_rank_prob(df: pd.DataFrame, col: str) -> np.ndarray:
    """レース内で合計 100% になるよう正規化した確率。"""
    out = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby('race_id').groups.items():
        v = pd.to_numeric(df.loc[idx, col], errors='coerce').to_numpy(dtype=float)
        v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
        s = v.sum()
        out[df.index.get_indexer(idx)] = v / s if s > 0 else 1.0 / max(len(idx), 1)
    return out


def _bootstrap_ci(values: np.ndarray, stat, n: int = 2000, seed: int = 7) -> tuple[float, float]:
    if len(values) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    draws = np.array([stat(values[i]) for i in idx])
    return (float(np.percentile(draws, 5)), float(np.percentile(draws, 95)))


def _roi_stat(payouts: np.ndarray) -> float:
    return float(payouts.mean() / STAKE * 100 - 100) if len(payouts) else float('nan')


def _pct(a: int, b: int) -> float:
    return round(a / b * 100, 2) if b else float('nan')


# ---------------------------------------------------- 馬単位（確率モデル評価）

def runner_metrics(runner: pd.DataFrame, results: pd.DataFrame) -> dict:
    """全馬・全レースでの確率モデル品質。AI と市場を同じ指標で並べる。"""
    res = results[['race_id', '馬名', '着順', '人気', '確定オッズ']].copy()
    res = res.rename(columns={'人気': '確定人気'})
    df = runner.drop(columns=[c for c in ('人気',) if c in runner.columns])
    df = df.merge(res, on=['race_id', '馬名'], how='inner').rename(columns={'確定人気': '人気'})
    df = df[pd.notna(df['着順'])].reset_index(drop=True)
    if df.empty:
        return {'照合頭数': 0}

    y_win = (df['着順'] == 1).astype(int).to_numpy()
    y_top3 = (df['着順'] <= 3).astype(int).to_numpy()

    ai_win = np.clip(pd.to_numeric(df['SIM勝率'], errors='coerce').fillna(0).to_numpy() / 100.0, 1e-6, 1)
    ai_top3 = np.clip(pd.to_numeric(df['SIM3着内率'], errors='coerce').fillna(0).to_numpy() / 100.0, 1e-6, 1)
    idx_prob = _race_softmax_rank_prob(df, 'AREru指数')

    # 市場ベースライン: 1/確定オッズ をレース内で正規化（控除率を抜いた素の市場確率）
    df['_impl'] = 1.0 / pd.to_numeric(df['確定オッズ'], errors='coerce')
    mkt_win = _race_softmax_rank_prob(df, '_impl')

    def block(p, y, name):
        return {
            'モデル': name,
            'logloss': round(_logloss(p, y), 5),
            'Brier': round(_brier(p, y), 5),
            'AUC': round(_auc(p, y), 4),
            '平均予測': round(float(p.mean()) * 100, 2),
            '実測率': round(float(y.mean()) * 100, 2),
        }

    # 1位指名の的中（レース単位）
    def top1_hit(col: str, ascending: bool) -> dict:
        hit = win_odds = 0
        races = 0
        payouts = []
        for rid, g in df.groupby('race_id'):
            v = pd.to_numeric(g[col], errors='coerce')
            if v.notna().sum() == 0:
                continue
            pick = g.loc[v.idxmin() if ascending else v.idxmax()]
            races += 1
            o = pick['確定オッズ']
            if pick['着順'] == 1:
                hit += 1
                payouts.append(float(o) * STAKE if pd.notna(o) else 0.0)
            else:
                payouts.append(0.0)
            if pd.notna(o):
                win_odds += float(o)
        pay = np.array(payouts, dtype=float)
        return {
            'レース数': races,
            '1着率': _pct(hit, races),
            'ROI': round(_roi_stat(pay), 2) if races else None,
            '平均オッズ': round(win_odds / races, 2) if races else None,
        }

    out = {
        '照合頭数': int(len(df)),
        '照合レース数': int(df['race_id'].nunique()),
        '勝率モデル': [block(ai_win, y_win, 'AI(SIM勝率)'),
                   block(idx_prob, y_win, 'AI(AREru指数のみ)'),
                   block(mkt_win, y_win, '市場(1/オッズ)')],
        '複勝モデル': [block(ai_top3, y_top3, 'AI(SIM3着内率)'),
                   block(mkt_win * 3, y_top3, '市場(1/オッズ×3)')],
        '1位指名': {
            'AI_SIM勝率1位': top1_hit('SIM勝率', ascending=False),
            'AI_AREru指数1位': top1_hit('AREru指数', ascending=False),
            '市場_1番人気': top1_hit('人気', ascending=True),
        },
    }

    # AI1位の勝率 / AI1〜3位の複勝率（ユーザー要求）
    for col, name in (('SIM勝率', 'SIM勝率'), ('AREru指数', 'AREru指数')):
        r1 = r3 = n_r = 0
        n3 = 0
        for rid, g in df.groupby('race_id'):
            g = g.sort_values(col, ascending=False)
            n_r += 1
            if g.iloc[0]['着順'] == 1:
                r1 += 1
            top3 = g.head(3)
            n3 += len(top3)
            r3 += int((top3['着順'] <= 3).sum())
        out[f'AI1位の勝率({name})'] = _pct(r1, n_r)
        out[f'AI1〜3位の複勝率({name})'] = _pct(r3, n3)

    # 較正（AI の勝率予測バンド別に実測勝率を見る）
    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 35), (35, 100)]
    calib = []
    p_pct = ai_win * 100
    for lo, hi in bands:
        m = (p_pct >= lo) & (p_pct < hi)
        if m.sum() == 0:
            continue
        calib.append({
            '予測勝率帯': f'{lo}-{hi}%',
            '頭数': int(m.sum()),
            '平均予測': round(float(p_pct[m].mean()), 2),
            '実測勝率': round(float(y_win[m].mean()) * 100, 2),
        })
    out['勝率較正'] = calib

    # 「人気だから AI 評価が高い」の逆流チェック
    sub = df[pd.notna(df['人気'])]
    if len(sub) > 10:
        def mean_rank_corr(col: str) -> float:
            vals = []
            for _, g in sub.groupby('race_id'):
                if len(g) < 3:
                    continue
                a = g[col].rank(ascending=False).to_numpy(dtype=float)
                b = g['人気'].rank().to_numpy(dtype=float)
                if a.std() == 0 or b.std() == 0:
                    continue
                vals.append(float(np.corrcoef(a, b)[0, 1]))
            return round(float(np.mean(vals)), 4) if vals else float('nan')

        out['市場との相関'] = {
            'AREru指数順位 vs 人気 の順位相関': mean_rank_corr('AREru指数'),
            'SIM勝率順位 vs 人気 の順位相関': mean_rank_corr('SIM勝率'),
        }
    return out


# ------------------------------------------------------- レース単位（運用成績）

def _pick_names(race_row, role: str) -> list[str]:
    try:
        cards = json.loads(race_row.get('ピックカード') or '[]')
    except Exception:
        return []
    return [str(c.get('馬名') or '').strip() for c in cards if str(c.get('役割') or '') == role]


def race_metrics(race: pd.DataFrame, results: pd.DataFrame) -> dict:
    res = results.set_index(['race_id', '馬名'])

    def lookup(rid: str, name: str):
        try:
            return res.loc[(str(rid), str(name).strip())]
        except KeyError:
            return None

    honmei = {'n': 0, 'win': 0, 'top3': 0, 'pay': [], 'odds': []}
    ana = {'n': 0, 'win': 0, 'top3': 0, 'pay': [], 'odds': []}
    danger = {'n': 0, 'out_of_top3': 0, 'win': 0}
    buy = {'n': 0, 'win': 0, 'top3': 0, 'pay': [], 'odds': [], 'ev': []}
    ev_bands: dict[str, list[float]] = {}
    conf_bands: dict[str, list[float]] = {}
    rank_bands: dict[str, list[float]] = {}

    for _, row in race.iterrows():
        rid = str(row.get('race_id'))
        hm = str(row.get('本命') or '').strip()
        rr = lookup(rid, hm) if hm else None
        if rr is not None:
            f = rr['着順']
            o = rr['確定オッズ']
            pay = float(o) * STAKE if (pd.notna(f) and f == 1 and pd.notna(o)) else 0.0
            honmei['n'] += 1
            honmei['win'] += int(pd.notna(f) and f == 1)
            honmei['top3'] += int(pd.notna(f) and f <= 3)
            honmei['pay'].append(pay)
            if pd.notna(o):
                honmei['odds'].append(float(o))

            ev = pd.to_numeric(pd.Series([row.get('期待値')]), errors='coerce').iloc[0]
            conf = pd.to_numeric(pd.Series([row.get('AI信頼度スコア')]), errors='coerce').iloc[0]
            grade = str(row.get('勝負ランク') or '-')
            if pd.notna(ev):
                lo = int(ev // 5 * 5)
                ev_bands.setdefault(f'{lo}-{lo + 5}', []).append(pay)
            if pd.notna(conf):
                lo = int(conf // 10 * 10)
                conf_bands.setdefault(f'{lo}-{lo + 10}', []).append(pay)
            rank_bands.setdefault(grade, []).append(pay)

            if str(row.get('投資判定') or '').startswith('買い'):
                buy['n'] += 1
                buy['win'] += int(pd.notna(f) and f == 1)
                buy['top3'] += int(pd.notna(f) and f <= 3)
                buy['pay'].append(pay)
                if pd.notna(o):
                    buy['odds'].append(float(o))
                if pd.notna(ev):
                    buy['ev'].append(float(ev))

        for name in _pick_names(row, '穴馬') + _pick_names(row, '注目馬'):
            ar = lookup(rid, name)
            if ar is None:
                continue
            f, o = ar['着順'], ar['確定オッズ']
            ana['n'] += 1
            ana['win'] += int(pd.notna(f) and f == 1)
            ana['top3'] += int(pd.notna(f) and f <= 3)
            ana['pay'].append(float(o) * STAKE if (pd.notna(f) and f == 1 and pd.notna(o)) else 0.0)
            if pd.notna(o):
                ana['odds'].append(float(o))

        dg = str(row.get('人気馬危険') or '').strip()
        if dg and dg not in ('なし', '-'):
            dr = lookup(rid, dg)
            if dr is not None and pd.notna(dr['着順']):
                danger['n'] += 1
                danger['out_of_top3'] += int(dr['着順'] > 3)
                danger['win'] += int(dr['着順'] == 1)

    def summarize(d: dict, label: str) -> dict:
        pay = np.array(d['pay'], dtype=float)
        lo, hi = _bootstrap_ci(pay, _roi_stat) if len(pay) else (float('nan'), float('nan'))
        return {
            '対象': label,
            '件数': d['n'],
            '勝率': _pct(d['win'], d['n']),
            '複勝率': _pct(d['top3'], d['n']),
            '回収率': round(float(pay.mean() / STAKE * 100), 2) if len(pay) else None,
            'ROI': round(_roi_stat(pay), 2) if len(pay) else None,
            'ROI_90%CI': [round(lo, 1), round(hi, 1)] if len(pay) else None,
            '平均オッズ': round(float(np.mean(d['odds'])), 2) if d['odds'] else None,
        }

    def band_table(bands: dict[str, list[float]], key: str) -> list[dict]:
        rows = []
        for k, v in sorted(bands.items()):
            pay = np.array(v, dtype=float)
            rows.append({key: k, '件数': len(v),
                         '回収率': round(float(pay.mean() / STAKE * 100), 2) if len(pay) else None,
                         '的中率': round(float((pay > 0).mean() * 100), 2) if len(pay) else None})
        return rows

    return {
        '本命': summarize(honmei, '本命単勝'),
        '穴馬': summarize(ana, '注目馬+穴馬 単勝'),
        'BUY': {**summarize(buy, 'BUY本命単勝'),
                '平均期待値': round(float(np.mean(buy['ev'])), 2) if buy['ev'] else None},
        '危険人気馬': {'件数': danger['n'],
                  '3着外に飛んだ率': _pct(danger['out_of_top3'], danger['n']),
                  '1着してしまった率': _pct(danger['win'], danger['n'])},
        '期待値別の実回収率': band_table(ev_bands, '期待値帯'),
        '信頼度別の実績': band_table(conf_bands, 'AI信頼度帯'),
        '勝負ランク別の実績': band_table(rank_bands, '勝負ランク'),
    }


def market_baseline(results: pd.DataFrame, race_ids: set[str]) -> dict:
    r = results[results['race_id'].isin(race_ids)]
    out = {}
    for pop in (1, 2, 3):
        g = r[r['人気'] == pop]
        pay = np.where(g['着順'] == 1, g['確定オッズ'].fillna(0) * STAKE, 0.0)
        out[f'{pop}番人気'] = {
            '件数': int(len(g)),
            '勝率': round(float((g['着順'] == 1).mean() * 100), 2) if len(g) else None,
            '複勝率': round(float((g['着順'] <= 3).mean() * 100), 2) if len(g) else None,
            'ROI': round(_roi_stat(pay), 2) if len(g) else None,
        }
    allh = r[pd.notna(r['確定オッズ'])]
    pay = np.where(allh['着順'] == 1, allh['確定オッズ'] * STAKE, 0.0)
    out['全馬均等単勝'] = {'件数': int(len(allh)), 'ROI': round(_roi_stat(pay), 2)}
    return out


# ---------------------------------------------------------------------- main

def run(label: str, dates: list[str], *, sim_runs: int, use_cache: bool) -> dict:
    history = load_history()
    results = load_results()
    races, runners = [], []
    for d in dates:
        t0 = time.time()
        got = replay_date(d, history, sim_runs=sim_runs, label=label, use_cache=use_cache)
        if got is None:
            print(f'[qbt] {d} skip (no runners)', flush=True)
            continue
        rc, rn = got
        races.append(rc)
        runners.append(rn)
        print(f'[qbt] {d} races={len(rc)} runners={len(rn)} {time.time() - t0:.1f}s', flush=True)
    race = pd.concat(races, ignore_index=True)
    runner = pd.concat(runners, ignore_index=True)
    race['race_id'] = race['race_id'].astype(str)
    runner['race_id'] = runner['race_id'].astype(str)
    runner['馬名'] = runner['馬名'].astype(str).str.strip()

    def section(tag: str, rc: pd.DataFrame, rn: pd.DataFrame) -> dict:
        if rc.empty:
            return {'開催日数': 0, 'レース数': 0, '確率モデル': {}, '運用成績': {}, '市場ベースライン': {}}
        return {
            '開催日数': int(rc['date'].nunique()),
            'レース数': int(rc['race_id'].nunique()),
            '確率モデル': runner_metrics(rn, results),
            '運用成績': race_metrics(rc, results),
            '市場ベースライン': market_baseline(results, set(rc['race_id'])),
        }

    train_r = race[race['date'] < HOLDOUT_FROM]
    hold_r = race[race['date'] >= HOLDOUT_FROM]
    train_n = runner[runner['date'] < HOLDOUT_FROM]
    hold_n = runner[runner['date'] >= HOLDOUT_FROM]

    return {
        'label': label,
        '設計': {'holdout開始日': HOLDOUT_FROM, 'SIM_RUNS': sim_runs,
               '対象日数': len(dates), '1点': STAKE,
               '環境': {k: os.environ.get(k) for k in
                      ('ARERU_LEGACY_SCORE', 'ARERU_LOGIC_PRESET', 'ARERU_PROB_V2',
                       'ARERU_AI_WEIGHT', 'ARERU_PAST_FEATURES', 'ARERU_PAST_PANEL',
                       'ARERU_PAST_MODEL') if os.environ.get(k)}},
        'full': section('full', race, runner),
        'train': section('train', train_r, train_n),
        'holdout': section('holdout', hold_r, hold_n),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='base')
    ap.add_argument('--sim-runs', type=int, default=3000)
    ap.add_argument('--dates', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()

    results = load_results()
    dates = a.dates.split(',') if a.dates else result_dates(results)
    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    from areru_engine import parse_date
    have = set(parse_date(runners['日付']).dt.strftime('%Y-%m-%d').dropna())
    dates = [d for d in dates if d in have]
    if a.limit:
        dates = dates[:a.limit]

    rep = run(a.label, dates, sim_runs=a.sim_runs, use_cache=not a.no_cache)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f'{a.label}.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(f'\nwrote {p}')
    for tag in ('full', 'train', 'holdout'):
        s = rep[tag]
        if not s['レース数']:
            continue
        print(f"\n--- {tag}: {s['レース数']}R ---")
        print('  本命      ', s['運用成績']['本命'])
        print('  BUY       ', s['運用成績']['BUY'])
        print('  市場1番人気', s['市場ベースライン']['1番人気'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
