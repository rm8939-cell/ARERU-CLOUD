#!/usr/bin/env python3
"""固定済み新S/A/B/BUY×S の再現性検証のみ（条件再調整禁止）。

  python3 scripts/validate_s_reproducibility.py
  python3 scripts/validate_s_reproducibility.py --write data/s_rank_reproducibility_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / 'data' / 'predictions_by_date'
RESULTS = ROOT / 'data' / 'results.csv'

TRAIN_END = '2026-07-28'
# frozen gates — do not tune here
S_MIN_ABILITY = 80.0
S_MIN_DATA_N = 3
S_MAX_ODDS = 50.0
A_SCORE_FLOOR = 60.0
B_SCORE_FLOOR = 48.0
C_SCORE_FLOOR = 38.0
S_SCORE_FLOOR = 66.0
BUY_EV_FLOOR = 100
BUY_CONF_FLOOR = 58
BUY_ABILITY_FLOOR = 65.0
BUY_ODDS_MAX = 50.0
BUY_REPRO = 40
EV_DEMOTE = 90.0


def fnum(s, default=np.nan):
    try:
        t = str(s).strip().replace('%', '').replace('倍', '').replace(',', '')
        if t in ('', 'nan', 'None', 'なし', '—', '-'):
            return default
        return float(t)
    except Exception:
        return default


def load() -> pd.DataFrame:
    rows = []
    for path in sorted(PRED_DIR.glob('predictions_*.csv')):
        day = re.search(r'(\d{4}-\d{2}-\d{2})', path.name)
        df = pd.read_csv(path, dtype=str)
        df['date'] = day.group(1) if day else ''
        rows.append(df)
    P = pd.concat(rows, ignore_index=True)
    for c in (
        'AI信頼度スコア', '能力差スコア', 'シミュレーション再現率',
        'シミュレーション勝率', '期待値', '本命オッズ', 'レース信頼度スコア', 'データ件数',
    ):
        P[c] = P[c].map(fnum)
    P['本命馬番_k'] = P['本命馬番'].map(
        lambda x: str(int(float(x))) if fnum(x) == fnum(x) else str(x).strip()
    )
    R = pd.read_csv(RESULTS, dtype=str)
    R['着順_n'] = pd.to_numeric(R['着順'], errors='coerce')
    W = R[R['着順_n'] == 1][['race_id', '馬番', '確定オッズ']].rename(
        columns={'馬番': 'win_umaban', '確定オッズ': 'win_odds'}
    )
    W['win_umaban'] = W['win_umaban'].astype(str).str.strip()
    W['win_odds'] = pd.to_numeric(W['win_odds'], errors='coerce')
    P = P.merge(W, on='race_id', how='inner')
    P = P[P['本命オッズ'].notna() & (P['本命オッズ'] > 0)].copy()
    P['hit'] = (P['本命馬番_k'] == P['win_umaban']).astype(int)
    P['payout'] = np.where(
        P['hit'] == 1, P['win_odds'].fillna(P['本命オッズ']).fillna(0) * 100, 0.0
    )
    return P.reset_index(drop=True)


def qualify_s(r):
    odds = r.get('本命オッズ')
    if odds == odds and odds is not None and odds > S_MAX_ODDS:
        return False
    return (r.get('能力差スコア') or 0) >= S_MIN_ABILITY and (r.get('データ件数') or 0) >= S_MIN_DATA_N


def buy_score(r):
    ev = r.get('期待値') or 0
    rc = r.get('レース信頼度スコア') or r.get('AI信頼度スコア') or 0
    ab = r.get('能力差スコア') or 0
    win = r.get('シミュレーション勝率') or 0
    odds = r.get('本命オッズ') or 0
    pen = 8 if win >= 30 else 0
    mid = 5 if 8 <= odds <= 35 else 0
    return ab * 0.55 + rc * 0.35 + max(0.0, min(ev, 112) - 100) * 0.5 + mid - pen


def rank_slots(n, by_venue):
    if by_venue:
        if n <= 6:
            return 1, 1
        if n <= 9:
            return 1, 2
        return 2, 3
    if n <= 12:
        return 1, 4
    if n <= 24:
        return 2, 5
    return 2, 6


def buy_cap(n, by_venue):
    if by_venue:
        if n <= 6:
            return 2
        if n <= 9:
            return 3
        return 4
    s, a = rank_slots(n, False)
    return s + a


def assign(df: pd.DataFrame) -> pd.DataFrame:
    records = df.to_dict('records')
    groups = defaultdict(list)
    for i, r in enumerate(records):
        src = str(r.get('source') or '').lower()
        if src == 'nar':
            key = (r['date'], 'nar', r.get('開催地') or '')
            bv = True
        else:
            key = (r['date'], 'jra', '_')
            bv = False
        groups[(key, bv)].append(i)
    rk = ['D'] * len(records)
    buy = ['見送り'] * len(records)
    for (_key, bv), idxs in groups.items():
        items = [(i, records[i]) for i in idxs]
        n = len(items)
        ss, aa = rank_slots(n, bv)
        cap = buy_cap(n, bv)
        ordered = sorted(items, key=lambda x: buy_score(x[1]), reverse=True)
        s_left, a_left = ss, aa
        assigned = {}
        for pos, (idx, r) in enumerate(ordered):
            score = r.get('レース信頼度スコア')
            if score != score or score is None:
                score = r.get('AI信頼度スコア') or 50
            ok = qualify_s(r)
            if s_left > 0 and ok and score >= S_SCORE_FLOOR:
                k = 'S'
                s_left -= 1
            elif a_left > 0 and score >= A_SCORE_FLOOR:
                k = 'A'
                a_left -= 1
            elif score >= B_SCORE_FLOOR:
                k = 'B'
            elif score >= C_SCORE_FLOOR:
                k = 'C'
            else:
                k = 'D'
            if pos >= ss + aa + max(2, n // 3) and k in ('S', 'A', 'B'):
                k = 'C' if score >= C_SCORE_FLOOR else 'D'
            assigned[idx] = k
        for idx, r in items:
            k = assigned[idx]
            ev = r.get('期待値')
            if ev == ev and ev is not None and ev < EV_DEMOTE and k in ('S', 'A'):
                k = 'B'
            if k == 'S' and not qualify_s(r):
                k = 'A'
            rk[idx] = k
            assigned[idx] = k
        cands = []
        for idx, r in items:
            if assigned[idx] not in ('S', 'A'):
                continue
            ev = r.get('期待値')
            conf = r.get('AI信頼度スコア') or 0
            rc = r.get('レース信頼度スコア') or conf
            repro = r.get('シミュレーション再現率') or 0
            ab = r.get('能力差スコア') or 0
            odds = r.get('本命オッズ')
            if ev != ev or ev is None:
                continue
            if ev < BUY_EV_FLOOR or conf < BUY_CONF_FLOOR or rc < BUY_CONF_FLOOR or repro < BUY_REPRO:
                continue
            if ab < BUY_ABILITY_FLOOR:
                continue
            if odds == odds and odds > BUY_ODDS_MAX:
                continue
            cands.append((idx, r))
        cands.sort(key=lambda x: buy_score(x[1]), reverse=True)
        for i, _ in cands[:cap]:
            buy[i] = '買い'
    out = df.copy()
    out['_rk'] = rk
    out['_buy'] = buy
    return out


def metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {
            'n': 0, 'hits': 0, 'hit_rate': None, 'recovery': None,
            'avg_odds': None, 'avg_ev': None, 'profit': None, 'ge_100': None,
        }
    hits = int(df['hit'].sum())
    inv = n * 100.0
    pay = float(df['payout'].sum())
    rec = round(pay / inv * 100, 1)
    return {
        'n': n,
        'hits': hits,
        'hit_rate': round(hits / n * 100, 1),
        'recovery': rec,
        'avg_odds': round(float(df['本命オッズ'].mean()), 2),
        'avg_ev': round(float(df['期待値'].mean()), 1) if df['期待値'].notna().any() else None,
        'profit': int(pay - inv),
        'ge_100': rec >= 100.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', default='data/s_rank_reproducibility_report.json')
    args = ap.parse_args()

    P = load()
    ranked = assign(P)
    train = ranked[ranked['date'] <= TRAIN_END]
    test = ranked[ranked['date'] > TRAIN_END]

    cohorts = {
        '新S': lambda d: d[d['_rk'] == 'S'],
        '新A': lambda d: d[d['_rk'] == 'A'],
        '新B': lambda d: d[d['_rk'] == 'B'],
        '新BUY×S': lambda d: d[(d['_buy'] == '買い') & (d['_rk'] == 'S')],
    }

    report = {
        'frozen_logic': {
            'S': {'能力差': '>=80', 'データ件数': '>=3', 'オッズ': '<=50'},
            'note': '条件固定。再現性検証のみ。再調整禁止。',
        },
        'split': {
            'train': f'date <= {TRAIN_END}',
            'test': f'date > {TRAIN_END}',
            'train_n_races': int(len(train)),
            'test_n_races': int(len(test)),
            'train_date_range': [str(train['date'].min()), str(train['date'].max())],
            'test_date_range': [str(test['date'].min()), str(test['date'].max())],
        },
        'results': {},
    }

    print('FROZEN LOGIC — reproducibility only')
    print(
        f"train <= {TRAIN_END} n={len(train)} | "
        f"test > {TRAIN_END} n={len(test)}"
    )
    print(f"{'cohort':10s} {'period':7s} {'n':>4s} {'hit%':>6s} {'rec%':>7s} {'odds':>7s} {'EV':>6s}")
    for name, fn in cohorts.items():
        tr, te, al = metrics(fn(train)), metrics(fn(test)), metrics(fn(ranked))
        report['results'][name] = {'all': al, 'train': tr, 'test': te}
        for period, m in [('all', al), ('train', tr), ('test', te)]:
            print(
                f"{name:10s} {period:7s} {m['n']:4d} {str(m['hit_rate']):>6s} "
                f"{str(m['recovery']):>7s} {str(m['avg_odds']):>7s} {str(m['avg_ev']):>6s}"
            )

    s_tr = report['results']['新S']['train']['recovery']
    s_te = report['results']['新S']['test']['recovery']
    bs_tr = report['results']['新BUY×S']['train']['recovery']
    bs_te = report['results']['新BUY×S']['test']['recovery']
    holdout = {
        'reference_all_sample_claims': {'新S': 108.4, '新BUY×S': 121.4},
        '新S_train_recovery': s_tr,
        '新S_test_recovery': s_te,
        '新BUY×S_train_recovery': bs_tr,
        '新BUY×S_test_recovery': bs_te,
        '新S_test_ge_100': bool(s_te is not None and s_te >= 100),
        '新BUY×S_test_ge_100': bool(bs_te is not None and bs_te >= 100),
    }
    if not holdout['新S_test_ge_100'] or not holdout['新BUY×S_test_ge_100']:
        holdout['verdict'] = '過学習の可能性'
        holdout['verdict_detail'] = (
            f'検証期間で新S={s_te}% / 新BUY×S={bs_te}%。'
            f'学習期間は新S={s_tr}% / 新BUY×S={bs_tr}%。'
            '条件は再調整せず、別期間で100%以上を維持できなかった。'
        )
    else:
        holdout['verdict'] = '検証期間でも100%以上を維持'
        holdout['verdict_detail'] = f'検証期間で新S={s_te}% / 新BUY×S={bs_te}%'
    report['holdout_check'] = holdout
    print('\nHOLDOUT:', json.dumps(holdout, ensure_ascii=False, indent=2))

    path = Path(args.write)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
