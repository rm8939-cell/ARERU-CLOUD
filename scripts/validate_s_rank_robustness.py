#!/usr/bin/env python3
"""S/A/B 改善の頑健性検証（過学習チェック）。

使い方:
  python3 scripts/validate_s_rank_robustness.py
  python3 scripts/validate_s_rank_robustness.py --write data/s_rank_robustness_report.json

検証軸:
  1. S/A/B サンプル数
  2. 期間別回収率
  3. 競馬場別回収率
  4. 人気帯別
  5. オッズ帯別
  6. 能力差帯別
  7. 的中率・平均払戻・平均オッズ
  8. S条件の感度分析
  9. 特定期間/競馬場への集中度
  10. 学習/検証のアウトオブサンプル
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

MIN_N_SLICE = 8  # これ未満の帯は参考扱い


def fnum(s, default=np.nan):
    try:
        t = str(s).strip().replace('%', '').replace('倍', '').replace(',', '')
        if t in ('', 'nan', 'None', 'なし', '—', '-'):
            return default
        return float(t)
    except Exception:
        return default


def load_joined() -> pd.DataFrame:
    rows = []
    for path in sorted(PRED_DIR.glob('predictions_*.csv')):
        day = re.search(r'(\d{4}-\d{2}-\d{2})', path.name)
        df = pd.read_csv(path, dtype=str)
        df['date'] = day.group(1) if day else ''
        rows.append(df)
    P = pd.concat(rows, ignore_index=True)
    for c in (
        'AI信頼度スコア', '能力差スコア', '展開読みやすさ', 'シミュレーション再現率',
        'シミュレーション勝率', '期待値', '本命オッズ', 'レース信頼度スコア', 'データ件数',
        '本命人気', '荒れ度',
    ):
        P[c] = P[c].map(fnum) if c in P.columns else np.nan
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
    P['hit'] = (P['本命馬番_k'] == P['win_umaban']).astype(int)
    # 払戻は確定オッズ優先（的中時）。未的中は0。
    P['payout_odds'] = np.where(P['hit'] == 1, P['win_odds'].fillna(P['本命オッズ']), np.nan)
    P['payout'] = np.where(P['hit'] == 1, P['payout_odds'].fillna(0) * 100, 0.0)
    P['investment'] = 100.0
    # 評価は本命オッズがあるレースに限定（現実の投票可能性）
    P = P[P['本命オッズ'].notna() & (P['本命オッズ'] > 0)].copy()
    P['week'] = pd.to_datetime(P['date']).dt.to_period('W').astype(str)
    P['source'] = P['source'].astype(str).str.lower()
    return P.reset_index(drop=True)


def stats(df: pd.DataFrame, label: str = '') -> dict:
    n = len(df)
    if n == 0:
        return {'label': label, 'n': 0}
    hits = int(df['hit'].sum())
    inv = float(df['investment'].sum())
    pay = float(df['payout'].sum())
    hit_odds = df.loc[df['hit'] == 1, 'payout_odds']
    return {
        'label': label,
        'n': n,
        'hits': hits,
        'hit_rate': round(hits / n * 100.0, 1),
        'recovery': round(pay / inv * 100.0, 1) if inv else None,
        'avg_odds': round(float(df['本命オッズ'].mean()), 2),
        'avg_payout_per_bet': round(pay / n, 1),  # 1レース100円あたり平均払戻
        'avg_hit_payout': round(float(hit_odds.mean()) * 100, 1) if len(hit_odds) else None,
        'avg_ev': round(float(df['期待値'].mean()), 1) if df['期待値'].notna().any() else None,
        'profit': int(pay - inv),
        'reliable': n >= MIN_N_SLICE,
    }


def fmt(s: dict) -> str:
    if s.get('n', 0) == 0:
        return f"{s.get('label','')}: n=0"
    flag = '' if s.get('reliable', True) else ' [n小]'
    return (
        f"{s['label']}: n={s['n']:4d} hit={s['hit_rate']:5.1f}% "
        f"rec={s['recovery']:6.1f}% odds={s['avg_odds']:6.2f} "
        f"avg_pay={s['avg_payout_per_bet']:6.1f} "
        f"hit_pay={s['avg_hit_payout']} EV={s['avg_ev']}{flag}"
    )


# ---- rank assignment (aligned with production-ish slots) ----
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


def qualify_factory(ab=80, ai=72, repro=62, nmin=3, odds_max=50.0, win_min=None, win_max=None, pace_min=None):
    def q(r):
        odds = r.get('本命オッズ')
        if odds == odds and odds is not None and odds_max is not None and odds > odds_max:
            return False
        if win_min is not None and (r.get('シミュレーション勝率') or 0) < win_min:
            return False
        if win_max is not None and (r.get('シミュレーション勝率') or 0) > win_max:
            return False
        if pace_min is not None and (r.get('展開読みやすさ') or 0) < pace_min:
            return False
        return (
            (r.get('能力差スコア') or 0) >= ab
            and (r.get('AI信頼度スコア') or 0) >= ai
            and (r.get('シミュレーション再現率') or 0) >= repro
            and (r.get('データ件数') or 0) >= nmin
        )
    return q


def score_ability_first(r):
    ev = r.get('期待値') or 0
    rc = r.get('レース信頼度スコア') or r.get('AI信頼度スコア') or 0
    ab = r.get('能力差スコア') or 0
    win = r.get('シミュレーション勝率') or 0
    odds = r.get('本命オッズ') or 0
    pen = 8 if win >= 30 else 0
    mid = 5 if 8 <= odds <= 35 else 0
    return ab * 0.55 + rc * 0.35 + max(0.0, min(ev, 112) - 100) * 0.5 + mid - pen


def assign_ranks(df: pd.DataFrame, qualify_fn, score_fn=score_ability_first,
                 a_floor=60.0, s_floor=66.0, ev_demote=90.0) -> pd.DataFrame:
    """日付×source でスロット割当。戻り値に _rk 列。"""
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
    for (_key, bv), idxs in groups.items():
        items = [(i, records[i]) for i in idxs]
        n = len(items)
        ss, aa = rank_slots(n, bv)
        ordered = sorted(items, key=lambda x: score_fn(x[1]), reverse=True)
        s_left, a_left = ss, aa
        assigned = {}
        for pos, (idx, r) in enumerate(ordered):
            score = r.get('レース信頼度スコア')
            if score != score or score is None:
                score = r.get('AI信頼度スコア') or 50
            ok = qualify_fn(r)
            if s_left > 0 and ok and score >= s_floor:
                k = 'S'
                s_left -= 1
            elif a_left > 0 and score >= a_floor:
                k = 'A'
                a_left -= 1
            elif score >= 48:
                k = 'B'
            elif score >= 38:
                k = 'C'
            else:
                k = 'D'
            if pos >= ss + aa + max(2, n // 3) and k in ('S', 'A', 'B'):
                k = 'C' if score >= 38 else 'D'
            assigned[idx] = k
        for idx, r in items:
            k = assigned[idx]
            ev = r.get('期待値')
            if ev == ev and ev is not None and ev < ev_demote and k in ('S', 'A'):
                k = 'B'
            if k == 'S' and not qualify_fn(r):
                k = 'A'
            rk[idx] = k
    out = df.copy()
    out['_rk'] = rk
    return out


def slice_table(df: pd.DataFrame, col_fn, labels_order=None) -> list:
    tmp = df.copy()
    tmp['_band'] = col_fn(tmp)
    out = []
    bands = labels_order or sorted(tmp['_band'].dropna().unique(), key=str)
    for b in bands:
        g = tmp[tmp['_band'] == b]
        out.append(stats(g, str(b)))
    return out


def concentration(df: pd.DataFrame, by: str) -> dict:
    """的中払戻の集中度。上位k件が回収に占める割合。"""
    if df.empty or df['hit'].sum() == 0:
        return {'n': len(df), 'hits': 0}
    hits = df[df['hit'] == 1].sort_values('payout', ascending=False)
    total_pay = float(df['payout'].sum())
    inv = float(df['investment'].sum())
    top = {}
    for k in (1, 2, 3, 5):
        share = float(hits['payout'].head(k).sum() / total_pay) if total_pay else 0
        # そのk件を0にした場合の回収
        zeroed = df.copy()
        zeroed.loc[hits.head(k).index, 'payout'] = 0
        top[f'top{k}_payout_share'] = round(share * 100, 1)
        top[f'rec_without_top{k}'] = round(
            float(zeroed['payout'].sum() / inv * 100), 1
        ) if inv else None
    # by group contribution
    grp = df.groupby(by).agg(n=('hit', 'size'), pay=('payout', 'sum'), inv=('investment', 'sum'))
    grp['rec'] = grp['pay'] / grp['inv'] * 100
    grp = grp.sort_values('pay', ascending=False)
    top_groups = []
    for name, row in grp.head(8).iterrows():
        top_groups.append({
            'group': str(name),
            'n': int(row['n']),
            'payout_share': round(float(row['pay'] / total_pay * 100), 1) if total_pay else 0,
            'recovery': round(float(row['rec']), 1),
        })
    return {
        'n': int(len(df)),
        'hits': int(df['hit'].sum()),
        'recovery': round(float(df['payout'].sum() / inv * 100), 1) if inv else None,
        **top,
        'top_groups': top_groups,
    }


def oos_eval(df: pd.DataFrame, qualify_fn, train_end='2026-07-28') -> dict:
    """学習期間で定義を固定し、検証期間の成績だけを見る（定義探索自体は別途）。"""
    tr = df[df['date'] <= train_end]
    te = df[df['date'] > train_end]
    # pure mask（スロット無し）で定義の中身を評価
    def mask_stats(part):
        m = part.apply(lambda r: qualify_fn(r), axis=1)
        return stats(part[m], 'mask')
    # slotted
    tr_a = assign_ranks(tr, qualify_fn)
    te_a = assign_ranks(te, qualify_fn)
    return {
        'train_end': train_end,
        'train_mask': mask_stats(tr),
        'test_mask': mask_stats(te),
        'train_S': stats(tr_a[tr_a['_rk'] == 'S'], 'S'),
        'test_S': stats(te_a[te_a['_rk'] == 'S'], 'S'),
        'train_n_all': int(len(tr)),
        'test_n_all': int(len(te)),
    }


def sensitivity(df: pd.DataFrame) -> list:
    """単純な1軸ずつの感度。複雑组合は避ける。"""
    base = dict(ab=80, ai=72, repro=62, nmin=3, odds_max=50.0)
    variants = [('BASE current', dict(base))]
    # ability
    for ab in (70, 75, 80, 88):
        variants.append((f'ab>={ab}', {**base, 'ab': ab}))
    # AI
    for ai in (0, 55, 60, 65, 72, 78):
        variants.append((f'AI>={ai}', {**base, 'ai': ai}))
    # repro
    for rp in (0, 45, 50, 62, 70):
        variants.append((f'repro>={rp}', {**base, 'repro': rp}))
    # n
    for nmin in (0, 2, 3, 4):
        variants.append((f'n>={nmin}', {**base, 'nmin': nmin}))
    # odds max
    for od in (10, 15, 20, 30, 50, 999):
        variants.append((f'odds<={od}', {**base, 'odds_max': None if od >= 999 else float(od)}))
    # win gates (to show damage)
    variants.append(('+win>=15', {**base, 'win_min': 15}))
    variants.append(('+win>=20', {**base, 'win_min': 20}))
    variants.append(('+win<25', {**base, 'win_max': 25}))
    # pace (unreachable)
    variants.append(('+pace>=65', {**base, 'pace_min': 65}))
    variants.append(('+pace>=50', {**base, 'pace_min': 50}))
    # simplest candidates
    variants.append(('SIMPLE ab>=80 only', dict(ab=80, ai=0, repro=0, nmin=0, odds_max=None)))
    variants.append(('SIMPLE ab>=80 n>=3', dict(ab=80, ai=0, repro=0, nmin=3, odds_max=None)))
    variants.append(('SIMPLE ab>=80 n>=3 odds<=50', dict(ab=80, ai=0, repro=0, nmin=3, odds_max=50.0)))
    variants.append(('SIMPLE ab>=80 AI>=60 n>=3', dict(ab=80, ai=60, repro=0, nmin=3, odds_max=None)))
    variants.append(('SIMPLE ab>=80 AI>=72 n>=3', dict(ab=80, ai=72, repro=0, nmin=3, odds_max=None)))

    # dedupe labels keeping order
    seen = set()
    uniq = []
    for label, kw in variants:
        if label in seen:
            continue
        seen.add(label)
        uniq.append((label, kw))

    out = []
    train_end = '2026-07-28'
    for label, kw in uniq:
        q = qualify_factory(**kw)
        # pure mask all / train / test
        m_all = df.apply(lambda r: q(r), axis=1)
        m_tr = df['date'] <= train_end
        m_te = df['date'] > train_end
        s_all = stats(df[m_all], label)
        s_tr = stats(df[m_all & m_tr], label + '/train')
        s_te = stats(df[m_all & m_te], label + '/test')
        # slotted S
        assigned = assign_ranks(df, q)
        s_slot = stats(assigned[assigned['_rk'] == 'S'], label + '/slotS')
        out.append({
            'label': label,
            'params': {k: (None if v is None else v) for k, v in kw.items()},
            'mask_all': s_all,
            'mask_train': s_tr,
            'mask_test': s_te,
            'slot_S': s_slot,
            # 頑健スコア: test回収を重視。trainだけ高いのは減点
            'oos_ok': (
                s_te.get('n', 0) >= 15
                and (s_te.get('recovery') or 0) >= 80
                and (s_all.get('recovery') or 0) >= 90
            ),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', default='')
    args = ap.parse_args()

    df = load_joined()
    print(f'joined eval races: {len(df)} dates={df["date"].min()}..{df["date"].max()} venues={df["開催地"].nunique()}')

    # Current production-like assignment
    q_curr = qualify_factory(ab=80, ai=72, repro=62, nmin=3, odds_max=50.0)
    ranked = assign_ranks(df, q_curr)

    report = {
        'meta': {
            'n_eval': int(len(df)),
            'date_min': str(df['date'].min()),
            'date_max': str(df['date'].max()),
            'min_n_slice': MIN_N_SLICE,
            'current_S': {
                '能力差': '>=80', 'AI': '>=72', '再現性': '>=62',
                'データ件数': '>=3', 'オッズ': '<=50',
            },
        }
    }

    # 1) sample sizes
    print('\n=== 1) S/A/B サンプル数 ===')
    by_rank = {}
    for rk in ('S', 'A', 'B', 'C', 'D'):
        s = stats(ranked[ranked['_rk'] == rk], rk)
        by_rank[rk] = s
        print(' ', fmt(s))
    report['by_rank'] = by_rank

    S = ranked[ranked['_rk'] == 'S']
    A = ranked[ranked['_rk'] == 'A']
    B = ranked[ranked['_rk'] == 'B']

    # 2) period
    print('\n=== 2) 期間別（週次） S ===')
    weeks = sorted(df['week'].unique())
    period_s = slice_table(S, lambda x: x['week'], weeks)
    period_all_s_mask = slice_table(df[df.apply(lambda r: q_curr(r), axis=1)], lambda x: x['week'], weeks)
    for s in period_s:
        print(' ', fmt(s))
    report['period_S_slotted'] = period_s
    report['period_S_mask'] = period_all_s_mask

    # also half-split
    print('\n--- 前後半 ---')
    for name, part in [
        ('first_half date<=07-28', S[S['date'] <= '2026-07-28']),
        ('second_half date>=07-29', S[S['date'] >= '2026-07-29']),
    ]:
        print(' ', fmt(stats(part, name)))
    report['half_S'] = {
        'train': stats(S[S['date'] <= '2026-07-28'], '<=07-28'),
        'test': stats(S[S['date'] >= '2026-07-29'], '>=07-29'),
    }

    # 3) venue
    print('\n=== 3) 競馬場別 S ===')
    venue_s = slice_table(S, lambda x: x['開催地'])
    venue_s_sorted = sorted(venue_s, key=lambda x: (-(x.get('n') or 0), x.get('label')))
    for s in venue_s_sorted:
        if s['n']:
            print(' ', fmt(s))
    report['venue_S'] = venue_s_sorted
    print(' source JRA/NAR:')
    for src in ('jra', 'nar'):
        print(' ', fmt(stats(S[S['source'] == src], src)))
    report['source_S'] = {
        'jra': stats(S[S['source'] == 'jra'], 'jra'),
        'nar': stats(S[S['source'] == 'nar'], 'nar'),
    }

    # 4) popularity
    print('\n=== 4) 人気帯別（S / 全体マスク能力差80） ===')
    def pop_band(x):
        p = x['本命人気']
        return pd.cut(
            p, bins=[0, 1, 3, 5, 9, 99],
            labels=['1人気', '2-3人気', '4-5人気', '6-9人気', '10人気〜'],
            right=True,
        ).astype(str)

    pop_order = ['1人気', '2-3人気', '4-5人気', '6-9人気', '10人気〜']
    print(' S slotted:')
    pop_s = slice_table(S, pop_band, pop_order)
    for s in pop_s:
        print(' ', fmt(s))
    ab80 = df[df['能力差スコア'] >= 80]
    print(' ability>=80 mask:')
    pop_ab = slice_table(ab80, pop_band, pop_order)
    for s in pop_ab:
        print(' ', fmt(s))
    report['popularity_S'] = pop_s
    report['popularity_ability80'] = pop_ab

    # 5) odds bands
    print('\n=== 5) オッズ帯別 ===')
    def odds_band(x):
        o = x['本命オッズ']
        return pd.cut(
            o, bins=[0, 3, 5, 8, 12, 20, 50, 999],
            labels=['<=3', '3-5', '5-8', '8-12', '12-20', '20-50', '>50'],
            right=True,
        ).astype(str)

    odds_order = ['<=3', '3-5', '5-8', '8-12', '12-20', '20-50', '>50']
    print(' S:')
    odds_s = slice_table(S, odds_band, odds_order)
    for s in odds_s:
        print(' ', fmt(s))
    print(' ability>=80:')
    odds_ab = slice_table(ab80, odds_band, odds_order)
    for s in odds_ab:
        print(' ', fmt(s))
    report['odds_S'] = odds_s
    report['odds_ability80'] = odds_ab

    # 6) ability bands (all ranked + raw)
    print('\n=== 6) 能力差帯別（全レース / ランク別） ===')
    def ab_band(x):
        # discrete levels observed: 32,45,48,62,75,88
        a = x['能力差スコア']
        return pd.cut(
            a, bins=[-1, 40, 50, 70, 80, 100],
            labels=['<=40', '41-50', '51-70', '71-80', '>=80(88)'],
            right=True,
        ).astype(str)

    ab_order = ['<=40', '41-50', '51-70', '71-80', '>=80(88)']
    print(' all main:')
    ab_all = slice_table(df, ab_band, ab_order)
    for s in ab_all:
        print(' ', fmt(s))
    report['ability_all'] = ab_all
    for rk, part in [('S', S), ('A', A), ('B', B)]:
        print(f' {rk}:')
        for s in slice_table(part, ab_band, ab_order):
            if s['n']:
                print('  ', fmt(s))

    # 7 already in stats

    # 8 sensitivity
    print('\n=== 8) 感度分析（mask all / train / test） ===')
    sens = sensitivity(df)
    # sort by test recovery among n_test>=12, then simplicity
    def sens_key(x):
        te = x['mask_test']
        return (
            1 if x['oos_ok'] else 0,
            te.get('recovery') or 0,
            x['mask_all'].get('recovery') or 0,
            -(len(str(x['params']))),
        )
    sens_sorted = sorted(sens, key=sens_key, reverse=True)
    print(f"{'label':40s} {'all':>22s} {'train':>22s} {'test':>22s} oos")
    for x in sens_sorted:
        def short(s):
            if not s.get('n'):
                return 'n=0'
            return f"n={s['n']:3d} rec={s.get('recovery'):6.1f}"
        print(
            f"{x['label']:40s} {short(x['mask_all']):22s} {short(x['mask_train']):22s} "
            f"{short(x['mask_test']):22s} {x['oos_ok']}"
        )
    report['sensitivity'] = sens_sorted

    # 9 concentration
    print('\n=== 9) 偶然性チェック（払戻集中） ===')
    conc_s = concentration(S, '開催地')
    conc_ab = concentration(ab80, '開催地')
    conc_week = concentration(S, 'week')
    print(' S slotted payout concentration:', {k: conc_s[k] for k in conc_s if k != 'top_groups'})
    print(' S top venues:', conc_s['top_groups'][:5])
    print(' ability>=80 concentration:', {k: conc_ab[k] for k in conc_ab if k != 'top_groups'})
    print(' ability>=80 top venues:', conc_ab['top_groups'][:5])
    report['concentration_S'] = conc_s
    report['concentration_ability80'] = conc_ab
    report['concentration_S_by_week'] = concentration(S, 'week')

    # leave-one-venue-out recovery for S
    print('\n--- leave-one-venue-out S recovery ---')
    lovo = []
    for v in sorted(S['開催地'].unique()):
        part = S[S['開催地'] != v]
        s = stats(part, f'without {v}')
        lovo.append(s)
        print(' ', fmt(s))
    report['leave_one_venue_out_S'] = lovo

    # 10 OOS
    print('\n=== 10) アウトオブサンプル ===')
    oos = oos_eval(df, q_curr)
    print(' current def:', json.dumps(oos, ensure_ascii=False, indent=2))
    report['oos_current'] = oos

    # Compare simplest durable candidates on OOS
    simple_defs = {
        'ab>=80': qualify_factory(80, 0, 0, 0, None),
        'ab>=80 n>=3': qualify_factory(80, 0, 0, 3, None),
        'ab>=80 n>=3 odds<=50': qualify_factory(80, 0, 0, 3, 50),
        'ab>=80 AI>=60 n>=3': qualify_factory(80, 60, 0, 3, None),
        'ab>=80 AI>=72 n>=3': qualify_factory(80, 72, 0, 3, None),
        'ab>=80 AI>=72 repro>=62 n>=3 odds<=50 (current)': q_curr,
        'ab>=80 AI>=72 n>=3 odds<=50 (drop repro)': qualify_factory(80, 72, 0, 3, 50),
    }
    print('\n--- OOS comparison (mask) ---')
    oos_cmp = {}
    for name, q in simple_defs.items():
        o = oos_eval(df, q)
        oos_cmp[name] = {
            'train': o['train_mask'],
            'test': o['test_mask'],
            'train_S': o['train_S'],
            'test_S': o['test_S'],
        }
        tr, te = o['train_mask'], o['test_mask']
        print(
            f"  {name:55s} train n={tr.get('n',0):3d} rec={tr.get('recovery')} | "
            f"test n={te.get('n',0):3d} rec={te.get('recovery')}"
        )
    report['oos_simple_defs'] = oos_cmp

    # Final verdict helper
    print('\n=== VERDICT CANDIDATES ===')
    # Prefer: test rec not collapsed, leave-one-venue-out stable, low top1 share, simple
    verdict_notes = []
    curr_te = oos['test_mask'].get('recovery')
    curr_tr = oos['train_mask'].get('recovery')
    if (curr_te or 0) < 60:
        verdict_notes.append(
            f'現行S定義は検証期間回収 {curr_te}% で崩れており、学習期間 {curr_tr}% への依存が大きい。'
        )
    if conc_s.get('top1_payout_share', 0) >= 30:
        verdict_notes.append(
            f"S払戻の {conc_s.get('top1_payout_share')}% が最大1的中に集中（偶然性高）。"
        )
    # find defs where both train and test >= 70 and n_test>=15
    durable = []
    for name, o in oos_cmp.items():
        tr, te = o['train'], o['test']
        if tr.get('n', 0) >= 20 and te.get('n', 0) >= 15:
            if (tr.get('recovery') or 0) >= 70 and (te.get('recovery') or 0) >= 70:
                durable.append((name, tr, te))
    if durable:
        verdict_notes.append('学習・検証の両方で回収≥70%を満たす単純定義あり:')
        for name, tr, te in durable:
            verdict_notes.append(
                f"  - {name}: train rec={tr['recovery']} (n={tr['n']}), test rec={te['recovery']} (n={te['n']})"
            )
    else:
        verdict_notes.append(
            '学習・検証の両方で回収≥70%かつ十分なnを満たす定義は見つからず。'
            '現状データでは「長期+EVのS」を統計的に確定できない。'
        )
        # best by min(train,test) recovery among simple
        scored = []
        for name, o in oos_cmp.items():
            tr, te = o['train'], o['test']
            if tr.get('n', 0) < 20 or te.get('n', 0) < 12:
                continue
            scored.append((
                min(tr.get('recovery') or 0, te.get('recovery') or 0),
                (tr.get('recovery') or 0) + (te.get('recovery') or 0),
                name, tr, te,
            ))
        scored.sort(reverse=True)
        if scored:
            verdict_notes.append('相対的に崩れにくい単純定義（min(train,test)最大）:')
            for row in scored[:5]:
                _, _, name, tr, te = row
                verdict_notes.append(
                    f"  - {name}: train {tr['recovery']}% (n={tr['n']}) / test {te['recovery']}% (n={te['n']})"
                )

    # ability band is the cleanest signal?
    ab_band_s = {s['label']: s for s in ab_all}
    top_ab = ab_band_s.get('>=80(88)', {})
    verdict_notes.append(
        f"能力差帯別では >=80(88) が全期間回収 {top_ab.get('recovery')}% (n={top_ab.get('n')})。"
        f"ただし leave-one-big-hit / 検証後半で大きく低下しうる。"
    )

    report['verdict_notes'] = verdict_notes
    for line in verdict_notes:
        print(line)

    # Recommended stance
    recommendation = {
        'principle': '再現性優先。全体回収100%超だけを根拠に条件を複雑化しない。',
        'signal': '能力差トップ帯（離散値88 / >=80）が唯一の相対優位シグナル。',
        'not_confirmed': '現行の複合S条件（AI72+再現62+オッズ50）のアウトオブサンプル+EVは未確立。',
        'suggested_S': (
            'S必須は能力差>=80 と データ件数>=3 に限定し、'
            'AI/再現/オッズは買い厳選側のソフト条件に回す。'
            'ただし検証期間でも+EVは未達のため、UI上のSは「相対上位」であり '
            '「長期期待値プラス確定」とは断言しない。'
        ),
    }
    report['recommendation'] = recommendation
    print('\nRECOMMENDATION:', json.dumps(recommendation, ensure_ascii=False, indent=2))

    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\nwrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
