#!/usr/bin/env python3
"""Sランク到達不能の根因診断 + 旧/新ロジック本命単勝バックテスト。

使い方:
  python3 scripts/diagnose_s_rank.py
  python3 scripts/diagnose_s_rank.py --write data/s_rank_rootcause_backtest.json

検証内容:
  1. 旧 S 必須条件の通過率（展開安定≥65 が 0 件であること）
  2. 展開読みやすさの上限と荒れ指数の分布
  3. 旧 / PR中間 / 新(EV主導) の S/A/B 件数・的中・回収・平均オッズ・平均EV
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


def _fnum(s, default=np.nan):
    try:
        t = str(s).strip().replace('%', '').replace('倍', '').replace(',', '')
        if t in ('', 'nan', 'None', 'なし', '—', '-'):
            return default
        return float(t)
    except Exception:
        return default


def _load_predictions() -> pd.DataFrame:
    rows = []
    for path in sorted(PRED_DIR.glob('predictions_*.csv')):
        day = re.search(r'(\d{4}-\d{2}-\d{2})', path.name)
        df = pd.read_csv(path, dtype=str)
        df['date'] = day.group(1) if day else ''
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _attach_results(P: pd.DataFrame) -> pd.DataFrame:
    for c in (
        'AI信頼度スコア', '能力差スコア', '展開読みやすさ', 'シミュレーション再現率',
        'シミュレーション勝率', '期待値', '本命オッズ', 'レース信頼度スコア', 'データ件数',
    ):
        P[c + '_n'] = P[c].map(_fnum) if c in P.columns else np.nan
    P['本命馬番_k'] = P['本命馬番'].map(
        lambda x: str(int(float(x))) if _fnum(x) == _fnum(x) else str(x).strip()
    )
    R = pd.read_csv(RESULTS, dtype=str)
    R['着順_n'] = pd.to_numeric(R['着順'], errors='coerce')
    W = R[R['着順_n'] == 1][['race_id', '馬番', '確定オッズ']].rename(
        columns={'馬番': 'win_umaban', '確定オッズ': 'win_odds'}
    )
    W['win_umaban'] = W['win_umaban'].astype(str).str.strip()
    W['win_odds'] = pd.to_numeric(W['win_odds'], errors='coerce')
    P = P.merge(W, on='race_id', how='left')
    P['has_result'] = P['win_umaban'].notna()
    P['hit'] = (P['has_result'] & (P['本命馬番_k'] == P['win_umaban'])).astype(int)
    P['payout'] = np.where(
        P['hit'] == 1,
        P['win_odds'].fillna(P['本命オッズ_n']).fillna(0) * 100,
        0.0,
    )
    return P


def _pace_diagnosis(P: pd.DataFrame) -> dict:
    pace = P['展開読みやすさ_n']
    # 展開予想 JSON から荒れ指数
    chaos_vals = []
    for raw in P.get('展開予想', pd.Series(dtype=str)).fillna(''):
        if not isinstance(raw, str) or not raw.strip().startswith('{'):
            continue
        try:
            d = json.loads(raw.replace('NaN', 'null'))
            v = _fnum((d or {}).get('荒れ指数'))
            if v == v:
                chaos_vals.append(v)
        except Exception:
            continue
    ch = pd.Series(chaos_vals, dtype=float)
    return {
        '展開読みやすさ_max': float(pace.max()) if pace.notna().any() else None,
        '展開読みやすさ_ge65_count': int((pace >= 65).sum()),
        '展開読みやすさ_ge60_count': int((pace >= 60).sum()),
        '荒れ指数_n': int(len(ch)),
        '荒れ指数_min': float(ch.min()) if len(ch) else None,
        '荒れ指数_le35_count': int((ch <= 35).sum()) if len(ch) else 0,
        'formula_note': (
            'pace=50 + (chaos<=35:+22 | <=55:+10 | >=80:-18 | >=65:-10) '
            '+ 展開相性(+12/-14) + AI総評(+8/-8). '
            'chaos<=35 が 0 件のため +22 が発火せず観測 max=60 < 旧閾値65。'
        ),
    }


def _old_gate_table(P: pd.DataFrame) -> dict:
    gates = {
        'AI信頼度>=72': P['AI信頼度スコア_n'] >= 72,
        '能力差>=70': P['能力差スコア_n'] >= 70,
        '展開安定>=65': P['展開読みやすさ_n'] >= 65,
        'データ件数>=3': P['データ件数_n'] >= 3,
        '再現性>=62': P['シミュレーション再現率_n'] >= 62,
    }
    out = {}
    for k, m in gates.items():
        out[k] = {'pass': int(m.sum()), 'pass_pct': round(float(m.mean() * 100), 1)}
    all_ok = pd.concat(gates, axis=1).all(axis=1)
    except_pace = pd.concat({k: v for k, v in gates.items() if '展開' not in k}, axis=1).all(axis=1)
    out['ALL_old'] = {'pass': int(all_ok.sum()), 'pass_pct': round(float(all_ok.mean() * 100), 2)}
    out['ALL_except_pace'] = {'pass': int(except_pace.sum())}
    return out


def _rank_slots(n, by_venue):
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


def _buy_cap(n, by_venue):
    if by_venue:
        if n <= 6:
            return 2
        if n <= 9:
            return 3
        return 4
    s, a = _rank_slots(n, False)
    return s + a


def _q_old(r):
    return (
        (r['AI信頼度スコア_n'] or 0) >= 72
        and (r['能力差スコア_n'] or 0) >= 70
        and (r['展開読みやすさ_n'] or 0) >= 65
        and (r['データ件数_n'] or 0) >= 3
        and (r['シミュレーション再現率_n'] or 0) >= 62
    )


def _q_new(r):
    """再現性優先の最小S条件（OOSで+EV未確立のため複合条件は持たない）。"""
    odds = r['本命オッズ_n']
    if odds == odds and odds > 50:
        return False
    return (
        (r['能力差スコア_n'] or 0) >= 80
        and (r['データ件数_n'] or 0) >= 3
    )


def _sc_old(r):
    ev = r['期待値_n'] or 0
    rc = r['レース信頼度スコア_n'] or r['AI信頼度スコア_n'] or 0
    return rc * 0.65 + max(0.0, ev - 100) * 1.8


def _sc_new(r):
    ev = r['期待値_n'] or 0
    rc = r['レース信頼度スコア_n'] or r['AI信頼度スコア_n'] or 0
    ab = r['能力差スコア_n'] or 0
    win = r['シミュレーション勝率_n'] or 0
    odds = r['本命オッズ_n'] or 0
    pen = 8 if win >= 30 else 0
    mid = 5 if 8 <= odds <= 35 else 0
    return ab * 0.55 + rc * 0.35 + max(0.0, min(ev, 112) - 100) * 0.5 + mid - pen


def _assign(P, qualify_fn, score_fn, a_floor, s_floor, buy_ev, buy_conf, buy_ab, buy_odds, ev_demote):
    records = P.to_dict('records')
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
        ss, aa = _rank_slots(n, bv)
        cap = _buy_cap(n, bv)
        ordered = sorted(items, key=lambda x: score_fn(x[1]), reverse=True)
        s_left, a_left = ss, aa
        assigned = {}
        for pos, (idx, r) in enumerate(ordered):
            score = r['レース信頼度スコア_n'] if r['レース信頼度スコア_n'] == r['レース信頼度スコア_n'] else (
                r['AI信頼度スコア_n'] or 50
            )
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
            ev = r['期待値_n']
            if ev == ev and ev is not None and ev < ev_demote and k in ('S', 'A'):
                k = 'B'
            if k == 'S' and not qualify_fn(r):
                k = 'A'
            rk[idx] = k
            assigned[idx] = k
        cands = []
        for idx, r in items:
            if assigned[idx] not in ('S', 'A'):
                continue
            ev = r['期待値_n']
            conf = r['AI信頼度スコア_n'] or 0
            rc = r['レース信頼度スコア_n'] or conf
            repro = r['シミュレーション再現率_n'] or 0
            ab = r['能力差スコア_n'] or 0
            odds = r['本命オッズ_n']
            if ev != ev or ev is None:
                continue
            if ev < buy_ev or conf < buy_conf or rc < buy_conf or repro < 40:
                continue
            if buy_ab is not None and ab < buy_ab:
                continue
            if buy_odds is not None and odds == odds and odds > buy_odds:
                continue
            cands.append((idx, r))
        cands.sort(key=lambda x: score_fn(x[1]), reverse=True)
        for i, _ in cands[:cap]:
            buy[i] = '買い'
    out = P.copy()
    out['_rk'] = rk
    out['_buy'] = buy
    return out


def _metrics(df: pd.DataFrame) -> dict:
    E = df[df['has_result'] & df['本命オッズ_n'].notna() & (df['本命オッズ_n'] > 0)]
    rows = {}
    for key in ('S', 'A', 'B'):
        g = E[E['_rk'] == key]
        if g.empty:
            rows[key] = {'n': 0}
            continue
        inv = len(g) * 100
        pay = float(g['payout'].sum())
        rows[key] = {
            'n': int(len(g)),
            'hits': int(g['hit'].sum()),
            'hit_rate': round(float(g['hit'].mean() * 100), 1),
            'recovery': round(pay / inv * 100, 1),
            'avg_odds': round(float(g['本命オッズ_n'].mean()), 2),
            'avg_ev': round(float(g['期待値_n'].mean()), 1) if g['期待値_n'].notna().any() else None,
            'profit': int(pay - inv),
        }
    b = E[E['_buy'] == '買い']
    if b.empty:
        rows['BUY'] = {'n': 0}
        rows['BUY_S'] = {'n': 0}
    else:
        inv = len(b) * 100
        pay = float(b['payout'].sum())
        rows['BUY'] = {
            'n': int(len(b)),
            'hit_rate': round(float(b['hit'].mean() * 100), 1),
            'recovery': round(pay / inv * 100, 1),
            'avg_odds': round(float(b['本命オッズ_n'].mean()), 2),
            'avg_ev': round(float(b['期待値_n'].mean()), 1) if b['期待値_n'].notna().any() else None,
        }
        bs = b[b['_rk'] == 'S']
        if bs.empty:
            rows['BUY_S'] = {'n': 0}
        else:
            rows['BUY_S'] = {
                'n': int(len(bs)),
                'hit_rate': round(float(bs['hit'].mean() * 100), 1),
                'recovery': round(float(bs['payout'].sum() / (len(bs) * 100) * 100), 1),
            }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', default='', help='JSON 出力先')
    args = ap.parse_args()

    P = _attach_results(_load_predictions())
    pace = _pace_diagnosis(P)
    gates = _old_gate_table(P)

    old = _assign(
        P, _q_old, _sc_old, a_floor=40, s_floor=66,
        buy_ev=108, buy_conf=58, buy_ab=None, buy_odds=None, ev_demote=100,
    )
    new = _assign(
        P, _q_new, _sc_new, a_floor=60, s_floor=66,
        buy_ev=100, buy_conf=58, buy_ab=65, buy_odds=50, ev_demote=90,
    )
    old_m = _metrics(old)
    new_m = _metrics(new)

    print('=== 旧S条件 通過率 ===')
    for k, v in gates.items():
        print(f"  {k}: {v}")
    print('=== 展開読みやすさ 到達不能の根拠 ===')
    for k, v in pace.items():
        print(f"  {k}: {v}")
    print('=== バックテスト 本命単勝 ===')
    print('OLD:', old_m)
    print('NEW:', new_m)

    improved = (
        new_m['S'].get('n', 0) > 0
        and new_m['S'].get('recovery', 0) > old_m['A'].get('recovery', 0)
        and new_m['BUY'].get('recovery', 0) > old_m['BUY'].get('recovery', 0)
        and new_m['S'].get('recovery', 0) >= new_m['A'].get('recovery', 0) >= new_m['B'].get('recovery', 0)
    )
    print('IMPROVEMENT_CONFIRMED', improved)

    report = {
        'improvement_confirmed': improved,
        'old_S_gates': gates,
        'pace_unreachable_proof': pace,
        'old_logic': old_m,
        'new_logic': new_m,
        'new_S_definition': {
            '能力差': '>=80',
            'データ件数': '>=3',
            '本命オッズ': '<=50 or NA',
            'removed': ['展開安定>=65', '勝率下限', 'AI>=72', '再現>=62'],
            'meaning': '相対上位ラベル。OOS長期+EVは未確立（s_rank_robustness_report.json）。',
        },
        'notes': [
            '主指標は本命単勝。全体100%超だけで条件を複雑化しない。',
            '旧Sは展開安定>=65 が数学的に通過不能で常に0件。',
            '能力差>=80 は相対優位だが、検証期間では回収崩壊。詳細は頑健性レポートへ。',
        ],
    }
    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
