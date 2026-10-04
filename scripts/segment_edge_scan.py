#!/usr/bin/env python3
"""AI指数が市場に上乗せできるセグメントが存在するかを走査する。

全体で上乗せゼロでも、特定条件（JRA/地方・頭数・人気帯・履歴件数）では
情報を持つ可能性がある。train で係数を出し、holdout で確認する。
係数の符号が train と holdout で一致し、かつ holdout の対数損失が
市場単独より改善したセグメントだけを「信号あり」とみなす。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
HOLDOUT_FROM = '2026-08-10'

from market_edge_probe import fit_conditional_logit, predict_cl, race_logloss  # noqa: E402
from policy_lab import load  # noqa: E402


def blend_grid(df_tr: pd.DataFrame, df_ho: pd.DataFrame) -> list[dict]:
    """p ∝ p_mkt^a * exp(w·z_idx) の w を振って、素直に効くかを見る。"""
    rows = []
    for w in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        out = {}
        for frame, tag in ((df_tr, 'train'), (df_ho, 'holdout')):
            eta = 1.0 * np.log(frame['p_mkt'].clip(1e-6)) + w * frame['z_idx']
            s = pd.Series(eta, index=frame.index)
            e = np.exp(s - s.groupby(frame['race_id']).transform('max'))
            p = (e / e.groupby(frame['race_id']).transform('sum')).to_numpy()
            out[f'{tag}_logloss'] = round(
                race_logloss(p, frame['win'].to_numpy(), frame['race_id'].to_numpy()), 5)
            d = pd.DataFrame({'p': p, 'win': frame['win'].to_numpy(),
                              'o': frame['結果オッズ'].to_numpy(), 'g': frame['race_id'].to_numpy()})
            pick = d.loc[d.groupby('g')['p'].idxmax()]
            pay = np.where(pick['win'] == 1, np.nan_to_num(pick['o']) * 100, 0.0)
            out[f'{tag}_ROI'] = round(float(pay.mean()) - 100, 2)
            out[f'{tag}_1位的中'] = round(float(pick['win'].mean()) * 100, 2)
        rows.append({'AI重みw': w, **out})
    return rows


SEGMENTS = {
    'JRA': lambda d: d['source'].astype(str) == 'jra',
    '地方': lambda d: d['source'].astype(str) == 'nar',
    '少頭数(≤9)': lambda d: d['頭数'] <= 9,
    '中頭数(10-13)': lambda d: (d['頭数'] >= 10) & (d['頭数'] <= 13),
    '多頭数(≥14)': lambda d: d['頭数'] >= 14,
    '上位人気(1-3)': lambda d: d['確定人気'] <= 3,
    '中人気(4-8)': lambda d: (d['確定人気'] >= 4) & (d['確定人気'] <= 8),
    '人気薄(≥9)': lambda d: d['確定人気'] >= 9,
}


def main() -> int:
    df = load()
    tr = df[df['date'] < HOLDOUT_FROM].reset_index(drop=True)
    ho = df[df['date'] >= HOLDOUT_FROM].reset_index(drop=True)

    grid = blend_grid(tr, ho)
    print('=== AI指数のブレンド重み走査（p ∝ 市場確率 × exp(w·z指数)）===')
    for r in grid:
        print(' ', json.dumps(r, ensure_ascii=False))

    print('\n=== セグメント別の上乗せ検定 ===')
    seg_rows = []
    for name, fn in SEGMENTS.items():
        # セグメント条件はレース単位で揃える（馬単位条件はレースを丸ごと採る）
        t = tr[fn(tr).fillna(False)]
        h = ho[fn(ho).fillna(False)]
        t = tr[tr['race_id'].isin(t['race_id'])]
        h = ho[ho['race_id'].isin(h['race_id'])]
        if t['race_id'].nunique() < 80 or h['race_id'].nunique() < 40:
            seg_rows.append({'セグメント': name, '判定': 'サンプル不足',
                             'train_R': int(t['race_id'].nunique()),
                             'holdout_R': int(h['race_id'].nunique())})
            continue
        cols = ['log_mkt', 'z_idx']
        for f in (t, h):
            f['log_mkt'] = np.log(f['p_mkt'].clip(1e-6))
        b_full = fit_conditional_logit(t[cols].to_numpy(float), t['win'].to_numpy(),
                                       t['race_id'].to_numpy())
        b_mkt = fit_conditional_logit(t[['log_mkt']].to_numpy(float), t['win'].to_numpy(),
                                      t['race_id'].to_numpy())
        p_full = predict_cl(h[cols].to_numpy(float), b_full, h['race_id'].to_numpy())
        p_mkt = predict_cl(h[['log_mkt']].to_numpy(float), b_mkt, h['race_id'].to_numpy())
        ll_full = race_logloss(p_full, h['win'].to_numpy(), h['race_id'].to_numpy())
        ll_mkt = race_logloss(p_mkt, h['win'].to_numpy(), h['race_id'].to_numpy())
        # holdout 単独でも係数を推定し、符号が train と一致するかを見る
        b_ho = fit_conditional_logit(h[cols].to_numpy(float), h['win'].to_numpy(),
                                     h['race_id'].to_numpy())
        same_sign = bool(np.sign(b_full[1]) == np.sign(b_ho[1]) and abs(b_full[1]) > 1e-3)
        seg_rows.append({
            'セグメント': name,
            'train_R': int(t['race_id'].nunique()), 'holdout_R': int(h['race_id'].nunique()),
            'train係数_AI指数': round(float(b_full[1]), 4),
            'holdout係数_AI指数': round(float(b_ho[1]), 4),
            '符号一致': same_sign,
            'holdout_logloss改善': round(ll_mkt - ll_full, 5),
            '判定': '信号あり' if (same_sign and ll_mkt - ll_full > 0.005) else '信号なし',
        })
    for r in seg_rows:
        print(' ', json.dumps(r, ensure_ascii=False))

    out = {'ブレンド重み走査': grid, 'セグメント検定': seg_rows}
    p = DATA / 'quality_backtest' / 'segment_edge_scan.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\nwrote', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
