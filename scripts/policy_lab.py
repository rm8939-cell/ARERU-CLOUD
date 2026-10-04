#!/usr/bin/env python3
"""本命選定・期待値・BUY 方針の候補を、同じ馬単位パネル上で比較する実験台。

エンジンを触る前に「どの方針なら holdout で改善するか」をここで確かめる。
採用した方針だけを areru_engine / ev_analysis に実装する。

重要な前提
  - 市場確率は Σ(1/オッズ) で正規化する。控除率を抜かないと期待値 100% が
    「トントン」にならず、BUY 条件が実質ザルになる。
  - 予想時点の単勝オッズのみ使用。確定オッズは払戻計算だけに使う。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
HOLDOUT_FROM = '2026-08-10'
STAKE = 100


def load() -> pd.DataFrame:
    cdir = DATA / 'quality_backtest_cache' / 'prod_old'
    rn = pd.concat([pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
                    for f in sorted(glob.glob(str(cdir / 'runner_*.csv')))], ignore_index=True)
    res = pd.read_csv(DATA / 'results.csv', encoding='utf-8-sig', low_memory=False)
    res['race_id'] = res['race_id'].astype(str)
    res['馬名'] = res['馬名'].astype(str).str.strip()
    res = res[['race_id', '馬名', '着順', '確定オッズ', '人気']].rename(
        columns={'確定オッズ': '結果オッズ', '人気': '確定人気'})
    rn['race_id'] = rn['race_id'].astype(str)
    rn['馬名'] = rn['馬名'].astype(str).str.strip()
    df = rn.drop(columns=[c for c in ('人気',) if c in rn.columns]).merge(
        res, on=['race_id', '馬名'], how='inner')
    df = df[pd.notna(df['着順'])].copy()
    df['win'] = (df['着順'] == 1).astype(float)
    df['top3'] = (df['着順'] <= 3).astype(float)
    df['odds'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    full = df.groupby('race_id')['odds'].transform(lambda s: s.notna().all() & (s > 1.0).all())
    df = df[full.fillna(False)].reset_index(drop=True)
    df['raw_impl'] = 1.0 / df['odds']
    df['overround'] = df.groupby('race_id')['raw_impl'].transform('sum')
    df['p_mkt'] = df['raw_impl'] / df['overround']          # 控除率を抜いた市場確率
    df['p_sim'] = pd.to_numeric(df['SIM勝率'], errors='coerce').fillna(0) / 100.0
    df['p_sim'] = df['p_sim'] / df.groupby('race_id')['p_sim'].transform('sum').replace(0, np.nan)
    v = pd.to_numeric(df['AREru指数'], errors='coerce')
    df['z_idx'] = ((v - v.groupby(df['race_id']).transform('mean'))
                   / v.groupby(df['race_id']).transform('std').replace(0, np.nan)).fillna(0.0)
    return df


def softmax_in_race(df: pd.DataFrame, eta: np.ndarray) -> np.ndarray:
    s = pd.Series(eta, index=df.index)
    g = s.groupby(df['race_id'])
    e = np.exp(s - g.transform('max'))
    return (e / e.groupby(df['race_id']).transform('sum')).to_numpy()


def eval_policy(df: pd.DataFrame, pick_col: str, *, bet_mask: pd.Series | None = None,
                label: str = '') -> dict:
    """各レースで pick_col が最大の馬に単勝 100 円。bet_mask で見送りを表現。"""
    idx = df.groupby('race_id')[pick_col].idxmax()
    picks = df.loc[idx]
    if bet_mask is not None:
        picks = picks[bet_mask.loc[picks.index].fillna(False)]
    if picks.empty:
        return {'方針': label, '件数': 0}
    pay = np.where(picks['win'] == 1, picks['結果オッズ'].fillna(0) * STAKE, 0.0)
    roi = pay.mean() / STAKE * 100 - 100
    rng = np.random.default_rng(11)
    bs = pay[rng.integers(0, len(pay), size=(2000, len(pay)))].mean(axis=1) / STAKE * 100 - 100
    return {
        '方針': label,
        '件数': int(len(picks)),
        '勝率': round(float(picks['win'].mean()) * 100, 2),
        '複勝率': round(float(picks['top3'].mean()) * 100, 2),
        '回収率': round(float(pay.mean() / STAKE * 100), 2),
        'ROI': round(float(roi), 2),
        'ROI_90%CI': [round(float(np.percentile(bs, 5)), 1), round(float(np.percentile(bs, 95)), 1)],
        '平均オッズ': round(float(picks['結果オッズ'].mean()), 2),
        '平均人気': round(float(picks['確定人気'].mean()), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='policy_lab')
    a = ap.parse_args()
    df = load()
    tr = df[df['date'] < HOLDOUT_FROM]
    ho = df[df['date'] >= HOLDOUT_FROM]

    print(f'train {tr["race_id"].nunique()}R  holdout {ho["race_id"].nunique()}R')
    print(f'控除率（Σ1/オッズ の中央値）: {df.groupby("race_id")["overround"].first().median():.3f}')

    # --- train でブレンド重みと鋭さを学習（holdout は一切触らない） ----------
    from market_edge_probe import fit_conditional_logit
    Xtr = tr[['raw_impl']].copy()
    Xtr['log_mkt'] = np.log(tr['p_mkt'].clip(1e-6))
    Xtr['z_idx'] = tr['z_idx']
    beta = fit_conditional_logit(Xtr[['log_mkt', 'z_idx']].to_numpy(dtype=float),
                                 tr['win'].to_numpy(), tr['race_id'].to_numpy())
    print(f'train 学習係数: log_market={beta[0]:.4f}  AI指数={beta[1]:.4f}')

    for frame, tag in ((tr, 'train'), (ho, 'holdout')):
        f = frame.copy()
        f['p_blend'] = softmax_in_race(
            f, beta[0] * np.log(f['p_mkt'].clip(1e-6)) + beta[1] * f['z_idx'])
        # 期待値（真の損益分岐 = 100）
        f['ev_true_sim'] = f['p_sim'] * f['odds'] * 100
        f['ev_true_blend'] = f['p_blend'] * f['odds'] * 100
        # 現行実装の期待値（控除率を抜かない = 水増し）
        f['ev_current_style'] = f['raw_impl'] * f['odds'] * 100

        rows = [
            eval_policy(f, 'p_mkt', label='市場1番人気（基準）'),
            eval_policy(f, 'p_sim', label='現行 SIM勝率 最大'),
            eval_policy(f, 'SIM3着内率', label='現行 本命（SIM3着内率 最大）'),
            eval_policy(f, 'AREru指数', label='AI指数 最大'),
            eval_policy(f, 'p_blend', label='市場+AI指数 の学習ブレンド 最大'),
            eval_policy(f, 'ev_true_sim', label='EV(SIM) 最大 = 現行EVの思想'),
            eval_policy(f, 'p_mkt',
                        bet_mask=(f['ev_true_blend'] > 100),
                        label='ブレンドEV>100 のレースだけ 1番人気'),
            eval_policy(f, 'p_blend',
                        bet_mask=(f['ev_true_blend'] > 100),
                        label='ブレンドEV>100 のときだけ ブレンド最大'),
            eval_policy(f, 'p_blend',
                        bet_mask=(f['p_blend'] >= 0.35),
                        label='ブレンド勝率35%以上だけ'),
        ]
        print(f'\n=== {tag} ===')
        for r in rows:
            print(' ', json.dumps(r, ensure_ascii=False))

        if tag == 'holdout':
            # 現行の水増しEVが何%の馬を「買い水準」に見せるか
            n_over = int((f['ev_current_style'] >= 108).sum())
            n_true = int((f['ev_true_sim'] >= 108).sum())
            print(f'\n  控除率を抜かない式で EV>=108 になる頭数: {n_over} / {len(f)}')
            print(f'  控除率を抜いた式で  EV>=108 になる頭数: {n_true} / {len(f)}')
            print(f'  市場確率×オッズ の平均（= 真の基準値）: '
                  f'{float((f["p_mkt"] * f["odds"]).mean()) * 100:.1f}%')
            print(f'  1/オッズ×オッズ の平均（= 現行の基準値）: '
                  f'{float((f["raw_impl"] * f["odds"]).mean()) * 100:.1f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
