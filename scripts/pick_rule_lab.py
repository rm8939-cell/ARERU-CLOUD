#!/usr/bin/env python3
"""穴馬・危険人気馬の定義候補を、確率v2 のパネル上で比較する。

どちらも「人気薄だから」「AI順位が低いから」では決めない。
穴馬は複勝圏に来る確率で、危険人気馬は同じ人気帯の平均と比べて評価する。
人気帯を揃えないと「3番人気を選べば48%は飛ぶ」を実力と誤認してしまう。
"""
from __future__ import annotations

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


def load(label: str) -> pd.DataFrame:
    cdir = DATA / 'quality_backtest_cache' / label
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
    df['p_fin'] = pd.to_numeric(df['SIM勝率'], errors='coerce')
    df['p3_fin'] = pd.to_numeric(df['SIM3着内率'], errors='coerce')
    df['rank_win'] = df.groupby('race_id')['p_fin'].rank(ascending=False, method='first')
    df['rank_p3'] = df.groupby('race_id')['p3_fin'].rank(ascending=False, method='first')
    # 控除率を抜いた市場勝率。オッズの絶対値ではなくレース内の支持率で見る。
    impl = 1.0 / df['odds']
    df['p_mkt'] = impl / df.groupby('race_id')['_impl_sum'].transform('first') * 100 \
        if '_impl_sum' in df.columns else impl / df.groupby('race_id')[
            'odds'].transform(lambda s: (1.0 / s).sum()) * 100
    return df


def pop_matched_baseline(df: pd.DataFrame, picks: pd.DataFrame, col: str) -> float:
    """選ばれた馬と同じ人気構成の馬が、平均でどうなるかの基準値。"""
    base = df.groupby('確定人気')[col].mean()
    w = picks['確定人気'].value_counts(normalize=True)
    common = base.index.intersection(w.index)
    if len(common) == 0:
        return float('nan')
    return float((base.loc[common] * w.loc[common]).sum() / w.loc[common].sum())


def eval_ana(df: pd.DataFrame, picks: pd.DataFrame, label: str) -> dict:
    if picks.empty:
        return {'定義': label, '件数': 0}
    pay = np.where(picks['win'] == 1, picks['結果オッズ'].fillna(0) * STAKE, 0.0)
    return {
        '定義': label,
        '件数': int(len(picks)),
        '複勝率': round(float(picks['top3'].mean()) * 100, 2),
        '同人気帯の複勝率': round(pop_matched_baseline(df, picks, 'top3') * 100, 2),
        '勝率': round(float(picks['win'].mean()) * 100, 2),
        '単勝ROI': round(float(pay.mean()) - 100, 2),
        '平均人気': round(float(picks['確定人気'].mean()), 2),
        '平均オッズ': round(float(picks['結果オッズ'].mean()), 2),
    }


def eval_danger(df: pd.DataFrame, picks: pd.DataFrame, label: str) -> dict:
    if picks.empty:
        return {'定義': label, '件数': 0}
    out3 = (picks['着順'] > 3).mean()
    base = 1.0 - pop_matched_baseline(df, picks, 'top3')
    return {
        '定義': label,
        '件数': int(len(picks)),
        '3着外率': round(float(out3) * 100, 2),
        '同人気帯の3着外率': round(float(base) * 100, 2),
        '上乗せpp': round(float(out3 - base) * 100, 2),
        '1着してしまった率': round(float((picks['着順'] == 1).mean()) * 100, 2),
        '平均人気': round(float(picks['確定人気'].mean()), 2),
    }


def main() -> int:
    df = load('prob_v2')
    tr = df[df['date'] < HOLDOUT_FROM]
    ho = df[df['date'] >= HOLDOUT_FROM]

    for frame, tag in ((tr, 'train'), (ho, 'holdout')):
        f = frame.copy()
        print(f'\n########## {tag}  {f["race_id"].nunique()}R ##########')

        print('\n--- 穴馬（本命・対抗を除いた相手）の定義候補 ---')
        others = f[f['rank_win'] >= 3]
        rows = [
            eval_ana(f, others[others['rank_p3'] <= 5],
                     '複勝確率 3〜5位（確率順の相手）'),
            eval_ana(f, others[others['確定人気'] >= 6],
                     '人気6番手以降（旧: 人気薄だから穴）'),
            eval_ana(f, others.sort_values('p3_fin', ascending=False)
                     .groupby('race_id').head(3),
                     '複勝確率 上位3頭'),
            eval_ana(f, others[(others['p3_fin'] >= 30) & (others['確定人気'] >= 4)],
                     '複勝確率30%以上 かつ 4番人気以降'),
        ]
        for r in rows:
            print(' ', json.dumps(r, ensure_ascii=False))

        print('\n--- 危険人気馬の定義候補（同じ人気帯との差で見る）---')
        rows = [
            eval_danger(f, f[f['確定人気'] <= 3], '1〜3番人気すべて（基準確認）'),
            eval_danger(f, f[(f['確定人気'] <= 3) & (f['rank_p3'] >= 4)],
                        '上位人気だが複勝確率4位以下'),
            eval_danger(f, f[(f['確定人気'] <= 3) & (f['odds'] >= 5.0)],
                        '上位人気だがオッズ5倍以上（弱い人気馬）'),
            eval_danger(f, f[(f['確定人気'] == 1) & (f['odds'] >= 4.0)],
                        '1番人気だがオッズ4倍以上'),
            eval_danger(f, f[(f['確定人気'] <= 3) & (f['頭数'] >= 14)],
                        '上位人気 × 多頭数'),
            eval_danger(f, f[(f['確定人気'] == 1) & (f['p_mkt'] < 30)],
                        '1番人気だが市場勝率30%未満'),
            eval_danger(f, f[(f['確定人気'] <= 2) & (f['p_mkt'] < 25)],
                        '1〜2番人気だが市場勝率25%未満'),
            eval_danger(f, f[(f['確定人気'] <= 3) & (f['p_mkt'] < 22)],
                        '1〜3番人気だが市場勝率22%未満'),
            eval_danger(f, f[(f['確定人気'] <= 3) & (f['p_mkt'] < 22) & (f['頭数'] >= 12)],
                        '1〜3番人気 × 市場勝率22%未満 × 12頭立て以上'),
        ]
        for r in rows:
            print(' ', json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
