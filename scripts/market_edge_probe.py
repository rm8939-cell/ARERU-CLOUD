#!/usr/bin/env python3
"""AI が市場（単勝オッズ）に対して上乗せ情報を持つかを検定する。

レース内 softmax（条件付きロジット）で勝ち馬を当てるモデルを train で学習し、
holdout の対数尤度・AUC・1位的中で比較する。

  M0  市場のみ            : η = b * log(市場暗示確率)
  M1  AI指数のみ          : η = b * z(AREru指数)
  M2  市場 + AI指数        : 上乗せがあるか
  M3  市場 + 各因子        : どの因子に上乗せがあるか
  M4  市場 + SIM3着内率     : 展開/ラップ由来の情報に上乗せがあるか

市場オッズは予想時点の `単勝オッズ`（runners.csv 由来）のみを使う。
確定オッズは払戻計算にしか使わない。
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


def load_panel(label: str) -> pd.DataFrame:
    cdir = DATA / 'quality_backtest_cache' / label
    files = sorted(glob.glob(str(cdir / 'runner_*.csv')))
    if not files:
        raise SystemExit(f'{cdir} にキャッシュがありません。先に quality_backtest.py を実行してください')
    rn = pd.concat([pd.read_csv(f, encoding='utf-8-sig', low_memory=False) for f in files],
                   ignore_index=True)
    res = pd.read_csv(DATA / 'results.csv', encoding='utf-8-sig', low_memory=False)
    res['race_id'] = res['race_id'].astype(str)
    res['馬名'] = res['馬名'].astype(str).str.strip()
    res = res[['race_id', '馬名', '着順', '確定オッズ']].rename(columns={'確定オッズ': '結果オッズ'})
    rn['race_id'] = rn['race_id'].astype(str)
    rn['馬名'] = rn['馬名'].astype(str).str.strip()
    df = rn.merge(res, on=['race_id', '馬名'], how='inner')
    df = df[pd.notna(df['着順'])].copy()
    df['win'] = (df['着順'] == 1).astype(float)
    df['top3'] = (df['着順'] <= 3).astype(float)
    # 予想時点オッズのみ。欠損レースは丸ごと落とす（埋めない）。
    df['odds'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    ok = df.groupby('race_id')['odds'].transform(lambda s: s.notna().all() & (s > 1.0).all())
    df = df[ok.fillna(False)].reset_index(drop=True)
    df['impl'] = 1.0 / df['odds']
    df['impl'] = df.groupby('race_id')['impl'].transform(lambda s: s / s.sum())
    df['log_impl'] = np.log(df['impl'].clip(1e-6))
    return df


def zscore_in_race(df: pd.DataFrame, col: str) -> np.ndarray:
    v = pd.to_numeric(df[col], errors='coerce')
    g = v.groupby(df['race_id'])
    out = (v - g.transform('mean')) / g.transform('std').replace(0, np.nan)
    return out.fillna(0.0).to_numpy(dtype=float)


def _cl_loss_grad(beta, X, y, gidx, n_g, l2):
    eta = X @ beta
    m = np.full(n_g, -np.inf)
    np.maximum.at(m, gidx, eta)
    e = np.exp(eta - m[gidx])
    s = np.zeros(n_g)
    np.add.at(s, gidx, e)
    p = e / s[gidx]
    nll = -(float(eta[y == 1].sum()) - float((m + np.log(s)).sum())) / n_g
    grad = -(X.T @ (y - p)) / n_g + l2 * beta
    return nll + 0.5 * l2 * float(beta @ beta), grad


def fit_conditional_logit(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          *, l2: float = 1e-4, iters: int = 4000, lr: float = 0.08) -> np.ndarray:
    """レース内 softmax の最尤推定（Adam）。収束しない重みは採用しない。"""
    _uniq, gidx = np.unique(groups, return_inverse=True)
    n_g = len(_uniq)
    beta = np.zeros(X.shape[1], dtype=float)
    m_t = np.zeros_like(beta)
    v_t = np.zeros_like(beta)
    prev = None
    for it in range(1, iters + 1):
        loss, grad = _cl_loss_grad(beta, X, y, gidx, n_g, l2)
        m_t = 0.9 * m_t + 0.1 * grad
        v_t = 0.999 * v_t + 0.001 * grad ** 2
        mh = m_t / (1 - 0.9 ** it)
        vh = v_t / (1 - 0.999 ** it)
        beta -= lr * mh / (np.sqrt(vh) + 1e-8)
        if prev is not None and abs(prev - loss) < 1e-9:
            break
        prev = loss
    return beta


def predict_cl(X: np.ndarray, beta: np.ndarray, groups: np.ndarray) -> np.ndarray:
    uniq, gidx = np.unique(groups, return_inverse=True)
    eta = X @ beta
    m = np.full(len(uniq), -np.inf)
    np.maximum.at(m, gidx, eta)
    e = np.exp(eta - m[gidx])
    s = np.zeros(len(uniq))
    np.add.at(s, gidx, e)
    return e / s[gidx]


def race_logloss(p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """勝ち馬1頭あたりの -log p（レース単位の多値対数損失）。"""
    m = y == 1
    return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1.0))))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def top1_rate(p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    d = pd.DataFrame({'p': p, 'y': y, 'g': groups})
    hit = d.loc[d.groupby('g')['p'].idxmax(), 'y'].sum()
    return float(hit / d['g'].nunique() * 100)


FEATURE_SETS: dict[str, list[str]] = {
    'M0_市場のみ': ['log_impl'],
    'M1_AI指数のみ': ['z_idx'],
    'M2_市場+AI指数': ['log_impl', 'z_idx'],
    'M3_市場+6因子': ['log_impl', 'z_perf', 'z_upset', 'z_cons', 'z_trend', 'z_value', 'z_ctx'],
    'M4_市場+SIM3着内率': ['log_impl', 'z_sim3'],
    'M5_市場+市場非依存4因子': ['log_impl', 'z_perf', 'z_cons', 'z_trend', 'z_ctx'],
    'M6_全部': ['log_impl', 'z_idx', 'z_perf', 'z_cons', 'z_trend', 'z_ctx', 'z_sim3'],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='prod_old')
    a = ap.parse_args()

    df = load_panel(a.label)
    df['z_idx'] = zscore_in_race(df, 'AREru指数')
    df['z_perf'] = zscore_in_race(df, '因子_performance')
    df['z_upset'] = zscore_in_race(df, '因子_upset')
    df['z_cons'] = zscore_in_race(df, '因子_consistency')
    df['z_trend'] = zscore_in_race(df, '因子_trend')
    df['z_value'] = zscore_in_race(df, '因子_value')
    df['z_ctx'] = zscore_in_race(df, '因子_context')
    df['z_sim3'] = zscore_in_race(df, 'SIM3着内率')

    tr = df[df['date'] < HOLDOUT_FROM].reset_index(drop=True)
    ho = df[df['date'] >= HOLDOUT_FROM].reset_index(drop=True)
    print(f'train {tr["race_id"].nunique()}R / {len(tr)}頭   holdout {ho["race_id"].nunique()}R / {len(ho)}頭')

    rows = []
    for name, cols in FEATURE_SETS.items():
        Xtr = tr[cols].to_numpy(dtype=float)
        Xho = ho[cols].to_numpy(dtype=float)
        beta = fit_conditional_logit(Xtr, tr['win'].to_numpy(), tr['race_id'].to_numpy())
        p_tr = predict_cl(Xtr, beta, tr['race_id'].to_numpy())
        p_ho = predict_cl(Xho, beta, ho['race_id'].to_numpy())
        rows.append({
            'model': name,
            '係数': {c: round(float(b), 4) for c, b in zip(cols, beta)},
            'train_logloss': round(race_logloss(p_tr, tr['win'].to_numpy(), tr['race_id'].to_numpy()), 5),
            'holdout_logloss': round(race_logloss(p_ho, ho['win'].to_numpy(), ho['race_id'].to_numpy()), 5),
            'holdout_AUC': round(auc(p_ho, ho['win'].to_numpy()), 4),
            'holdout_1位的中': round(top1_rate(p_ho, ho['win'].to_numpy(), ho['race_id'].to_numpy()), 2),
        })

    base = next(r for r in rows if r['model'] == 'M0_市場のみ')
    for r in rows:
        r['holdout_logloss差_vs市場'] = round(r['holdout_logloss'] - base['holdout_logloss'], 5)

    # 本番の SIM勝率 をそのまま確率として使った場合（参考）
    sim = ho['SIM勝率'].astype(float).to_numpy() / 100.0
    sim = np.clip(sim, 1e-6, None)
    g = ho['race_id'].to_numpy()
    s = pd.Series(sim).groupby(g).transform('sum').to_numpy()
    sim_n = sim / s
    rows.append({
        'model': '参考_本番SIM勝率そのまま',
        '係数': {},
        'train_logloss': None,
        'holdout_logloss': round(race_logloss(sim_n, ho['win'].to_numpy(), g), 5),
        'holdout_AUC': round(auc(sim_n, ho['win'].to_numpy()), 4),
        'holdout_1位的中': round(top1_rate(sim_n, ho['win'].to_numpy(), g), 2),
        'holdout_logloss差_vs市場': round(race_logloss(sim_n, ho['win'].to_numpy(), g) - base['holdout_logloss'], 5),
    })

    out = {'設計': {'label': a.label, 'holdout開始': HOLDOUT_FROM,
                  'train_races': int(tr['race_id'].nunique()),
                  'holdout_races': int(ho['race_id'].nunique()),
                  '注': 'オッズは予想時点の単勝オッズのみ。確定オッズは不使用。'},
           'results': rows}
    (DATA / 'quality_backtest').mkdir(parents=True, exist_ok=True)
    p = DATA / 'quality_backtest' / f'market_edge_{a.label}.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    print('wrote', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
