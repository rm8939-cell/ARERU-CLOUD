#!/usr/bin/env python3
"""「市場オッズに上乗せできる情報」が既存データから本当に作れるかを検定する。

現行エンジンの特徴（着順・人気の再符号化）は holdout で市場を改善しない。
ここでは、人気・着順に依存しない素材＝走破タイムから速度指数を作り、
同じ条件付きロジットで上乗せ分を測る。

リーク対策
  - コース基準タイムは「そのレース日より前」の履歴だけで作る（expanding）。
  - 馬ごとの特徴も「そのレース日より前」の走りだけを使う。
  - 市場は予想時点の単勝オッズのみ。確定オッズは使わない。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
HOLDOUT_FROM = '2026-08-10'
MIN_KEY_SAMPLES = 20

from market_edge_probe import (  # noqa: E402
    auc, fit_conditional_logit, load_panel, predict_cl, race_logloss, top1_rate,
)

# クラスの上下。中央・地方をまたいで同じ物差しに乗せる（高いほど上）。
CLASS_PATTERNS: list[tuple[str, float]] = [
    (r'G1|GI(?![IV])|ジーワン', 10.0), (r'G2|GII(?!I)', 9.0), (r'G3|GIII', 8.0),
    (r'Jpn1|JpnI(?![IV])', 9.5), (r'Jpn2|JpnII(?!I)', 8.5), (r'Jpn3|JpnIII', 7.5),
    (r'重賞', 8.0), (r'オープン|ＯＰ|OP特別', 7.0), (r'リステッド|L\b', 7.2),
    (r'3勝|1600万', 6.0), (r'2勝|1000万', 5.0), (r'1勝|500万', 4.0),
    (r'未勝利', 2.0), (r'新馬|メイクデビュー', 1.5),
    (r'\bA1\b|Ａ１', 6.5), (r'\bA2\b|Ａ２', 6.0), (r'\bA3\b|Ａ３', 5.6),
    (r'\bA\b|Ａ級', 6.0), (r'\bB1\b|Ｂ１', 5.0), (r'\bB2\b|Ｂ２', 4.6),
    (r'\bB3\b|Ｂ３', 4.3), (r'\bB\b|Ｂ級', 4.6), (r'\bC1\b|Ｃ１', 3.6),
    (r'\bC2\b|Ｃ２', 3.2), (r'\bC3\b|Ｃ３', 2.9), (r'\bC\b|Ｃ級', 3.2),
    (r'\bD\b|Ｄ級', 2.4),
]


def class_rank(name: str) -> float:
    s = str(name or '')
    for pat, v in CLASS_PATTERNS:
        if re.search(pat, s):
            return v
    return 3.5


def to_seconds(v) -> float:
    s = str(v or '').strip()
    m = re.fullmatch(r'(\d+):(\d+)\.(\d)', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10.0
    m = re.fullmatch(r'(\d+)\.(\d+)\.(\d)', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10.0
    return float('nan')


def norm_venue(v) -> str:
    s = re.sub(r'\d+', '', str(v or '')).strip()
    s = re.sub(r'[（(][^）)]*[）)]', '', s).strip()
    return s


def prepare_history() -> pd.DataFrame:
    from history_index import build_master_history
    h = build_master_history().copy()
    h['sec'] = h['タイム'].map(to_seconds)
    d = h['距離'].astype(str)
    h['surface'] = d.str[:1]
    h['dist_m'] = pd.to_numeric(d.str.extract(r'(\d+)')[0], errors='coerce')
    h['venue'] = h['場'].map(norm_venue)
    h['cls'] = h['レース名'].map(class_rank)
    h['finish'] = pd.to_numeric(h['着順'], errors='coerce')
    h['margin'] = pd.to_numeric(h['着差'], errors='coerce')
    h['field'] = pd.to_numeric(h['頭数'], errors='coerce')
    h = h[h['surface'].isin(['芝', 'ダ'])]
    h = h[np.isfinite(h['sec']) & np.isfinite(h['dist_m'])]
    h['key'] = h['venue'] + '|' + h['surface'] + '|' + h['dist_m'].astype(int).astype(str)
    return h.reset_index(drop=True)


def course_baseline(h: pd.DataFrame) -> pd.DataFrame:
    """コース別の基準タイム。各レース日より前のデータだけで作る。"""
    agg = (h.groupby(['key', '_date'])['sec']
             .agg(n='size', s='sum', ss=lambda x: float((x ** 2).sum()))
             .reset_index()
             .sort_values(['key', '_date']))
    g = agg.groupby('key', sort=False)
    agg['cn'] = g['n'].cumsum() - agg['n']
    agg['cs'] = g['s'].cumsum() - agg['s']
    agg['css'] = g['ss'].cumsum() - agg['ss']
    mean = agg['cs'] / agg['cn'].replace(0, np.nan)
    var = agg['css'] / agg['cn'].replace(0, np.nan) - mean ** 2
    agg['base_mean'] = mean
    agg['base_std'] = np.sqrt(var.clip(lower=1e-6))
    agg['base_n'] = agg['cn']
    return agg[['key', '_date', 'base_mean', 'base_std', 'base_n']]


def horse_features(h: pd.DataFrame) -> pd.DataFrame:
    """各走の時点で「その走りまで（当該走を含む）」の累積特徴。"""
    base = course_baseline(h)
    h = h.merge(base, on=['key', '_date'], how='left')
    ok = h['base_n'] >= MIN_KEY_SAMPLES
    # 速いほど高い指数。距離で割って 1 ハロンあたりに直してから標準化。
    h['spd'] = np.where(ok, -(h['sec'] - h['base_mean']) / h['base_std'], np.nan)
    h['spd'] = h['spd'].clip(-4, 4)
    h = h.sort_values(['_horse', '_date']).reset_index(drop=True)
    g = h.groupby('_horse', sort=False)
    out = pd.DataFrame({
        '_horse': h['_horse'],
        '_date': h['_date'],
        'f_spd_last': h['spd'],
        'f_spd_best3': g['spd'].transform(lambda s: s.rolling(3, min_periods=1).max()),
        'f_spd_avg3': g['spd'].transform(lambda s: s.rolling(3, min_periods=1).mean()),
        'f_cls_last': h['cls'],
        'f_cls_best3': g['cls'].transform(lambda s: s.rolling(3, min_periods=1).max()),
        'f_margin_last': h['margin'],
        'f_dist_last': h['dist_m'],
        'f_surface_last': h['surface'],
        'f_venue_last': h['venue'],
        'f_field_last': h['field'],
        'f_runs': g.cumcount() + 1,
    })
    out['f_prev_date'] = h['_date']
    return out


def attach_features(panel: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """各出走馬に、レース日より前の最新の履歴特徴を貼る（merge_asof）。"""
    from areru_engine import clean_name
    p = panel.copy()
    p['_horse'] = p['馬名'].map(clean_name)
    p['_date'] = pd.to_datetime(p['date'])
    p = p.sort_values('_date').reset_index(drop=True)
    f = feats.sort_values('_date').reset_index(drop=True)
    merged = pd.merge_asof(p, f, on='_date', by='_horse', allow_exact_matches=False)
    merged['f_layoff'] = (merged['_date'] - merged['f_prev_date']).dt.days
    return merged


def zscore_in_race(df: pd.DataFrame, col: str) -> np.ndarray:
    v = pd.to_numeric(df[col], errors='coerce')
    g = v.groupby(df['race_id'])
    out = (v - g.transform('mean')) / g.transform('std').replace(0, np.nan)
    return out.fillna(0.0).to_numpy(dtype=float)


CANDIDATES: dict[str, list[str]] = {
    'M0_市場のみ': ['log_impl'],
    'S1_市場+速度指数(直近)': ['log_impl', 'z_spd_last'],
    'S2_市場+速度指数(直近3best)': ['log_impl', 'z_spd_best3'],
    'S3_市場+速度指数(直近3平均)': ['log_impl', 'z_spd_avg3'],
    'S4_市場+クラス': ['log_impl', 'z_cls_best3'],
    'S5_市場+距離変化': ['log_impl', 'z_dist_chg'],
    'S6_市場+休養明け': ['log_impl', 'z_layoff'],
    'S7_市場+出走回数': ['log_impl', 'z_runs'],
    'S8_市場+速度3best+クラス': ['log_impl', 'z_spd_best3', 'z_cls_best3'],
    'S9_市場+速度3best+クラス+休養+距離変化':
        ['log_impl', 'z_spd_best3', 'z_cls_best3', 'z_layoff', 'z_dist_chg'],
    'S10_速度指数のみ(市場なし)': ['z_spd_best3'],
    'S11_市場+現行AI指数+速度3best': ['log_impl', 'z_idx', 'z_spd_best3'],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='prod_old')
    a = ap.parse_args()

    panel = load_panel(a.label)
    h = prepare_history()
    feats = horse_features(h)
    df = attach_features(panel, feats)

    df['dist_chg'] = 0.0  # 今走の距離は data に無い。前走距離の変化幅で代用（後述の限界）
    df['z_idx'] = zscore_in_race(df, 'AREru指数')
    df['z_spd_last'] = zscore_in_race(df, 'f_spd_last')
    df['z_spd_best3'] = zscore_in_race(df, 'f_spd_best3')
    df['z_spd_avg3'] = zscore_in_race(df, 'f_spd_avg3')
    df['z_cls_best3'] = zscore_in_race(df, 'f_cls_best3')
    df['z_layoff'] = zscore_in_race(df, 'f_layoff')
    df['z_runs'] = zscore_in_race(df, 'f_runs')
    df['z_dist_chg'] = zscore_in_race(df, 'f_dist_last')

    cov = {
        '速度指数が付いた割合': round(float(df['f_spd_best3'].notna().mean()) * 100, 1),
        '履歴が紐づいた割合': round(float(df['f_runs'].notna().mean()) * 100, 1),
        '頭数': int(len(df)), 'レース数': int(df['race_id'].nunique()),
    }
    print('カバレッジ:', cov)

    tr = df[df['date'] < HOLDOUT_FROM].reset_index(drop=True)
    ho = df[df['date'] >= HOLDOUT_FROM].reset_index(drop=True)

    rows = []
    for name, cols in CANDIDATES.items():
        Xtr = tr[cols].to_numpy(dtype=float)
        Xho = ho[cols].to_numpy(dtype=float)
        beta = fit_conditional_logit(Xtr, tr['win'].to_numpy(), tr['race_id'].to_numpy())
        p_ho = predict_cl(Xho, beta, ho['race_id'].to_numpy())
        p_tr = predict_cl(Xtr, beta, tr['race_id'].to_numpy())
        rows.append({
            'model': name,
            '係数': {c: round(float(b), 4) for c, b in zip(cols, beta)},
            'train_logloss': round(race_logloss(p_tr, tr['win'].to_numpy(), tr['race_id'].to_numpy()), 5),
            'holdout_logloss': round(race_logloss(p_ho, ho['win'].to_numpy(), ho['race_id'].to_numpy()), 5),
            'holdout_AUC': round(auc(p_ho, ho['win'].to_numpy()), 4),
            'holdout_1位的中': round(top1_rate(p_ho, ho['win'].to_numpy(), ho['race_id'].to_numpy()), 2),
        })
    base = rows[0]['holdout_logloss']
    for r in rows:
        r['holdout_logloss改善'] = round(base - r['holdout_logloss'], 5)

    out = {'設計': {'label': a.label, 'holdout開始': HOLDOUT_FROM,
                  'train_races': int(tr['race_id'].nunique()),
                  'holdout_races': int(ho['race_id'].nunique()),
                  'カバレッジ': cov,
                  '注': 'コース基準タイム・馬の特徴ともレース日より前のみ。確定オッズ不使用。'},
           'results': rows}
    p = DATA / 'quality_backtest' / f'feature_signal_{a.label}.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in rows:
        print(f"{r['model']:42s} holdout_ll={r['holdout_logloss']:.5f} "
              f"改善={r['holdout_logloss改善']:+.5f} AUC={r['holdout_AUC']:.4f} "
              f"1位={r['holdout_1位的中']:.2f}% 係数={r['係数']}")
    print('wrote', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
