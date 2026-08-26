"""未使用の取得済み特徴量を、旧ガウスSIMの指数へ加算する。

段階SIMは起動しない（ARERU_LEGACY_SCORE=1 のまま）。BUY閾値は変更しない。
重みはドメイン知識の固定値で、ROIグリッドサーチはしない。
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

UNUSED_SCORE_FEATURES = frozenset({
    'burden', 'sgate', 'sstyle', 'slayoff', 'sweight', 'sjockey', 'sfield',
})


def _num(x):
    return pd.to_numeric(x, errors='coerce')


def _clean(x) -> str:
    return re.sub(r'[\s\u3000]+', '', str(x or '')).strip()


def _parse_weight(v) -> float:
    s = str(v or '').strip()
    m = re.search(r'(\d{3,4})', s)
    return float(m.group(1)) if m else float('nan')


def _surface_of(dist) -> str:
    s = str(dist or '')
    if s.startswith('芝'):
        return '芝'
    if s.startswith('ダ'):
        return 'ダ'
    return ''


def _infer_style(finishes: np.ndarray, pops: np.ndarray) -> float:
    ok = ~np.isnan(finishes)
    if not ok.any():
        return 0.5
    f = finishes[ok]
    p = pops[ok] if pops is not None and len(pops) == len(finishes) else np.full_like(f, np.nan)
    frontish = []
    for fi, pi in zip(f, p):
        if not np.isnan(pi) and pi <= 3 and fi <= 3:
            frontish.append(0.25)
        elif not np.isnan(pi) and pi >= 8 and fi <= 3:
            frontish.append(0.75)
        elif fi <= 2:
            frontish.append(0.35)
        elif fi >= 10:
            frontish.append(0.6)
        else:
            frontish.append(0.5)
    return float(np.clip(np.mean(frontish), 0.05, 0.95))


def _hist_horse(history, horse, target):
    if history is None or getattr(history, 'empty', True):
        return None
    h = history[(history['_horse'] == _clean(horse)) & (history['_date'] < target)]
    if h.empty:
        return None
    return h.sort_values('_date', ascending=False)


def _layoff_days(history, horse, target) -> float:
    h = _hist_horse(history, horse, target)
    if h is None or h.empty:
        return float('nan')
    last = h['_date'].max()
    try:
        return float((pd.Timestamp(target) - pd.Timestamp(last)).days)
    except Exception:
        return float('nan')


def _weight_delta(history, horse, target) -> float:
    h = _hist_horse(history, horse, target)
    if h is None or len(h) < 2:
        return float('nan')
    w = h.head(2)['馬体重'].map(_parse_weight).to_numpy(dtype=float)
    if np.isnan(w).any():
        return float('nan')
    return float(w[0] - w[1])


def _jockey_place_adj(history, jockey, venue, target) -> float:
    if history is None or getattr(history, 'empty', True) or not str(jockey or '').strip():
        return 0.0
    j = _clean(jockey)
    h = history[(history['_date'] < target) & (history['騎手'].map(_clean) == j)].tail(40)
    if len(h) < 5:
        return 0.0
    fin = _num(h['着順'])
    place = float((fin <= 3).mean())
    bonus = (place - 0.22) * 6.0
    same = h[h['場'].astype(str) == str(venue)]
    if len(same) >= 3:
        bonus += (float((_num(same['着順']) <= 3).mean()) - 0.22) * 4.0
    return float(np.clip(bonus, -2.5, 3.0))


def _guess_surface(g: pd.DataFrame, history, target) -> str:
    for _, row in g.iterrows():
        h = _hist_horse(history, row.get('馬名'), target)
        if h is None or h.empty:
            continue
        surf = _surface_of(h.iloc[0].get('距離'))
        if surf:
            return surf
    return 'ダ'


def apply_unused_score_extras(g: pd.DataFrame, history, target, venue: str) -> pd.DataFrame:
    """レース内の AREru指数へ未使用特徴の固定加点を足す。段階SIMは使わない。"""
    from areru_engine import ablation_enabled

    if g is None or g.empty:
        return g
    active = [f for f in UNUSED_SCORE_FEATURES if ablation_enabled(f)]
    if not active:
        return g

    g = g.copy()
    n = len(g)
    kgs = _num(g['斤量']) if '斤量' in g.columns else pd.Series(np.nan, index=g.index)
    field_kg = float(kgs.mean()) if kgs.notna().any() else float('nan')
    surface = _guess_surface(g, history, target)

    new_scores = []
    new_reasons = []
    for idx, row in g.iterrows():
        bonus = 0.0
        reasons: list[str] = []

        if ablation_enabled('burden') and pd.notna(field_kg) and pd.notna(kgs.loc[idx]):
            delta = float(kgs.loc[idx]) - field_kg
            if delta >= 3.0:
                bonus -= 1.6
                reasons.append('斤量過多')
            elif delta >= 1.5:
                bonus -= 0.8
            elif delta <= -2.0:
                bonus += 1.0
                reasons.append('斤量軽量')
            elif delta <= -1.0:
                bonus += 0.5

        if ablation_enabled('sgate'):
            try:
                w = int(float(row.get('枠')))
            except (TypeError, ValueError):
                w = None
            if w is not None and n > 0:
                inner = w <= max(2, n // 4)
                outer = w >= max(n - 2, n * 3 // 4)
                if surface == 'ダ':
                    if inner:
                        bonus += 1.1
                        reasons.append('ダート内枠')
                    elif outer:
                        bonus -= 0.7
                elif surface == '芝':
                    if outer:
                        bonus += 0.7
                    elif inner:
                        bonus -= 0.3

        if ablation_enabled('sweight'):
            wd = _weight_delta(history, row.get('馬名'), target)
            if not np.isnan(wd):
                if abs(wd) >= 12:
                    bonus -= 1.6
                    reasons.append('馬体重大幅増減')
                elif abs(wd) >= 8:
                    bonus -= 0.8
                elif -6 <= wd <= -2:
                    bonus += 0.5

        if ablation_enabled('sjockey'):
            jb = _jockey_place_adj(history, row.get('騎手'), venue, target)
            if abs(jb) >= 0.4:
                bonus += jb * 0.45
                if jb >= 1.2:
                    reasons.append('騎手複勝率高')
                elif jb <= -1.2:
                    reasons.append('騎手複勝率低')

        if ablation_enabled('slayoff'):
            lay = _layoff_days(history, row.get('馬名'), target)
            if not np.isnan(lay):
                if lay >= 84:
                    bonus -= 1.8
                    reasons.append('長期休養明け')
                elif lay >= 56:
                    bonus -= 0.8
                    reasons.append('休み明け')
                elif 14 <= lay <= 42:
                    bonus += 0.5

        if ablation_enabled('sstyle'):
            finishes = np.array([_num(row.get(f'着順{i}')) for i in range(1, 6)], dtype=float)
            pops = np.array([_num(row.get(f'人気{i}')) for i in range(1, 6)], dtype=float)
            style = _infer_style(finishes, pops)
            if n >= 12 and style >= 0.62:
                bonus += 0.7
                reasons.append('多頭数差し向き')
            elif n <= 8 and style <= 0.35:
                bonus += 0.7
                reasons.append('少頭数先行向き')

        if ablation_enabled('sfield'):
            h = _hist_horse(history, row.get('馬名'), target)
            if h is not None and not h.empty and '頭数' in h.columns:
                heads = _num(h['頭数'])
                fin = _num(h['着順'])
                if n >= 12:
                    big = h[(heads >= 12) & fin.notna()]
                    if len(big) >= 2 and float((_num(big['着順']) <= 5).mean()) >= 0.5:
                        bonus += 0.8
                        reasons.append('多頭数実績')
                elif n <= 8:
                    small = h[(heads <= 8) & fin.notna()]
                    if len(small) >= 2 and float((_num(small['着順']) <= 5).mean()) >= 0.5:
                        bonus += 0.5

        bonus = float(np.clip(bonus, -4.0, 4.0))
        new_scores.append(float(row.get('AREru指数') or 0) + bonus)
        new_reasons.append(reasons)

    g['AREru指数'] = np.clip(np.array(new_scores, dtype=float), 0.0, 100.0).round(2)
    if '理由' in g.columns:
        merged = []
        for old, extra in zip(g['理由'].astype(str).tolist(), new_reasons):
            bits = [x for x in extra if x]
            if not bits:
                merged.append(old)
                continue
            add = ' / '.join(bits[:2])
            if old and old not in ('nan', '総合評価'):
                merged.append(f'{old} / {add}')
            else:
                merged.append(add)
        g['理由'] = merged
    return g


def xsel_features_from_env() -> set[str]:
    raw = str(os.environ.get('ARERU_XSEL_FEATURES') or '').strip()
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(',') if x.strip()}
