"""履歴マッチング・騎手統計・runners 補完ユーティリティ。"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from areru_engine import DATA_DIR, clean_name, num, parse_date

PAST_DETAIL_COLS = [
    *(f'{c}{i}' for i in range(1, 6) for c in ('タイム', '着差', '馬場')),
]


def _blank(v) -> bool:
    s = str(v or '').strip()
    return not s or s.lower() in ('nan', '--', 'none', 'null')


def _safe_int(v):
    s = str(v or '').strip()
    m = re.search(r'\d+', s)
    return int(m.group(0)) if m else None


@lru_cache(maxsize=1)
def load_raw_all_history() -> pd.DataFrame:
    p = DATA_DIR / 'all_history.csv'
    if not p.exists():
        return pd.DataFrame()
    h = pd.read_csv(p, encoding='utf-8-sig')
    h['_date'] = parse_date(h['年月日'])
    h['_horse'] = h['馬名'].map(clean_name)
    return h


def build_master_history(base: pd.DataFrame | None = None) -> pd.DataFrame:
    """all_history + results.csv を統合したマスタ履歴。"""
    from areru_engine import expand_scoring_history
    if base is None:
        base = load_raw_all_history()
    return expand_scoring_history(base)


def _slot_match_row(
    horse: str,
    finish,
    pop,
    venue: str,
    history: pd.DataFrame,
    target,
) -> pd.Series | None:
    """着順×人気×場 で履歴1行を特定。"""
    if history is None or history.empty:
        return None
    fin_i = _safe_int(finish)
    pop_i = _safe_int(pop)
    if fin_i is None:
        return None
    h = history[(history['_horse'] == horse) & (history['_date'] < target)]
    if h.empty:
        return None
    for _, row in h.iterrows():
        rf = _safe_int(row.get('着順'))
        rp = _safe_int(row.get('人気'))
        rv = str(row.get('場') or '').strip()
        if rf != fin_i:
            continue
        if pop_i is not None and rp is not None and rp != pop_i:
            continue
        if venue and rv and venue not in rv and rv not in venue:
            continue
        return row
    # 着順のみフォールバック
    for _, row in h.iterrows():
        if _safe_int(row.get('着順')) == fin_i:
            return row
    return None


def enrich_runner_history_fields(row, history: pd.DataFrame | None, target) -> dict:
    """runners 行の past1..5 スロットに タイム/着差/馬場 をスロットマッチで補完。"""
    out = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
    if history is None or getattr(history, 'empty', True):
        return out
    horse = clean_name(out.get('馬名'))
    for i in range(1, 6):
        fin = out.get(f'着順{i}')
        if _blank(fin):
            continue
        pop = out.get(f'人気{i}')
        venue = str(out.get(f'場{i}') or '').strip()
        matched = _slot_match_row(horse, fin, pop, venue, history, target)
        if matched is None:
            continue
        for col, key in (('タイム', 'タイム'), ('着差', '着差'), ('馬場', '馬場')):
            field = f'{col}{i}'
            if not _blank(out.get(field)):
                continue
            val = matched.get(key)
            if not _blank(val):
                out[field] = val
    return out


def backfill_runners_past_detail(runners: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """runners.csv 全体に タイム/着差/馬場 をスロットマッチで一括補完。"""
    if history is None:
        history = build_master_history()
    df = runners.copy()
    for c in PAST_DETAIL_COLS:
        if c not in df.columns:
            df[c] = ''
    dates = parse_date(df['日付'])
    filled = 0
    for idx in df.index:
        target = pd.Timestamp(dates.loc[idx]) if pd.notna(dates.loc[idx]) else None
        if target is None:
            continue
        before = enrich_runner_history_fields(df.loc[idx], history, target)
        for c in PAST_DETAIL_COLS:
            v = before.get(c)
            if not _blank(v) and _blank(df.at[idx, c]):
                df.at[idx, c] = v
                filled += 1
    return df


@lru_cache(maxsize=1)
def build_jockey_stats_index() -> dict[str, dict]:
    """all_history から騎手統計インデックス（全体・場別）。"""
    h = load_raw_all_history()
    if h.empty or '騎手' not in h.columns:
        return {}
    idx: dict[str, dict] = {}
    h = h.copy()
    h['_jockey'] = h['騎手'].map(clean_name)
    for jockey, g in h.groupby('_jockey'):
        if not jockey:
            continue
        fin = num(g['着順'])
        place = float((fin <= 3).mean()) if fin.notna().any() else 0.22
        win = float((fin == 1).mean()) if fin.notna().any() else 0.08
        venues: dict[str, dict] = {}
        for venue, vg in g.groupby(g['場'].astype(str)):
            vf = num(vg['着順'])
            if vf.notna().sum() >= 2:
                venues[venue] = {
                    'place': float((vf <= 3).mean()),
                    'n': int(vf.notna().sum()),
                }
        idx[jockey] = {'place': place, 'win': win, 'n': int(len(g)), 'venues': venues}
    return idx


def jockey_bonus_from_index(jockey: str, venue: str) -> float:
    """騎手統計インデックスから補正（legacy 側は呼ばない）。"""
    stats = build_jockey_stats_index()
    j = clean_name(jockey)
    if not j or j not in stats:
        return 0.0
    s = stats[j]
    if s['n'] < 3:
        return 0.0
    bonus = (s['place'] - 0.22) * 18
    vstats = s.get('venues', {}).get(str(venue))
    if vstats and vstats.get('n', 0) >= 2:
        bonus += (vstats['place'] - 0.22) * 8
    return float(np.clip(bonus, -6, 8))


def feature_fill_report(runners: pd.DataFrame, history: pd.DataFrame | None = None) -> dict:
    """特徴量の充足率レポート。"""
    if history is None:
        history = build_master_history()
    n = len(runners)
    report: dict = {'runners_rows': n, 'columns': {}}
    cols_core = [
        '着順1', '人気1', '場1', 'レース名1', '単勝オッズ', '人気', '騎手', '斤量', '枠',
        'タイム1', '着差1', '馬場1',
    ]
    for c in cols_core:
        if c not in runners.columns:
            report['columns'][c] = {'filled': 0, 'rate': 0.0}
            continue
        filled = (~runners[c].map(_blank)).sum()
        report['columns'][c] = {'filled': int(filled), 'rate': round(filled / max(n, 1) * 100, 1)}

    dates = parse_date(runners['日付'])
    slot_detail = 0
    jockey_hit = 0
    hist_horses = set(history['_horse']) if '_horse' in history.columns else set()
    jockey_idx = build_jockey_stats_index()
    for idx, row in runners.iterrows():
        target = pd.Timestamp(dates.loc[idx]) if pd.notna(dates.loc[idx]) else None
        if target is None:
            continue
        enriched = enrich_runner_history_fields(row, history, target)
        if not _blank(enriched.get('タイム1')) or not _blank(enriched.get('着差1')):
            slot_detail += 1
        j = clean_name(row.get('騎手'))
        if j and j in jockey_idx:
            jockey_hit += 1
    report['slot_matched_detail_any'] = {
        'filled': slot_detail,
        'rate': round(slot_detail / max(n, 1) * 100, 1),
    }
    report['jockey_in_index'] = {
        'filled': jockey_hit,
        'rate': round(jockey_hit / max(n, 1) * 100, 1),
    }
    report['history_horses'] = len(hist_horses)
    return report
