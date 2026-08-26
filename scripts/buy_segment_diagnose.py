#!/usr/bin/env python3
"""NEW BUY の条件別 ROI 分解。

閾値で件数を削る診断ではなく、どの層が損失を生んでいるかを
train / holdout で分けて見る。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
CACHE = DATA / 'rca_logic_cache'
OUT = DATA / 'buy_segment_report.json'
STAKE = 100


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _dates(logic: str) -> list[str]:
    return sorted({p.name.split('_')[-1].replace('.csv', '') for p in CACHE.glob(f'pred_{logic}_*.csv')})


def _split(dates: list[str]):
    cut = max(1, int(len(dates) * 0.70))
    return dates[:cut], dates[cut:]


def _num(v):
    return pd.to_numeric(v, errors='coerce')


def _odds_band(o) -> str:
    if o is None or (isinstance(o, float) and np.isnan(o)):
        return '不明'
    x = float(o)
    if x < 3:
        return '1.0-2.9'
    if x < 5:
        return '3.0-4.9'
    if x < 8:
        return '5.0-7.9'
    if x < 12:
        return '8.0-11.9'
    if x < 20:
        return '12.0-19.9'
    if x < 40:
        return '20.0-39.9'
    return '40.0+'


def _pop_band(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '不明'
    x = float(p)
    if x <= 1:
        return '1番人気'
    if x <= 3:
        return '2-3番人気'
    if x <= 6:
        return '4-6番人気'
    if x <= 9:
        return '7-9番人気'
    return '10番人気以下'


def _win_band(w) -> str:
    if w is None or (isinstance(w, float) and np.isnan(w)):
        return '不明'
    x = float(w)
    if x < 6:
        return '<6%'
    if x < 10:
        return '6-10%'
    if x < 15:
        return '10-15%'
    if x < 22:
        return '15-22%'
    return '22%+'


def _edge_band(e) -> str:
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return '不明'
    x = float(e)
    if x < 0:
        return '負エッジ'
    if x < 1.5:
        return '0-1.5pp'
    if x < 3:
        return '1.5-3pp'
    if x < 5:
        return '3-5pp'
    return '5pp+'


def _dist_band(m) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return '不明'
    x = float(m)
    if x < 1300:
        return '短距離(<1300)'
    if x < 1700:
        return 'マイル(1300-1699)'
    if x < 2100:
        return '中距離(1700-2099)'
    return '長距離(2100+)'


def _gate_band(w, n) -> str:
    try:
        w = int(float(w))
    except (TypeError, ValueError):
        return '不明'
    if not n or n <= 0:
        return str(w)
    if w <= max(2, n // 4):
        return '内枠'
    if w >= max(n - 2, int(n * 3 / 4)):
        return '外枠'
    return '中枠'


def _style_label(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return '不明'
    x = float(s)
    if x <= 0.28:
        return '逃げ'
    if x <= 0.45:
        return '先行'
    if x <= 0.70:
        return '差し'
    return '追込'


def _wdelta_band(d) -> str:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return '不明'
    x = float(d)
    if x <= -8:
        return '大幅減(-8kg超)'
    if x < -2:
        return '減(-8〜-2)'
    if x <= 2:
        return 'ほぼ変わらず'
    if x < 8:
        return '増(+2〜+8)'
    return '大幅増(+8kg超)'


def _track_band(t: str) -> str:
    s = str(t or '').strip()
    if not s or s in ('nan', 'None', '--'):
        return '不明'
    if '不' in s:
        return '不良'
    if '重' in s and '稍' not in s:
        return '重'
    if '稍' in s:
        return '稍重'
    if '良' in s:
        return '良'
    return s[:8]


def _summ(rows: list[dict], label: str) -> dict:
    n = len(rows)
    empty = {
        'label': label, 'BUY件数': 0, '的中件数': 0, '的中率': 0.0,
        '平均オッズ': None, '払戻合計': 0.0, '投資額': 0, 'ROI': 0.0, '回収率': 0.0,
        '損失額': 0.0,
    }
    if n == 0:
        return empty
    hits = sum(1 for r in rows if r.get('的中'))
    pay = float(sum(r.get('払戻') or 0 for r in rows))
    inv = n * STAKE
    odds = [r['オッズ'] for r in rows if r.get('オッズ') is not None]
    roi = pay / inv * 100 - 100 if inv else 0.0
    return {
        'label': label,
        'BUY件数': n,
        '的中件数': hits,
        '的中率': round(hits / n * 100, 2),
        '平均オッズ': round(sum(odds) / len(odds), 2) if odds else None,
        '払戻合計': round(pay, 1),
        '投資額': inv,
        'ROI': round(roi, 2),
        '回収率': round(pay / inv * 100, 2) if inv else 0.0,
        '損失額': round(inv - pay, 1),
    }


def _load_buy(logic: str, results: pd.DataFrame, dates: list[str]) -> list[dict]:
    from areru_engine import clean_name
    from ev_analysis import score_horse_ev
    from race_sim import infer_style, parse_weight, dist_meters, surface_of

    runners = pd.read_csv(DATA / 'runners.csv', encoding='utf-8-sig', low_memory=False)
    runners['race_id'] = runners['race_id'].astype(str)
    runners['_horse'] = runners['馬名'].map(clean_name)
    hist = pd.read_csv(DATA / 'all_history.csv', encoding='utf-8-sig', low_memory=False)
    from areru_engine import parse_date
    hist['_date'] = parse_date(hist['年月日'])
    hist['_horse'] = hist['馬名'].map(clean_name)

    out = []
    for d in dates:
        p = CACHE / f'pred_{logic}_{d}.csv'
        if not p.exists():
            continue
        pred = pd.read_csv(p, encoding='utf-8-sig')
        day_res = results[results['date'] == d]
        day_run = runners[runners['日付'].astype(str) == d] if '日付' in runners.columns else runners
        target = pd.Timestamp(d)
        for _, row in pred.iterrows():
            if not str(row.get('投資判定') or '').startswith('買い'):
                continue
            rid = str(row.get('race_id', ''))
            horse = str(row.get('本命') or '').strip()
            hn = clean_name(horse)
            rr = day_res[(day_res['race_id'].astype(str) == rid) & (day_res['馬名'].map(clean_name) == hn)]
            pred_odds = _num(row.get('本命オッズ'))
            odds = float(pred_odds) if pd.notna(pred_odds) else None
            hit = False
            finish = None
            if not rr.empty:
                finish = float(_num(rr.iloc[0]['着順'])) if pd.notna(_num(rr.iloc[0]['着順'])) else None
                o = _num(rr.iloc[0]['確定オッズ'])
                if pd.notna(o):
                    odds = float(o)
                hit = finish == 1.0
            pay = (odds * STAKE) if hit and odds else 0.0

            sim = _num(row.get('シミュレーション勝率'))
            fair = _num(row.get('AI適正オッズ'))
            conf = _num(row.get('AI信頼度スコア'))
            repro = _num(row.get('シミュレーション再現率'))
            n_data = int(_num(row.get('本命データ件数')) or _num(row.get('データ件数')) or 0)
            apt = _num(row.get('能力差スコア'))
            reasons = str(row.get('本命理由') or '')
            market = float(pred_odds) if pd.notna(pred_odds) else odds
            scored = {}
            if market and pd.notna(sim):
                scored = score_horse_ev(
                    market, float(sim) if pd.notna(sim) else None,
                    float(fair) if pd.notna(fair) else None,
                    float(conf) if pd.notna(conf) else 50.0,
                    float(repro) if pd.notna(repro) else 50.0,
                    n_data,
                    float(apt) if pd.notna(apt) else 50.0,
                    reasons,
                )
            adj = scored.get('補正勝率')
            implied = scored.get('市場暗示勝率')
            edge = (adj - implied) if adj is not None and implied is not None else None

            rg = day_run[(day_run['race_id'] == rid) & (day_run['_horse'] == hn)]
            field_n = int((day_run['race_id'] == rid).sum()) if not day_run.empty else 0
            waku = jockey = kg = None
            finishes = pops = None
            if not rg.empty:
                g = rg.iloc[0]
                waku = g.get('枠')
                jockey = str(g.get('騎手') or '').strip()
                kg = _num(g.get('斤量'))
                finishes = np.array([_num(g.get(f'着順{i}')) for i in range(1, 6)], dtype=float)
                pops = np.array([_num(g.get(f'人気{i}')) for i in range(1, 6)], dtype=float)
            style = infer_style(finishes, pops) if finishes is not None else float('nan')

            # 当日公式レコード（診断用。予想には使わない）
            ht = hist[(hist['_horse'] == hn) & (hist['_date'].dt.normalize() == target.normalize())]
            dist = track = wdelta = surface = None
            dist_m = float('nan')
            if not ht.empty:
                rec = ht.iloc[0]
                dist = str(rec.get('距離') or '')
                track = str(rec.get('馬場') or '')
                surface = surface_of(dist)
                dist_m = dist_meters(dist)
                today_w = parse_weight(rec.get('馬体重'))
                prev = hist[(hist['_horse'] == hn) & (hist['_date'] < target)].sort_values('_date', ascending=False)
                if not prev.empty:
                    prev_w = parse_weight(prev.iloc[0].get('馬体重'))
                    if not np.isnan(today_w) and not np.isnan(prev_w):
                        wdelta = today_w - prev_w
            if (not jockey) or jockey in ('nan', 'None'):
                if not ht.empty:
                    jockey = str(ht.iloc[0].get('騎手') or '').strip()

            # 展開相性から脚質補完
            style_txt = _style_label(style)
            detail = str(row.get('展開相性') or '')
            if style_txt == '不明':
                try:
                    import json as _json
                    dj = _json.loads(row.get('本命詳細') or '{}')
                    pac = str(dj.get('展開相性') or '')
                    if '逃' in pac:
                        style_txt = '逃げ'
                    elif '先行' in pac:
                        style_txt = '先行'
                    elif '差' in pac:
                        style_txt = '差し'
                    elif '追' in pac:
                        style_txt = '追込'
                except Exception:
                    pass

            pop = _num(row.get('本命人気'))
            out.append({
                'logic': logic,
                'date': d,
                'race_id': rid,
                'venue': str(row.get('開催地') or ''),
                'source': str(row.get('source') or ''),
                '本命': horse,
                '的中': hit,
                'オッズ': odds,
                '払戻': pay,
                '人気': float(pop) if pd.notna(pop) else None,
                '勝負ランク': str(row.get('勝負ランク') or ''),
                '推定勝率': adj,
                'sim勝率': float(sim) if pd.notna(sim) else None,
                '実エッジ': edge,
                '期待値': float(_num(row.get('期待値'))) if pd.notna(_num(row.get('期待値'))) else None,
                'オッズ帯': _odds_band(odds),
                '人気帯': _pop_band(pop),
                'AIランク': str(row.get('勝負ランク') or '不明').upper()[:1] or '不明',
                '勝率帯': _win_band(adj),
                'エッジ帯': _edge_band(edge),
                '芝ダ': surface or '不明',
                '距離帯': _dist_band(dist_m),
                '馬場': _track_band(track),
                '競馬場': str(row.get('開催地') or '不明'),
                '騎手': jockey or '不明',
                '枠帯': _gate_band(waku, field_n),
                '脚質': style_txt,
                '馬体重帯': _wdelta_band(wdelta),
            })
    return out


def _by_dim(rows: list[dict], key: str, train: set[str], hold: set[str]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) or '不明')].append(r)
    out = {}
    for k, rs in sorted(groups.items(), key=lambda x: -len(x[1])):
        tr = [x for x in rs if x['date'] in train]
        ho = [x for x in rs if x['date'] in hold]
        out[k] = {
            'full': _summ(rs, k),
            'train': _summ(tr, k),
            'holdout': _summ(ho, k),
        }
    return out


def _drag(rows: list[dict], key: str, baseline_roi: float) -> list[dict]:
    """損失額が大きく、かつ ROI が全体より悪い層。"""
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) or '不明')].append(r)
    items = []
    for k, rs in groups.items():
        sm = _summ(rs, k)
        items.append({
            '条件': f'{key}={k}',
            **sm,
            '全体より悪い': sm['ROI'] < baseline_roi,
            'ROI差_vs全体': round(sm['ROI'] - baseline_roi, 2),
        })
    items.sort(key=lambda x: x['損失額'], reverse=True)
    return items


def main():
    results = pd.read_csv(DATA / 'results.csv', encoding='utf-8-sig', low_memory=False)
    results['date'] = results['date'].astype(str)
    results['race_id'] = results['race_id'].astype(str)
    dates = _dates('NEW')
    train, hold = _split(dates)
    train_s, hold_s = set(train), set(hold)
    print(f'[seg] dates={len(dates)} train={len(train)} holdout={len(hold)}', flush=True)

    report = {
        '検証設計': {
            '開催日数': len(dates),
            'train': train,
            'holdout': hold,
            '評価': 'BUYのみ・確定オッズ・1点100円',
            '閾値探索': '禁止',
        },
        'overall': {},
        'segments': {},
        'loss_leaders': {},
        'holdout_confirmed_damage': [],
    }
    rows_by = {}
    for logic in ('OLD', 'NEW'):
        print(f'[seg] load {logic}', flush=True)
        rows_by[logic] = _load_buy(logic, results, dates)
        all_r = rows_by[logic]
        report['overall'][logic] = {
            'full': _summ(all_r, f'{logic}/full'),
            'train': _summ([r for r in all_r if r['date'] in train_s], f'{logic}/train'),
            'holdout': _summ([r for r in all_r if r['date'] in hold_s], f'{logic}/holdout'),
        }
        print(report['overall'][logic]['full'], flush=True)

    dims = (
        'オッズ帯', '人気帯', 'AIランク', '勝率帯', 'エッジ帯',
        '芝ダ', '距離帯', '馬場', '競馬場', '騎手', '枠帯', '脚質', '馬体重帯',
    )
    new_rows = rows_by['NEW']
    base_roi = report['overall']['NEW']['full']['ROI']
    report['segments']['NEW'] = {d: _by_dim(new_rows, d, train_s, hold_s) for d in dims}

    for d in dims:
        report['loss_leaders'][d] = _drag(new_rows, d, base_roi)[:12]

    # holdout でも悪化が再現する層（件数十分）
    confirmed = []
    for d in dims:
        table = report['segments']['NEW'][d]
        for k, blk in table.items():
            tr, ho, fu = blk['train'], blk['holdout'], blk['full']
            if fu['BUY件数'] < 20:
                continue
            if tr['BUY件数'] < 12 or ho['BUY件数'] < 8:
                continue
            if tr['ROI'] < base_roi and ho['ROI'] < report['overall']['NEW']['holdout']['ROI']:
                confirmed.append({
                    '条件': f'{d}={k}',
                    'full': fu,
                    'train': tr,
                    'holdout': ho,
                    'train_vs_NEW': round(tr['ROI'] - report['overall']['NEW']['train']['ROI'], 2),
                    'holdout_vs_NEW': round(ho['ROI'] - report['overall']['NEW']['holdout']['ROI'], 2),
                    '損失額': fu['損失額'],
                })
            elif tr['ROI'] < -70 and ho['ROI'] < -70 and fu['BUY件数'] >= 25:
                confirmed.append({
                    '条件': f'{d}={k}',
                    'full': fu,
                    'train': tr,
                    'holdout': ho,
                    'train_vs_NEW': round(tr['ROI'] - report['overall']['NEW']['train']['ROI'], 2),
                    'holdout_vs_NEW': round(ho['ROI'] - report['overall']['NEW']['holdout']['ROI'], 2),
                    '損失額': fu['損失額'],
                    'note': '両期間とも大幅マイナス',
                })
    confirmed.sort(key=lambda x: x['損失額'], reverse=True)
    report['holdout_confirmed_damage'] = confirmed

    OUT.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'overall': report['overall'],
        'damage': [{'条件': c['条件'], 'n': c['full']['BUY件数'], 'ROI': c['full']['ROI'],
                    'train': c['train']['ROI'], 'holdout': c['holdout']['ROI'],
                    '損失': c['損失額']} for c in confirmed[:20]],
    }, ensure_ascii=False, indent=2))
    print(f'📁 {OUT}')


if __name__ == '__main__':
    main()
