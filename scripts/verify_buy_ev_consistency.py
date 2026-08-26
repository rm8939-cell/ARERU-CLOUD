#!/usr/bin/env python3
"""本番ロジック固定のうえ、BUY限定の旧/新比較と期待値公式の検算。

期待値% の定義（修正後）:
  補正勝率 p_adj（0-1）= 市場暗示勝率 + take × (採用AI勝率 − 市場暗示勝率)
  期待値%  = p_adj × 本命オッズ × 100
           = 補正勝率(%) × オッズ
  例: 補正勝率 18.9% × オッズ 6.5倍 → 122.85% → 表示 123%

推定勝率（BUYカード）= 補正勝率。SIM生勝率×オッズは「生チケットEV」であり表示期待値ではない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
DATA = BASE / 'data'
CACHE = DATA / 'rca_logic_cache'
OUT = DATA / 'buy_ev_verify_report.json'
STAKE = 100


def _load_results():
    r = pd.read_csv(DATA / 'results.csv', encoding='utf-8-sig', low_memory=False)
    r['date'] = r['date'].astype(str)
    return r


def _dates_for(logic: str) -> list[str]:
    return sorted({p.name.split('_')[-1].replace('.csv', '') for p in CACHE.glob(f'pred_{logic}_*.csv')})


def _settle(pred: pd.DataFrame, results: pd.DataFrame, date: str, buy_only: bool) -> list[dict]:
    from areru_engine import clean_name
    day = results[results['date'] == date]
    rows = []
    df = pred
    if buy_only:
        df = pred[pred['投資判定'].astype(str).str.startswith('買い')]
    for _, row in df.iterrows():
        rid = str(row.get('race_id', ''))
        horse = str(row.get('本命', '')).strip()
        ev = pd.to_numeric(row.get('期待値'), errors='coerce')
        sim = pd.to_numeric(row.get('シミュレーション勝率'), errors='coerce')
        adj = pd.to_numeric(row.get('補正勝率'), errors='coerce') if '補正勝率' in pred.columns else float('nan')
        odds_p = pd.to_numeric(row.get('本命オッズ'), errors='coerce')
        g = day[(day['race_id'].astype(str) == rid) & (day['馬名'].map(clean_name) == clean_name(horse))]
        hit = False
        odds = float(odds_p) if pd.notna(odds_p) else None
        if not g.empty:
            fin = float(pd.to_numeric(g.iloc[0]['着順'], errors='coerce'))
            o = float(pd.to_numeric(g.iloc[0]['確定オッズ'], errors='coerce'))
            if pd.notna(o):
                odds = o
            hit = fin == 1
        pay = (odds * STAKE) if hit and odds is not None else 0.0
        rows.append({
            'date': date, 'hit': hit, 'odds': odds, 'ev': float(ev) if pd.notna(ev) else None,
            'sim': float(sim) if pd.notna(sim) else None,
            'adj': float(adj) if pd.notna(adj) else None,
            'pay': pay,
        })
    return rows


def _summ(rows: list[dict], label: str) -> dict:
    n = len(rows)
    if n == 0:
        return {'label': label, 'BUY件数': 0, '的中率': 0.0, '回収率': 0.0, 'ROI': 0.0, 'サンプル数': 0}
    hits = sum(1 for r in rows if r['hit'])
    ret = sum(r['pay'] for r in rows)
    inv = n * STAKE
    return {
        'label': label,
        'サンプル数': n,
        'BUY件数': n,
        '的中件数': hits,
        '的中率': round(hits / n * 100, 2),
        '回収率': round(ret / inv * 100, 2),
        'ROI': round(ret / inv * 100 - 100, 2),
    }


def _split(dates: list[str]):
    cut = max(1, int(len(dates) * 0.70))
    return dates[:cut], dates[cut:]


def frozen_buy(logic: str, results, dates, train, hold) -> dict:
    bets = []
    for d in dates:
        p = CACHE / f'pred_{logic}_{d}.csv'
        if not p.exists():
            continue
        pred = pd.read_csv(p, encoding='utf-8-sig')
        bets.extend(_settle(pred, results, d, buy_only=True))
    out = {
        'full': _summ(bets, f'{logic}/frozen/full'),
        'train': _summ([b for b in bets if b['date'] in train], f'{logic}/frozen/train'),
        'holdout': _summ([b for b in bets if b['date'] in hold], f'{logic}/frozen/holdout'),
    }
    return out


def refinalize_buy(logic: str, results, dates, train, hold) -> dict:
    from ev_analysis import finalize_predictions_df
    bets = []
    mismatches = []
    n_checked = 0
    for d in dates:
        p = CACHE / f'pred_{logic}_{d}.csv'
        if not p.exists():
            continue
        pred = pd.read_csv(p, encoding='utf-8-sig')
        fixed = finalize_predictions_df(pred)
        # EV consistency on BUY
        buy = fixed[fixed['投資判定'].astype(str).str.startswith('買い')]
        for _, r in buy.iterrows():
            n_checked += 1
            adj = pd.to_numeric(r.get('補正勝率'), errors='coerce')
            odds = pd.to_numeric(r.get('本命オッズ'), errors='coerce')
            ev = pd.to_numeric(r.get('期待値'), errors='coerce')
            chk = pd.to_numeric(r.get('期待値検算'), errors='coerce') if '期待値検算' in fixed.columns else float('nan')
            if pd.notna(adj) and pd.notna(odds) and pd.notna(ev):
                from ev_analysis import _soft_display_ev
                recon_raw = adj * float(odds)
                recon_disp = _soft_display_ev(recon_raw)
                # 表示は tanh 圧縮後。生EV（補正勝率×オッズ）と表示の差は仕様。
                if abs(float(ev) - recon_disp) > 1.5:
                    mismatches.append({
                        'date': d, 'ev': float(ev), 'adj': float(adj), 'odds': float(odds),
                        'recon_raw': round(recon_raw, 1), 'recon_disp': recon_disp,
                        '検算': None if pd.isna(chk) else float(chk),
                    })
        bets.extend(_settle(fixed, results, d, buy_only=True))
    out = {
        'full': _summ(bets, f'{logic}/honestEV/full'),
        'train': _summ([b for b in bets if b['date'] in train], f'{logic}/honestEV/train'),
        'holdout': _summ([b for b in bets if b['date'] in hold], f'{logic}/honestEV/holdout'),
        'ev_mismatch_n': len(mismatches),
        'ev_checked_buy': n_checked,
        'ev_mismatch_samples': mismatches[:8],
    }
    return out


def formula_doc() -> dict:
    from ev_analysis import display_ev_from_winrate_odds, score_horse_ev
    ex = score_horse_ev(
        market=14.8, win_pct=22.0, fair=4.5, conf=70.0, repro=65.0, n=4, apt=60.0, reasons=''
    )
    example = display_ev_from_winrate_odds(9.5, 14.8)
    return {
        '定義_生EV': '生EV% = 補正勝率(%) × オッズ（= 推定勝率 × オッズ）',
        '定義_表示EV': '表示EV% = clip(100 + 26*tanh((生EV-100)/30), 78, 124)',
        '補正勝率': (
            '市場暗示勝率(1/オッズ) + take × (採用AI勝率 − 市場暗示勝率)。'
            'take は信頼度・再現率・データ件数・適性・穴抑制。控除は take で織込。'
        ),
        'ユーザー例': example,
        '注記': (
            'BUYカード「期待値123%」は生140.6%の圧縮表示。'
            '推定勝率9.5%×オッズ14.8倍は生EVと一致し、表示とは一致しない。'
            'ROI改善が holdout で確認できるまでこの表示は変更しない。'
        ),
        '合成例_score_horse_ev': ex,
    }


def main():
    results = _load_results()
    dates = _dates_for('OLD')
    train, hold = _split(dates)
    report = {
        '本番ロジック固定': {
            'main_render.yaml': 'ARERU_LEGACY_SCORE=1 → 本番は旧ロジック固定',
            '比較の旧': 'ARERU_LEGACY_SCORE=1（ガウスSIM・詳細特徴OFF）',
            '比較の新': 'ARERU_LEGACY_SCORE=0（段階SIM+詳細特徴）',
            '評価対象': '投資判定が買い のレースのみ（本命全体的中率は使わない）',
            '開催日数': len(dates),
            'train': train,
            'holdout': hold,
        },
        '期待値公式': formula_doc(),
        'frozen_BUY': {},
        'honestEV_BUY': {},
        '採用': {},
    }
    for logic in ('OLD', 'NEW'):
        print(f'[frozen] {logic}', flush=True)
        report['frozen_BUY'][logic] = frozen_buy(logic, results, dates, set(train), set(hold))
        print(report['frozen_BUY'][logic]['full'], flush=True)
    for logic in ('OLD', 'NEW'):
        print(f'[honestEV re-finalize] {logic}', flush=True)
        report['honestEV_BUY'][logic] = refinalize_buy(logic, results, dates, set(train), set(hold))
        print(report['honestEV_BUY'][logic]['full'], 'mismatch', report['honestEV_BUY'][logic]['ev_mismatch_n'], flush=True)

    def delta(a, b):
        return {
            'ROI差_pp': round(a['ROI'] - b['ROI'], 2),
            '的中率差_pp': round(a['的中率'] - b['的中率'], 2),
            'BUY差': a['BUY件数'] - b['BUY件数'],
        }

    fr_n, fr_o = report['frozen_BUY']['NEW'], report['frozen_BUY']['OLD']
    ho_n, ho_o = report['honestEV_BUY']['NEW'], report['honestEV_BUY']['OLD']
    report['frozen_vs'] = {k: delta(fr_n[k], fr_o[k]) for k in ('full', 'train', 'holdout')}
    report['honestEV_vs'] = {k: delta(ho_n[k], ho_o[k]) for k in ('full', 'train', 'holdout')}

    # adopt only if BUY ROI improves on train AND holdout after honest EV
    ok = (
        ho_n['train']['ROI'] > ho_o['train']['ROI']
        and ho_n['holdout']['ROI'] > ho_o['holdout']['ROI']
        and ho_n['full']['BUY件数'] >= 100
        and ho_o['full']['BUY件数'] >= 100
    )
    report['採用'] = {
        'adopt_NEW': ok,
        '理由': (
            'honest EV 後も train/holdout の BUY ROI が旧を上回る' if ok
            else 'honest EV 後に旧を安定して上回らないため新は不採用（本番は検証結果に従う）'
        ),
        'frozen_NEW_ROI差_full': report['frozen_vs']['full']['ROI差_pp'],
        'honest_NEW_ROI差_full': report['honestEV_vs']['full']['ROI差_pp'],
        'honest_NEW_ROI差_holdout': report['honestEV_vs']['holdout']['ROI差_pp'],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['採用'], ensure_ascii=False, indent=2))
    print(f'📁 {OUT}')


if __name__ == '__main__':
    main()
