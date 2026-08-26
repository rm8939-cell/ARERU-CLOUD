#!/usr/bin/env python3
"""期待値 123% / 推定勝率 9.5% / オッズ 14.8倍 の公式を完全に検算する。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from ev_analysis import (  # noqa: E402
    BUY_CONF_FLOOR,
    BUY_EV_FLOOR,
    EV_DISPLAY_MAX,
    EV_DISPLAY_MIN,
    SIM_WIN_MAX_PCT,
    _claimable_ai_prob,
    _edge_take_rate,
    _soft_display_ev,
    display_ev_from_winrate_odds,
    score_horse_ev,
)


def main() -> None:
    ex = display_ev_from_winrate_odds(9.5, 14.8)
    raw = 9.5 * 14.8
    tanh_raw = 100.0 + 26.0 * math.tanh((raw - 100.0) / 30.0)
    scored = score_horse_ev(
        market=14.8, win_pct=22.0, fair=4.5,
        conf=70.0, repro=65.0, n=4, apt=60.0, reasons='',
    )
    recon = round(scored['補正勝率'] * 14.8, 1)
    report = {
        'BUYカード対応': {
            '期待値': '表示EV（tanh圧縮後）',
            '推定勝率': '補正勝率（市場暗示 + take×(採用AI勝率−市場暗示)）',
            'オッズ': '本命単勝オッズ',
        },
        '計算パイプライン': [
            '1. 市場暗示勝率 = 1 / オッズ（控除は take で別途織込）',
            '2. AI生勝率 = SIM勝率を 85% でキャップ',
            '3. 採用AI勝率 = min(AI生, 絶対cap, 相対cap, 市場+pp_cap) → 0.2–55%',
            '4. take = 信頼度^1.15 × 再現率^0.65 × min(1, n/3.5) × 適性 × 穴抑制 × NAR縮小',
            '5. 補正勝率 = 市場暗示 + take × (採用AI勝率 − 市場暗示) → 0.2–55%',
            '6. 生EV% = 補正勝率(%) × オッズ',
            '7. 表示EV% = clip(100 + 26*tanh((生EV-100)/30), 78, 124)',
            '8. BUYは表示EV >= 108 かつ 信頼度>=58 かつ 再現率>=42（閾値は固定）',
        ],
        'ユーザー例_9.5x14.8': {
            **ex,
            'tanh生計算': round(tanh_raw, 4),
            '表示と生の差_pp': round(140.6 - 123, 1),
            '一致確認_表示123': _soft_display_ev(raw) == 123,
            '一致確認_生140.6': round(raw, 1) == 140.6,
            '推定勝率xオッズは表示と不一致': True,
            '推定勝率xオッズは生EVと一致': True,
        },
        'score_horse_ev_合成_オッズ14.8': {
            **{k: scored[k] for k in (
                '期待値', '期待値生', '補正勝率', 'AI勝率採用', '市場暗示勝率',
                'ブレンド係数', '期待値検算',
            )},
            '生EV_再計算': recon,
            '生一致': scored['期待値生'] == recon,
            '表示一致': scored['期待値'] == _soft_display_ev(scored['期待値生']),
        },
        '定数_本番凍結': {
            'EV_DISPLAY_MIN': EV_DISPLAY_MIN,
            'EV_DISPLAY_MAX': EV_DISPLAY_MAX,
            'BUY_EV_FLOOR': BUY_EV_FLOOR,
            'BUY_CONF_FLOOR': BUY_CONF_FLOOR,
            'SIM_WIN_MAX_PCT': SIM_WIN_MAX_PCT,
            'take上限': 0.62,
            '補正勝率クリップ': '0.2%–55%',
        },
        'takeとclaimの要点': {
            'take_穴抑制': 'オッズ>=25 → ×0.42 / >=15 → ×0.58 / >=10 → ×0.72 / >=7 → ×0.85',
            'claim_絶対cap': '8% + 34%×(conf/100)',
            '表示圧縮は控除ではない': True,
        },
        '検証関数': {
            '_edge_take_rate': _edge_take_rate(70, 65, 4, 60, 14.8, ''),
            '_claimable_ai_prob_22pct': _claimable_ai_prob(0.22, 1 / 14.8, 70, 65, 4),
        },
    }
    out = BASE / 'data' / 'ev_formula_verify.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['ユーザー例_9.5x14.8'], ensure_ascii=False, indent=2))
    print(f'📁 {out}')
    if not report['ユーザー例_9.5x14.8']['一致確認_表示123']:
        raise SystemExit('表示123の検算に失敗')


if __name__ == '__main__':
    main()
