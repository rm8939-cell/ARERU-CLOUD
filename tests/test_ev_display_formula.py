"""期待値表示公式の単体検算。

BUYカード:
  推定勝率 = 補正勝率(%)
  オッズ   = 本命単勝オッズ
  生EV%    = 推定勝率 × オッズ
  表示EV%  = clip(100 + 26*tanh((生EV-100)/30), 78, 124)

例: 9.5% × 14.8倍 = 生 140.6% → 表示 123%
"""
from __future__ import annotations

import math
import unittest

from ev_analysis import (
    BUY_EV_FLOOR,
    EV_DISPLAY_MAX,
    EV_DISPLAY_MIN,
    _soft_display_ev,
    display_ev_from_winrate_odds,
    score_horse_ev,
)


class TestEvDisplayFormula(unittest.TestCase):
    def test_example_95_times_148_displays_123(self):
        raw = 9.5 * 14.8
        self.assertAlmostEqual(raw, 140.6, places=1)
        compressed = 100.0 + 26.0 * math.tanh((raw - 100.0) / 30.0)
        self.assertAlmostEqual(compressed, 122.7, places=1)
        self.assertEqual(_soft_display_ev(raw), 123)
        doc = display_ev_from_winrate_odds(9.5, 14.8)
        self.assertEqual(doc['生EV'], 140.6)
        self.assertEqual(doc['表示EV'], 123)

    def test_display_not_equal_to_raw_product(self):
        raw = 9.5 * 14.8
        self.assertNotEqual(_soft_display_ev(raw), round(raw))
        self.assertLess(_soft_display_ev(raw), raw)

    def test_clip_bounds(self):
        self.assertGreaterEqual(_soft_display_ev(10.0), EV_DISPLAY_MIN)
        self.assertLessEqual(_soft_display_ev(400.0), EV_DISPLAY_MAX)
        self.assertEqual(_soft_display_ev(100.0), 100)

    def test_score_horse_ev_raw_equals_adj_times_odds(self):
        ex = score_horse_ev(
            market=14.8, win_pct=22.0, fair=4.5,
            conf=70.0, repro=65.0, n=4, apt=60.0, reasons='',
        )
        self.assertTrue(ex['期待値あり'])
        recon = round(ex['補正勝率'] * 14.8, 1)
        self.assertAlmostEqual(ex['期待値生'], recon, places=1)
        self.assertEqual(ex['期待値検算'], recon)
        self.assertEqual(ex['期待値'], _soft_display_ev(ex['期待値生']))

    def test_buy_floor_unchanged(self):
        self.assertEqual(BUY_EV_FLOOR, 108)


if __name__ == '__main__':
    unittest.main()
