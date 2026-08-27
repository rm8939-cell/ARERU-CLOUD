"""3候補較正はフラグOFFで無効果。BUYフィルタではない。"""
from __future__ import annotations

import os
import unittest

from race_sim import calib_flag, gate_band, is_sashi_style, three_calib_adj


class TestThreeCalibs(unittest.TestCase):
    def setUp(self):
        for k in list(os.environ):
            if k.startswith('ARERU_CALIB_') or k in ('ARERU_LOGIC_PRESET', 'ARERU_LEGACY_SCORE'):
                os.environ.pop(k, None)

    def tearDown(self):
        for k in list(os.environ):
            if k.startswith('ARERU_CALIB_'):
                os.environ.pop(k, None)

    def test_style_and_gate_bands_match_diagnosis(self):
        self.assertFalse(is_sashi_style(0.45))
        self.assertTrue(is_sashi_style(0.46))
        self.assertTrue(is_sashi_style(0.70))
        self.assertFalse(is_sashi_style(0.71))
        self.assertEqual(gate_band(1, 12), '内枠')
        self.assertEqual(gate_band(6, 12), '中枠')
        self.assertEqual(gate_band(12, 12), '外枠')

    def test_off_is_noop(self):
        d, plus, minus = three_calib_adj(0.6, 1, 12, 15.0)
        self.assertEqual(d, 0.0)
        self.assertEqual(plus, [])
        self.assertEqual(minus, [])

    def test_sashi_inner_only(self):
        os.environ['ARERU_CALIB_SASHI_INNER'] = '1'
        d, _, minus = three_calib_adj(0.6, 1, 12, 6.0)
        self.assertLess(d, 0)
        self.assertIn('差し×内枠', minus)
        # 先行×内枠は動かない
        d2, _, _ = three_calib_adj(0.3, 1, 12, 6.0)
        self.assertEqual(d2, 0.0)
        # 差し×中枠は動かない
        d3, _, _ = three_calib_adj(0.6, 6, 12, 6.0)
        self.assertEqual(d3, 0.0)

    def test_odds_inner_only(self):
        os.environ['ARERU_CALIB_ODDS_INNER'] = '1'
        d, _, minus = three_calib_adj(0.3, 1, 12, 15.0)
        self.assertLess(d, 0)
        self.assertIn('12-20倍×内枠', minus)
        d2, _, _ = three_calib_adj(0.3, 1, 12, 6.0)
        self.assertEqual(d2, 0.0)
        d3, _, _ = three_calib_adj(0.3, 6, 12, 15.0)
        self.assertEqual(d3, 0.0)

    def test_sashi_sweet_only(self):
        os.environ['ARERU_CALIB_SASHI_SWEET'] = '1'
        d, plus, _ = three_calib_adj(0.6, 6, 12, 6.5)
        self.assertGreater(d, 0)
        self.assertIn('差し×5-8倍×中枠', plus)
        d2, _, _ = three_calib_adj(0.6, 1, 12, 6.5)
        self.assertEqual(d2, 0.0)
        d3, _, _ = three_calib_adj(0.6, 6, 12, 15.0)
        self.assertEqual(d3, 0.0)

    def test_flags_independent(self):
        os.environ['ARERU_CALIB_SASHI_INNER'] = '1'
        os.environ['ARERU_CALIB_ODDS_INNER'] = '1'
        d, _, minus = three_calib_adj(0.6, 1, 12, 15.0)
        self.assertLess(d, -5.0)
        self.assertEqual(len(minus), 2)

    def test_buy_floor_unchanged(self):
        from ev_analysis import BUY_CONF_FLOOR, BUY_EV_FLOOR
        self.assertEqual(BUY_EV_FLOOR, 108)
        self.assertEqual(BUY_CONF_FLOOR, 58)
        self.assertFalse(calib_flag('SASHI_INNER'))

    def test_rejected_flags_stay_off_in_production_env(self):
        os.environ['ARERU_LEGACY_SCORE'] = '1'
        from areru_engine import ablation_enabled, legacy_score_enabled
        self.assertTrue(legacy_score_enabled())
        for feat in ('jockey', 'time', 'margin', 'track', 'weight', 'burden', 'sgate'):
            self.assertFalse(ablation_enabled(feat), feat)


if __name__ == '__main__':
    unittest.main()
