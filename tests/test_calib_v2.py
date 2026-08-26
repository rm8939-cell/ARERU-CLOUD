"""PRESET=D は BUY 除外フィルタではなく、holdout確認済みの本命較正。"""
from __future__ import annotations

import os
import unittest


class TestCalibV2(unittest.TestCase):
    def setUp(self):
        for k in list(os.environ):
            if k.startswith('ARERU_ABL_') or k in ('ARERU_LEGACY_SCORE', 'ARERU_LOGIC_PRESET'):
                os.environ.pop(k, None)

    def tearDown(self):
        os.environ.pop('ARERU_LOGIC_PRESET', None)
        os.environ.pop('ARERU_LEGACY_SCORE', None)

    def test_preset_d_keeps_stage_features(self):
        os.environ['ARERU_LEGACY_SCORE'] = '0'
        os.environ['ARERU_LOGIC_PRESET'] = 'D'
        from importlib import reload
        import areru_engine
        reload(areru_engine)
        self.assertTrue(areru_engine.calib_v2_enabled())
        self.assertTrue(areru_engine.ablation_enabled('jockey'))
        self.assertTrue(areru_engine.ablation_enabled('weight'))
        self.assertFalse(areru_engine.ablation_enabled('burden'))

    def test_dirt_inner_gate_flips_on_d(self):
        from race_sim import gate_bias
        os.environ.pop('ARERU_LOGIC_PRESET', None)
        self.assertGreater(gate_bias('大井', 1, 12, 'ダ'), 0)
        os.environ['ARERU_LOGIC_PRESET'] = 'D'
        self.assertLess(gate_bias('大井', 1, 12, 'ダ'), 0)
        self.assertGreater(gate_bias('大井', 12, 12, 'ダ'), 0)

    def test_buy_floor_unchanged(self):
        from ev_analysis import BUY_EV_FLOOR, BUY_CONF_FLOOR
        self.assertEqual(BUY_EV_FLOOR, 108)
        self.assertEqual(BUY_CONF_FLOOR, 58)


if __name__ == '__main__':
    unittest.main()
