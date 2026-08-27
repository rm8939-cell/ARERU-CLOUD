"""単特徴実験フラグは本番で無効。KEEP_GAUSS / HISTORY_EXPAND は明示ONのみ。"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _clear():
    for k in list(os.environ):
        if k.startswith('ARERU_ABL_') or k.startswith('ARERU_CALIB_') or k in (
            'ARERU_LEGACY_SCORE', 'ARERU_LOGIC_PRESET', 'ARERU_XSEL_FEATURES',
            'ARERU_KEEP_GAUSS',
        ):
            os.environ.pop(k, None)


class TestProductionUntouched(unittest.TestCase):
    def test_render_stays_legacy_no_abl(self):
        text = (ROOT / 'render.yaml').read_text(encoding='utf-8')
        self.assertIn('ARERU_LEGACY_SCORE', text)
        self.assertIn('value: "1"', text)
        self.assertNotIn('ARERU_KEEP_GAUSS', text)
        self.assertNotIn('ARERU_ABL_TIME', text)
        self.assertNotIn('ARERU_ABL_HISTORY_EXPAND', text)
        self.assertNotIn('ARERU_LOGIC_PRESET', text)

    def test_buy_floors_unchanged(self):
        from ev_analysis import BUY_CONF_FLOOR, BUY_EV_FLOOR
        self.assertEqual(BUY_EV_FLOOR, 108)
        self.assertEqual(BUY_CONF_FLOOR, 58)

    def test_recency_head_unchanged(self):
        from areru_engine import RECENCY
        self.assertEqual(list(RECENCY[:5]), [1.0, 0.82, 0.65, 0.48, 0.34])


class TestKeepGaussAndFlags(unittest.TestCase):
    def setUp(self):
        _clear()

    def tearDown(self):
        _clear()

    def test_defaults_stay_old_gaussian(self):
        os.environ['ARERU_LEGACY_SCORE'] = '1'
        from areru_engine import (
            ablation_enabled, keep_gauss_enabled, use_stage_sim,
        )
        self.assertFalse(keep_gauss_enabled())
        self.assertFalse(use_stage_sim())
        self.assertFalse(ablation_enabled('time'))
        self.assertFalse(ablation_enabled('margin'))
        self.assertFalse(ablation_enabled('track'))
        self.assertFalse(ablation_enabled('course'))
        self.assertFalse(ablation_enabled('history_expand'))
        self.assertFalse(ablation_enabled('sweight'))
        self.assertFalse(ablation_enabled('sjockey'))

    def test_abl_time_without_keep_gauss_would_switch_stage(self):
        os.environ['ARERU_LEGACY_SCORE'] = '1'
        os.environ['ARERU_ABL_TIME'] = '1'
        from areru_engine import use_stage_sim, keep_gauss_enabled
        self.assertFalse(keep_gauss_enabled())
        self.assertTrue(use_stage_sim())

    def test_keep_gauss_blocks_stage_switch(self):
        os.environ['ARERU_LEGACY_SCORE'] = '1'
        os.environ['ARERU_ABL_TIME'] = '1'
        os.environ['ARERU_KEEP_GAUSS'] = '1'
        from areru_engine import use_stage_sim
        self.assertFalse(use_stage_sim())

    def test_history_expand_never_implicit(self):
        os.environ['ARERU_LEGACY_SCORE'] = '0'
        from areru_engine import ablation_enabled
        self.assertFalse(ablation_enabled('history_expand'))
        os.environ['ARERU_ABL_HISTORY_EXPAND'] = '1'
        self.assertTrue(ablation_enabled('history_expand'))


class TestSingleFeatureScoreHooks(unittest.TestCase):
    def setUp(self):
        _clear()
        os.environ['ARERU_LEGACY_SCORE'] = '1'

    def tearDown(self):
        _clear()

    def _row(self):
        return {
            '馬名': 'テスト馬',
            'race_id': '202601010101',
            '着順1': 2, '着順2': 3, '着順3': 4, '着順4': 5, '着順5': 6,
            '人気1': 3, '人気2': 4, '人気3': 5, '人気4': 6, '人気5': 7,
            '場1': '大井', '場2': '大井', '場3': '大井', '場4': '大井', '場5': '大井',
            'レース名1': 'C1', 'レース名2': 'C1', 'レース名3': 'C1', 'レース名4': 'C1', 'レース名5': 'C1',
        }

    def _hist(self, n=8):
        rows = []
        for i in range(n):
            rows.append({
                '馬名': 'テスト馬',
                '_horse': 'テスト馬',
                '_date': pd.Timestamp('2026-07-01') - pd.Timedelta(days=14 * (i + 1)),
                '着順': 1 if i >= 5 else (2 + i),
                '人気': 8,
                '場': '大井',
                'レース名': 'C1',
                '距離': 'ダ1600',
                'タイム': '1:40.0',
                '着差': 'ハナ',
                '馬場': '良',
                '頭数': 12,
            })
        h = pd.DataFrame(rows)
        return h

    def test_history_expand_changes_score_only_when_on(self):
        from areru_engine import load_weights, score_runner
        row = self._row()
        hist = self._hist(8)
        target = pd.Timestamp('2026-08-01')
        w = load_weights()
        s0, _, _ = score_runner(row, hist, target, w)
        os.environ['ARERU_ABL_HISTORY_EXPAND'] = '1'
        s1, _, _ = score_runner(row, hist, target, w)
        self.assertNotEqual(round(s0, 4), round(s1, 4))

    def test_time_bonus_keep_gauss_changes_score(self):
        from areru_engine import load_weights, score_runner
        row = self._row()
        row['タイム1'] = '1:42.0'
        row['タイム2'] = '1:43.5'
        hist = self._hist(3)
        target = pd.Timestamp('2026-08-01')
        w = load_weights()
        s0, _, _ = score_runner(row, hist, target, w)
        os.environ['ARERU_ABL_TIME'] = '1'
        os.environ['ARERU_KEEP_GAUSS'] = '1'
        s1, _, why = score_runner(row, hist, target, w)
        self.assertGreater(s1, s0)
        self.assertTrue(any('タイム' in x for x in why))


if __name__ == '__main__':
    unittest.main()
