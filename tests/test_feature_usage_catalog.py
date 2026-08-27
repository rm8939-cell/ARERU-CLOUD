"""本番凍結と特徴量カタログの契約。予測ロジックは変えない。"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestProductionFreeze(unittest.TestCase):
    def test_render_stays_legacy(self):
        text = (ROOT / 'render.yaml').read_text(encoding='utf-8')
        self.assertIn('ARERU_LEGACY_SCORE', text)
        self.assertIn('value: "1"', text)
        self.assertNotIn('ARERU_CALIB_SASHI_INNER', text)
        self.assertNotIn('ARERU_LOGIC_PRESET', text)

    def test_three_candidates_rejected(self):
        verdict = json.loads((ROOT / 'data' / 'three_calib_verdict.json').read_text(encoding='utf-8'))
        self.assertEqual(verdict['採用候補'], [])
        self.assertEqual(verdict['保留'], [])
        self.assertEqual(set(verdict['不採用']), {'SASHI_INNER', 'ODDS_INNER', 'SASHI_SWEET'})
        self.assertFalse(verdict['本番ロジックを変更してよいか'])
        self.assertIn('OLD', verdict['本番'])
        self.assertTrue(verdict.get('確定'))

    def test_calib_flags_default_off(self):
        for k in list(os.environ):
            if k.startswith('ARERU_CALIB_') or k in ('ARERU_LOGIC_PRESET',):
                os.environ.pop(k, None)
        from race_sim import calib_flag, three_calib_adj
        self.assertFalse(calib_flag('SASHI_INNER'))
        self.assertFalse(calib_flag('ODDS_INNER'))
        self.assertFalse(calib_flag('SASHI_SWEET'))
        d, plus, minus = three_calib_adj(0.6, 1, 12, 15.0)
        self.assertEqual(d, 0.0)
        self.assertEqual(plus, [])
        self.assertEqual(minus, [])

    def test_buy_floors_unchanged(self):
        from ev_analysis import BUY_CONF_FLOOR, BUY_EV_FLOOR
        self.assertEqual(BUY_EV_FLOOR, 108)
        self.assertEqual(BUY_CONF_FLOOR, 58)


class TestFeatureUsageCatalog(unittest.TestCase):
    def test_catalog_has_four_lists(self):
        path = ROOT / 'data' / 'feature_usage_catalog.json'
        self.assertTrue(path.exists(), 'feature_usage_catalog.json が必要')
        data = json.loads(path.read_text(encoding='utf-8'))
        for key in ('currently_used', 'acquired_unused', 'sparse_or_unusable', 'verifiable'):
            self.assertIn(key, data)
            self.assertIsInstance(data[key], list)
            self.assertGreater(len(data[key]), 0, key)
        self.assertEqual(data['frozen']['three_candidates_verdict'], '不採用確定')
        self.assertTrue(data['frozen']['buy_filters_forbidden'])
        self.assertFalse(data['frozen']['production_change_allowed'])
        self.assertEqual(data['frozen']['production'], 'OLD (ARERU_LEGACY_SCORE=1)')
        ids_used = {x['id'] for x in data['currently_used']}
        self.assertIn('past_finish', ids_used)
        self.assertIn('market_odds', ids_used)
        sparse_ids = {x['id'] for x in data['sparse_or_unusable']}
        self.assertIn('pass_pace_last3f', sparse_ids)
        self.assertIn('runners_finish_today', sparse_ids)
        self.assertIn('hist_weight', sparse_ids)
        unused_ids = {x['id'] for x in data['acquired_unused']}
        self.assertTrue({'runners_kg', 'runners_waku', 'runners_jockey'} <= unused_ids)

    def test_catalog_csv_exists(self):
        path = ROOT / 'data' / 'feature_usage_catalog.csv'
        self.assertTrue(path.exists())
        text = path.read_text(encoding='utf-8')
        self.assertIn('currently_used', text)
        self.assertIn('acquired_unused', text)
        self.assertIn('sparse_or_unusable', text)
        self.assertIn('verifiable', text)

    def test_calibration_report_has_both_splits(self):
        path = ROOT / 'data' / 'calibration_report.json'
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding='utf-8'))
        old = data['calibration']['OLD']
        for period in ('train', 'holdout'):
            self.assertIn(period, old)
            self.assertGreater(old[period]['n_honmei'], 100)
            for model in ('sim', 'adj', 'market'):
                self.assertIsNotNone(old[period][model]['brier'])


if __name__ == '__main__':
    unittest.main()
