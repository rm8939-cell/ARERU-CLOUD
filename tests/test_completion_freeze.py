"""完成凍結: 採用候補は空。新規探索しない。本番は勝手に変えない。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestCompletionFreeze(unittest.TestCase):
    def test_no_adopt_candidates_and_no_auto_prod_change(self):
        data = json.loads((ROOT / 'data' / 'completion_freeze.json').read_text(encoding='utf-8'))
        self.assertEqual(data['採用候補'], [])
        self.assertEqual(data['探索'], '停止')
        self.assertFalse(data['本番']['変更してよいか'])
        self.assertEqual(data['本番']['ロジック'], 'OLD')
        rejected = {x['id'] for x in data['検証済みファクター']}
        self.assertIn('NEW', rejected)
        self.assertIn('SASHI_INNER', rejected)
        self.assertIn('WEIGHT', rejected)
        self.assertIn('JOCKEY', rejected)
        self.assertIn('HISTORY', rejected)
        self.assertTrue(all(x['判定'] == '不採用' for x in data['検証済みファクター']))
        self.assertTrue(any(x['id'] == '1' and x['必須'] for x in data['完成までに必要な作業']))

    def test_render_this_branch_stays_legacy_without_new_flags(self):
        text = (ROOT / 'render.yaml').read_text(encoding='utf-8')
        self.assertIn('ARERU_LEGACY_SCORE', text)
        self.assertIn('value: "1"', text)
        self.assertNotIn('ARERU_LOGIC_PRESET', text)
        self.assertNotIn('ARERU_KEEP_GAUSS', text)
        self.assertNotIn('ARERU_ABL_', text)
        self.assertNotIn('ARERU_CALIB_', text)

    def test_buy_floors_unchanged(self):
        from ev_analysis import BUY_CONF_FLOOR, BUY_EV_FLOOR
        self.assertEqual(BUY_EV_FLOOR, 108)
        self.assertEqual(BUY_CONF_FLOOR, 58)

    def test_catalog_points_to_freeze(self):
        cat = json.loads((ROOT / 'data' / 'feature_usage_catalog.json').read_text(encoding='utf-8'))
        self.assertFalse(cat['frozen']['production_change_allowed'])
        self.assertEqual(cat['frozen'].get('exploration'), 'stopped')
        self.assertEqual(cat['frozen'].get('adopt_candidates'), [])


if __name__ == '__main__':
    unittest.main()
