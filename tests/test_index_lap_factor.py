"""指数×ラップ合成ファクターの単体テスト。"""
from __future__ import annotations

import os
import unittest


class TestIndexLapFactor(unittest.TestCase):
    def test_score_neutral_lap_keeps_index(self):
        from race_sim import index_lap_score
        self.assertEqual(index_lap_score(80, 50), 80.0)

    def test_score_high_lap_boosts(self):
        from race_sim import index_lap_score
        # 80 * (1 + 0.25 * 0.4) = 88
        self.assertEqual(index_lap_score(80, 70), 88.0)

    def test_score_low_lap_cuts(self):
        from race_sim import index_lap_score
        # 80 * (1 + 0.25 * -0.4) = 72
        self.assertEqual(index_lap_score(80, 30), 72.0)

    def test_score_missing_lap_falls_back_to_index(self):
        from race_sim import index_lap_score
        self.assertEqual(index_lap_score(66.5, None), 66.5)

    def test_score_missing_index_is_none(self):
        from race_sim import index_lap_score
        self.assertIsNone(index_lap_score(None, 70))

    def test_enabled_default_on(self):
        from race_sim import index_lap_enabled
        os.environ.pop('ARERU_INDEX_LAP', None)
        self.assertTrue(index_lap_enabled())

    def test_enabled_can_disable(self):
        from race_sim import index_lap_enabled
        os.environ['ARERU_INDEX_LAP'] = '0'
        try:
            self.assertFalse(index_lap_enabled())
        finally:
            os.environ.pop('ARERU_INDEX_LAP', None)

    def test_aptitude_rows_include_index_lap(self):
        from race_display import aptitude_rows
        rows = aptitude_rows({
            '指数ラップ': 88.0,
            '指数ラップラベル': '88.0（指数80×ラップ押し上げ）',
            'ラップ適合度': 70,
            'ラップ適性表示': '後傾寄り適性',
            '展開適性': '差し向き',
        })
        names = [r['項目'] for r in rows]
        self.assertIn('指数ラップ', names)
        self.assertEqual(names[0], '指数ラップ')
        self.assertEqual(rows[0]['バー'], 88.0)

    def test_label_mentions_boost(self):
        from race_sim import index_lap_label
        lab = index_lap_label(88.0, 80, 70)
        self.assertIn('ラップ押し上げ', lab)
        self.assertIn('88.0', lab)


if __name__ == '__main__':
    unittest.main()
