from __future__ import annotations

import os
import unittest

import pandas as pd


class TestScoreExtras(unittest.TestCase):
    def test_legacy_off_does_not_change_score(self):
        os.environ['ARERU_LEGACY_SCORE'] = '1'
        os.environ.pop('ARERU_LOGIC_PRESET', None)
        for k in list(os.environ):
            if k.startswith('ARERU_ABL_'):
                os.environ.pop(k, None)
        from score_extras import apply_unused_score_extras
        g = pd.DataFrame({
            '馬名': ['A', 'B'],
            'AREru指数': [50.0, 48.0],
            '斤量': [56.0, 54.0],
            '枠': [1, 8],
            '理由': ['総合評価', '総合評価'],
        })
        out = apply_unused_score_extras(g, None, '2026-08-01', '大井')
        self.assertEqual(list(out['AREru指数']), [50.0, 48.0])

    def test_preset_x_burden_changes_score(self):
        os.environ['ARERU_LEGACY_SCORE'] = '1'
        os.environ['ARERU_LOGIC_PRESET'] = 'X'
        from importlib import reload
        import areru_engine
        reload(areru_engine)
        from score_extras import apply_unused_score_extras
        g = pd.DataFrame({
            '馬名': ['A', 'B'],
            'AREru指数': [50.0, 48.0],
            '斤量': [58.0, 52.0],
            '枠': [1, 8],
            '騎手': ['甲', '乙'],
            '着順1': [2, 8],
            '人気1': [2, 3],
            '理由': ['総合評価', '総合評価'],
        })
        out = apply_unused_score_extras(g, None, '2026-08-01', '大井')
        self.assertNotEqual(list(out['AREru指数']), [50.0, 48.0])
        # 軽い B が相対的に上がる
        self.assertGreater(float(out.iloc[1]['AREru指数']) - 48.0, float(out.iloc[0]['AREru指数']) - 50.0)


if __name__ == '__main__':
    unittest.main()
