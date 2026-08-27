"""完成凍結: 本番は OLD 固定。採用候補は空。表示定義と BUY 床を変えない。"""
from __future__ import annotations

import ast
import csv
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestCompletionFreeze(unittest.TestCase):
    def test_no_adopt_candidates(self):
        data = json.loads((ROOT / 'data' / 'completion_freeze.json').read_text(encoding='utf-8'))
        self.assertEqual(data['探索'], '停止')
        self.assertEqual(data['採用候補'], [])
        self.assertFalse(data['本番']['変更してよいか'])
        self.assertEqual(data['本番']['ロジック'], 'OLD')
        self.assertEqual(data['本番']['ARERU_LEGACY_SCORE'], '1')
        self.assertFalse(data['表示仕様_凍結']['tanhを変えてよいか'])

    def test_factor_ledger_columns_and_no_adopts(self):
        path = ROOT / 'data' / 'verified_factors.csv'
        with path.open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        required = {
            'ファクター名', '対象件数', 'train_ROI', 'holdout_ROI',
            '全期間ROI', '改善幅_holdout_vs_OLD', '再現性', '判定',
        }
        self.assertTrue(required.issubset(rows[0].keys()))
        ids = {r['id'] for r in rows}
        for need in (
            'OLD', 'NEW', 'SASHI_INNER', 'ODDS_INNER', 'SASHI_SWEET',
            'WEIGHT', 'JOCKEY', 'STYLE', 'GATE', 'TRACK', 'COURSE', 'HISTORY',
        ):
            self.assertIn(need, ids)
        for r in rows:
            if r['id'] == 'OLD':
                self.assertEqual(r['判定'], '基準')
            else:
                self.assertEqual(r['判定'], '不採用')


class TestProductionLock(unittest.TestCase):
    def test_legacy_pinned_on_render_and_gha(self):
        render = (ROOT / 'render.yaml').read_text(encoding='utf-8')
        gha = (ROOT / '.github' / 'workflows' / 'nar-daily-data.yml').read_text(encoding='utf-8')
        self.assertIn('ARERU_LEGACY_SCORE', render)
        self.assertIn('value: "1"', render)
        self.assertIn('ARERU_LEGACY_SCORE: "1"', gha)
        self.assertIn('ARERU_ENABLE_GENERATION', render)
        self.assertIn('value: "0"', render)
        forbidden = (
            'ARERU_LOGIC_PRESET',
            'ARERU_KEEP_GAUSS',
            'ARERU_ABL_',
            'ARERU_CALIB_',
        )
        for flag in forbidden:
            self.assertNotIn(flag, render)
            self.assertNotIn(flag, gha)

    def test_buy_floors_and_tanh_unchanged(self):
        from ev_analysis import (
            BUY_CONF_FLOOR,
            BUY_EV_FLOOR,
            EV_DISPLAY_MAX,
            EV_DISPLAY_MIN,
            _soft_display_ev,
        )
        self.assertEqual(BUY_EV_FLOOR, 108)
        self.assertEqual(BUY_CONF_FLOOR, 58)
        self.assertEqual(EV_DISPLAY_MIN, 78)
        self.assertEqual(EV_DISPLAY_MAX, 124)
        raw = 9.5 * 14.8
        expected = int(round(min(124, max(78, 100.0 + 26.0 * math.tanh((raw - 100.0) / 30.0)))))
        self.assertEqual(_soft_display_ev(raw), expected)
        self.assertEqual(_soft_display_ev(140.6), 123)

    def test_web_fallback_uses_buy_floor_not_100(self):
        text = (ROOT / 'web_app.py').read_text(encoding='utf-8')
        self.assertIn('BUY_EV_FLOOR', text)
        self.assertNotIn('if ev>=100:', text)

    def test_buy_template_does_not_fallback_to_sim_win(self):
        html = (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('表示期待値', html)
        self.assertIn('勝ちを保証するものではありません', html)
        self.assertIn('近走指数順位', html)
        # 推定勝率は補正勝率のみ。生SIMへ落とさない
        self.assertNotIn("or t.get('シミュレーション勝率')", html)
        self.assertIn('生の期待値＝推定勝率×オッズ', html)

    def test_index_rank_copied_from_pick_card(self):
        from ev_analysis import apply_expected_value
        record = {
            '本命': 'テスト馬',
            '本命オッズ': 8.0,
            '本命人気': 4,
            'オッズ取得済': True,
            'ピックカード一覧': [
                {'馬名': 'テスト馬', '役割': '本命', '近走指数順位': 3, '勝率': 12.0},
            ],
        }
        apply_expected_value(record)
        self.assertEqual(int(record['近走指数順位']), 3)

    def test_legacy_default_without_env_is_new_so_pin_is_required(self):
        import os
        from areru_engine import legacy_score_enabled
        old = os.environ.pop('ARERU_LEGACY_SCORE', None)
        try:
            self.assertFalse(legacy_score_enabled())
        finally:
            if old is not None:
                os.environ['ARERU_LEGACY_SCORE'] = old

    def test_no_new_scoring_functions_in_this_tree(self):
        """完成作業でスコア本体へ実験フラグを足していないこと。"""
        engine = ast.parse((ROOT / 'areru_engine.py').read_text(encoding='utf-8'))
        names = {n.name for n in ast.walk(engine) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn('legacy_score_enabled', names)


if __name__ == '__main__':
    unittest.main()
