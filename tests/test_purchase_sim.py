"""購入シミュレーション（★＋期待値100%以上）の組み立て。"""
from __future__ import annotations

import unittest


class TestPurchaseSimulation(unittest.TestCase):
    def test_star_axis_plus_ev_engine_rows(self):
        from web_app import _build_purchase_simulation
        race = {
            'race_id': '202601010101',
            '期待値': 120,
            '投資判定': '買い',
            '予想馬': [
                {'役割': '本命', '馬番表示': '5', '馬番': '5'},
                {'役割': '対抗', '馬番表示': '14', '馬番': '14'},
                {'役割': '注目馬', '馬番表示': '2', '馬番': '2'},
            ],
            '推奨馬券一覧': [
                {'券種': 'ワイド', '馬番表示': '5-14', '馬番': [5, 14], '的中率': 15.7, '実オッズ': 11.4},
                {'券種': 'ワイド', '馬番表示': '2-14', '馬番': [2, 14], '的中率': 19.0, '実オッズ': 6.2},
                {'券種': 'ワイド', '馬番表示': '1-8', '馬番': [1, 8], '的中率': 8.0, '実オッズ': 10.0},  # EV80 skip
                {'券種': '馬連', '馬番表示': '5-11', '馬番': [5, 11], '的中率': 1.9, '実オッズ': 51.8},
            ],
        }
        rows = _build_purchase_simulation(race, limit=10)
        self.assertTrue(rows)
        self.assertTrue(any(r.get('おすすめ') for r in rows))
        # 期待値 = 的中率% × オッズ
        wide = next(r for r in rows if r.get('券種') == 'ワイド' and '5-14' in str(r.get('馬番表示')))
        self.assertEqual(wide.get('期待値'), 179.0)
        # 100%未満は入らない
        joined = ' '.join(str(r.get('馬番表示')) for r in rows)
        self.assertNotIn('1-8', joined)
        self.assertLessEqual(len(rows), 10)

    def test_template_has_purchase_sim_ui(self):
        from pathlib import Path
        board = (Path(__file__).resolve().parents[1] / 'templates' / 'partials' / 'race_article_body.html').read_text(encoding='utf-8')
        css = (Path(__file__).resolve().parents[1] / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('購入シミュレーション', board)
        self.assertIn('sim-ev', board)
        self.assertIn('sim-kind', board)
        self.assertIn('期待値100%以上', board)
        self.assertIn('sim-row', css)


if __name__ == '__main__':
    unittest.main()
