"""全馬AI順位と表示印。BUY判定は既存ロジックのまま。"""
from __future__ import annotations

import unittest

from web_app import _stamp_ai_field_ranks, _stamp_buy_display


class TestFieldMarks(unittest.TestCase):
    def test_all_horses_ranked_and_marked(self):
        race = {
            '本命': 'イチ',
            '投資判定': '見送り',
            'AI一覧': [
                {'馬名': 'ゴ', 'AREru指数': 40, '馬番': 5},
                {'馬名': 'イチ', 'AREru指数': 80, '馬番': 1, '役割': '本命'},
                {'馬名': 'ニ', 'AREru指数': 70, '馬番': 2},
                {'馬名': 'サン', 'AREru指数': 60, '馬番': 3},
                {'馬名': 'ヨン', 'AREru指数': 55, '馬番': 4},
                {'馬名': 'ロク', 'AREru指数': 30, '馬番': 6},
            ],
        }
        _stamp_ai_field_ranks(race)
        _stamp_buy_display([race])
        names = [p['馬名'] for p in race['AI一覧']]
        self.assertEqual(names, ['イチ', 'ニ', 'サン', 'ヨン', 'ゴ', 'ロク'])
        self.assertEqual([p['AI順位'] for p in race['AI一覧']], [1, 2, 3, 4, 5, 6])
        self.assertEqual(race['AI一覧'][0]['表示印名'], '本命')
        self.assertEqual(race['AI一覧'][1]['表示印名'], '対抗')
        self.assertEqual(race['AI一覧'][2]['表示印名'], '単穴')
        self.assertEqual(race['AI一覧'][3]['表示印名'], '連下')
        self.assertEqual(race['AI一覧'][4]['表示印名'], '連下')
        self.assertEqual(race['AI一覧'][5]['表示印名'], '見送り')
        self.assertFalse(any(p.get('BUY表示') for p in race['AI一覧']))

    def test_buy_stays_on_engine_honmei_not_rank(self):
        race = {
            '本命': '本命馬',
            '投資判定': '買い',
            '予想馬': [{'馬名': '本命馬', '役割': '本命'}],
            'AI一覧': [
                {'馬名': '指数1位', 'AREru指数': 90, '馬番': 2},
                {'馬名': '本命馬', 'AREru指数': 50, '馬番': 1, '役割': '本命'},
            ],
        }
        _stamp_ai_field_ranks(race)
        _stamp_buy_display([race])
        by_name = {p['馬名']: p for p in race['AI一覧']}
        self.assertEqual(by_name['指数1位']['AI順位'], 1)
        self.assertEqual(by_name['指数1位']['表示印名'], '本命')
        self.assertFalse(by_name['指数1位'].get('BUY表示'))
        self.assertEqual(by_name['本命馬']['AI順位'], 2)
        self.assertEqual(by_name['本命馬']['表示印名'], '対抗')
        self.assertTrue(by_name['本命馬'].get('BUY表示'))

    def test_tie_breaks_stable_on_popularity_then_ban(self):
        race = {
            'AI一覧': [
                {'馬名': 'B', 'AREru指数': 50, '人気': 4, '馬番': 8},
                {'馬名': 'A', 'AREru指数': 50, '人気': 2, '馬番': 3},
            ],
        }
        _stamp_ai_field_ranks(race)
        self.assertEqual([p['馬名'] for p in race['AI一覧']], ['A', 'B'])


if __name__ == '__main__':
    unittest.main()
