"""レース分析UIは表示専用。AI順位・BUY・スコアは変えない。"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault('ARERU_SKIP_BOOT', '1')
os.environ.setdefault('ARERU_LEGACY_SCORE', '1')
os.environ.setdefault('ARERU_ENABLE_GENERATION', '0')

from web_app import (
    _grade_from_race_name,
    _stamp_ai_field_ranks,
    _stamp_buy_display,
    _stamp_race_analysis_display,
)


def _sample_race(name='3歳未勝利', grade_name=None):
    race = {
        '本命': '本命馬',
        'レース名': grade_name or name,
        '投資判定': '買い',
        '展開予想データ': {
            '想定ペース': 'ハイ',
            '逃げ有利度': 28,
            '先行有利度': 42,
            '差し有利度': 72,
            '追込有利度': 68,
            '有利枠': '外枠',
            '荒れ指数': 80,
            'AI総評': '差しが届きやすい',
            '逃げ馬数': 1,
            '先行馬数': 5,
            '差し馬数': 3,
            '追込馬数': 1,
        },
        '予想馬': [{'馬名': '本命馬', '役割': '本命'}],
        'AI一覧': [
            {
                '馬名': '指数1位', 'AREru指数': 90, '馬番': 2,
                '展開相性': '差し・標準', 'ラップ適性': '後傾寄り適性',
                '勝率': 12.0, '複勝率': 36.0, 'カード期待値': 110,
                'カード': {
                    '距離適性': '○', 'コース適性': '－', '馬場適性': '○',
                    '上がり評価': 68, '上がり順位': 1, '期待値': 110,
                    'AI信頼度スコア': 60, 'AI評価': 90,
                },
            },
            {
                '馬名': '本命馬', 'AREru指数': 50, '馬番': 1, '役割': '本命',
                '展開相性': '先行・標準',
                'カード': {'勝率': 18.0, '複勝率': 40.0, '期待値': 101},
            },
            {'馬名': 'サン', 'AREru指数': 40, '馬番': 3},
            {'馬名': 'ヨン', 'AREru指数': 35, '馬番': 4},
            {'馬名': 'ゴ', 'AREru指数': 30, '馬番': 5},
            {'馬名': 'ロク', 'AREru指数': 20, '馬番': 6},
        ],
    }
    _stamp_ai_field_ranks(race)
    _stamp_buy_display([race])
    _stamp_race_analysis_display(race, {})
    return race


class TestRaceAnalysisUi(unittest.TestCase):
    def test_analysis_does_not_change_ranks_or_buy(self):
        race = _sample_race()
        by_name = {p['馬名']: p for p in race['AI一覧']}
        self.assertEqual(by_name['指数1位']['AI順位'], 1)
        self.assertEqual(by_name['指数1位']['表示印名'], '本命')
        self.assertFalse(by_name['指数1位'].get('BUY表示'))
        self.assertEqual(by_name['本命馬']['AI順位'], 2)
        self.assertTrue(by_name['本命馬'].get('BUY表示'))
        self.assertEqual([p['AI順位'] for p in race['表示グループ']['見送り']], [6])

    def test_same_analysis_keys_for_stakes_and_maiden(self):
        maiden = _sample_race('3歳未勝利')
        g1 = _sample_race('宝塚記念(GI)', '宝塚記念(GI)')
        g2 = _sample_race('札幌記念(GII)', '札幌記念(GII)')
        g3 = _sample_race('小倉記念(GIII)', '小倉記念(GIII)')
        keys = set(maiden['レース分析'].keys())
        for race in (g1, g2, g3):
            self.assertEqual(set(race['レース分析'].keys()), keys)
        self.assertEqual(g1['重賞グレード'], 'GⅠ')
        self.assertEqual(g2['重賞グレード'], 'GⅡ')
        self.assertEqual(g3['重賞グレード'], 'GⅢ')
        self.assertFalse(maiden['重賞レース'])
        self.assertIsNone(maiden['レース分析']['勝ち時計予想'])
        self.assertIsNone(g1['レース分析']['勝ち時計予想'])

    def test_lap_map_from_existing_style_only(self):
        race = _sample_race()
        names = {d['馬名'] for d in race['ラップマップ']}
        self.assertIn('指数1位', names)
        self.assertIn('本命馬', names)
        self.assertNotIn('ロク', names)
        top = next(d for d in race['ラップマップ'] if d['馬名'] == '指数1位')
        self.assertEqual(top['x'], 66.7)
        self.assertGreater(top['y'], 50)

    def test_missing_fields_stay_empty_not_invented(self):
        race = _sample_race()
        weak = next(p for p in race['AI一覧'] if p['馬名'] == 'ロク')
        self.assertIsNone(weak.get('ラップ適合度'))
        self.assertIsNone(weak.get('重賞実績'))
        self.assertIsNone(weak.get('クラス適性'))
        self.assertIsNone(weak.get('前走内容'))

    def test_grade_parser(self):
        self.assertEqual(_grade_from_race_name('天皇賞(春)(GI)'), 'GⅠ')
        self.assertEqual(_grade_from_race_name('札幌記念(GII)'), 'GⅡ')
        self.assertEqual(_grade_from_race_name('小倉記念(GIII)'), 'GⅢ')
        self.assertEqual(_grade_from_race_name('佐賀皐月賞(重賞)'), '重賞')
        self.assertEqual(_grade_from_race_name('3歳未勝利'), '')


if __name__ == '__main__':
    unittest.main()
