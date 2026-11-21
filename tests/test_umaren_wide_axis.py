"""馬連・ワイド軸の買い目生成テスト。"""
from __future__ import annotations

import unittest


class TestUmarenWideAxis(unittest.TestCase):
    def _race(self):
        return {
            '期待値': 110,
            '投資判定': '買い',
            'source': 'jra',
            '推奨券種': '三連単',
            '予想馬': [
                {'役割': '本命', '馬番表示': '①', '馬番': '1'},
                {'役割': '対抗', '馬番表示': '⑤', '馬番': '5'},
                {'役割': '注目馬', '馬番表示': '⑧', '馬番': '8'},
            ],
        }

    def test_axis_is_wide_then_umaren_without_tansho(self):
        from web_app import _axis_umaren_wide_tickets
        tickets = _axis_umaren_wide_tickets(self._race())
        kinds = [t['券種'] for t in tickets if not t.get('参考')]
        self.assertEqual(kinds[0], 'ワイド')
        self.assertIn('馬連', kinds)
        self.assertNotIn('単勝', kinds)
        wide = next(t for t in tickets if t['券種'] == 'ワイド' and not t.get('参考'))
        self.assertIn('①-⑤', wide['馬番表示'])

    def test_prefer_kind_overrides_sanrentan(self):
        from web_app import _axis_umaren_wide_tickets, _prefer_umaren_wide_kind
        race = self._race()
        tickets = _axis_umaren_wide_tickets(race)
        kind = _prefer_umaren_wide_kind(race, tickets)
        self.assertIn(kind, ('ワイド', '馬連'))

    def test_keeps_engine_wide_or_umaren(self):
        from web_app import _prefer_umaren_wide_kind
        race = self._race()
        race['推奨券種'] = '馬連'
        self.assertEqual(_prefer_umaren_wide_kind(race, []), '馬連')

    def test_ignores_unmarked_and_sim_reference(self):
        from web_app import _axis_umaren_wide_tickets
        race = self._race()
        # 印なしの上位馬・印非依存SIM買い目が混ざっても相手に使わない
        race['予想馬'] = list(race['予想馬']) + [
            {'役割': '', '馬番表示': '⑫', '馬番': '12', '馬名': 'ノーマーク'},
        ]
        race['ワイド買い目'] = '①-⑫｜⑤-⑫'
        race['馬連買い目'] = '①-⑫'
        tickets = _axis_umaren_wide_tickets(race)
        self.assertTrue(tickets)
        self.assertTrue(all(not t.get('参考') for t in tickets))
        joined = ' '.join(str(t.get('馬番表示') or '') for t in tickets)
        self.assertNotIn('⑫', joined)
        self.assertNotIn('12', joined)
        self.assertIn('①', joined)
        self.assertIn('⑤', joined)

    def test_partners_come_from_display_marks_only(self):
        from web_app import _axis_umaren_wide_tickets
        race = {
            '期待値': 105,
            '投資判定': '見送り',
            '表示グループ': {
                '本命': [{'表示印': '◎', '表示印名': '本命', '馬番表示': '③', '馬番': '3'}],
                '対抗': [{'表示印': '○', '表示印名': '対抗', '馬番表示': '⑦', '馬番': '7'}],
                '単穴': [{'表示印': '▲', '表示印名': '単穴', '馬番表示': '⑨', '馬番': '9'}],
                '連下': [],
                '見送り': [{'表示印': '', '表示印名': '見送り', '馬番表示': '⑪', '馬番': '11'}],
            },
            '予想馬': [
                {'役割': '', '馬番表示': '⑪', '馬番': '11'},
            ],
            'ワイド買い目': '③-⑪',
        }
        tickets = _axis_umaren_wide_tickets(race)
        joined = ' '.join(str(t.get('馬番表示') or '') for t in tickets)
        self.assertIn('③', joined)
        self.assertIn('⑦', joined)
        self.assertNotIn('⑪', joined)


if __name__ == '__main__':
    unittest.main()
