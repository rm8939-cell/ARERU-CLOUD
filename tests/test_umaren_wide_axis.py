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


if __name__ == '__main__':
    unittest.main()
