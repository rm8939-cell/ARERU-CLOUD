"""文字化けテキストの表示サニタイズと履歴取得エンコーディング。"""
from __future__ import annotations

import unittest


class TestMojibakeSanitize(unittest.TestCase):
    def test_clean_text_drops_replacement_char(self):
        from race_display import clean_text, is_mojibake
        bad = '2罩恰�育Μ'
        self.assertTrue(is_mojibake(bad))
        self.assertIsNone(clean_text(bad))
        self.assertEqual(clean_text(bad, fallback='データ不足'), 'データ不足')
        self.assertEqual(clean_text('東京'), '東京')
        self.assertEqual(clean_text('未勝利'), '未勝利')

    def test_honmei_view_skips_garbled_prev(self):
        from race_display import build_honmei_view
        view = build_honmei_view(
            {
                '前走内容': '1着 4��井��1 3罩割札筝�1',
                'ラップ適性表示': '前傾不利気味',
                '展開適性': '差し・標準',
            },
            {'AIレース診断': {'レースタイプ': '前半厳しい消耗戦'}},
            {},
        )
        self.assertIsNotNone(view)
        self.assertNotIn('�', view)
        self.assertNotIn('罩', view)
        self.assertIn('ラップ適性', view)

    def test_history_fetch_does_not_force_eucjp(self):
        import inspect
        from netkeiba_client import NetkeibaClient
        src = inspect.getsource(NetkeibaClient.fetch_horse_history)
        self.assertNotIn('encoding="euc-jp"', src)
        self.assertNotIn("encoding='euc-jp'", src)


if __name__ == '__main__':
    unittest.main()
