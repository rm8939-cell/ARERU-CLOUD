"""レース画面キャッシュ: 予想結果は変えず、2回目を速くする。"""
from __future__ import annotations

import os
import re
import unittest

os.environ.setdefault('ARERU_SKIP_BOOT', '1')
os.environ.setdefault('ARERU_LEGACY_SCORE', '1')
os.environ.setdefault('ARERU_ENABLE_GENERATION', '0')

from web_app import app, _PAGE_HTML_CACHE, _PREP_RACES_CACHE, _clear_runtime_caches


URL = '/?date=2026-08-29&history=1&source=jra&mode=predict'


def _fingerprint(html: str) -> dict:
    judges = re.findall(r'data-race-id="([^"]*)"[^>]*data-race-judge="([^"]*)"', html)
    if not judges:
        judges = re.findall(
            r'data-race-judge="([^"]*)"[^>]*data-race-id="([^"]*)"',
            html,
        )
        judges = [(b, a) for a, b in judges]
    buys = re.findall(r'🔥 BUY', html)
    ai = re.findall(r'>AI(\d+)<', html)
    honmei = re.findall(r'◎本命', html)
    return {
        'ui': 'data-ui="areu-app-v20"' in html,
        'judges': judges,
        'buy_count': len(buys),
        'ai_ranks': ai[:40],
        'honmei': len(honmei),
        'bytes': len(html.encode('utf-8')),
    }


class TestRacePageCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        _clear_runtime_caches()

    def test_second_load_hits_cache_and_matches_predictions(self):
        _clear_runtime_caches()
        first = self.client.get(URL)
        self.assertEqual(first.status_code, 200)
        html1 = first.get_data(as_text=True)
        fp1 = _fingerprint(html1)
        self.assertTrue(fp1['ui'])
        self.assertGreater(fp1['buy_count'], 0)
        self.assertTrue(fp1['judges'])
        header1 = first.headers.get('X-ARERU-Perf', '')
        self.assertIn('cache=miss', header1)

        second = self.client.get(URL)
        self.assertEqual(second.status_code, 200)
        html2 = second.get_data(as_text=True)
        fp2 = _fingerprint(html2)
        self.assertEqual(fp1, fp2)
        header2 = second.headers.get('X-ARERU-Perf', '')
        self.assertIn('cache=hit', header2)
        # ヒット時は計算をやり直さない
        m1 = re.search(r'total=([\d.]+)', header1)
        m2 = re.search(r'total=([\d.]+)', header2)
        self.assertIsNotNone(m1)
        self.assertIsNotNone(m2)
        self.assertLess(float(m2.group(1)), float(m1.group(1)))
        self.assertLess(float(m2.group(1)), 800.0)

    def test_force_refresh_skips_html_lookup(self):
        _clear_runtime_caches()
        first = self.client.get(URL)
        self.assertIn('cache=miss', first.headers.get('X-ARERU-Perf', ''))
        self.assertTrue(_PAGE_HTML_CACHE)
        # 通常の再読込はヒット。キャッシュを捨てたあとだけ miss に戻る。
        second = self.client.get(URL)
        self.assertIn('cache=hit', second.headers.get('X-ARERU-Perf', ''))
        _clear_runtime_caches()
        third = self.client.get(URL)
        self.assertIn('cache=miss', third.headers.get('X-ARERU-Perf', ''))
        self.assertEqual(
            _fingerprint(first.get_data(as_text=True)),
            _fingerprint(third.get_data(as_text=True)),
        )


if __name__ == '__main__':
    unittest.main()
