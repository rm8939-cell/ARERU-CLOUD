"""P0-2 unit / smoke tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from areru_engine import venue_from_race_id
from netkeiba_client import NetkeibaClient, venue_from_netkeiba_id
from refresh_data import (
    RUNNER_COLS,
    available_dates,
    latest_weekend,
    load_existing_runners,
    merge_runners,
    _normalize_runners,
)
import pandas as pd


class VenueParseTests(unittest.TestCase):
    def test_netkeiba_id(self):
        self.assertEqual(venue_from_race_id("202602011101"), "函館")
        self.assertEqual(venue_from_netkeiba_id("202605050801"), "東京")

    def test_legacy_jra_url(self):
        rid = "https://www.jra.go.jp/JRADB/accessS.html?CNAME=pw01sde0110202602060420260712%2F94"
        # pw01sde + RR(01) + VV(10) → 小倉
        self.assertEqual(venue_from_race_id(rid), "小倉")


class PastFiveTests(unittest.TestCase):
    def test_filters_future_and_limits_five(self):
        client = NetkeibaClient(sleep=0)
        hist = [
            {"日付": "2026-07-19", "着順": "1", "人気": "1"},
            {"日付": "2026-07-12", "着順": "2", "人気": "3"},
            {"日付": "2026-07-05", "着順": "4", "人気": "5"},
            {"年月日": "2026年6月28日", "着順": "3", "人気": "2"},
            {"日付": "2026-06-21", "着順": "6", "人気": "8"},
            {"日付": "2026-06-14", "着順": "1", "人気": "4"},
            {"日付": "2026-06-07", "着順": "9", "人気": "10"},
        ]
        out = client.past_five_for_score(hist, "2026-07-18")
        self.assertEqual(out["着順1"], "2")
        self.assertEqual(out["人気1"], "3")
        self.assertEqual(out["着順2"], "4")
        self.assertEqual(out["着順3"], "3")
        self.assertEqual(out["着順4"], "6")
        self.assertEqual(out["着順5"], "1")
        self.assertEqual(out["人気5"], "4")


class RefreshHelpersTests(unittest.TestCase):
    def test_latest_weekend(self):
        found = ["2026-07-19", "2026-07-18", "2026-07-12", "2026-07-11"]
        self.assertEqual(latest_weekend(found), ["2026-07-18", "2026-07-19"])

    def test_merge_replaces_same_date(self):
        base = _normalize_runners(pd.DataFrame([{
            "race_id": "old", "日付": "2026-07-18", "レース": 1,
            "馬名": "旧馬", "実着順": "",
            "着順1": 1, "人気1": 1, "着順2": "", "人気2": "",
            "着順3": "", "人気3": "", "着順4": "", "人気4": "",
            "着順5": "", "人気5": "",
        }]))
        new = _normalize_runners(pd.DataFrame([{
            "race_id": "new", "日付": "2026-07-18", "レース": 1,
            "馬名": "新馬", "実着順": "",
            "着順1": 2, "人気1": 2, "着順2": "", "人気2": "",
            "着順3": "", "人気3": "", "着順4": "", "人気4": "",
            "着順5": "", "人気5": "",
        }]))
        merged = merge_runners(base, new)
        self.assertEqual(list(merged["馬名"]), ["新馬"])
        self.assertEqual(available_dates(merged), ["2026-07-18"])


class ClientParseTests(unittest.TestCase):
    def test_list_race_ids_parses_html(self):
        client = NetkeibaClient(sleep=0)
        html = '<a href="shutuba.html?race_id=202602011101">1R</a>' \
               '<a href="result.html?race_id=202602011102">2R</a>'
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_resp.encoding = "utf-8"
        with patch.object(client.session, "get", return_value=mock_resp):
            ids = client.list_race_ids("20260718")
        self.assertEqual(ids, ["202602011101", "202602011102"])

    def test_fetch_shutuba_umaban_fallback(self):
        client = NetkeibaClient(sleep=0)
        html = """
        <html><head><title>1R 2026年7月18日</title></head><body>
        <table class="Shutuba_Table">
          <tr class="HorseList" id="tr_3">
            <td class="Waku"></td><td class="Umaban"></td>
            <td><span class="HorseName"><a href="/horse/2024104687">テストホース</a></span>
            <select name="3"></select></td>
          </tr>
        </table>
        </body></html>
        """
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_resp.encoding = "utf-8"
        with patch.object(client.session, "get", return_value=mock_resp):
            data = client.fetch_shutuba("202602011101")
        self.assertEqual(data["日付"], "2026-07-18")
        self.assertEqual(len(data["horses"]), 1)
        self.assertEqual(data["horses"][0]["馬番"], "3")
        self.assertEqual(data["horses"][0]["horse_id"], "2024104687")


class WebDatesTests(unittest.TestCase):
    def test_dates_union_runners_and_predictions(self):
        import web_app

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            arch = data / "predictions_by_date"
            arch.mkdir(parents=True)
            runners = data / "runners.csv"
            pd.DataFrame({"日付": ["2026-07-18", "2026-07-12"]}).to_csv(
                runners, index=False, encoding="utf-8-sig"
            )
            (arch / "predictions_2026-07-19.csv").write_text("race_id\n1\n", encoding="utf-8")
            (arch / "predictions_2026-07-04.csv").write_text("race_id\n1\n", encoding="utf-8")

            old_data, old_arch, old_runners, old_legacy = (
                web_app.DATA, web_app.ARCH, web_app.RUNNERS, web_app.LEGACY
            )
            try:
                web_app.DATA = data
                web_app.ARCH = arch
                web_app.RUNNERS = runners
                web_app.LEGACY = data / "missing.csv"
                got = web_app.dates()
            finally:
                web_app.DATA, web_app.ARCH, web_app.RUNNERS, web_app.LEGACY = (
                    old_data, old_arch, old_runners, old_legacy
                )
        self.assertEqual(
            got,
            ["2026-07-19", "2026-07-18", "2026-07-12", "2026-07-04"],
        )


class LiveSmokeTests(unittest.TestCase):
    """実ネットワークが使える環境向けの軽いスモーク。失敗しても他テストは通る。"""

    def test_discover_includes_latest_weekend(self):
        try:
            client = NetkeibaClient(sleep=0.15)
            dates = client.discover_kaisai_dates(lookback=14, lookahead=7)
        except Exception as e:
            self.skipTest(f"network unavailable: {e}")
        self.assertIn("2026-07-18", dates)
        self.assertIn("2026-07-19", dates)
        self.assertGreaterEqual(len(client.list_race_ids("20260718")), 1)


if __name__ == "__main__":
    unittest.main()
