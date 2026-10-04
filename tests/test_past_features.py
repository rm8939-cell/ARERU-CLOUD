import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import past_feature_model as pfm  # noqa: E402
import past_features as pf  # noqa: E402


class TestParsers(unittest.TestCase):
    def test_time(self):
        self.assertAlmostEqual(pf.parse_time("2:31.9"), 151.9, places=3)
        self.assertAlmostEqual(pf.parse_time("58.4"), 58.4, places=3)
        self.assertTrue(np.isnan(pf.parse_time("")))

    def test_distance(self):
        self.assertEqual(pf.parse_distance("芝2500"), ("芝", 2500.0))
        self.assertEqual(pf.parse_distance("ダ1400"), ("ダ", 1400.0))
        self.assertEqual(pf.parse_distance("")[0], "")

    def test_going_is_not_weather(self):
        # 旧実装は天気列を馬場として読んでいた。天気語は馬場として解釈されない。
        self.assertEqual(pf.parse_going("良"), 0.0)
        self.assertEqual(pf.parse_going("不良"), 3.0)
        self.assertTrue(np.isnan(pf.parse_going("晴")))
        self.assertTrue(np.isnan(pf.parse_going("雨")))

    def test_pace_and_passing(self):
        self.assertEqual(pf.parse_pace("30.0-35.4"), (30.0, 35.4))
        self.assertEqual(pf.parse_passing("12-12-11-10"), [12.0, 12.0, 11.0, 10.0])
        self.assertEqual(pf.parse_passing(""), [])

    def test_weight_and_finish(self):
        self.assertEqual(pf.parse_weight("438(+2)"), (438.0, 2.0))
        self.assertEqual(pf.parse_finish("3"), 3.0)
        # 中止・除外は着順として扱わない。
        self.assertTrue(np.isnan(pf.parse_finish("中止")))

    def test_turn(self):
        self.assertEqual(pf.turn_of("東京"), -1.0)
        self.assertEqual(pf.turn_of("中山"), 1.0)
        self.assertTrue(np.isnan(pf.turn_of("帯広")))


def _hist(rows):
    base = {k: "" for k in ("年月日", "場", "レース", "レース名", "頭数", "人気",
                            "着順", "騎手", "斤量", "距離", "馬場", "馬体重",
                            "タイム", "着差", "通過", "ペース", "上り", "天気",
                            "備考", "オッズ", "枠番", "馬番", "賞金")}
    out = []
    for r in rows:
        d = dict(base)
        d.update(r)
        out.append(d)
    df = pd.DataFrame(out)
    df["horse_id"] = "h1"
    return df


class TestNoLeakage(unittest.TestCase):
    def setUp(self):
        self.rows = _hist([
            {"年月日": "2026-01-10", "場": "1中山1", "レース": "5", "レース名": "1勝クラス",
             "頭数": "12", "人気": "3", "着順": "2", "距離": "芝1600", "馬場": "良",
             "タイム": "1:34.0", "上り": "34.5", "通過": "5-5", "ペース": "35.0-35.2"},
            {"年月日": "2026-03-10", "場": "2中山3", "レース": "7", "レース名": "1勝クラス",
             "頭数": "14", "人気": "2", "着順": "1", "距離": "芝1600", "馬場": "良",
             "タイム": "1:33.0", "上り": "33.9", "通過": "4-4", "ペース": "35.1-34.8"},
            {"年月日": "2026-06-10", "場": "3中山2", "レース": "9", "レース名": "2勝クラス",
             "頭数": "16", "人気": "1", "着順": "1", "距離": "芝1800", "馬場": "重",
             "タイム": "1:48.0", "上り": "35.5", "通過": "2-2-2", "ペース": "36.0-35.9"},
        ])

    def test_build_features_sees_only_the_past(self):
        h = pf.add_split_columns(_typed(self.rows))
        cut = pd.Timestamp("2026-04-01")
        past = h[h["date"] < cut]
        self.assertEqual(len(past), 2)
        base = pf.course_baselines(h, cut)
        # 基準も cut より前の行だけから作られる。6月の重馬場1800は入らない。
        self.assertFalse(((base["dist"] == 1800.0) & (base["going"] == 2.0)).any())

    def test_baseline_excludes_future_rows(self):
        h = pf.add_split_columns(_typed(self.rows))
        early = pf.course_baselines(h, pd.Timestamp("2026-02-01"))
        late = pf.course_baselines(h, pd.Timestamp("2026-12-01"))
        self.assertLessEqual(int(early["t_n"].sum() if not early.empty else 0),
                             int(late["t_n"].sum() if not late.empty else 0))

    def test_missing_data_is_nan_not_zero(self):
        h = pf.add_split_columns(_typed(self.rows))
        cut = pd.Timestamp("2026-12-01")
        past = pf.attach_baselines(h[h["date"] < cut].copy(),
                                   pf.course_baselines(h, cut),
                                   pf.pace_baseline(h, cut), None)
        feats = pf.build_features(past, {"date": cut, "venue": "中山", "surface": "芝",
                                         "dist": 2000.0, "going": 0.0, "class_lv": 5.0})
        # 2000mの経験が無いので同距離適性は NaN。0点ではない。
        self.assertTrue(np.isnan(feats["同距離適性"]))
        self.assertEqual(feats["同距離適性件数"], 0.0)


def _typed(rows: pd.DataFrame) -> pd.DataFrame:
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "h1.json"
        p.write_text(json.dumps(rows.drop(columns=["horse_id"]).to_dict("records"),
                                ensure_ascii=False), encoding="utf-8")
        old = pf.HORSE_CACHE
        pf.HORSE_CACHE = Path(d)
        try:
            return pf.load_history_frame(["h1"])
        finally:
            pf.HORSE_CACHE = old


class TestModelApplication(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("ARERU_PAST_FEATURES", None)
        pfm.reset_cache()

    def test_disabled_by_default(self):
        os.environ.pop("ARERU_PAST_FEATURES", None)
        self.assertFalse(pfm.enabled())

    def test_race_z_treats_missing_as_average(self):
        z = pfm.race_z(np.array([1.0, 2.0, 3.0, np.nan]))
        self.assertEqual(z[3], 0.0)
        self.assertAlmostEqual(float(z[:3].mean()), 0.0, places=9)

    def test_race_z_constant_is_flat(self):
        z = pfm.race_z(np.array([2.0, 2.0, 2.0]))
        self.assertTrue(np.allclose(z, 0.0))

    def test_linear_term_none_without_model(self):
        os.environ["ARERU_PAST_FEATURES"] = "1"
        pfm.reset_cache()
        self.assertIsNone(pfm.linear_term("nonexistent", ["A", "B"]))


if __name__ == "__main__":
    unittest.main()
