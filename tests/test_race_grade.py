import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import race_grade as rg  # noqa: E402
from ev_analysis import decide_buy_skip  # noqa: E402


class EnvGuard(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("ARERU_BUY_V2")
        rg.reset_cache()

    def tearDown(self):
        if self._env is None:
            os.environ.pop("ARERU_BUY_V2", None)
        else:
            os.environ["ARERU_BUY_V2"] = self._env
        rg.reset_cache()


class TestEnabled(EnvGuard):
    def test_off_by_default(self):
        os.environ.pop("ARERU_BUY_V2", None)
        self.assertFalse(rg.enabled())

    def test_on_values(self):
        for v in ("1", "true", "YES", "on"):
            os.environ["ARERU_BUY_V2"] = v
            self.assertTrue(rg.enabled(), v)

    def test_off_values(self):
        for v in ("", "0", "false", "no"):
            os.environ["ARERU_BUY_V2"] = v
            self.assertFalse(rg.enabled(), v)


class TestSignals(EnvGuard):
    def test_small_field_is_none(self):
        self.assertIsNone(rg.signals([50, 30, 20]))

    def test_zero_probs_is_none(self):
        self.assertIsNone(rg.signals([0, 0, 0, 0, 0]))

    def test_shape_signals(self):
        s = rg.signals([40, 20, 15, 10, 8, 7])
        self.assertAlmostEqual(s["1位2位差"], 0.2, places=6)
        self.assertAlmostEqual(s["上位3集中度"], 0.75, places=6)
        self.assertAlmostEqual(s["AI1位信頼度"], 0.4, places=6)

    def test_market_aligned_probs_give_zero_deviation(self):
        # 勝率がオッズの単調変換なら、人気との乖離はゼロになる
        odds = [2.0, 4.0, 6.0, 10.0, 20.0, 40.0]
        probs = [1.0 / o for o in odds]
        s = rg.signals(probs, odds)
        self.assertAlmostEqual(s["人気乖離"], 0.0, places=9)

    def test_deviation_is_positive_when_ai_disagrees(self):
        odds = [2.0, 4.0, 6.0, 10.0, 20.0, 40.0]
        probs = [10, 10, 10, 10, 10, 50]  # 最低人気を最上位に評価
        s = rg.signals(probs, odds)
        self.assertGreater(s["人気乖離"], 0.5)

    def test_missing_odds_leaves_market_signals_neutral(self):
        s = rg.signals([40, 20, 15, 10, 8, 7], None)
        self.assertEqual(s["人気乖離"], 0.0)
        self.assertEqual(s["穴馬候補"], 0.0)

    def test_longshot_candidate_uses_only_long_odds(self):
        odds = [1.5, 3.0, 12.0, 15.0, 30.0, 50.0]
        probs = [45, 25, 18, 6, 4, 2]
        s = rg.signals(probs, odds)
        self.assertAlmostEqual(s["穴馬候補"], 0.18, places=6)


class TestSimStability(EnvGuard):
    def test_dominant_favourite_is_stable(self):
        self.assertGreater(rg.sim_stability([0.9, 0.04, 0.03, 0.03]), 0.9)

    def test_flat_race_is_unstable(self):
        self.assertLess(rg.sim_stability([0.2] * 5), 0.5)

    def test_deterministic_for_same_seed(self):
        p = [0.4, 0.25, 0.15, 0.1, 0.1]
        self.assertEqual(rg.sim_stability(p, seed=7), rg.sim_stability(p, seed=7))


class TestGrade(EnvGuard):
    def test_returns_known_class(self):
        g = rg.grade([40, 20, 15, 10, 8, 7], [2.2, 4.5, 6.0, 9.0, 12.0, 14.0])
        self.assertIsNotNone(g)
        self.assertIn(g["レース区分"], ["買うべき", "少額なら買える", "見送る"])
        self.assertTrue(0.0 <= g["予想しやすさ"] <= 100.0)

    def test_clear_favourite_is_easier_than_muddle(self):
        odds = [1.6, 4.0, 7.0, 11.0, 16.0, 22.0, 30.0, 45.0]
        clear = rg.grade([55, 15, 10, 7, 5, 4, 2, 2], odds)
        muddle = rg.grade([16, 15, 14, 13, 12, 11, 10, 9], odds)
        self.assertGreater(clear["予想しやすさ"], muddle["予想しやすさ"])

    def test_small_field_returns_none(self):
        self.assertIsNone(rg.grade([60, 30, 10]))

    def test_missing_model_returns_none(self):
        os.environ["ARERU_GRADE_MODEL"] = "/nonexistent/race_grade_model.json"
        rg.reset_cache()
        try:
            self.assertIsNone(rg.grade([40, 20, 15, 10, 8, 7]))
        finally:
            os.environ.pop("ARERU_GRADE_MODEL", None)
            rg.reset_cache()


class TestBuyDecision(EnvGuard):
    def test_class_overrides_expected_value(self):
        os.environ["ARERU_BUY_V2"] = "1"
        # 期待値が低くても、読みやすいレースなら買い
        d = decide_buy_skip(70.0, 40.0, 30.0, True, race_class="買うべき")
        self.assertEqual(d["投資判定"], "買い")
        # 期待値が高くても、混戦・荒れそうなら見送り
        d = decide_buy_skip(140.0, 90.0, 90.0, True, race_class="見送る")
        self.assertEqual(d["投資判定"], "見送り")

    def test_small_class_is_its_own_decision(self):
        os.environ["ARERU_BUY_V2"] = "1"
        d = decide_buy_skip(100.0, 60.0, 60.0, True, race_class="少額なら買える")
        self.assertEqual(d["投資判定"], "少額")

    def test_without_class_falls_back_to_legacy_gate(self):
        os.environ.pop("ARERU_BUY_V2", None)
        d = decide_buy_skip(140.0, 90.0, 90.0, True, race_conf=90.0)
        self.assertEqual(d["投資判定"], "買い")
        d = decide_buy_skip(70.0, 40.0, 30.0, True, race_conf=40.0)
        self.assertEqual(d["投資判定"], "見送り")

    def test_no_odds_is_pending(self):
        d = decide_buy_skip(None, 50.0, 50.0, False)
        self.assertEqual(d["投資判定"], "判定待ち")


class TestModelFile(EnvGuard):
    def test_model_has_required_keys(self):
        m = rg.load_model()
        self.assertIsNotNone(m)
        for k in ("特徴量", "平均", "標準偏差", "予想しやすさ係数",
                  "荒れやすさ係数", "買うべき下限", "見送り下限", "荒れ上限"):
            self.assertIn(k, m)
        self.assertEqual(len(m["予想しやすさ係数"]), len(m["特徴量"]) + 1)
        self.assertEqual(len(m["荒れやすさ係数"]), len(m["特徴量"]) + 1)

    def test_constant_feature_does_not_blow_up(self):
        # 人気乖離は市場のみのモデルでは常に0。標準偏差で割って発散しないこと
        g = rg.grade([40, 20, 15, 10, 8, 7], [2.5, 5.0, 6.7, 10.0, 12.5, 14.3])
        self.assertTrue(np.isfinite(g["予想しやすさ"]))
        self.assertTrue(np.isfinite(g["荒れやすさ"]))


if __name__ == "__main__":
    unittest.main()
