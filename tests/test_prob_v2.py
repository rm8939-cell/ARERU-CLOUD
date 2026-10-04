"""確率v2（AI独自確率と市場確率の分離）の不変条件。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import areru_engine  # noqa: E402
import ev_analysis  # noqa: E402


class PlackettLuceTest(unittest.TestCase):
    def test_place_probs_sum_to_theoretical_totals(self):
        for win in ([50, 25, 15, 10], [40, 20, 15, 10, 8, 7], [20] * 5):
            p = np.array(win, dtype=float)
            p = p / p.sum() * 100
            top2, top3 = areru_engine._place_probs_from_win(p)
            self.assertAlmostEqual(float(top2.sum()), 200.0, places=4)
            self.assertAlmostEqual(float(top3.sum()), 300.0, places=4)

    def test_place_probs_are_monotonic_in_win_probability(self):
        p = np.array([45.0, 25.0, 18.0, 12.0])
        top2, top3 = areru_engine._place_probs_from_win(p)
        self.assertTrue((top3 >= top2 - 1e-9).all())
        self.assertTrue((top2 >= p - 1e-9).all())
        # 勝率が高い馬ほど複勝確率も高い
        self.assertTrue((np.diff(top3) < 0).all())


class MarketProbabilityTest(unittest.TestCase):
    def _frame(self, odds):
        return pd.DataFrame({'単勝オッズ': odds})

    def test_strict_mode_normalises_away_the_takeout(self):
        odds = [2.0, 4.0, 6.0, 12.0, 20.0]
        pct = areru_engine._market_win_pct(self._frame(odds), len(odds), strict=True)
        self.assertAlmostEqual(float(pct.sum()), 100.0, places=6)
        # 正規化前は合計が 100% を超える（これが控除率）
        self.assertGreater(sum(1.0 / o for o in odds), 1.0)
        # 期待値の基準: 市場確率 × オッズ は全頭で同じ値になり、必ず 100% 未満
        ev = np.array(pct) / 100.0 * np.array(odds) * 100
        self.assertTrue(np.allclose(ev, ev[0]))
        self.assertLess(float(ev[0]), 100.0)

    def test_strict_mode_rejects_placeholder_odds(self):
        # 地方の未取得マーカー 1.0 倍が混ざったら市場確率を作らない
        self.assertIsNone(
            areru_engine._market_win_pct(self._frame([1.0, 4.0, 6.0, 12.0]), 4, strict=True))
        self.assertIsNone(
            areru_engine._market_win_pct(self._frame([2.0, np.nan, 6.0, 12.0]), 4, strict=True))

    def test_legacy_mode_keeps_previous_lenient_behaviour(self):
        odds = [2.0, np.nan, 6.0, 12.0, 20.0]
        pct = areru_engine._market_win_pct(self._frame(odds), len(odds),
                                           win_fallback=np.array([30, 10, 25, 20, 15.0]),
                                           strict=False)
        self.assertIsNotNone(pct)
        self.assertAlmostEqual(float(pct.sum()), 100.0, places=6)


class BlendTest(unittest.TestCase):
    def test_zero_measured_weight_returns_the_market_untouched(self):
        ai = np.array([60.0, 20.0, 15.0, 5.0])
        mkt = np.array([30.0, 30.0, 25.0, 15.0])
        out = areru_engine._blend_ai_market(ai, mkt, 0.0)
        self.assertTrue(np.allclose(out, mkt))

    def test_positive_weight_moves_towards_the_ai_opinion(self):
        ai = np.array([60.0, 20.0, 15.0, 5.0])
        mkt = np.array([25.0, 25.0, 25.0, 25.0])
        out = areru_engine._blend_ai_market(ai, mkt, 0.4)
        self.assertAlmostEqual(float(out.sum()), 100.0, places=6)
        self.assertGreater(out[0], mkt[0])
        self.assertLess(out[3], mkt[3])

    def test_default_weight_is_zero_because_no_edge_was_measured(self):
        os.environ.pop('ARERU_AI_WEIGHT', None)
        self.assertEqual(areru_engine.measured_ai_weight(), 0.0)


class ExpectedValueBaselineTest(unittest.TestCase):
    def test_market_prob_makes_break_even_exactly_100(self):
        # 市場確率をそのまま採用した馬は、定義上 期待値 100% ちょうどになる
        out = ev_analysis.score_horse_ev(
            market=5.0, win_pct=20.0, fair=5.0, conf=70.0, repro=70.0, n=4, apt=50.0,
            market_prob=20.0)
        self.assertAlmostEqual(out['期待値生'], 100.0, delta=0.6)

    def test_without_market_prob_the_baseline_is_inflated_by_the_takeout(self):
        # 実データの控除率（Σ1/オッズ の中央値 1.266）を再現したオッズ表を作る
        probs = np.array([0.34, 0.22, 0.16, 0.11, 0.07, 0.05, 0.03, 0.02])
        overround = 1.266
        odds = 1.0 / (probs * overround)
        self.assertAlmostEqual(float((1.0 / odds).sum()), overround, places=6)

        target = 1  # 2番人気の馬で比べる
        legacy = ev_analysis.score_horse_ev(
            market=float(odds[target]), win_pct=float(probs[target] * 100),
            fair=1.0 / probs[target], conf=70.0, repro=70.0, n=4, apt=50.0)
        honest = ev_analysis.score_horse_ev(
            market=float(odds[target]), win_pct=float(probs[target] * 100),
            fair=1.0 / probs[target], conf=70.0, repro=70.0, n=4, apt=50.0,
            market_prob=float(probs[target] * 100))
        # 控除率を抜いた基準では「市場どおりの評価」は 100/1.266 = 79% にしかならない。
        # 旧式はこれを 90% 台に見せるため、BUY の閾値が実質ザルになる。
        self.assertAlmostEqual(honest['期待値生'], 100.0 / overround, delta=1.0)
        self.assertGreater(legacy['期待値生'] - honest['期待値生'], 10.0)


if __name__ == '__main__':
    unittest.main()
