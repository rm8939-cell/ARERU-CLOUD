"""レース分析UI: 既存データ転記のみ。予想・BUY・AI順位は変えない。"""
from __future__ import annotations

import copy
import os
import unittest

os.environ.setdefault('ARERU_SKIP_BOOT', '1')
os.environ.setdefault('ARERU_LEGACY_SCORE', '1')
os.environ.setdefault('ARERU_ENABLE_GENERATION', '0')

from web_app import (
    _grade_from_race_name,
    _stamp_ai_field_ranks,
    _stamp_race_analysis_display,
)


def _horse(name, ban, role='', style_txt='先行向き', **extra):
    card = {
        '展開相性': style_txt,
        'ラップ適性': '平均ペース適性',
        '勝率': extra.pop('勝率', 12.0),
        '複勝率': extra.pop('複勝率', 30.0),
        '期待値': extra.pop('期待値', 90),
        'AI評価': extra.pop('AI評価', 70),
        'AI信頼度スコア': extra.pop('信頼度', 60),
        '判断根拠': extra.pop('判断根拠', []),
    }
    p = {
        '馬名': name,
        '馬番': ban,
        '馬番表示': ban,
        '役割': role,
        'BUY表示': extra.pop('BUY表示', False),
        'カード': card,
        'AREru指数': extra.pop('AREru指数', 70 - int(ban or 1)),
    }
    p.update(extra)
    return p


def _race(name='', pace='ミドル', horses=None):
    pace_d = {
        '想定ペース': pace,
        '逃げ有利度': 52,
        '先行有利度': 58,
        '差し有利度': 55,
        '追込有利度': 48,
        '有利枠': '内枠',
        '荒れ指数': 40,
        'AI総評': '平均的な流れ。脚質の偏りは小さい。',
        '逃げ馬数': 2,
        '先行馬数': 4,
        '差し馬数': 5,
        '追込馬数': 3,
    }
    if pace == 'ハイ':
        pace_d.update({'逃げ有利度': 28, '先行有利度': 42, '差し有利度': 72, '追込有利度': 68})
    elif pace == 'スロー':
        pace_d.update({'逃げ有利度': 78, '先行有利度': 70, '差し有利度': 38, '追込有利度': 30})
    horses = horses or [
        _horse('本命馬', 1, '本命', '先行向き', BUY表示=True, AREru指数=90),
        _horse('対抗馬', 2, '対抗', '差し向き', AREru指数=80),
        _horse('穴馬', 3, '穴馬', '追込向き', AREru指数=70),
    ]
    race = {
        'race_id': '202609040911',
        'レース名': name,
        '投資判定': '買い',
        '期待値': 120,
        '展開予想データ': pace_d,
        'AI一覧': horses,
        '予想馬': list(horses),
    }
    return race


class TestRaceAnalysisDisplay(unittest.TestCase):
    def test_does_not_change_ranks_or_buy(self):
        race = _race('3歳以上2勝クラス')
        _stamp_ai_field_ranks(race)
        before = [
            (p.get('AI順位'), p.get('BUY表示'), p.get('表示印名'), p.get('馬名'))
            for p in race['AI一覧']
        ]
        ev = race['期待値']
        judge = race['投資判定']
        _stamp_race_analysis_display(race, {})
        after = [
            (p.get('AI順位'), p.get('BUY表示'), p.get('表示印名'), p.get('馬名'))
            for p in race['AI一覧']
        ]
        self.assertEqual(before, after)
        self.assertEqual(race['期待値'], ev)
        self.assertEqual(race['投資判定'], judge)

    def test_same_board_keys_for_maiden_and_g1(self):
        maiden = _race('3歳未勝利')
        g1 = _race('天皇賞(秋)(GI)')
        for r in (maiden, g1):
            _stamp_ai_field_ranks(r)
            _stamp_race_analysis_display(r, {})
        self.assertEqual(set(maiden['レース分析']), set(g1['レース分析']))
        self.assertEqual(g1['重賞グレード'], 'GⅠ')
        self.assertTrue(g1['重賞レース'])
        self.assertFalse(maiden['重賞レース'])
        self.assertIsNone(maiden['レース分析']['勝ち時計予想'])
        self.assertIsNone(g1['レース分析']['勝ち時計予想'])
        self.assertIsNone(maiden['レース分析']['馬場傾向'])

    def test_lap_rank_uses_fit_not_ai_order(self):
        race = _race('阪神11R', pace='ハイ', horses=[
            _horse('逃げ馬', 1, '本命', '逃げ残り', BUY表示=True, AREru指数=99),
            _horse('差し馬', 8, '対抗', '差し向き', AREru指数=80),
            _horse('追込馬', 12, '穴馬', '追込向き', AREru指数=70),
        ])
        _stamp_ai_field_ranks(race)
        _stamp_race_analysis_display(race, {})
        ranked = race['ラップ適合ランキング']
        self.assertTrue(ranked)
        fits = [p['ラップ適合度'] for p in ranked]
        self.assertEqual(fits, sorted(fits, reverse=True))
        # ハイペースでは差し/追込の適合度が逃げより高い（既存 lap_aptitude）
        by_name = {p['馬名']: p['ラップ適合度'] for p in ranked}
        self.assertGreater(by_name['追込馬'], by_name['逃げ馬'])
        self.assertTrue(race['ラップマップ'])
        xs = {pt['馬名']: pt['x'] for pt in race['ラップマップ']}
        self.assertGreater(xs['逃げ馬'], xs['追込馬'])  # 右が前方

    def test_missing_fields_stay_none(self):
        race = {
            'race_id': 'x',
            'レース名': '',
            '展開予想データ': {},
            'AI一覧': [_horse('無名', 4, style_txt='')],
        }
        _stamp_ai_field_ranks(race)
        _stamp_race_analysis_display(race, {})
        anal = race['レース分析']
        self.assertIsNone(anal['想定ペース'])
        self.assertIsNone(anal['勝ち時計予想'])
        self.assertEqual(anal['注目ポイント'], [])
        self.assertEqual(race['ラップマップ'], [])

    def test_board_is_shared_and_does_not_invent_clock(self):
        maiden = _race('3歳以上2勝クラス')
        g1 = _race('スプリンターズS(GI)')
        for r in (maiden, g1):
            _stamp_ai_field_ranks(r)
            _stamp_race_analysis_display(r, {})
            self.assertIn('レース分析', r)
            self.assertIn('ラップマップ', r)
            self.assertIn('ラップ適合ランキング', r)
            self.assertIsNone(r['レース分析']['勝ち時計予想'])
            self.assertIsNone(r['レース分析']['馬場傾向'])
        self.assertTrue(g1['重賞レース'])
        self.assertEqual(g1['重賞グレード'], 'GⅠ')

    def test_grade_parser(self):
        self.assertEqual(_grade_from_race_name('桜花賞(GI)'), 'GⅠ')
        self.assertEqual(_grade_from_race_name('毎日王冠(GII)'), 'GⅡ')
        self.assertEqual(_grade_from_race_name('しらさぎS(GIII)'), 'GⅢ')
        self.assertEqual(_grade_from_race_name('3歳未勝利'), '')

    def test_template_v28_is_responsive_and_valid(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[1] / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('data-ui="areu-app-v28"', html)
        self.assertIn('data-ra-ui="v28"', html)
        self.assertIn('ra-metrics', html)
        self.assertIn('全馬 詳細データ', html)
        self.assertIn('ラップ適合度 × 展開ポジション', html)
        self.assertNotIn('var(--i', html)
        self.assertNotIn('% 3', html)
        self.assertIn('@media (max-width:899px)', html)
        self.assertIn('@media (min-width:900px)', html)
        self.assertIn('grid-template-columns:1fr 1fr', html)
        self.assertIn('color:#087443', html)
        self.assertIn('background:#159447', html)
        self.assertIn('background:#E8EEF0', html)
        self.assertIn('background:#CBD8E3', html)
        self.assertIn('color:#526174', html)
        self.assertIn('border:1px solid #DDE5E2', html)
        self.assertIn('border-radius:18px', html)


if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
