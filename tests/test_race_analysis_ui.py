"""レース分析UI: 既存データ転記のみ。予想・BUY・AI順位は変えない。"""
from __future__ import annotations

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

    def test_template_v29_is_responsive_and_valid(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[1] / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.assertIn('data-ui="areu-app-v29"', html)
        self.assertIn('data-ra-ui="v29"', html)
        self.assertIn('ra-metrics', html)
        self.assertIn('全頭AI評価', html)
        self.assertIn('AIレース診断', html)
        self.assertIn('AI予想（◎○▲△）', html)
        self.assertIn('展開予想（全馬ポジション）', html)
        self.assertIn('本命馬の評価詳細', html)
        self.assertIn('今回の注目ポイント', html)
        self.assertIn('全馬の展開予測・適性一覧', html)
        self.assertIn('ラップ適合度 × 展開ポジション', html)
        self.assertNotIn('section-title gold">推奨馬券', html)
        self.assertNotIn('label">軍資金', html)
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

    def test_badges_use_jra_frame_colors_from_waku(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[1] / 'templates' / 'index.html').read_text(encoding='utf-8')
        for cls, color in (
            ('.wk-1', '#fff'), ('.wk-2', '#111'), ('.wk-3', '#e53935'), ('.wk-4', '#1565c0'),
            ('.wk-5', '#f4c20d'), ('.wk-6', '#2e7d32'), ('.wk-7', '#ef6c00'), ('.wk-8', '#ec407a'),
        ):
            self.assertIn(f'{cls}{{background:{color}', html)
        # 展開ゾーン・散布図・適合ランキングなど複数箇所で枠色を使う
        self.assertGreaterEqual(html.count("wk-{{_wk|int}}"), 2)

    def test_lap_map_carries_waku_for_badge_color(self):
        race = _race('中山8R', pace='ハイ', horses=[
            _horse('逃げ馬', 1, '本命', '逃げ残り', 枠番='1', AREru指数=99),
            _horse('差し馬', 10, '対抗', '差し向き', 枠番='6', AREru指数=80),
        ])
        _stamp_ai_field_ranks(race)
        _stamp_race_analysis_display(race, {})
        waku = {pt['馬名']: pt['枠番'] for pt in race['ラップマップ']}
        self.assertEqual(waku['逃げ馬'], '1')
        self.assertEqual(waku['差し馬'], '6')
        ranked = {p['馬名']: p.get('枠番') for p in race['ラップ適合ランキング']}
        self.assertEqual(ranked['差し馬'], '6')

    def test_zones_cover_all_runners(self):
        race = _race('新潟1R', horses=[
            _horse('A', 1, '本命', '逃げ残り', AREru指数=90),
            _horse('B', 2, '対抗', '先行向き', AREru指数=80),
            _horse('C', 3, '穴馬', '差し向き', AREru指数=70),
            _horse('D', 4, '', '追込向き', AREru指数=60),
            _horse('E', 5, '', '', AREru指数=50),
        ])
        _stamp_ai_field_ranks(race)
        _stamp_race_analysis_display(race, {})
        zones = race['展開ゾーン']
        total = sum(len(zones[z]) for z in ('逃げ', '先行', '中団', '後方', '不明'))
        self.assertEqual(total, 5)
        self.assertEqual(len(zones['逃げ']), 1)
        self.assertEqual(len(zones['先行']), 1)
        self.assertEqual(len(zones['中団']), 1)
        self.assertEqual(len(zones['後方']), 1)
        self.assertEqual(len(zones['不明']), 1)
        self.assertIsNotNone(race.get('AIレース診断'))
        self.assertIsNotNone(race.get('本命評価'))
        for p in race['AI一覧']:
            self.assertIn('条件変更', p)
            self.assertIn('適性詳細', p)
            # ブリンカーはデータ列が無いので必ずデータ不足
            labels = {c.get('ラベル') for c in p['条件変更']}
            self.assertTrue(any('ブリンカー' in str(x) or '馬具' in str(x) for x in labels))

    def test_condition_change_does_not_auto_boost(self):
        from race_display import detect_condition_changes
        horse = _horse('検査馬', 7, style_txt='先行向き')
        race = {
            '日付': '2026-09-27',
            '開催地': '中山',
            '距離': '芝1600',
            '芝ダ': '芝',
        }
        changes = detect_condition_changes(horse, race, {})
        for ch in changes:
            # 効果はスコアを勝手にいじらない（単純プラス禁止）
            effect = str(ch.get('効果') or '')
            self.assertNotRegex(effect, r'\+\d+')
            self.assertNotIn('自動加点', effect)


class TestRaceDisplayHelpers(unittest.TestCase):
    def test_parse_surface_dist(self):
        from race_display import parse_surface_dist
        self.assertEqual(parse_surface_dist('ダ1400'), ('ダート', 1400))
        self.assertEqual(parse_surface_dist('芝 2000m'), ('芝', 2000))
        self.assertEqual(parse_surface_dist(''), (None, None))

    def test_aptitude_marks_ai_usage(self):
        from race_display import aptitude_rows
        p = _horse('適性馬', 1, style_txt='先行向き', ラップ適合度=72, 展開適性='先行向き', コース適性='○')
        rows = {r['項目']: r for r in aptitude_rows(p)}
        self.assertTrue(rows['ラップ適性']['AI評価に使用'])
        self.assertTrue(rows['展開適性']['AI評価に使用'])
        self.assertFalse(rows['前走内容']['AI評価に使用'])
        self.assertEqual(rows['前走内容']['表示'], 'データ不足')


if __name__ == '__main__':
    unittest.main()
