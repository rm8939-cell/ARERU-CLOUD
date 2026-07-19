"""期待値計算・AI自己評価・日別成績ダッシュボード。"""
from __future__ import annotations

import math
import re

import pandas as pd

# 表示側でもエンジンと同じ上限・下限を適用（既存CSVの100%/1.0倍を補正）
SIM_WIN_MAX_PCT = 98.0
AI_FAIR_ODDS_MIN = 1.1
# 表示期待値のハード上限（100%＝市場オッズと同値の勝率想定）
EV_DISPLAY_MAX = 124
EV_DISPLAY_MIN = 78


def parse_odds_value(v):
    """オッズ文字列/数値を float へ。欠損は None。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if pd.isna(v):
            return None
        x = float(v)
        return x if x > 0 else None
    s = str(v).strip().replace('倍', '').replace('%', '')
    if not s or s.lower() in (
        'nan', 'none', 'なし', '—', '-', 'オッズ接続後に算出', '券種別オッズ待ち'
    ):
        return None
    try:
        x = float(s)
        return x if x > 0 else None
    except Exception:
        return None


def _clamp(x, a=0.0, b=100.0) -> float:
    return float(max(a, min(b, x)))


def ev_tone(ev_pct):
    """期待値の色区分: 115+緑 / 105-114黄 / 95-104グレー / 未満赤"""
    try:
        v = float(ev_pct)
    except (TypeError, ValueError):
        return 'ev-none'
    if v >= 115:
        return 'ev-buy'
    if v >= 105:
        return 'ev-consider'
    if v >= 95:
        return 'ev-neutral'
    return 'ev-skip'


def ev_label(tone):
    return {
        'ev-buy': '買い',
        'ev-consider': '検討',
        'ev-neutral': '普通',
        'ev-skip': '見送り',
        'ev-none': '—',
    }.get(tone, '—')


def ev_plain_label(display_ev: float | None) -> str:
    """一覧向けの短い言葉。"""
    if display_ev is None:
        return '—'
    v = float(display_ev)
    if v >= 120:
        return '割安'
    if v >= 112:
        return 'やや割安'
    if v >= 100:
        return '普通'
    if v >= 92:
        return 'やや割高'
    return '割高'


def stars_for_score(score: int) -> str:
    filled = max(1, min(5, (int(score) + 19) // 20))
    return '★' * filled + '☆' * (5 - filled)


def _count_past_races(record: dict) -> int:
    for key in ('本命データ件数', 'データ件数'):
        try:
            v = int(float(record.get(key)))
            if v >= 0:
                return min(5, v)
        except (TypeError, ValueError):
            pass
    n = 0
    for i in range(1, 6):
        v = record.get(f'着順{i}')
        s = str(v or '').strip()
        if not s or s.lower() in ('nan', 'none', 'なし'):
            continue
        if re.search(r'\d', s):
            n += 1
    if n:
        return n
    # predictions CSV に着順が無い場合は理由から推定
    reasons = str(record.get('本命理由') or '')
    if 'サンプル少' in reasons:
        return 1
    if '履歴少' in reasons:
        return 2
    return 3


def _main_pick_card(record: dict) -> dict:
    for c in record.get('ピックカード一覧') or []:
        if isinstance(c, dict) and c.get('役割') == '本命':
            return c
    return {}


def _aptitude_score(record: dict, pick: dict) -> tuple[float, dict]:
    """ラップ・上がり・展開・距離・コース・クラス・地方補正の適合度 0-100。"""
    reasons = str(record.get('本命理由') or '') + ' / ' + str(pick.get('プラス材料') or '')
    minus = str(pick.get('不安材料') or '') + ' / ' + reasons
    score = 50.0
    detail = {}

    # 距離・コース
    dist_ok = pick.get('距離適性') == '○' or '距離適性' in reasons
    course_ok = pick.get('コース適性') == '○' or 'コース適性' in reasons
    score += 8 if dist_ok else (-4 if '距離' in minus else 0)
    score += 8 if course_ok else 0
    detail['距離適性'] = 70 if dist_ok else 45
    detail['コース適性'] = 70 if course_ok else 45

    # ラップ・展開
    lap = str(pick.get('ラップ適性') or pick.get('展開相性') or '')
    if '好相性' in lap or '適性' in lap and '不利' not in lap:
        score += 8
        detail['ラップ適性'] = 72
    elif '不利' in lap:
        score -= 8
        detail['ラップ適性'] = 35
    else:
        detail['ラップ適性'] = 50
    if '好相性' in str(pick.get('展開相性') or ''):
        score += 6
        detail['展開一致率'] = 75
    elif '不利' in str(pick.get('展開相性') or ''):
        score -= 6
        detail['展開一致率'] = 35
    else:
        detail['展開一致率'] = 50

    # 上がり
    try:
        last3 = float(pick.get('上がり評価')) if pick.get('上がり評価') is not None else None
    except (TypeError, ValueError):
        last3 = None
    if last3 is not None:
        if last3 >= 62:
            score += 7
        elif last3 <= 40:
            score -= 6
        detail['上がり性能'] = _clamp(last3)
    else:
        detail['上がり性能'] = 50

    # クラス・地方
    if 'クラス条件好転' in reasons:
        score += 5
        detail['クラス補正'] = 65
    elif 'クラス' in minus:
        score -= 3
        detail['クラス補正'] = 40
    else:
        detail['クラス補正'] = 50
    if '地方実績を中央換算' in reasons or '地方実績中心' in reasons:
        score -= 12
        detail['地方→中央補正'] = 30
    else:
        detail['地方→中央補正'] = 55

    if 'サンプル少' in reasons or '履歴少' in reasons:
        score -= 10

    return _clamp(score), detail


def calc_simulation_reproducibility(record: dict, pick: dict | None = None) -> float:
    """シミュレーション再現率 0-100。安定・サンプル・勝率の現実性から推定。"""
    pick = pick or _main_pick_card(record)
    n = _count_past_races(record)
    reasons = str(record.get('本命理由') or '')
    win = parse_odds_value(record.get('シミュレーション勝率'))
    place = parse_odds_value(record.get('シミュレーション3着内率'))
    score = 38.0
    score += min(28.0, n * 5.5)
    if '安定感' in reasons:
        score += 12
    if 'サンプル少' in reasons or '履歴少' in reasons:
        score -= 18
    if win is not None:
        if 8 <= win <= 38:
            score += 12
        elif win > 50:
            score -= 18
        elif win < 5:
            score -= 10
    if place is not None and win is not None and place >= win * 1.8:
        score += 6  # 複勝圏の厚み
    if '地方実績' in reasons:
        score -= 8
    apt, _ = _aptitude_score(record, pick)
    score += (apt - 50) * 0.25
    return round(_clamp(score, 8, 92), 1)


def calc_ai_confidence(record: dict, pick: dict | None = None, repro: float | None = None) -> float:
    """AI信頼度 0-100（期待値とは独立）。"""
    pick = pick or _main_pick_card(record)
    n = _count_past_races(record)
    repro = calc_simulation_reproducibility(record, pick) if repro is None else float(repro)
    apt, _ = _aptitude_score(record, pick)
    win = parse_odds_value(record.get('シミュレーション勝率'))
    fair = parse_odds_value(record.get('AI適正オッズ'))
    market = parse_odds_value(record.get('本命オッズ') or record.get('現在オッズ'))
    reasons = str(record.get('本命理由') or '')

    score = 0.0
    score += min(22.0, n * 4.4)                         # データ件数
    score += repro * 0.28                                 # 再現率
    score += apt * 0.22                                   # 適性群
    if win is not None and 10 <= win <= 42:
        score += 12
    elif win is not None and win > 55:
        score -= 15
    # 市場と乖離しすぎる適正は信頼を落とす
    if market and fair and fair > 0:
        ratio = market / fair
        if ratio > 8:
            score -= 20
        elif ratio > 4:
            score -= 12
        elif 0.7 <= ratio <= 2.2:
            score += 8
    if '地方実績を中央換算' in reasons or '地方実績中心' in reasons:
        score -= 10
    if 'サンプル少' in reasons:
        score -= 8
    rank = str(record.get('勝負ランク') or '').upper()
    if rank == 'S':
        score += 4
    elif rank == 'C':
        score -= 4
    return round(_clamp(score, 5, 95), 1)


def _edge_take_rate(conf: float, repro: float, n: int, apt: float, market: float, reasons: str) -> float:
    """AIが主張するエッジのうち、期待値に取り込む割合（0〜0.55）。"""
    take = (
        (conf / 100.0) ** 1.15
        * (max(8.0, repro) / 100.0) ** 0.65
        * min(1.0, n / 3.5)
        * (0.50 + 0.50 * apt / 100.0)
    )
    # 大穴は推定誤差が大きいのでさらに抑制
    if market >= 25:
        take *= 0.42
    elif market >= 15:
        take *= 0.58
    elif market >= 10:
        take *= 0.72
    elif market >= 7:
        take *= 0.85
    if n <= 1:
        take *= 0.55
    elif n == 2:
        take *= 0.75
    if '地方実績を中央換算' in reasons or '地方実績中心' in reasons:
        take *= 0.70
    if 'サンプル少' in reasons or '履歴少' in reasons:
        take *= 0.72
    return _clamp(take, 0.04, 0.62)


def _claimable_ai_prob(ai_p: float, implied: float, conf: float, repro: float, n: int) -> float:
    """市場から大きく離れたAI勝率は、信頼度に応じてだけ認める。"""
    abs_cap = 0.08 + 0.34 * (conf / 100.0)  # 8%〜42%
    rel_cap = implied * (
        1.08 + 0.95 * (conf / 100.0) * math.sqrt(max(0.05, repro / 100.0))
    )
    max_pp = 0.015 + 0.14 * (conf / 100.0) * (repro / 100.0) * min(1.0, n / 4.0)
    pp_cap = implied + max_pp
    return _clamp(min(ai_p, abs_cap, rel_cap, pp_cap), 0.002, 0.55)


def _soft_display_ev(raw_ev: float) -> int:
    """極端な生EVを tanh で圧縮し、ユーザーが信じやすい帯に落とす。"""
    edge = float(raw_ev) - 100.0
    # |edge|が大きいほど伸びにくく、概ね 78〜124%
    compressed = 100.0 + 26.0 * math.tanh(edge / 30.0)
    return int(round(_clamp(compressed, EV_DISPLAY_MIN, EV_DISPLAY_MAX)))


def score_horse_ev(
    market: float | None,
    win_pct: float | None,
    fair: float | None,
    conf: float,
    repro: float,
    n: int,
    apt: float,
    reasons: str = '',
) -> dict:
    """単頭の信頼度補正期待値。100%＝現在オッズと同値の勝率想定。"""
    empty = {
        '期待値': None, '期待値生': None, '期待値表示': '—', '期待値エッジ': None,
        '期待値トーン': 'ev-none', '期待値ラベル': '—', '期待値コメント': '—',
        '期待値あり': False, '補正勝率': None, 'ブレンド係数': None,
        'AI適正オッズ補正': None,
    }
    market = parse_odds_value(market)
    if market is None or market <= 0:
        return empty

    if win_pct is not None and win_pct > 0:
        ai_p = min(0.85, float(win_pct) / 100.0)
    elif fair is not None and fair > 0:
        ai_p = min(0.85, 1.0 / float(fair))
    else:
        return empty

    implied = 1.0 / market  # 100% EV の基準（控除は別途信頼度で織り込み）
    ai_eff = _claimable_ai_prob(ai_p, implied, conf, repro, n)
    take = _edge_take_rate(conf, repro, n, apt, market, reasons)
    edge_p = ai_eff - implied
    adj_p = implied + edge_p * take
    adj_p = _clamp(adj_p, 0.002, 0.55)

    raw_ev = market * adj_p * 100.0
    display_ev = _soft_display_ev(raw_ev)
    fair_adj = max(AI_FAIR_ODDS_MIN, 1.0 / max(adj_p, 1e-6))

    return {
        '期待値': display_ev,
        '期待値生': int(round(raw_ev)),
        '期待値表示': f'{display_ev}%',
        '期待値エッジ': display_ev - 100,
        '期待値トーン': ev_tone(display_ev),
        '期待値ラベル': ev_label(ev_tone(display_ev)),
        '期待値コメント': ev_plain_label(display_ev),
        '期待値あり': True,
        '補正勝率': round(adj_p * 100, 1),
        'ブレンド係数': round(take, 3),
        'AI適正オッズ補正': round(fair_adj, 1),
        'AI勝率採用': round(ai_eff * 100, 1),
        '市場暗示勝率': round(implied * 100, 1),
    }


def calc_confidence_adjusted_ev(record: dict) -> dict:
    """信頼度・再現率・適性でAI勝率を縮約した期待値。

    考慮: AI勝率 / 現在オッズ / AI適正オッズ / 再現率 / データ件数 /
          ラップ・上がり・展開・距離・コース・クラス・地方→中央
    """
    market = parse_odds_value(record.get('本命オッズ') or record.get('現在オッズ'))
    fair = parse_odds_value(record.get('AI適正オッズ'))
    win_pct = parse_odds_value(record.get('シミュレーション勝率'))
    pick = _main_pick_card(record)
    n = _count_past_races(record)
    repro = calc_simulation_reproducibility(record, pick)
    conf = calc_ai_confidence(record, pick, repro=repro)
    apt, apt_detail = _aptitude_score(record, pick)
    reasons = str(record.get('本命理由') or '') + ' / ' + str(pick.get('不安材料') or '')

    base = {
        'AI信頼度スコア': conf,
        'シミュレーション再現率': repro,
        '適性スコア': round(apt, 1),
        '適性内訳': apt_detail,
        'データ件数': n,
    }
    scored = score_horse_ev(market, win_pct, fair, conf, repro, n, apt, reasons)
    scored.update(base)
    return scored


def decide_buy_skip(ev: float | None, confidence: float, repro: float, has_odds: bool) -> dict:
    """期待回収率で買い／見送りを決定（表示ランクと一致）。

    100%以上 → 買い / 100%未満 → 見送り
    """
    if not has_odds or ev is None:
        return {
            '一覧判定': '判定待ち', '一覧判定トーン': 'wait',
            '投資判定': '判定待ち', '投資判定アイコン': '⚪', '投資判定トーン': 'wait',
            '投資判定表示': '判定待ち',
        }
    e = float(ev)
    if e >= 100:
        return {
            '一覧判定': '買い', '一覧判定トーン': 'buy',
            '投資判定': '買い', '投資判定アイコン': '🟢', '投資判定トーン': 'buy',
            '投資判定表示': '買い',
        }
    return {
        '一覧判定': '見送り', '一覧判定トーン': 'skip',
        '投資判定': '見送り', '投資判定アイコン': '🔴', '投資判定トーン': 'skip',
        '投資判定表示': '見送り',
    }


def rank_from_expected_value(ev) -> str:
    """期待回収率から S〜D / 見送り を決定。

    S：120%以上 / A：115〜119 / B：110〜114 / C：105〜109 / D：100〜104
    見送り：100%未満（空文字）
    """
    try:
        if ev is None or ev == '' or str(ev).strip() in ('—', '-', 'なし', 'nan', 'None'):
            return ''
        e = float(str(ev).replace('%', '').strip())
    except (TypeError, ValueError):
        return ''
    if e >= 120:
        return 'S'
    if e >= 115:
        return 'A'
    if e >= 110:
        return 'B'
    if e >= 105:
        return 'C'
    if e >= 100:
        return 'D'
    return ''


def apply_ev_rank_and_labels(record: dict) -> dict:
    """期待回収率にランク・買い判定・表示ラベルを揃える。"""
    from areru_engine import RANK_LABELS, RANK_CLASSES

    ev = record.get('期待値')
    if ev is None:
        raw = record.get('レース期待回収率') or record.get('期待値表示') or ''
        try:
            ev = float(str(raw).replace('%', '').strip()) if str(raw).strip() not in ('', '—', 'なし') else None
        except (TypeError, ValueError):
            ev = None
    rk = rank_from_expected_value(ev)
    record['勝負ランク'] = rk
    if rk:
        record['勝負ランク表示'] = f'{rk}'
        record['BET判定'] = RANK_LABELS.get(rk, '')
        record['BETクラス'] = RANK_CLASSES.get(rk, '')
    else:
        record['勝負ランク表示'] = '見送り'
        record['BET判定'] = '見送り'
        record['BETクラス'] = 'skip'

    has_odds = bool(record.get('オッズ取得済'))
    if ev is not None:
        record['期待値あり'] = True
        decision = decide_buy_skip(float(ev), float(record.get('AI信頼度スコア') or 50),
                                   float(record.get('シミュレーション再現率') or 50), True)
    else:
        decision = decide_buy_skip(None, 50, 50, has_odds)
    record.update(decision)

    if ev is not None:
        ev_i = int(round(float(ev)))
        record['期待値'] = ev_i
        record['期待値表示'] = f'{ev_i}%'
        record['期待回収率表示'] = f'{ev_i}%'
        record['期待回収率短'] = f'{ev_i}%'
        record['期待値トーン'] = 'ev-buy' if ev_i >= 100 else 'ev-skip'
        record['レース期待回収率'] = ev_i
    else:
        record['期待回収率表示'] = '—'
        record['期待回収率短'] = '—'
        record['期待値トーン'] = 'ev-none'
    return record


def build_ai_buy_reasons(record: dict, limit: int = 3) -> list[str]:
    """詳細用の短い買い理由（最大3）。"""
    reasons: list[str] = []

    def add(msg: str):
        msg = str(msg or '').strip()
        if msg and msg not in reasons and len(reasons) < limit:
            reasons.append(msg)

    ev = record.get('期待値')
    try:
        edge = int(round(float(ev) - 100)) if ev is not None else None
    except (TypeError, ValueError):
        edge = None
    if edge is not None and edge > 0:
        add(f'単勝期待値＋{edge}%')
    elif edge is not None and edge < 0:
        add(f'単勝期待値{edge}%')

    for pick in (record.get('予想馬') or [])[:1]:
        for tip in (pick.get('要点') or [])[:2]:
            add(str(tip))
    if not reasons:
        for c in (record.get('ピックカード一覧') or [])[:1]:
            if not isinstance(c, dict):
                continue
            try:
                idx = int(float(c.get('近走指数順位')))
                if idx == 1:
                    add('近走指数トップ')
                elif idx <= 3:
                    add(f'近走指数{idx}位')
            except (TypeError, ValueError):
                pass
            mo = c.get('単勝オッズ')
            fo = c.get('AI適正オッズ')
            try:
                if mo and fo and float(mo) > float(fo) * 1.15:
                    add('人気との乖離が大きい')
            except (TypeError, ValueError):
                pass
            for p in (c.get('プラス材料一覧') or []):
                add(str(p))
                if len(reasons) >= limit:
                    break
    while len(reasons) < min(2, limit) and record.get('投資判定') == '買い':
        for filler in ('条件面のバランスが良い', '相手関係でも残せる', '再現性のある評価'):
            add(filler)
            if len(reasons) >= min(2, limit):
                break
        break
    return reasons[:limit]


def calc_expected_value(market_odds, fair_odds):
    """後方互換: 単純比。極端値は市場連動キャップで抑制。"""
    m = parse_odds_value(market_odds)
    f = parse_odds_value(fair_odds)
    if m is None or f is None or f <= 0:
        return {
            '期待値': None, '期待値生': None, '期待値表示': '—', '期待値エッジ': None,
            '期待値トーン': 'ev-none', '期待値ラベル': '—', '期待値あり': False,
            '期待値コメント': '—',
        }
    # 単純比でも「市場の1.25倍以上割安」は認めない
    fair_adj = max(float(f), float(m) / 1.25, AI_FAIR_ODDS_MIN)
    raw = (float(m) / fair_adj) * 100.0
    disp = _soft_display_ev(raw)
    return {
        '期待値': disp,
        '期待値生': int(round(raw)),
        '期待値表示': f'{disp}%',
        '期待値エッジ': disp - 100,
        '期待値トーン': ev_tone(disp),
        '期待値ラベル': ev_label(ev_tone(disp)),
        '期待値コメント': ev_plain_label(disp),
        '期待値あり': True,
        'AI適正オッズ補正': round(fair_adj, 1),
    }


def ai_confidence_stars(record: dict) -> str:
    """一覧用AI信頼度 ★。スコアがあればそれを使用。"""
    score = record.get('AI信頼度スコア')
    try:
        if score is not None:
            return stars_for_score(int(float(score)))
    except (TypeError, ValueError):
        pass
    return stars_for_score(50)


def finish_num(fin):
    s = str(fin or '').strip()
    if not s or s in ('結果待ち', '取消', '除外', '中止', '—', '－'):
        return None
    m = re.match(r'(\d+)', s)
    return int(m.group(1)) if m else None


def normalize_fair_odds_fields(record: dict) -> dict:
    """仮想勝率99%以上・適正オッズ1.0倍固定を表示前に補正。"""
    win = parse_odds_value(record.get('シミュレーション勝率'))
    fair = parse_odds_value(record.get('AI適正オッズ'))
    cap = min(SIM_WIN_MAX_PCT, 100.0 / AI_FAIR_ODDS_MIN)
    changed = False
    if win is not None and win > cap:
        win = cap
        changed = True
    if fair is not None and fair < AI_FAIR_ODDS_MIN:
        fair = AI_FAIR_ODDS_MIN
        changed = True
    if win is not None and (fair is None or changed):
        # 勝率と適正オッズを整合（下限1.1倍）
        fair = max(100.0 / win, AI_FAIR_ODDS_MIN) if win > 0 else AI_FAIR_ODDS_MIN
    if fair is not None and win is None:
        win = min(100.0 / fair, cap) if fair > 0 else cap
    if win is not None:
        record['シミュレーション勝率'] = round(win, 1)
    if fair is not None:
        record['AI適正オッズ'] = round(fair, 1)
    return record


def _rescore_pick_cards(record: dict, conf: float, repro: float, n: int, apt: float) -> None:
    """ピックカードの期待値再計算＋判断根拠の充実。"""
    from pick_rationale import enrich_pick_card

    reasons = str(record.get('本命理由') or '')
    for card in record.get('ピックカード一覧') or []:
        if not isinstance(card, dict):
            continue
        # 既存CSVでも【AI信頼度】【理由】【コメント】を埋める
        enrich_pick_card(card, record)
        horse_conf = float(card.get('AI信頼度スコア') or conf)
        scored = score_horse_ev(
            card.get('単勝オッズ'),
            card.get('勝率'),
            card.get('AI適正オッズ'),
            horse_conf,
            repro,
            n,
            apt,
            reasons + ' / ' + str(card.get('不安材料') or ''),
        )
        if scored.get('期待値あり'):
            card['期待値'] = scored['期待値']
            card['期待値表示'] = scored['期待値表示']
            card['期待値生'] = scored.get('期待値生')
            if scored.get('AI適正オッズ補正') is not None:
                card['AI適正オッズ表示'] = scored['AI適正オッズ補正']
    danger = record.get('危険人気カード')
    if isinstance(danger, dict) and danger:
        scored = score_horse_ev(
            danger.get('単勝オッズ'),
            danger.get('勝率'),
            danger.get('AI適正オッズ'),
            max(20.0, conf - 8),
            repro,
            n,
            apt,
            reasons,
        )
        if scored.get('期待値あり'):
            danger['期待値'] = scored['期待値']


def apply_expected_value(record: dict) -> dict:
    """信頼度補正期待値・AI信頼度・再現率・買い判定を一括付与。"""
    normalize_fair_odds_fields(record)
    fair = parse_odds_value(record.get('AI適正オッズ'))
    market = parse_odds_value(record.get('本命オッズ'))
    if fair is not None:
        record['AI適正オッズ'] = round(max(fair, AI_FAIR_ODDS_MIN), 1)
    record['現在オッズ'] = round(market, 1) if market is not None else None
    record['オッズ取得済'] = market is not None

    ev = calc_confidence_adjusted_ev(record)
    record['期待値'] = ev['期待値']
    record['期待値生'] = ev.get('期待値生')
    record['期待値表示'] = ev['期待値表示']
    record['期待値エッジ'] = ev.get('期待値エッジ')
    record['期待値トーン'] = ev['期待値トーン']
    record['期待値ラベル'] = ev['期待値ラベル']
    record['期待値コメント'] = ev.get('期待値コメント') or '—'
    record['期待値あり'] = ev['期待値あり']
    record['AI信頼度スコア'] = ev.get('AI信頼度スコア')
    record['シミュレーション再現率'] = ev.get('シミュレーション再現率')
    record['適性スコア'] = ev.get('適性スコア')
    record['補正勝率'] = ev.get('補正勝率')
    record['データ件数'] = ev.get('データ件数')
    record['ブレンド係数'] = ev.get('ブレンド係数')
    if ev.get('AI適正オッズ補正') is not None:
        record['AI適正オッズ表示'] = ev['AI適正オッズ補正']
    record['AI信頼度'] = stars_for_score(int(float(ev.get('AI信頼度スコア') or 50)))

    conf = float(ev.get('AI信頼度スコア') or 0)
    repro = float(ev.get('シミュレーション再現率') or 0)
    n = int(ev.get('データ件数') or _count_past_races(record))
    apt = float(ev.get('適性スコア') or 50)
    _rescore_pick_cards(record, conf, repro, n, apt)

    decision = decide_buy_skip(
        ev.get('期待値'),
        conf,
        repro,
        bool(record.get('オッズ取得済')),
    )
    # CSVの旧投資判定は上書き（一覧と詳細の矛盾をなくす）
    record.update(decision)
    record['レース期待回収率'] = ev['期待値'] if ev.get('期待値あり') else ''

    # 一覧・詳細共通の簡潔表示パック
    from pick_rationale import build_display_picks
    record['予想馬'] = build_display_picks(record)
    if record['予想馬']:
        record['本命短表示'] = record['予想馬'][0].get('表示行') or record.get('本命表示') or '—'
    else:
        ban = record.get('本命馬番表示') or ''
        name = str(record.get('本命') or '').strip()
        record['本命短表示'] = f'◎{ban} {name}'.strip() if (ban or name) else '—'

    # ランク＝期待回収率（矛盾表示をなくす）
    apply_ev_rank_and_labels(record)
    record['AI買い理由'] = build_ai_buy_reasons(record, limit=3)

    if ev.get('期待値あり') and (ev.get('期待値生') or 0) >= 160:
        import logging
        logging.getLogger('areru').warning(
            'EV_HIGH race_id=%s horse=%s market=%s fair=%s raw=%s display=%s '
            'conf=%s repro=%s win=%s blend=%s',
            record.get('race_id'), record.get('本命'),
            record.get('現在オッズ'), record.get('AI適正オッズ'),
            ev.get('期待値生'), record.get('期待値'),
            conf, repro,
            record.get('シミュレーション勝率'), ev.get('ブレンド係数'),
        )
    return record


def build_ai_self_eval(r, review, clean_horse, rival_odds=None):
    """結果確定レース向け: AI自己評価。"""
    empty = {
        'あり': False,
        '採点': None,
        '星': '',
        '本命評価': '—',
        '対抗評価': '—',
        '印順位': '—',
        '期待値行': [],
        '改善ポイント': [],
        'サマリー': '',
    }
    if not r.get('結果確定') or not review:
        return empty

    hon = next((x for x in review if x.get('印') == '◎'), None)
    tai = next((x for x in review if x.get('印') == '○'), None)
    hon_fin = finish_num(hon.get('着順')) if hon else None
    tai_fin = finish_num(tai.get('着順')) if tai else None

    honmei_txt = (
        f"◎ → {hon['着順']}"
        if hon and hon.get('着順') not in ('', '結果待ち')
        else '◎ → —'
    )
    taikou_txt = (
        f"○ → {tai['着順']}"
        if tai and tai.get('着順') not in ('', '結果待ち')
        else '○ → —'
    )

    marked = []
    for x in review:
        n = finish_num(x.get('着順'))
        if n is not None:
            marked.append((n, x.get('印', ''), x.get('馬名', '')))
    marked.sort(key=lambda t: t[0])
    mark_n = len(review)
    mark_rank = None
    hon_name = clean_horse(hon.get('馬名', '')) if hon else ''
    for i, (_fin, mk, name) in enumerate(marked, start=1):
        if mk == '◎' or clean_horse(name) == hon_name:
            mark_rank = i
            break
    mark_rank_txt = (
        f'{mark_n}頭中{mark_rank}位'
        if mark_rank
        else (f'{mark_n}頭中—' if mark_n else '—')
    )

    ev_rows = []
    main_ev = calc_expected_value(r.get('本命オッズ'), r.get('AI適正オッズ'))
    if main_ev['期待値あり']:
        edge = main_ev['期待値エッジ']
        ev_rows.append({
            '印': '◎',
            '表示': f"{'+' if edge >= 0 else ''}{edge}%",
            '値': edge,
            'トーン': main_ev['期待値トーン'],
        })
    else:
        ev_rows.append({'印': '◎', '表示': '—', '値': None, 'トーン': 'ev-none'})

    o_row = next((x for x in r.get('印一覧', []) if x.get('印') == '○'), None)
    o_market = parse_odds_value(rival_odds) or (
        parse_odds_value(o_row.get('単勝オッズ')) if o_row else None
    )
    o_fair = parse_odds_value(o_row.get('AI適正オッズ')) if o_row else None
    # 印に適正オッズが無い場合のみ、3着内率から控えめに概算（大穴は出さない）
    if o_fair is None and o_row:
        o_place = parse_odds_value(o_row.get('3着内率'))
        if o_place and o_place >= 20:
            win_pct = max(5.0, min(45.0, o_place / 2.5))
            o_fair = 100.0 / win_pct
    o_ev = None
    if o_market and o_fair and 1.2 <= o_fair <= 40 and o_market <= 50:
        o_ev = calc_expected_value(o_market, o_fair)
    if o_ev and o_ev['期待値あり']:
        o_edge = max(-80, min(120, o_ev['期待値エッジ']))
        ev_rows.append({
            '印': '○',
            '表示': f"{'+' if o_edge >= 0 else ''}{o_edge}%",
            '値': o_edge,
            'トーン': ev_tone(100 + o_edge),
        })
    else:
        ev_rows.append({'印': '○', '表示': '—', '値': None, 'トーン': 'ev-none'})

    score = 0
    if hon_fin == 1:
        score += 40
    elif hon_fin == 2:
        score += 28
    elif hon_fin == 3:
        score += 18
    elif hon_fin and hon_fin <= 5:
        score += 8
    if tai_fin == 1:
        score += 25
    elif tai_fin == 2:
        score += 20
    elif tai_fin == 3:
        score += 14
    elif tai_fin and tai_fin <= 5:
        score += 6
    top3_marks = sum(1 for n, _, __ in marked if n <= 3)
    score += min(20, top3_marks * 8)
    if mark_rank == 1:
        score += 10
    elif mark_rank == 2:
        score += 6
    elif mark_rank == 3:
        score += 3
    if main_ev['期待値あり']:
        if main_ev['期待値'] >= 120 and hon_fin and hon_fin <= 3:
            score += 5
        elif main_ev['期待値'] >= 100 and hon_fin and hon_fin <= 5:
            score += 3
        elif main_ev['期待値'] < 90 and hon_fin and hon_fin >= 6:
            score += 2
    score = int(max(0, min(100, score)))

    tips = []
    chaos = parse_odds_value(r.get('荒れ度')) or 0
    reason = str(r.get('本命理由', '') or '')
    if hon_fin and hon_fin >= 6 and tai_fin and tai_fin <= 2:
        tips.append('本命と対抗の序列を再検討')
    if hon_fin and hon_fin >= 8:
        tips.append('人気補正が強すぎ')
    if hon_fin and hon_fin <= 3 and tai_fin and tai_fin <= 3:
        tips.append('展開予測は良好')
    if top3_marks == 0:
        tips.append('印の上位捕捉が不足')
    elif top3_marks >= 2:
        tips.append('印の上位捕捉は良好')
    if chaos >= 60 and top3_marks >= 1:
        tips.append('波乱対応は機能')
    elif chaos >= 60 and top3_marks == 0:
        tips.append('差し馬補正不足')
    if main_ev['期待値あり'] and main_ev['期待値'] >= 120 and (not hon_fin or hon_fin >= 5):
        tips.append('高期待値でも着順乖離あり')
    if main_ev['期待値あり'] and main_ev['期待値'] < 90 and hon_fin and hon_fin == 1:
        tips.append('過小評価した本命を取りこぼし')
    if '過小評価' in reason or '妙味' in reason:
        if hon_fin and hon_fin <= 3:
            tips.append('穴目線の選定は有効')
        elif hon_fin and hon_fin >= 8:
            tips.append('穴目線の振れ幅を抑制')
    if not tips:
        tips.append('総合バランスは良好' if score >= 80 else '軸選定と相手選定の再学習が必要')
    tips = list(dict.fromkeys(tips))[:4]

    return {
        'あり': True,
        '採点': score,
        '星': stars_for_score(score),
        '本命評価': honmei_txt,
        '対抗評価': taikou_txt,
        '印順位': mark_rank_txt,
        '期待値行': ev_rows,
        '改善ポイント': tips,
        'サマリー': f"AI採点 {score}点 · {honmei_txt} · {taikou_txt}",
    }


def day_performance(records, verification=None, safe_pct=None):
    """結果検証上部: 本日成績ダッシュボード。"""
    verification = verification or {}
    if safe_pct is None:
        safe_pct = lambda num, den: round(float(num) / float(den) * 100, 1) if den else 0.0

    ranks = {'S': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0}
    verified = []
    ev_vals = []
    for r in records or []:
        rk = str(r.get('勝負ランク', '') or '').upper()
        if rk in ranks:
            ranks[rk] += 1
        if r.get('結果確定'):
            verified.append(r)
        ev = r.get('期待値')
        if ev is not None:
            try:
                ev_vals.append(float(ev))
            except Exception:
                pass

    main_wins = 0
    main_places = 0
    main_n = 0
    for r in verified:
        review = r.get('結果一覧') or []
        hon = next((x for x in review if x.get('印') == '◎'), None)
        n = finish_num(hon.get('着順')) if hon else None
        if n is None:
            continue
        main_n += 1
        if n == 1:
            main_wins += 1
        if n <= 3:
            main_places += 1

    recovery = verification.get('recovery') if verification.get('has_data') else None
    profit = verification.get('profit') if verification.get('has_data') else None
    tone = verification.get('tone', 'roi-bad') if verification.get('has_data') else 'roi-bad'
    if verification.get('scope') != 'day':
        day = str(verification.get('selected_date') or '')
        for row in verification.get('daily') or []:
            if str(row.get('date')) == day:
                recovery = row.get('recovery')
                profit = row.get('profit')
                tone = row.get('tone', 'roi-bad')
                break

    avg_ev = round(sum(ev_vals) / len(ev_vals)) if ev_vals else None
    return {
        'has_data': bool(records),
        'S': ranks['S'],
        'A': ranks['A'],
        'B': ranks['B'],
        'C': ranks['C'],
        'D': ranks['D'],
        '件数': len(records or []),
        '検証数': len(verified),
        '本命勝率': safe_pct(main_wins, main_n) if main_n else None,
        '複勝率': safe_pct(main_places, main_n) if main_n else None,
        '回収率': recovery,
        '利益': profit,
        '利益表示': (
            f"{'+' if (profit or 0) >= 0 else ''}{int(profit):,}円"
            if profit is not None
            else '—'
        ),
        '期待値': avg_ev,
        '期待値表示': f'{avg_ev}%' if avg_ev is not None else '—',
        '期待値トーン': ev_tone(avg_ev) if avg_ev is not None else 'ev-none',
        '回収率トーン': tone if recovery is not None else 'roi-bad',
        '本命母数': main_n,
    }


def load_score_odds(arch_path, date_str: str, norm_race_id, clean_horse_fn) -> dict:
    """scores_{date}.csv の 単勝オッズ → {(race_id, 馬名): odds}"""
    if not date_str:
        return {}
    path = arch_path / f'scores_{date_str}.csv'
    if not path.exists():
        return {}
    try:
        sdf = pd.read_csv(path).fillna('')
    except Exception:
        return {}
    if '馬名' not in sdf.columns or '単勝オッズ' not in sdf.columns:
        return {}
    out = {}
    for _, x in sdf.iterrows():
        odds = parse_odds_value(x.get('単勝オッズ'))
        if odds is None:
            continue
        rid = norm_race_id(x.get('race_id', ''))
        name = clean_horse_fn(x.get('馬名', ''))
        if rid and name:
            out[(rid, name)] = odds
    return out
