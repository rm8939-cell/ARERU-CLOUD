"""期待値計算・AI自己評価・日別成績ダッシュボード。"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# 表示側でもエンジンと同じ上限・下限を適用（既存CSVの100%/1.0倍を補正）
SIM_WIN_MAX_PCT = 98.0
AI_FAIR_ODDS_MIN = 1.1
# 表示期待値のハード上限（100%＝市場オッズと同値の勝率想定）
EV_DISPLAY_MAX = 124
EV_DISPLAY_MIN = 78

# 買い厳選: 期待回収率の最低ライン（これ未満は候補に入れない）
BUY_EV_FLOOR = 108
# レース信頼度の最低ライン
BUY_CONF_FLOOR = 58

# Sランク厳格条件（すべて満たした場合のみ S。不足は A へ降格）
S_MIN_AI_CONF = 72.0       # AI信頼度が非常に高い
S_MIN_ABILITY_GAP = 70.0   # 能力差が明確
S_MIN_PACE_STABLE = 65.0   # 展開予測が安定
S_MIN_DATA_N = 3           # データ件数が十分
S_MIN_REPRO = 62.0         # 予測の再現性が高い

RANK_PERF_PATH = Path(__file__).resolve().parent / 'data' / 'rank_performance.json'


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
    # 結果検証に基づく軽い補正（自己学習ではなく固定テーブル）
    score += _result_correction_delta(record)
    rank = str(record.get('勝負ランク') or '').upper()
    if rank == 'S':
        score += 4
    elif rank == 'C':
        score -= 4
    return round(_clamp(score, 5, 95), 1)


@lru_cache(maxsize=1)
def _load_result_correction_table() -> dict:
    """analysis_result.csv から当たりやすい/外れやすい条件を集計。

    自己学習ではなく、検証結果の単純な的中率差を ±補正値に落とすだけ。
    """
    path = Path(__file__).resolve().parent / 'data' / 'analysis_result.csv'
    out = {
        'venue': {},   # venue -> delta
        'rank': {},    # S/A/B/C/D -> delta
        'source': {},  # nar/jra -> delta
        'global': 0.0,
    }
    if not path.exists() or path.stat().st_size < 64:
        return out
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', usecols=lambda c: c in (
            'hit', '開催地', '勝負ランク', 'source', 'bet_type', '購入対象',
        ))
    except Exception:
        return out
    if df.empty or 'hit' not in df.columns:
        return out
    # 本命行を優先（券種ノイズを減らす）
    if 'bet_type' in df.columns:
        main = df[df['bet_type'].astype(str) == '本命']
        if len(main) >= 30:
            df = main
    if '購入対象' in df.columns:
        bought = df[df['購入対象'].astype(str).isin(('1', '1.0', 'True', 'true'))]
        if len(bought) >= 20:
            df = bought
    try:
        hits = pd.to_numeric(df['hit'], errors='coerce').fillna(0)
        base = float(hits.mean()) if len(hits) else 0.0
    except Exception:
        return out
    if base <= 0:
        return out
    out['global'] = round(base, 4)

    def _delta(series_hit: pd.Series, min_n: int = 12) -> float:
        if len(series_hit) < min_n:
            return 0.0
        rate = float(series_hit.mean())
        # 全体比 ±0.15 → 補正 ±6 程度に圧縮
        return round(_clamp((rate - base) * 40.0, -6.0, 6.0), 2)

    if '開催地' in df.columns:
        for venue, g in df.groupby(df['開催地'].astype(str)):
            d = _delta(pd.to_numeric(g['hit'], errors='coerce').fillna(0))
            if d:
                out['venue'][str(venue)] = d
    if '勝負ランク' in df.columns:
        for rk, g in df.groupby(df['勝負ランク'].astype(str).str.upper()):
            if rk not in ('S', 'A', 'B', 'C', 'D'):
                continue
            d = _delta(pd.to_numeric(g['hit'], errors='coerce').fillna(0), min_n=8)
            if d:
                out['rank'][rk] = d
    if 'source' in df.columns:
        for src, g in df.groupby(df['source'].astype(str).str.lower()):
            if src not in ('nar', 'jra'):
                continue
            d = _delta(pd.to_numeric(g['hit'], errors='coerce').fillna(0), min_n=15)
            if d:
                out['source'][src] = d
    return out


def _result_correction_delta(record: dict) -> float:
    """結果検証テーブルから小さな補正値を返す（±8以内）。"""
    try:
        table = _load_result_correction_table()
    except Exception:
        return 0.0
    delta = 0.0
    venue = str(record.get('開催地') or '').strip()
    if venue and venue in table.get('venue', {}):
        delta += float(table['venue'][venue])
    src = str(record.get('source') or '').lower()
    if not src:
        try:
            from areru_engine import source_from_race_id
            src = source_from_race_id(record.get('race_id', ''))
        except Exception:
            src = ''
    if src in table.get('source', {}):
        delta += float(table['source'][src]) * 0.5
    # 前段ランクがあれば参照（再計算前のヒント）
    rk = str(record.get('勝負ランク') or '').upper()
    if rk in table.get('rank', {}):
        delta += float(table['rank'][rk]) * 0.35
    return round(_clamp(delta, -8.0, 8.0), 2)


def _ability_gap_score(record: dict) -> float:
    """本命と対抗の能力差 0-100（差が大きいほど読みやすい）。"""
    cards = [c for c in (record.get('ピックカード一覧') or []) if isinstance(c, dict)]
    main = next((c for c in cards if c.get('役割') == '本命'), None)
    rival = next((c for c in cards if c.get('役割') == '対抗'), None)
    if not main:
        return 45.0

    def _idx(c):
        for k in ('AI評価', 'AI信頼度スコア', '近走指数順位'):
            try:
                v = float(c.get(k))
                if k == '近走指数順位':
                    return max(0.0, 100.0 - (v - 1.0) * 12.0)
                return v
            except (TypeError, ValueError):
                continue
        return None

    m = _idx(main)
    if m is None:
        return 45.0
    if not rival:
        return _clamp(50.0 + (m - 50.0) * 0.4)
    r = _idx(rival)
    if r is None:
        return _clamp(50.0 + (m - 50.0) * 0.3)
    gap = m - r
    # 指数順位なら既にスケール済み。評価差 8〜25 が理想帯
    if gap >= 18:
        return 88.0
    if gap >= 10:
        return 75.0
    if gap >= 5:
        return 62.0
    if gap >= 0:
        return 48.0
    return 32.0


def _pace_clarity_score(record: dict) -> float:
    """展開の読みやすさ 0-100。"""
    pace = record.get('展開予想データ')
    if not isinstance(pace, dict):
        try:
            import json
            raw = record.get('展開予想')
            if isinstance(raw, str) and raw.strip().startswith('{'):
                pace = json.loads(raw.replace('NaN', 'null'))
            else:
                pace = {}
        except Exception:
            pace = {}
    score = 50.0
    try:
        chaos = float(pace.get('荒れ指数')) if pace.get('荒れ指数') is not None else None
    except (TypeError, ValueError):
        chaos = None
    if chaos is not None:
        if chaos <= 35:
            score += 22
        elif chaos <= 55:
            score += 10
        elif chaos >= 80:
            score -= 18
        elif chaos >= 65:
            score -= 10
    pick = _main_pick_card(record)
    phase = str(pick.get('展開相性') or '')
    if '好相性' in phase or '有利' in phase:
        score += 12
    elif '不利' in phase:
        score -= 14
    summary = str(pace.get('AI総評') or '')
    if '読みやすい' in summary or '先行有利' in summary or '明確' in summary:
        score += 8
    if '混戦' in summary or '不明' in summary:
        score -= 8
    return _clamp(score)


def _venue_bias_match_score(record: dict) -> float:
    """競馬場バイアス一致率 0-100（結果補正＋適性）。"""
    base = 50.0
    delta = _result_correction_delta(record)
    # delta +6 → +18 程度
    base += delta * 3.0
    pick = _main_pick_card(record)
    apt, detail = _aptitude_score(record, pick)
    base += (apt - 50.0) * 0.35
    course = float(detail.get('コース適性') or 50)
    base += (course - 50.0) * 0.25
    return _clamp(base)


def calc_race_confidence(record: dict) -> dict:
    """レース信頼度の総合評価。

    AI信頼度・能力差・展開の読みやすさ・データ件数・
    シミュレーション一致率・競馬場バイアス一致率を合成。
    """
    pick = _main_pick_card(record)
    n = _count_past_races(record)
    repro = float(record.get('シミュレーション再現率') or calc_simulation_reproducibility(record, pick))
    conf = float(record.get('AI信頼度スコア') or calc_ai_confidence(record, pick, repro=repro))
    ability = _ability_gap_score(record)
    pace = _pace_clarity_score(record)
    data_score = _clamp(n * 18.0 + 10.0)  # 0〜5件 → 〜100
    bias = _venue_bias_match_score(record)
    correction = _result_correction_delta(record)

    score = (
        conf * 0.28
        + ability * 0.18
        + pace * 0.15
        + data_score * 0.12
        + repro * 0.15
        + bias * 0.12
        + correction * 0.4
    )
    score = round(_clamp(score, 8, 96), 1)
    return {
        'レース信頼度スコア': score,
        '能力差スコア': round(ability, 1),
        '展開読みやすさ': round(pace, 1),
        'データ件数スコア': round(data_score, 1),
        'シミュレーション一致率': round(repro, 1),
        '競馬場バイアス一致率': round(bias, 1),
        '結果補正': correction,
    }


def rank_from_race_confidence(score) -> str:
    """レース信頼度から暫定 S〜D。

    注意: S は qualify_s_rank() を通さないと確定しない（不足時は A へ降格）。
    """
    try:
        if score is None or score == '' or str(score).strip() in ('—', '-', 'なし', 'nan', 'None'):
            return 'D'
        s = float(score)
    except (TypeError, ValueError):
        return 'D'
    if s >= 78:
        return 'S'  # 暫定。後段で厳格条件チェック
    if s >= 68:
        return 'A'
    if s >= 58:
        return 'B'
    if s >= 48:
        return 'C'
    return 'D'


def qualify_s_rank(record: dict) -> tuple[bool, dict]:
    """Sランク厳格判定。5条件すべて満たしたときだけ True。

    ・AI信頼度が非常に高い
    ・能力差が明確
    ・展開予測が安定
    ・データ件数が十分
    ・予測の再現性が高い
    """
    # 信頼度パックが未計算なら補完
    if record.get('能力差スコア') is None or record.get('展開読みやすさ') is None:
        pack = calc_race_confidence(record)
        for k, v in pack.items():
            record.setdefault(k, v)

    try:
        conf = float(record.get('AI信頼度スコア') or 0)
    except (TypeError, ValueError):
        conf = 0.0
    try:
        ability = float(record.get('能力差スコア') or 0)
    except (TypeError, ValueError):
        ability = 0.0
    try:
        pace = float(record.get('展開読みやすさ') or 0)
    except (TypeError, ValueError):
        pace = 0.0
    try:
        n = int(float(record.get('データ件数') or _count_past_races(record) or 0))
    except (TypeError, ValueError):
        n = 0
    try:
        repro = float(
            record.get('シミュレーション再現率')
            or record.get('シミュレーション一致率')
            or 0
        )
    except (TypeError, ValueError):
        repro = 0.0

    checks = {
        'AI信頼度': conf >= S_MIN_AI_CONF,
        '能力差': ability >= S_MIN_ABILITY_GAP,
        '展開安定': pace >= S_MIN_PACE_STABLE,
        'データ件数': n >= S_MIN_DATA_N,
        '再現性': repro >= S_MIN_REPRO,
    }
    detail = {
        '合格': all(checks.values()),
        '条件': checks,
        '値': {
            'AI信頼度': round(conf, 1),
            '能力差': round(ability, 1),
            '展開安定': round(pace, 1),
            'データ件数': n,
            '再現性': round(repro, 1),
        },
        '閾値': {
            'AI信頼度': S_MIN_AI_CONF,
            '能力差': S_MIN_ABILITY_GAP,
            '展開安定': S_MIN_PACE_STABLE,
            'データ件数': S_MIN_DATA_N,
            '再現性': S_MIN_REPRO,
        },
    }
    if not detail['合格']:
        failed = [k for k, ok in checks.items() if not ok]
        detail['降格理由'] = ' / '.join(failed) if failed else '条件不足'
    return bool(detail['合格']), detail


def finalize_race_rank(record: dict, provisional: str | None = None) -> str:
    """暫定ランクを確定。S条件不足は必ず A へ降格。"""
    rk = str(provisional or record.get('勝負ランク') or '').upper()
    if rk == 'S':
        ok, detail = qualify_s_rank(record)
        record['S判定'] = detail
        if not ok:
            record['S降格'] = True
            record['S降格理由'] = detail.get('降格理由') or '条件不足'
            return 'A'
        record['S降格'] = False
        return 'S'
    # S以外でも判定内訳を残す（改善用）
    if 'S判定' not in record:
        _, detail = qualify_s_rank(record)
        record['S判定'] = detail
        record['S降格'] = False
    return rk if rk in ('S', 'A', 'B', 'C', 'D') else 'D'


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


def decide_buy_skip(ev: float | None, confidence: float, repro: float, has_odds: bool,
                    race_conf: float | None = None) -> dict:
    """買い／見送りの暫定判定（最終は tighten_buy_selection で厳選）。

    期待回収率・信頼度が揃った候補のみ「買い」。足りなければ見送り。
    """
    if not has_odds or ev is None:
        return {
            '一覧判定': '判定待ち', '一覧判定トーン': 'wait',
            '投資判定': '判定待ち', '投資判定アイコン': '⚪', '投資判定トーン': 'wait',
            '投資判定表示': '判定待ち',
        }
    e = float(ev)
    conf = float(confidence or 0)
    rc = float(race_conf if race_conf is not None else conf)
    # 厳選: EV・信頼度・レース信頼度の三重ゲート
    if e >= BUY_EV_FLOOR and conf >= BUY_CONF_FLOOR and rc >= BUY_CONF_FLOOR and float(repro or 0) >= 42:
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
    """互換用: 期待回収率からの仮ランク（表示はレース信頼度を優先）。"""
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
    """レース信頼度で S〜D を付け、期待回収率で買い判定を揃える。"""
    from areru_engine import RANK_LABELS, RANK_CLASSES

    ev = record.get('期待値')
    if ev is None:
        raw = record.get('レース期待回収率') or record.get('期待値表示') or ''
        try:
            ev = float(str(raw).replace('%', '').strip()) if str(raw).strip() not in ('', '—', 'なし') else None
        except (TypeError, ValueError):
            ev = None

    # レース信頼度を算出し、勝負ランクの主根拠にする
    conf_pack = calc_race_confidence(record)
    record.update(conf_pack)
    rc = float(conf_pack.get('レース信頼度スコア') or 50)
    rk = rank_from_race_confidence(rc)
    # EVが極端に弱い（見送り帯）ならランクを1段落とす
    if ev is not None:
        try:
            e = float(ev)
            if e < 100 and rk in ('S', 'A'):
                rk = 'B'
            elif e < 95 and rk == 'B':
                rk = 'C'
            elif e < 90:
                rk = 'D' if rk != 'S' else 'C'
        except (TypeError, ValueError):
            pass

    # S は厳格条件を満たす場合のみ。不足は A へ降格
    rk = finalize_race_rank(record, rk)

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
    ai_conf = float(record.get('AI信頼度スコア') or 50)
    repro = float(record.get('シミュレーション再現率') or 50)
    if ev is not None:
        record['期待値あり'] = True
        decision = decide_buy_skip(float(ev), ai_conf, repro, True, race_conf=rc)
    else:
        decision = decide_buy_skip(None, ai_conf, repro, has_odds, race_conf=rc)
    record.update(decision)

    if ev is not None:
        ev_i = int(round(float(ev)))
        record['期待値'] = ev_i
        record['期待値表示'] = f'{ev_i}%'
        record['期待回収率表示'] = f'{ev_i}%'
        record['期待回収率短'] = f'{ev_i}%'
        record['期待値トーン'] = 'ev-buy' if ev_i >= BUY_EV_FLOOR else 'ev-skip'
        record['レース期待回収率'] = ev_i
    else:
        record['期待回収率表示'] = '—'
        record['期待回収率短'] = '—'
        record['期待値トーン'] = 'ev-none'
    return record


def _buy_score(record: dict) -> float:
    """厳選時の並び用スコア。"""
    try:
        ev = float(record.get('期待値') if record.get('期待値') is not None else 0)
    except (TypeError, ValueError):
        ev = 0.0
    try:
        rc = float(record.get('レース信頼度スコア') or record.get('AI信頼度スコア') or 0)
    except (TypeError, ValueError):
        rc = 0.0
    rank_bonus = {'S': 12, 'A': 8, 'B': 3, 'C': 0, 'D': -4}.get(
        str(record.get('勝負ランク') or '').upper(), 0
    )
    return rc * 0.65 + max(0.0, ev - 100.0) * 1.8 + rank_bonus


def _scope_key(record: dict, by_venue: bool) -> str:
    if by_venue:
        return str(record.get('開催地') or '不明')
    return '_day_'


def _rank_slots(n: int, by_venue: bool) -> tuple[int, int]:
    """(S枠, A枠)。地方は開催場単位、JRAは日次全体。"""
    if by_venue:
        # 地方: 6R→S1/A1、10-12R→S1〜2/A2〜3
        if n <= 6:
            return 1, 1
        if n <= 9:
            return 1, 2
        return 2, 3
    # JRA: 開催全体で S1〜2 / A4〜6
    if n <= 12:
        return 1, 4
    if n <= 24:
        return 2, 5
    return 2, 6


def _buy_cap(n: int, by_venue: bool) -> int:
    if by_venue:
        if n <= 6:
            return 2
        if n <= 9:
            return 3
        return 4
    # JRA: S+A の枠にほぼ合わせる
    s, a = _rank_slots(n, by_venue=False)
    return s + a


def tighten_buy_selection(races: list, by_venue: bool = False) -> list:
    """開催単位で S/A 枠と買いレース数を厳選する（買わないAI）。

    - 地方 (by_venue=True): 場ごとに S/A と買い数を制限
    - JRA (by_venue=False): 日次で S1〜2 / A4〜6、買いはその上位のみ
    """
    races = list(races or [])
    if not races:
        return races

    from areru_engine import RANK_LABELS, RANK_CLASSES
    from collections import defaultdict

    groups = defaultdict(list)
    for i, r in enumerate(races):
        groups[_scope_key(r, by_venue)].append((i, r))

    for _key, items in groups.items():
        n = len(items)
        s_slots, a_slots = _rank_slots(n, by_venue)
        buy_cap = _buy_cap(n, by_venue)

        # 信頼度順で相対ランクを再割当。
        # S は枠があっても厳格条件を満たすレースだけ（満たさなければ A へ）。
        ordered = sorted(items, key=lambda x: _buy_score(x[1]), reverse=True)
        assigned = {}
        s_left, a_left = s_slots, a_slots
        for pos, (idx, r) in enumerate(ordered):
            score = float(r.get('レース信頼度スコア') or r.get('AI信頼度スコア') or 50)
            ok_s, s_detail = qualify_s_rank(r)
            r['S判定'] = s_detail
            if s_left > 0 and ok_s and score >= 66:
                rk = 'S'
                s_left -= 1
                r['S降格'] = False
            elif a_left > 0 and score >= 40:
                rk = 'A'
                a_left -= 1
                # S枠候補だったが条件不足 → 明示的に A 降格
                if not ok_s and score >= 66:
                    r['S降格'] = True
                    r['S降格理由'] = s_detail.get('降格理由') or '条件不足'
                else:
                    r['S降格'] = False
            elif score >= 48:
                rk = 'B'
                r['S降格'] = False
            elif score >= 38:
                rk = 'C'
                r['S降格'] = False
            else:
                rk = 'D'
                r['S降格'] = False
            # 下位帯は S/A にしない
            if pos >= s_slots + a_slots + max(2, n // 3) and rk in ('S', 'A', 'B'):
                rk = 'C' if score >= 38 else 'D'
            # 最終ガード: S は必ず qualify 通過
            if rk == 'S' and not ok_s:
                rk = 'A'
                r['S降格'] = True
                r['S降格理由'] = s_detail.get('降格理由') or '条件不足'
            assigned[idx] = rk

        for idx, r in items:
            rk = assigned.get(idx, 'D')
            r['勝負ランク'] = rk
            r['勝負ランク表示'] = rk
            r['BET判定'] = RANK_LABELS.get(rk, '')
            r['BETクラス'] = RANK_CLASSES.get(rk, '')

        # 買い候補: S/A かつ EV・信頼度ゲート通過
        candidates = []
        for idx, r in items:
            rk = str(r.get('勝負ランク') or '')
            try:
                ev = float(r.get('期待値')) if r.get('期待値') is not None else None
            except (TypeError, ValueError):
                ev = None
            try:
                conf = float(r.get('AI信頼度スコア') or 0)
                rc = float(r.get('レース信頼度スコア') or conf)
                repro = float(r.get('シミュレーション再現率') or 0)
            except (TypeError, ValueError):
                conf, rc, repro = 0.0, 0.0, 0.0
            has_odds = bool(r.get('オッズ取得済')) or ev is not None
            if rk not in ('S', 'A'):
                continue
            if not has_odds or ev is None:
                continue
            if ev < BUY_EV_FLOOR or conf < BUY_CONF_FLOOR or rc < BUY_CONF_FLOOR or repro < 40:
                continue
            candidates.append((idx, r))

        candidates.sort(key=lambda x: _buy_score(x[1]), reverse=True)
        buy_ids = {idx for idx, _ in candidates[:buy_cap]}

        skip = {
            '一覧判定': '見送り', '一覧判定トーン': 'skip',
            '投資判定': '見送り', '投資判定アイコン': '🔴', '投資判定トーン': 'skip',
            '投資判定表示': '見送り',
        }
        buy = {
            '一覧判定': '買い', '一覧判定トーン': 'buy',
            '投資判定': '買い', '投資判定アイコン': '🟢', '投資判定トーン': 'buy',
            '投資判定表示': '買い',
        }
        for idx, r in items:
            if idx in buy_ids:
                r.update(buy)
            else:
                r.update(skip)

    return races


def refresh_rank_performance_log(analysis_csv: Path | None = None) -> dict:
    """S/A/B ごとの的中率・回収率を集計して data/rank_performance.json に記録。

    自己学習ではなく、結果検証に基づく判定基準改善用のログ。
    """
    import json
    from datetime import datetime, timezone, timedelta

    src = Path(analysis_csv) if analysis_csv else (
        Path(__file__).resolve().parent / 'data' / 'analysis_result.csv'
    )
    jst = timezone(timedelta(hours=9))
    out = {
        'updated_at': datetime.now(jst).isoformat(timespec='seconds'),
        'source': str(src.name),
        'note': 'S/A/Bの的中率・回収率。今後のS判定閾値改善に利用する（自動学習ではない）。',
        'by_rank': {},
        'thresholds_s': {
            'AI信頼度': S_MIN_AI_CONF,
            '能力差': S_MIN_ABILITY_GAP,
            '展開安定': S_MIN_PACE_STABLE,
            'データ件数': S_MIN_DATA_N,
            '再現性': S_MIN_REPRO,
        },
    }
    empty_rank = {
        'bets': 0, 'hits': 0, 'hit_rate': None,
        'investment': 0, 'payout': 0, 'recovery': None, 'profit': 0,
    }
    for rk in ('S', 'A', 'B'):
        out['by_rank'][rk] = dict(empty_rank)

    if not src.exists() or src.stat().st_size < 64:
        _write_rank_perf(out)
        return out

    try:
        df = pd.read_csv(src, encoding='utf-8-sig')
    except Exception as e:
        out['error'] = str(e)[:200]
        _write_rank_perf(out)
        return out

    if df.empty or '勝負ランク' not in df.columns:
        _write_rank_perf(out)
        return out

    # 購入対象がある場合は購入分のみ（無い場合は全行）
    base = df
    if '購入対象' in df.columns:
        bought = df[pd.to_numeric(df['購入対象'], errors='coerce').fillna(0).astype(int) == 1]
        if len(bought) >= 10:
            base = bought

    for rk in ('S', 'A', 'B'):
        g = base[base['勝負ランク'].astype(str).str.upper() == rk]
        if g.empty:
            continue
        hits = pd.to_numeric(g.get('hit'), errors='coerce').fillna(0)
        inv = float(pd.to_numeric(g.get('investment'), errors='coerce').fillna(0).sum())
        pay = float(pd.to_numeric(g.get('payout'), errors='coerce').fillna(0).sum())
        hit_n = int(hits.sum())
        bets = int(len(g))
        out['by_rank'][rk] = {
            'bets': bets,
            'hits': hit_n,
            'hit_rate': round(hit_n / bets * 100.0, 1) if bets else None,
            'investment': int(inv),
            'payout': int(pay),
            'recovery': round(pay / inv * 100.0, 1) if inv else None,
            'profit': int(pay - inv),
        }

    # 直近スナップショットを最大30件保持
    prev = {}
    try:
        if RANK_PERF_PATH.exists():
            prev = json.loads(RANK_PERF_PATH.read_text(encoding='utf-8'))
    except Exception:
        prev = {}
    history = list(prev.get('history') or [])
    history.append({
        'updated_at': out['updated_at'],
        'by_rank': out['by_rank'],
    })
    out['history'] = history[-30:]
    _write_rank_perf(out)
    return out


def _write_rank_perf(payload: dict) -> None:
    import json
    try:
        RANK_PERF_PATH.parent.mkdir(parents=True, exist_ok=True)
        RANK_PERF_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'[rank-perf] wrote {RANK_PERF_PATH.name}', flush=True)
    except Exception as e:
        print(f'[rank-perf] write fail: {e}', flush=True)


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

    # ランク＝レース信頼度、買い＝厳選前の暫定（一覧は apply_display_ranks で最終確定）
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
