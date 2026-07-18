"""期待値計算・AI自己評価・日別成績ダッシュボード。"""
from __future__ import annotations

import re

import pandas as pd


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


def ev_tone(ev_pct):
    """期待値の色区分: 120+緑 / 100-119黄 / 90-99グレー / 90未満赤"""
    try:
        v = float(ev_pct)
    except (TypeError, ValueError):
        return 'ev-none'
    if v >= 120:
        return 'ev-buy'
    if v >= 100:
        return 'ev-consider'
    if v >= 90:
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


def calc_expected_value(market_odds, fair_odds):
    """現在オッズ / AI適正オッズ × 100 → 期待値%。"""
    m = parse_odds_value(market_odds)
    f = parse_odds_value(fair_odds)
    if m is None or f is None or f <= 0:
        return {
            '期待値': None,
            '期待値表示': '—',
            '期待値エッジ': None,
            '期待値トーン': 'ev-none',
            '期待値ラベル': '—',
            '期待値あり': False,
        }
    ev = round(m / f * 100)
    edge = ev - 100
    tone = ev_tone(ev)
    return {
        '期待値': ev,
        '期待値表示': f'{ev}%',
        '期待値エッジ': edge,
        '期待値トーン': tone,
        '期待値ラベル': ev_label(tone),
        '期待値あり': True,
    }


def finish_num(fin):
    s = str(fin or '').strip()
    if not s or s in ('結果待ち', '取消', '除外', '中止', '—', '－'):
        return None
    m = re.match(r'(\d+)', s)
    return int(m.group(1)) if m else None


def stars_for_score(score: int) -> str:
    filled = max(1, min(5, (int(score) + 19) // 20))
    return '★' * filled + '☆' * (5 - filled)


def apply_expected_value(record: dict) -> dict:
    """予想レコードへ期待値フィールドを付与（AI適正オッズ列は保持）。"""
    fair = parse_odds_value(record.get('AI適正オッズ'))
    market = parse_odds_value(record.get('本命オッズ'))
    if fair is not None:
        record['AI適正オッズ'] = round(fair, 1)
    record['現在オッズ'] = round(market, 1) if market is not None else None
    ev = calc_expected_value(market, fair)
    record['期待値'] = ev['期待値']
    record['期待値表示'] = ev['期待値表示']
    record['期待値エッジ'] = ev['期待値エッジ']
    record['期待値トーン'] = ev['期待値トーン']
    record['期待値ラベル'] = ev['期待値ラベル']
    record['期待値あり'] = ev['期待値あり']
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

    ranks = {'S': 0, 'A': 0, 'B': 0, 'C': 0}
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
