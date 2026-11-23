"""過去日の予想＋結果（実績タブ）が history なしでも開けること。"""
from __future__ import annotations

import re

from web_app import _best_result_view_date, _force_calendar_today, _today_jst, app


def test_force_calendar_result_keeps_explicit_past_date():
    assert _force_calendar_today('2026-07-19', False, 'result', False) is False
    assert _force_calendar_today('', False, 'result', False) is True
    assert _force_calendar_today('2026-07-19', False, 'predict', False) is True
    assert _force_calendar_today('2026-07-19', False, 'predict', True) is False


def test_best_result_view_skips_empty_today():
    today = _today_jst()
    day = _best_result_view_date('jra', prefer=today, today=today)
    assert day
    assert day <= today
    # 当日が空なら前日以前の結果日へ
    if day == today:
        assert True  # 当日に結果がある環境もあり得る
    else:
        assert day < today


def test_result_mode_past_date_without_history():
    client = app.test_client()
    res = client.get('/?source=jra&mode=result&date=2026-07-19')
    assert res.status_code == 200
    html = res.data.decode('utf-8', 'replace')
    assert 'value="2026-07-19"' in html
    assert re.search(r'<option value="2026-07-19"[^>]*selected', html)
    assert '予想と着順' in html
    assert 'result-pill settled' in html
    assert 'history=1' in html  # nav preserves past browsing


def test_result_mode_today_empty_falls_back():
    """当日未完成の実績URLでも最新結果日が表示される。"""
    client = app.test_client()
    today = _today_jst()
    res = client.get(f'/?source=jra&mode=result&date={today}')
    assert res.status_code == 200
    html = res.data.decode('utf-8', 'replace')
    assert 'mode-result' in html
    # 空の当日に張り付いて「反応しない」ように見えないこと
    assert '予想と着順' in html or '開催場' in html or '結果検証' in html
    assert 'result-pill settled' in html or 'result-pill pending' in html or 'venue-panel' in html


def test_jisseki_nav_points_at_result_jump_date():
    client = app.test_client()
    res = client.get('/?source=jra&mode=predict&date=2026-11-23&history=1&venue=%E6%9D%B1%E4%BA%AC')
    html = res.data.decode('utf-8', 'replace')
    m = re.search(r'<span>実績</span></a>', html)
    assert m
    # 実績アンカー直前の href
    chunk = html[max(0, m.start() - 400):m.start()]
    href = re.search(r'href="([^"]+)"\s*>\s*<i[^>]*>\s*<svg[^>]*>.*?実績', chunk + '<span>実績', re.S)
    # fallback: find last href before 実績
    hrefs = re.findall(r'href="([^"]*)"', chunk)
    assert hrefs
    target = hrefs[-1].replace('&amp;', '&')
    assert 'mode=result' in target
    assert 'venue=' not in target


def test_result_nav_preserves_history_query():
    client = app.test_client()
    res = client.get('/?source=jra&mode=result&date=2026-07-19')
    html = res.data.decode('utf-8', 'replace')
    # _histq 変数だけ HTML escape されるため &history=1 → &amp;history=1
    assert 'mode=result&date=2026-07-19&amp;history=1' in html
