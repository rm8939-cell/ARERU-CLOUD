"""過去日の予想＋結果（実績タブ）が history なしでも開けること。"""
from __future__ import annotations

import re

from web_app import _force_calendar_today, app


def test_force_calendar_result_keeps_explicit_past_date():
    assert _force_calendar_today('2026-07-19', False, 'result', False) is False
    assert _force_calendar_today('', False, 'result', False) is True
    assert _force_calendar_today('2026-07-19', False, 'predict', False) is True
    assert _force_calendar_today('2026-07-19', False, 'predict', True) is False


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


def test_result_nav_preserves_history_query():
    client = app.test_client()
    res = client.get('/?source=jra&mode=result&date=2026-07-19')
    html = res.data.decode('utf-8', 'replace')
    # _histq 変数だけ HTML escape されるため &history=1 → &amp;history=1
    assert 'mode=result&date=2026-07-19&amp;history=1' in html
