"""前の日の10時以降に翌日カードを見られること。"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from daily_ops import (
    NEXT_DAY_VIEW_HOUR,
    can_view_predict_date,
    next_day_jst,
    past_next_day_view_time,
)
from web_app import _force_calendar_today


JST = timezone(timedelta(hours=9))


def test_next_day_helpers():
    assert next_day_jst('2026-11-27') == '2026-11-28'
    before = datetime(2026, 11, 27, 9, 59, tzinfo=JST)
    after = datetime(2026, 11, 27, 10, 0, tzinfo=JST)
    assert past_next_day_view_time(before) is False
    assert past_next_day_view_time(after) is True
    assert can_view_predict_date('2026-11-28', now=before) is False
    assert can_view_predict_date('2026-11-28', now=after) is True
    assert can_view_predict_date('2026-11-27', now=before) is True
    assert NEXT_DAY_VIEW_HOUR == 10


def test_force_calendar_allows_tomorrow_after_view_hour(monkeypatch):
    # 当日を固定して、翌日を明示したときに force しない
    monkeypatch.setattr('web_app._today_jst', lambda: '2026-11-27')
    monkeypatch.setattr('web_app.past_next_day_view_time', lambda now=None: True)
    monkeypatch.setattr('web_app.can_view_predict_date', lambda d, now=None: d == '2026-11-28')
    assert _force_calendar_today('2026-11-28', False, 'predict', False) is False
    # 閲覧不可なら当日へ寄せる
    monkeypatch.setattr('web_app.can_view_predict_date', lambda d, now=None: False)
    assert _force_calendar_today('2026-11-28', False, 'predict', False) is True


def test_resolve_fetch_status_next_day_generating(monkeypatch):
    from web_app import resolve_fetch_status

    monkeypatch.setattr('web_app._today_jst', lambda: '2026-11-27')
    monkeypatch.setattr('web_app.past_next_day_view_time', lambda now=None: True)
    monkeypatch.setattr('web_app._nar_pred_ready', lambda d, s='nar': False)
    monkeypatch.setattr('web_app._expire_stale_job', lambda s: {'state': 'idle'})
    monkeypatch.setattr('web_app._clear_stale_source_locks', lambda s: None)
    st, msg = resolve_fetch_status('2026-11-28', source='jra')
    assert st == 'generating'
    assert '明日' in msg


def test_predict_keeps_tomorrow_url(monkeypatch):
    """10時以降に date=明日 を開いても完成キャッシュ日へ落とさない。"""
    from web_app import app, next_day_jst

    monkeypatch.setattr('web_app._today_jst', lambda: '2026-11-27')
    monkeypatch.setattr('web_app.past_next_day_view_time', lambda now=None: True)
    monkeypatch.setattr(
        'web_app.can_view_predict_date',
        lambda d, now=None: d in ('2026-11-27', '2026-11-28') or d <= '2026-11-27',
    )
    tm = next_day_jst('2026-11-27')
    client = app.test_client()
    res = client.get(f'/?source=jra&mode=predict&date={tm}')
    assert res.status_code == 200
    html = res.data.decode()
    assert f'fx-datecur">{tm}<' in html or f'value="{tm}" selected' in html
    assert 'value="2026-11-23" selected' not in html
