#!/usr/bin/env python3
"""JRA・地方の表示健全性チェック。

これまで監視していたのは地方だけだったため、JRA が白画面になっても
ワークフローは success のままだった。中央・地方の両方を同じ基準で見る。

判定は「利用者に見えているか」に寄せてある:
- HTTP 200 で返っているか
- エラーページ（通信エラー）や「データ取得中」で止まっていないか
- レースカード（data-race-id）が描かれているか
  - 開催なしのお知らせが出ているときは正常
- 地方のように会場ピッカーが出る場合は、その先のレース一覧まで開けるか

exit 0: すべて OK
exit 1: いずれか失敗
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

RACE_ID_RE = re.compile(r'data-race-id="([^"]+)"')
VENUE_RE = re.compile(r'data-venue="([^"]+)"')
SELECTED_DATE_RE = re.compile(r'<option value="(\d{4}-\d{2}-\d{2})"[^>]*selected')
NO_MEETING_MARKERS = ('開催はありません', '本日は開催なし')
GENERATING_MARKERS = ('<h2>データ取得中</h2>', '初回データ')
ERROR_MARKERS = ('通信エラー',)


def fetch(url: str, timeout: int = 120) -> tuple[int, str, float]:
    req = urllib.request.Request(url, headers={'User-Agent': 'areru-display-monitor/2.0'})
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as res:
        body = res.read().decode('utf-8', errors='replace')
        return res.status, body, time.time() - started


def inspect(html: str) -> dict:
    return {
        'races': len(set(RACE_ID_RE.findall(html))),
        'venues': VENUE_RE.findall(html),
        'selected': (SELECTED_DATE_RE.findall(html) or [''])[0],
        'generating': any(m in html for m in GENERATING_MARKERS),
        'error': any(m in html for m in ERROR_MARKERS),
        'no_meeting': any(m in html for m in NO_MEETING_MARKERS),
        # 白画面の直接検知: <body> がほぼ空なら描画されていない
        'body_bytes': len(html.split('<body', 1)[-1]) if '<body' in html else 0,
    }


def check_source(base: str, source: str) -> dict:
    """1 ソース分の予想ページを見る。会場ピッカーが出たら 1 段深く追う。"""
    out: dict = {'ok': False, 'source': source}
    url = f'{base}/?source={source}&mode=predict'
    try:
        code, html, took = fetch(url)
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
        return out
    info = inspect(html)
    out.update({'code': code, 'took_sec': round(took, 1), 'url': url, **info})
    out['venues'] = len(info['venues'])

    # 地方は会場ピッカー経由なので、レース一覧まで開けることを確認する
    if code == 200 and not info['races'] and info['venues'] and info['selected']:
        venue = info['venues'][0]
        q = urllib.parse.urlencode(
            {'source': source, 'mode': 'predict', 'date': info['selected'], 'venue': venue},
        )
        try:
            code, html, took = fetch(f'{base}/?{q}')
            deep = inspect(html)
            out['venue_page'] = {
                'venue': venue,
                'code': code,
                'took_sec': round(took, 1),
                'races': deep['races'],
                'error': deep['error'],
                'generating': deep['generating'],
            }
            info = deep
            out['races'] = deep['races']
            out['error_page'] = deep['error']
            out['generating'] = deep['generating']
        except Exception as e:
            out['venue_page'] = {'venue': venue, 'error': f'{type(e).__name__}: {e}'}
            return out

    out['ok'] = bool(
        code == 200
        and not info['error']
        and not info['generating']
        and info['body_bytes'] > 2000
        and (info['races'] > 0 or info['no_meeting'])
    )
    return out


def main() -> int:
    base = (os.environ.get('APP_URL') or 'https://areru-cloud.onrender.com').rstrip('/')
    sources = [s for s in (os.environ.get('MONITOR_SOURCES') or 'jra,nar').split(',') if s.strip()]
    report = {
        'ok': False,
        'at': datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S%z'),
        'base': base,
        'checks': {},
    }

    try:
        code, body, took = fetch(f'{base}/healthz', timeout=120)
        health = json.loads(body)
        report['checks']['healthz'] = {
            'ok': code == 200 and bool(health.get('ok')),
            'code': code,
            'took_sec': round(took, 1),
            'cold_start': took > 20,
            'rss_mb': health.get('rss_mb'),
            'uptime_sec': health.get('uptime_sec'),
            'latest_pred_date': health.get('latest_pred_date'),
            'today_ready': health.get('today_ready'),
            'data_sync': health.get('data_sync'),
        }
    except Exception as e:
        report['checks']['healthz'] = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    for source in sources:
        report['checks'][source.strip()] = check_source(base, source.strip())

    report['ok'] = all(bool(c.get('ok')) for c in report['checks'].values())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
