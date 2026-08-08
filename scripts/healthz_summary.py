#!/usr/bin/env python3
"""/healthz のレスポンスを 1 行に潰して Actions のログに出す。

見たいのは白画面の前兆になる値だけ:
- rss_mb: Free の 512MB に近づくと Render がワーカーを SIGKILL する（exit 137）
- uptime_sec: 短いままなら再起動を繰り返している
- data_sync: データ取り込みが止まっていないか
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/healthz.json'
    try:
        with open(path, encoding='utf-8') as fh:
            health = json.load(fh)
    except Exception as e:
        print(f'  healthz unreadable: {type(e).__name__}: {e}')
        return 1
    sync = health.get('data_sync') or {}
    print(
        '  rss_mb={rss} uptime_sec={up} latest_pred={pred} today_ready={ready}'.format(
            rss=health.get('rss_mb'),
            up=health.get('uptime_sec'),
            pred=health.get('latest_pred_date'),
            ready=health.get('today_ready'),
        )
    )
    print(
        '  data_sync sha={sha} last_ok={ok} changed={changed} error={err}'.format(
            sha=sync.get('sha'),
            ok=sync.get('last_ok_at'),
            changed=sync.get('changed_files'),
            err=sync.get('last_error') or '-',
        )
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
