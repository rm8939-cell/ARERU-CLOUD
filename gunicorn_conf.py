"""Render Free (0.1 CPU / 512MB) 向けの gunicorn 設定。

方針
----
- ワーカーは 1 本。512MB では pandas を持ったプロセスを 2 本立てられない。
- ただしスレッドは 2 本。1 本だと重いページ描画中に /healthz まで詰まり、
  Render のヘルスチェックがこけてインスタンスごと落ちる（＝白画面）。
- preload_app=True。--max-requests でワーカーを作り直すとき、preload なしでは
  そのたびに pandas を import し直すことになり、0.1 CPU では数秒〜十数秒
  応答が止まる。preload しておけば fork するだけで済む。
- preload すると常駐スレッドは arbiter 側に立ってしまい fork 後に消えるので、
  データ同期とキープアライブは post_fork で立て直す。
"""
from __future__ import annotations

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1
worker_class = 'gthread'
threads = int(os.environ.get('ARERU_THREADS', '2'))
# 応答用のタイムアウト。重い生成は Render では動かさない（ARERU_ENABLE_GENERATION=0）。
timeout = int(os.environ.get('ARERU_TIMEOUT', '120'))
graceful_timeout = 30
# Render の zero-downtime 判定に間に合うよう、コールドスタート後の待ち受けを早く。
keepalive = 30
# メモリ断片化の定期回収。preload により作り直しは fork だけなので安い。
max_requests = int(os.environ.get('ARERU_MAX_REQUESTS', '400'))
max_requests_jitter = 80
preload_app = True
accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('ARERU_LOG_LEVEL', 'info')
# 実行時間とステータスを残す（Worker Timeout / 500 の切り分け用）
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(M)sms'


def post_fork(server, worker):
    """ワーカー内でデータ同期・キープアライブを起動する。"""
    try:
        import web_app

        web_app.start_background_services()
    except Exception as e:  # 起動失敗で配信自体を止めない
        server.log.error('background services failed: %s', e)


def worker_abort(worker):
    """Worker Timeout で SIGABRT される直前にメモリを残す（原因切り分け用）。"""
    try:
        import data_sync

        worker.log.error('worker aborted (timeout) rss=%sMB', data_sync.rss_mb())
    except Exception:
        pass
