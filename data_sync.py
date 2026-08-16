#!/usr/bin/env python3
"""稼働中の配信インスタンスへ GitHub の data/ を取り込む。

なぜ必要か
----------
これまで data/ は GitHub Actions が 1 日 8〜10 回コミットしており、その
コミットのたびに Render が再デプロイしていた。Render Free は
zero-downtime deploy に対応していないので、再デプロイのあいだサービスは
落ちる（実測: 2026-07-31 17:12 JST のデータコミット後、/healthz が
17:20〜17:34 JST のあいだ 90 秒 timeout を返し続けた）。ブラウザから見ると
これが「白画面」になる。

そこで data/ の変更では再デプロイしない（render.yaml の buildFilter と
コミットメッセージの [skip render]）ことにして、代わりに動いているプロセス
自身が GitHub のスナップショットを取りに行く。デプロイはコード変更のときだけ
起きるようになり、データ更新でサービスが落ちなくなる。

取り込み方
----------
公開リポジトリの tarball（codeload）へ If-None-Match 付きで GET し、変化が
無ければ 304（0 バイト）で終わる。api.github.com は未認証だと 60 req/hour で、
しかも Render の送信 IP は他テナントと共有されるため、ポーリングに使うと
すぐ枯れる。codeload はその制限の外にあり、条件付き GET が効く。

展開は data/ 配下だけ。内容が変わったファイルだけ os.replace で置き換える
ので、変化のないファイルの mtime は動かず、web_app 側の mtime ベースの
キャッシュも無駄に捨てられない。途中で失敗しても既存データは残る。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))
BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
STATE_PATH = DATA / '.data_sync.json'

DEFAULT_REPO = 'rm8939-cell/areru-cloud'
DEFAULT_BRANCH = 'main'
# tarball を落とす頻度。Render Free の 15 分スピンダウンよりは短くしたいが、
# データ更新自体は 1 日 10 回程度なので 5 分あれば十分速い。
DEFAULT_INTERVAL_SEC = 300
# 壊れた tarball を延々と展開しないための保険（現状の data/ は 50MB 未満）。
MAX_TARBALL_BYTES = 400 * 1024 * 1024

# 配信に使う data/ のうち、取り込まないもの。
# - 先頭がドットのファイルはインスタンスのローカル状態（ジョブ状態・ロック）
# - *.html はスクレイパのデバッグダンプで配信には使わない
_SKIP_NAME_RE = re.compile(r'(^\.|\.html$)')
# tarball 内の想定パス: <repo>-<sha>/data/....
_MEMBER_RE = re.compile(r'^[^/]+/(data/[^\0]+)$')

_LOCK = threading.RLock()
_STATE: dict = {
    'enabled': False,
    'etag': '',
    'sha': '',
    'last_ok_at': '',
    'last_change_at': '',
    'last_try_at': '',
    'last_error': '',
    'changed_files': 0,
    'syncs': 0,
    'running': False,
}
_THREAD: threading.Thread | None = None
_KEEPALIVE_THREAD: threading.Thread | None = None


def _env(name: str, default: str = '') -> str:
    return str(os.environ.get(name) or default).strip()


def _flag(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if raw in ('0', 'false', 'no', 'off'):
        return False
    return default


def on_render() -> bool:
    return _flag('RENDER', False) or bool(_env('RENDER_SERVICE_ID'))


def sync_enabled() -> bool:
    """既定では Render 上でのみ有効。ローカル開発では data/ を勝手に上書きしない。"""
    return _flag('ARERU_DATA_SYNC', on_render())


def repo_slug() -> str:
    explicit = _env('ARERU_DATA_REPO')
    if explicit:
        return explicit.replace('https://github.com/', '').strip('/')
    # Render は RENDER_GIT_REPO_SLUG を owner/name 形式で渡す
    slug = _env('RENDER_GIT_REPO_SLUG')
    if slug:
        return slug.strip('/')
    return DEFAULT_REPO


def branch_name() -> str:
    return _env('ARERU_DATA_BRANCH') or _env('RENDER_GIT_BRANCH') or DEFAULT_BRANCH


def interval_sec() -> int:
    try:
        return max(30, int(_env('ARERU_DATA_SYNC_INTERVAL', str(DEFAULT_INTERVAL_SEC))))
    except ValueError:
        return DEFAULT_INTERVAL_SEC


def verify_every() -> int:
    """何回に 1 回、ETag を無視して全体を突き合わせるか（0 で無効）。"""
    try:
        return max(0, int(_env('ARERU_DATA_SYNC_VERIFY_EVERY', '12')))
    except ValueError:
        return 12


def tarball_url() -> str:
    return f'https://codeload.github.com/{repo_slug()}/tar.gz/refs/heads/{branch_name()}'


def _now() -> str:
    return datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S%z')


def _headers(accept: str) -> dict:
    h = {'Accept': accept, 'User-Agent': 'areru-cloud-data-sync/1.0'}
    token = _env('ARERU_DATA_TOKEN') or _env('GITHUB_TOKEN')
    if token:
        h['Authorization'] = f'Bearer {token}'
    return h


def _load_state() -> None:
    """再起動をまたいで最後に取り込んだ内容を思い出す。"""
    try:
        raw = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    with _LOCK:
        for key in ('etag', 'sha', 'last_ok_at', 'last_change_at'):
            value = raw.get(key)
            if isinstance(value, str):
                _STATE[key] = value


def _save_state() -> None:
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            payload = {k: _STATE[k] for k in ('etag', 'sha', 'last_ok_at', 'last_change_at')}
        tmp = STATE_PATH.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload), encoding='utf-8')
        os.replace(tmp, STATE_PATH)
    except Exception:
        pass


def head_sha(timeout: int = 20) -> str:
    """対象ブランチの HEAD sha。

    ポーリングには使わない（api.github.com は未認証 60 req/hour で、Render の
    共有 IP ではすぐ枯れる）。実際に取り込みが起きたときだけ、どのコミットを
    配信しているか記録するために呼ぶ。
    """
    url = f'https://api.github.com/repos/{repo_slug()}/commits/{branch_name()}'
    req = urllib.request.Request(url, headers=_headers('application/vnd.github.sha'))
    with urllib.request.urlopen(req, timeout=timeout) as res:
        sha = res.read().decode('utf-8', 'replace').strip()
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise ValueError(f'unexpected sha response: {sha[:80]!r}')
    return sha


def _download_tarball(dest: Path, etag: str, timeout: int = 180) -> tuple[int, str]:
    """条件付き GET。変わっていなければ (0, etag) を返して何も落とさない。"""
    headers = _headers('application/octet-stream')
    if etag:
        headers['If-None-Match'] = etag
    req = urllib.request.Request(tarball_url(), headers=headers)
    total = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            if res.status == 304:
                return 0, etag
            new_etag = res.headers.get('ETag') or ''
            with dest.open('wb') as out:
                while True:
                    chunk = res.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_TARBALL_BYTES:
                        raise ValueError('tarball larger than expected; aborting')
                    out.write(chunk)
        return total, new_etag
    except urllib.error.HTTPError as e:
        # urllib は 304 を例外にすることがある（リダイレクトハンドラ経由）
        if e.code == 304:
            return 0, etag
        raise


def _target_path(rel: str) -> Path | None:
    """tarball 内の data/... を安全なローカルパスへ。範囲外や隠しファイルは弾く。"""
    parts = [p for p in rel.split('/') if p not in ('', '.')]
    if not parts or parts[0] != 'data' or '..' in parts:
        return None
    if _SKIP_NAME_RE.search(parts[-1]):
        return None
    target = (BASE / Path(*parts)).resolve()
    try:
        target.relative_to(DATA.resolve())
    except ValueError:
        return None
    return target


def _write_if_changed(target: Path, payload: bytes) -> bool:
    if target.exists() and target.stat().st_size == len(payload):
        try:
            if target.read_bytes() == payload:
                return False
        except OSError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix='.sync-')
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(payload)
        # mkstemp は 0600。読み取り専用の配信データなので通常の 0644 に戻す。
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True


def _extract_data(tar_path: Path) -> int:
    changed = 0
    with tarfile.open(tar_path, mode='r:gz') as tf:
        for member in tf:
            if not member.isfile():
                continue
            m = _MEMBER_RE.match(member.name)
            if not m:
                continue
            target = _target_path(m.group(1))
            if target is None:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with src:
                payload = src.read()
            if _write_if_changed(target, payload):
                changed += 1
    return changed


def sync_once(*, force: bool = False, on_update=None) -> dict:
    """GitHub の最新 data/ を取り込む。失敗しても既存データはそのまま残す。

    戻り値はそのまま /healthz や /api/data-sync のレスポンスに載る。
    """
    if not sync_enabled():
        return {'ok': True, 'skipped': 'disabled', **snapshot()}
    with _LOCK:
        if _STATE['running']:
            return {'ok': True, 'skipped': 'already running', **snapshot()}
        _STATE['running'] = True
        _STATE['last_try_at'] = _now()
        _STATE['enabled'] = True
    try:
        result = _sync_body(force=force, on_update=on_update)
    finally:
        with _LOCK:
            _STATE['running'] = False
    return {**result, **snapshot()}


def _sync_body(*, force: bool, on_update) -> dict:
    tmp_dir = None
    started = time.time()
    try:
        with _LOCK:
            etag = '' if force else _STATE['etag']
        tmp_dir = Path(tempfile.mkdtemp(prefix='areru-data-sync-'))
        tar_path = tmp_dir / 'snapshot.tar.gz'
        size, new_etag = _download_tarball(tar_path, etag)
        if size == 0:
            with _LOCK:
                _STATE['last_ok_at'] = _now()
                _STATE['last_error'] = ''
            print('[data-sync] up to date (304)', flush=True)
            return {'ok': True, 'changed': 0, 'skipped': 'unchanged'}
        changed = _extract_data(tar_path)
        took = time.time() - started
        with _LOCK:
            _STATE['etag'] = new_etag
            _STATE['last_ok_at'] = _now()
            _STATE['last_error'] = ''
            _STATE['changed_files'] = changed
            _STATE['syncs'] += 1
            if changed:
                _STATE['last_change_at'] = _now()
        print(
            f'[data-sync] ok changed={changed} tarball={size // 1024}KB {took:.1f}s',
            flush=True,
        )
        # どのコミットを配信しているかの記録。失敗しても取り込み自体は成功。
        # ポーリングでは叩かないので API の 60 req/hour には当たらない。
        try:
            with _LOCK:
                _STATE['sha'] = head_sha()
        except Exception as e:
            print(f'[data-sync] sha lookup skipped: {type(e).__name__}: {e}', flush=True)
        _save_state()
        if changed and callable(on_update):
            try:
                on_update(changed)
            except Exception as e:
                print(f'[data-sync] post-update hook failed: {e}', flush=True)
        with _LOCK:
            return {'ok': True, 'changed': changed, 'sha': _STATE['sha']}
    except Exception as e:
        msg = f'{type(e).__name__}: {e}'
        with _LOCK:
            _STATE['last_error'] = msg[:200]
        print(f'[data-sync] failed: {msg}', flush=True)
        return {'ok': False, 'error': msg[:200]}
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def snapshot() -> dict:
    with _LOCK:
        return {
            'enabled': sync_enabled(),
            'repo': repo_slug(),
            'branch': branch_name(),
            'sha': _STATE['sha'][:12],
            'etag': _STATE['etag'][:18],
            'last_ok_at': _STATE['last_ok_at'],
            'last_change_at': _STATE['last_change_at'],
            'last_try_at': _STATE['last_try_at'],
            'last_error': _STATE['last_error'],
            'changed_files': _STATE['changed_files'],
            'syncs': _STATE['syncs'],
            'running': _STATE['running'],
            'interval_sec': interval_sec(),
        }


def start_background_sync(on_update=None) -> bool:
    """起動直後に 1 回、その後は interval ごとに取り込む常駐スレッド。"""
    global _THREAD
    if not sync_enabled():
        print('[data-sync] disabled', flush=True)
        return False
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return False
        _load_state()

        def _loop() -> None:
            # デプロイ直後の checkout には最新データが入っているので、
            # 起動直後の混雑を避けて少しだけ待ってから取りに行く。
            time.sleep(float(_env('ARERU_DATA_SYNC_DELAY', '10') or 10))
            backoff = 0
            polls = 0
            every = verify_every()
            while True:
                # 通常は ETag が変わったときだけ落とす。ただしそれだと
                # 「上流は変わっていないのにローカルだけ欠けた」状態
                # （途中で落ちた展開など）を直せないので、定期的に
                # ETag を無視して全体を突き合わせる。
                force = every > 0 and polls > 0 and polls % every == 0
                result = sync_once(force=force, on_update=on_update)
                polls += 1
                if result.get('ok'):
                    backoff = 0
                else:
                    # GitHub 側の一時障害で叩き続けない
                    backoff = min(backoff * 2 or 60, 30 * 60)
                time.sleep(backoff or interval_sec())

        _THREAD = threading.Thread(target=_loop, daemon=True, name='data-sync')
        _THREAD.start()
    print(
        f'[data-sync] started repo={repo_slug()} branch={branch_name()} '
        f'interval={interval_sec()}s',
        flush=True,
    )
    return True


def keepalive_url() -> str:
    explicit = _env('ARERU_KEEPALIVE_URL')
    if explicit:
        return explicit.rstrip('/')
    external = _env('RENDER_EXTERNAL_URL')
    return external.rstrip('/') if external else ''


def keepalive_interval_sec() -> int:
    """Render Free は 15 分無アクセスでスピンダウンする。

    15 分の窓に対して 5 分間隔なら、1〜2 回失敗してもまだ間に合う。
    """
    try:
        return max(60, int(_env('ARERU_KEEPALIVE_INTERVAL', '300')))
    except ValueError:
        return 300


def start_keepalive() -> bool:
    """自分の公開 URL に定期アクセスしてスピンダウンを防ぐ。

    Render のスピンダウン判定は「ロードバランサ経由の inbound があるか」なので、
    自分自身への外向きリクエストでもカウントされる。プロセスが死んでいるときは
    当然効かないため、外部からの GitHub Actions ping（keepalive.yml）と併用する。
    """
    global _KEEPALIVE_THREAD
    if not _flag('ARERU_KEEPALIVE', on_render()):
        return False
    url = keepalive_url()
    if not url:
        print('[keepalive] no external URL; skipped', flush=True)
        return False
    with _LOCK:
        if _KEEPALIVE_THREAD is not None and _KEEPALIVE_THREAD.is_alive():
            return False

        def _loop() -> None:
            target = f'{url}/healthz'
            wait = keepalive_interval_sec()
            while True:
                time.sleep(wait)
                try:
                    req = urllib.request.Request(
                        target, headers={'User-Agent': 'areru-cloud-keepalive/1.0'},
                    )
                    with urllib.request.urlopen(req, timeout=30) as res:
                        res.read(256)
                    wait = keepalive_interval_sec()
                except Exception as e:
                    # 失敗をそのまま間隔ぶん放置すると 15 分の窓を割りうるので詰める
                    wait = 60
                    print(f'[keepalive] ping failed: {type(e).__name__}: {e}', flush=True)

        _KEEPALIVE_THREAD = threading.Thread(target=_loop, daemon=True, name='keepalive')
        _KEEPALIVE_THREAD.start()
    print(f'[keepalive] started url={url} interval={keepalive_interval_sec()}s', flush=True)
    return True


def rss_mb() -> float:
    """ワーカーの常駐メモリ（MB）。512MB 上限の Free で OOM 前兆を掴むため。"""
    try:
        with open('/proc/self/status', encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return 0.0


if __name__ == '__main__':
    os.environ.setdefault('ARERU_DATA_SYNC', '1')
    print(json.dumps(sync_once(force=True), ensure_ascii=False, indent=2))
