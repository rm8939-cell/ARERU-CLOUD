"""毎朝8時までに予想を完成させる運用ヘルパー。

- 08:00 JST まで: 開催取得 → AI予想の本生成
- 08:00 JST 以降: 基本ロジック固定。取消・除外・オッズのみ更新
- JRA: 開催日（土日・祝など netkeiba 検出日）のみ本生成
- 地方: 開催日のみ本生成
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

JST = timezone(timedelta(hours=9))
PREDICT_READY_HOUR = 8
DATA = Path(__file__).resolve().parent / 'data'
SEAL_DIR = DATA / 'predict_seals'
SEAL_DIR.mkdir(parents=True, exist_ok=True)


def now_jst() -> datetime:
    return datetime.now(JST)


def today_jst() -> str:
    return now_jst().date().isoformat()


def past_predict_deadline(now: datetime | None = None) -> bool:
    """予想本生成の締切（朝8時）を過ぎているか。"""
    t = now or now_jst()
    if t.tzinfo is None:
        t = t.replace(tzinfo=JST)
    return t.hour >= PREDICT_READY_HOUR


def seal_path(date_str: str) -> Path:
    return SEAL_DIR / f'{date_str}.json'


def read_seal(date_str: str) -> dict:
    p = seal_path(date_str)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_seal(
    date_str: str,
    *,
    sources: list[str] | None = None,
    note: str = '',
    mode: str = 'morning',
) -> dict:
    """当日予想を封印（以降は本ロジックを固定）。"""
    prev = read_seal(date_str)
    srcs = sorted(set((prev.get('sources') or []) + list(sources or [])))
    payload = {
        'date': date_str,
        'sealed_at': now_jst().isoformat(timespec='seconds'),
        'deadline_hour': PREDICT_READY_HOUR,
        'sources': srcs,
        'mode': mode,
        'note': note or prev.get('note') or '',
    }
    seal_path(date_str).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return payload


def is_sealed(date_str: str, source: str | None = None) -> bool:
    seal = read_seal(date_str)
    if not seal:
        return False
    if not source:
        return True
    srcs = [str(s).lower() for s in (seal.get('sources') or [])]
    return str(source).lower() in srcs or str(source).lower() == 'all'


def discover_has_meeting(date_str: str, source: str) -> bool:
    """netkeiba 検出で当日開催があるか。"""
    try:
        from netkeiba_client import NetkeibaClient
        found = NetkeibaClient(sleep=0.08).discover_kaisai_dates(
            lookback=2, lookahead=1, source=source,
        )
        return date_str in set(found or [])
    except Exception:
        return False


def jra_is_race_day(date_str: str | None = None) -> bool:
    """JRA 開催日か（検出優先。失敗時は土日のみ True）。"""
    day = date_str or today_jst()
    if discover_has_meeting(day, 'jra'):
        return True
    try:
        d = datetime.fromisoformat(day).date()
    except Exception:
        d = now_jst().date()
    # 検出失敗時のフォールバック: 土日
    return d.weekday() >= 5


def nar_is_race_day(date_str: str | None = None) -> bool:
    day = date_str or today_jst()
    if discover_has_meeting(day, 'nar'):
        return True
    # 地方はほぼ毎日。検出失敗時は本生成を試す（開催なしならパイプラインが完了扱い）
    return True


def source_is_race_day(source: str, date_str: str | None = None) -> bool:
    src = str(source or '').lower()
    if src == 'jra':
        return jra_is_race_day(date_str)
    if src == 'nar':
        return nar_is_race_day(date_str)
    return True


def should_full_predict(
    source: str,
    *,
    date_str: str | None = None,
    ready: bool = False,
    force_full: bool = False,
) -> bool:
    """本生成（ランク再計算を含む）を行ってよいか。

    - 08:00 前: 朝の本生成・再生成を許可（force_full で上書き可）
    - 08:00 以降かつ完成済み: 本生成禁止（force_full でも odds-only）
    """
    day = date_str or today_jst()
    src = str(source or '').lower()
    # 朝8時以降・完成済み → ハード凍結
    if ready and past_predict_deadline():
        return False
    if force_full:
        return True
    if ready and is_sealed(day, src):
        return False
    if not source_is_race_day(src, day):
        return False
    return True


def should_odds_only_update(
    source: str,
    *,
    date_str: str | None = None,
    ready: bool = False,
) -> bool:
    """締切後の最小更新（オッズ・取消）か。"""
    day = date_str or today_jst()
    if not ready:
        return False
    if is_sealed(day, source) or past_predict_deadline():
        return True
    return False


def auto_seal_if_ready(date_str: str, source: str, ready: bool) -> bool:
    """本生成成功後に封印（以降はオッズ等の最小更新のみ）。"""
    if not ready:
        return False
    day = date_str or today_jst()
    if is_sealed(day, source):
        return False
    write_seal(day, sources=[source], note='morning_predict_ready', mode='morning')
    return True
