"""最新開催日を自動取得し、runners.csv と predictions_by_date を更新する。

PO-3/PO-7: JRA・NAR の単勝/券種オッズを取得して runners.csv へ統合し、
AIスコア再計算（predictions 再生成）まで一気通貫で行う。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

JST = timezone(timedelta(hours=9))

import pandas as pd

from areru_engine import parse_date, source_from_race_id
from netkeiba_client import NetkeibaClient, infer_source

# パイプ実行時でも進捗が見えるようにする
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

DATA = Path("data")
RUNNERS = DATA / "runners.csv"
LEGACY = DATA / "score_test_data.csv"
PRED_DIR = DATA / "predictions_by_date"
ODDS_TICKETS = DATA / "odds_tickets"
PRED_DIR.mkdir(parents=True, exist_ok=True)
ODDS_TICKETS.mkdir(parents=True, exist_ok=True)

# Render Free 向け: 履歴ネット取得を抑えて時間内に完了させる
def _fast_nar_mode() -> bool:
    if str(os.environ.get("ARERU_FAST_NAR") or "").strip() in ("1", "true", "yes"):
        return True
    if str(os.environ.get("RENDER") or "").strip().lower() in ("true", "1"):
        return True
    return False


def _stage_log(stage: str, msg: str = "", **kv) -> None:
    bits = [f"[Local Update] {stage}"]
    if msg:
        bits.append(msg)
    for k, v in kv.items():
        if v is None or v == "":
            continue
        bits.append(f"{k}={v}")
    print(" ".join(bits), flush=True)


def resolve_date_token(token: str) -> str:
    """today / yesterday / tomorrow を JST の YYYY-MM-DD に展開する。"""
    s = str(token or "").strip()
    if not s:
        return s
    key = s.lower()
    today = datetime.now(JST).date()
    if key in ("today", "本日"):
        return today.isoformat()
    if key in ("yesterday", "昨日"):
        return (today - timedelta(days=1)).isoformat()
    if key in ("tomorrow", "明日"):
        return (today + timedelta(days=1)).isoformat()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    return s


def resolve_date_tokens(dates: list[str] | None) -> list[str] | None:
    if not dates:
        return dates
    out = []
    for d in dates:
        r = resolve_date_token(d)
        if r and r not in out:
            out.append(r)
    return out


RUNNER_COLS = [
    "race_id", "日付", "レース", "馬名", "馬番", "枠", "騎手", "斤量", "実着順",
    "着順1", "人気1", "場1", "レース名1",
    "着順2", "人気2", "場2", "レース名2",
    "着順3", "人気3", "場3", "レース名3",
    "着順4", "人気4", "場4", "レース名4",
    "着順5", "人気5", "場5", "レース名5",
    "単勝オッズ", "人気", "オッズ更新日時", "source",
]


def _rss_mb() -> float:
    """自プロセス RSS (MB)。ログ用。失敗時は -1。"""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS: bytes / Linux: KB
        if sys.platform == "darwin":
            return float(ru) / (1024 * 1024)
        return float(ru) / 1024
    except Exception:
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
    except Exception:
        pass
    return -1.0


def _log_mem(tag: str) -> None:
    mb = _rss_mb()
    if mb >= 0:
        print(f"  📊 MEM {mb:.1f} MB | {tag}", flush=True)
    else:
        print(f"  📊 MEM ? | {tag}", flush=True)


def _normalize_runners(df: pd.DataFrame) -> pd.DataFrame:
    for c in RUNNER_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[RUNNER_COLS].copy()
    df["日付"] = parse_date(df["日付"]).dt.strftime("%Y-%m-%d")
    df["レース"] = pd.to_numeric(df["レース"], errors="coerce")
    # 馬番は "4.0" にならないよう整数文字列へ
    ban = pd.to_numeric(df["馬番"], errors="coerce")
    df["馬番"] = ban.apply(lambda x: str(int(x)) if pd.notna(x) else "")
    waku = pd.to_numeric(df["枠"], errors="coerce")
    df["枠"] = waku.apply(lambda x: str(int(x)) if pd.notna(x) else "")
    pop = pd.to_numeric(df["人気"], errors="coerce")
    df["人気"] = pop.apply(lambda x: str(int(x)) if pd.notna(x) else "")
    odds = pd.to_numeric(df["単勝オッズ"], errors="coerce")
    df["単勝オッズ"] = odds.apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")
    # source 補完
    df["source"] = df.apply(
        lambda r: str(r["source"]).strip().lower()
        if str(r.get("source") or "").strip().lower() in ("jra", "nar")
        else source_from_race_id(r.get("race_id", "")),
        axis=1,
    )
    return df.dropna(subset=["日付", "馬名"]).reset_index(drop=True)


def load_existing_runners() -> pd.DataFrame:
    if RUNNERS.exists():
        return _normalize_runners(pd.read_csv(RUNNERS, encoding="utf-8-sig"))
    if LEGACY.exists():
        print(f"↪️  初回移行: {LEGACY} → {RUNNERS}")
        return _normalize_runners(pd.read_csv(LEGACY, encoding="utf-8-sig"))
    return pd.DataFrame(columns=RUNNER_COLS)


def available_dates(runners: pd.DataFrame, source: str | None = None) -> list[str]:
    if runners.empty:
        return []
    df = runners
    if source in ("jra", "nar"):
        df = df[df["source"].astype(str) == source]
    d = parse_date(df["日付"]).dropna().dt.strftime("%Y-%m-%d").unique().tolist()
    return sorted(d, reverse=True)


def _apply_win_odds(entries: list[dict], win_odds: dict[str, dict]) -> list[dict]:
    """馬番で単勝オッズを上書き。API未公開時は出馬表の値を残す。"""
    if not win_odds:
        return entries
    for e in entries:
        ban = str(e.get("馬番") or "").strip()
        info = win_odds.get(ban) or win_odds.get(ban.zfill(2))
        if not info:
            continue
        if info.get("単勝オッズ"):
            e["単勝オッズ"] = info["単勝オッズ"]
        if info.get("人気"):
            e["人気"] = info["人気"]
        if info.get("オッズ更新日時"):
            e["オッズ更新日時"] = info["オッズ更新日時"]
    return entries


def _save_ticket_odds(client: NetkeibaClient, race_id: str, source: str) -> None:
    """券種別オッズを data/odds_tickets/{race_id}.json に保存。"""
    rid = str(race_id)
    if not (rid.isdigit() and len(rid) == 12):
        return
    try:
        maps = client.fetch_ticket_odds_maps(rid, source=source)
        if not any(maps.get(k) for k in ("ワイド", "馬連", "三連複")):
            return
        (ODDS_TICKETS / f"{rid}.json").write_text(
            json.dumps(maps, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"  ⚠️ 券種オッズ保存失敗 {rid}: {e}")


def _count_odds_tickets_for_date(target: str, race_ids: list[str] | None = None) -> int:
    """指定日の odds_tickets JSON 保存件数。"""
    if race_ids is None:
        # race_id: YYYY + venue(2) + MMDD + RR
        ymd = target.replace("-", "")
        mmdd = ymd[4:]
        year = ymd[:4]
        n = 0
        if ODDS_TICKETS.exists():
            for p in ODDS_TICKETS.glob("*.json"):
                name = p.stem
                if len(name) == 12 and name.startswith(year) and name[6:10] == mmdd:
                    n += 1
        return n
    return sum(1 for rid in race_ids if (ODDS_TICKETS / f"{rid}.json").exists())


def build_date_runners(
    client: NetkeibaClient,
    target: str,
    source: str = "jra",
    include_results: bool = True,
    include_odds: bool = True,
    persist_cb=None,
    venues: list[str] | None = None,
) -> pd.DataFrame:
    """開催日の出走を取得。開催場単位で失敗しても他場は継続する。"""
    t0 = time.perf_counter()
    fast = _fast_nar_mode() and source == "nar"
    # Render Free: 全体予算（秒）。超えたら残会場はスキップして保存へ進む
    budget_sec = float(os.environ.get("ARERU_FETCH_BUDGET_SEC") or (240 if fast else 1500))
    if fast:
        include_results = False  # 結果ページ取得を省略して時間短縮
        try:
            client.sleep = min(float(getattr(client, "sleep", 0.25) or 0.25), 0.05)
        except Exception:
            pass
    _stage_log(
        "START",
        f"source={source} date={target}",
        fast=int(fast), budget_sec=int(budget_sec),
        include_results=int(include_results),
    )

    ymd = target.replace("-", "")
    t_list = time.perf_counter()
    try:
        race_ids = client.list_race_ids(ymd, source=source)
    except Exception as e:
        _stage_log(
            "開催取得",
            "ERROR",
            date=target, source=source, err=f"{type(e).__name__}: {e}",
            sec=f"{time.perf_counter()-t_list:.1f}",
        )
        print(
            f"[pipeline] 開催取得 失敗 date={target} source={source} "
            f"開催場数=0 レース数=0 保存件数=0 error={e}",
            flush=True,
        )
        return pd.DataFrame(columns=RUNNER_COLS)

    list_sec = time.perf_counter() - t_list
    if not race_ids:
        _stage_log(
            "開催取得", "0件", date=target, source=source,
            sec=f"{list_sec:.1f}",
        )
        print(
            f"[pipeline] 開催取得 失敗 date={target} source={source} "
            f"開催場数=0 レース数=0 保存件数=0",
            flush=True,
        )
        return pd.DataFrame(columns=RUNNER_COLS)

    venue_names = sorted(
        {
            (client.parse_race_id(rid).get("venue") or "?")
            for rid in race_ids
        }
    )
    _stage_log(
        "開催取得完了",
        date=target, source=source,
        開催場数=len(venue_names), レース数=len(race_ids),
        場=",".join(venue_names),
        sec=f"{list_sec:.1f}",
    )
    print(
        f"[pipeline] 開催取得 成功 date={target} source={source} "
        f"開催場数={len(venue_names)} レース数={len(race_ids)} "
        f"保存件数={len(race_ids)} 場={venue_names} sec={list_sec:.1f}",
        flush=True,
    )

    if venues:
        from netkeiba_client import normalize_venue_name
        want = {normalize_venue_name(v) for v in venues if str(v).strip()}
        filtered = []
        for rid in race_ids:
            meta = client.parse_race_id(rid)
            v = normalize_venue_name(meta.get("venue") or "")
            if v in want:
                filtered.append(rid)
        print(
            f"🎯 開催場フィルタ: {sorted(want)} → {len(filtered)}/{len(race_ids)}レース",
            flush=True,
        )
        race_ids = filtered
        if not race_ids:
            print(f"⚠️  {source.upper()} {target}: 指定開催場のレースなし")
            return pd.DataFrame(columns=RUNNER_COLS)

    # 開催場ごとにグループ化（1場失敗でも他場続行）
    by_venue: dict[str, list[str]] = {}
    for rid in sorted(race_ids, key=_race_sort_key):
        venue = client.parse_race_id(rid).get("venue") or "?"
        by_venue.setdefault(venue, []).append(rid)

    rows: list[dict] = []
    ok_n = fail_n = 0
    skipped_venues: list[str] = []
    ok_venues: list[str] = []
    t_races = time.perf_counter()
    odds_sec_total = 0.0
    hist_sec_total = 0.0

    print(f"========== 取得開始 ==========")
    print(
        f"📥 {source.upper()} {target}: {len(race_ids)}レース / "
        f"{len(by_venue)}開催場（場単位・失敗スキップ）"
        f"{' [FAST]' if fast else ''}...",
        flush=True,
    )

    for venue, v_rids in by_venue.items():
        elapsed = time.perf_counter() - t0
        if elapsed > budget_sec:
            skipped_venues.append(venue)
            _stage_log(
                "レース取得",
                f"BUDGET_SKIP venue={venue}",
                reason="time_budget",
                elapsed=f"{elapsed:.1f}",
                budget=int(budget_sec),
            )
            continue

        print(f"—— {venue} ——", flush=True)
        _stage_log("レース取得開始", venue=venue, races=len(v_rids))
        t_venue = time.perf_counter()
        venue_ok = 0
        venue_fail = 0
        venue_rows: list[dict] = []

        try:
            for i, rid in enumerate(v_rids, 1):
                if time.perf_counter() - t0 > budget_sec:
                    _stage_log(
                        "レース取得",
                        f"BUDGET_STOP mid-venue={venue}",
                        done=f"{i-1}/{len(v_rids)}",
                    )
                    break
                meta = client.parse_race_id(rid)
                race_no = meta.get("race_no") or "?"
                try:
                    rn_label = f"{int(float(race_no)):02d}R"
                except Exception:
                    rn_label = f"{race_no}R"
                print(f"  → {venue} {rn_label} 取得 [{i}/{len(v_rids)}] {rid}", flush=True)
                try:
                    entries = client.fetch_entries(rid, source=source)
                    if not entries:
                        raise RuntimeError("出馬表0頭（HTML変更/取消の可能性）")
                    results = {}
                    if include_results:
                        try:
                            results = client.fetch_results(rid, source=source) or {}
                        except Exception as e:
                            print(f"  ⚠️ {venue} {rn_label} 結果取得失敗（継続）: {e}", flush=True)
                            results = {}
                    t_odds = time.perf_counter()
                    # FAST: 出馬表の単勝表示を使い、別途オッズAPIは叩かない（時間短縮）
                    if include_odds and not fast:
                        win_odds = _fetch_win_odds_with_fallback(client, rid, source)
                    else:
                        win_odds = {}
                    odds_sec_total += time.perf_counter() - t_odds
                    entries = _apply_win_odds(entries, win_odds)
                    odds_n = sum(1 for e in entries if e.get("単勝オッズ"))
                    # FAST: 券種HTMLは重いのでスキップ（単勝のみ）
                    if include_odds and odds_n and not fast:
                        try:
                            _save_ticket_odds(client, rid, source)
                        except Exception as e:
                            print(f"  ⚠️ 券種保存スキップ {rid}: {e}", flush=True)
                    print(
                        f"  ✓ {venue} {rn_label} 出走{len(entries)}頭 "
                        f"オッズ{odds_n}頭 結果{len(results)}頭",
                        flush=True,
                    )
                    race_rows = []
                    for e in entries:
                        t_hist = time.perf_counter()
                        try:
                            # FAST: キャッシュのみ（ネット取得しない）→ Render時間内完了
                            hist = (
                                client.fetch_horse_history(
                                    e["horse_id"],
                                    use_cache=True,
                                    cache_only=fast,
                                )
                                if e.get("horse_id") else []
                            )
                        except Exception as e_hist:
                            print(f"  ⚠️ 履歴失敗 {e.get('馬名')}: {e_hist}", flush=True)
                            hist = []
                        hist_sec_total += time.perf_counter() - t_hist
                        score = client.past_five_for_score(hist, target)
                        finish = results.get(e["馬名"], "")
                        race_rows.append({
                            "race_id": rid,
                            "日付": e.get("日付") or target,
                            "レース": e.get("レース"),
                            "馬名": e["馬名"],
                            "馬番": e.get("馬番", ""),
                            "枠": e.get("枠", ""),
                            "騎手": e.get("騎手", ""),
                            "斤量": e.get("斤量", ""),
                            "実着順": finish or score.get("実着順", ""),
                            "単勝オッズ": e.get("単勝オッズ", ""),
                            "人気": e.get("人気", ""),
                            "オッズ更新日時": e.get("オッズ更新日時", ""),
                            "source": source,
                            **{k: score[k] for k in score if k != "実着順"},
                        })
                        if finish:
                            race_rows[-1]["実着順"] = finish
                    venue_rows.extend(race_rows)
                    venue_ok += 1
                    ok_n += 1
                    if persist_cb and race_rows:
                        try:
                            persist_cb(_normalize_runners(pd.DataFrame(race_rows)))
                            print(f"  💾 増分保存 {venue} {rn_label}", flush=True)
                        except Exception as e_save:
                            print(f"  ⚠️ 増分保存失敗（継続）: {e_save}", flush=True)
                except Exception as e:
                    venue_fail += 1
                    fail_n += 1
                    print(
                        f"  ✗ {venue} {rn_label} 失敗（次レースへ）: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    if source == "nar":
                        print(
                            f"[netkeiba-api] NAR レース取得エラー race_id={rid} "
                            f"venue={venue} race={rn_label} "
                            f"err={type(e).__name__}: {e}",
                            flush=True,
                        )
                    continue

            # 開催場として1Rも取れなければスキップ扱い
            if venue_ok == 0 and venue_fail > 0:
                skipped_venues.append(venue)
                _stage_log(
                    "レース取得",
                    f"VENUE_SKIP {venue}",
                    ok=0, fail=venue_fail,
                    sec=f"{time.perf_counter()-t_venue:.1f}",
                )
            else:
                ok_venues.append(venue)
                rows.extend(venue_rows)
                _stage_log(
                    "レース取得",
                    f"VENUE_OK {venue}",
                    ok=venue_ok, fail=venue_fail,
                    sec=f"{time.perf_counter()-t_venue:.1f}",
                )
        except Exception as e:
            # 開催場ごと丸ごと失敗 → スキップして次の場へ
            skipped_venues.append(venue)
            fail_n += max(1, len(v_rids) - venue_ok)
            print(
                f"  ✗ {venue} 開催場ごとスキップ: {type(e).__name__}: {e}",
                flush=True,
            )
            _stage_log(
                "レース取得",
                f"VENUE_ERROR_SKIP {venue}",
                err=f"{type(e).__name__}: {e}",
                sec=f"{time.perf_counter()-t_venue:.1f}",
            )
            continue

    races_sec = time.perf_counter() - t_races
    _stage_log(
        "レース取得完了",
        date=target, source=source,
        ok_venues=",".join(ok_venues) or "-",
        skipped_venues=",".join(skipped_venues) or "-",
        ok_races=ok_n, fail_races=fail_n,
        sec=f"{races_sec:.1f}",
        odds_sec=f"{odds_sec_total:.1f}",
        hist_sec=f"{hist_sec_total:.1f}",
    )
    print(f"========== 取得完了 ==========")
    print(
        f"✅ {source.upper()} {target}: 成功 {ok_n} / 失敗 {fail_n} / "
        f"場OK={ok_venues} 場SKIP={skipped_venues}",
        flush=True,
    )
    if source == "nar" and (fail_n or skipped_venues):
        print(
            f"[netkeiba-api] NAR 取得サマリ date={target} ok={ok_n} fail={fail_n} "
            f"skip_venues={skipped_venues}",
            flush=True,
        )
    odds_n = _count_odds_tickets_for_date(target, race_ids)
    print(
        f"[pipeline] レース取得 "
        f"{'成功' if ok_n > 0 else '失敗'} date={target} source={source} "
        f"成功={ok_n} 失敗={fail_n} 保存件数={ok_n} odds_json={odds_n} "
        f"sec={races_sec:.1f}",
        flush=True,
    )
    print(
        f"[pipeline] 保存 "
        f"{'成功' if ok_n > 0 else '失敗'} date={target} source={source} "
        f"runners_races={ok_n} odds_json={odds_n} 保存件数={ok_n}",
        flush=True,
    )
    _stage_log(
        "DB保存",
        "runners増分完了",
        rows=len(rows), races=ok_n,
        total_sec=f"{time.perf_counter()-t0:.1f}",
    )
    return _normalize_runners(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=RUNNER_COLS)


def _race_sort_key(race_id: str) -> tuple:
    """開催場コード → レース番号の順（場ごとに全Rを順番取得するため）。"""
    rid = str(race_id)
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})", rid)
    if not m:
        return ("zzz", 999, rid)
    _year, venue, a, b, race_no = m.groups()
    try:
        rn = int(race_no)
    except Exception:
        rn = 999
    return (venue, rn, rid)


def _fetch_win_odds_with_fallback(client: NetkeibaClient, rid: str, src: str) -> dict:
    """単勝オッズ取得。APIが空でも例外で止めず、出馬表スクレイピングへフォールバック。"""
    win: dict = {}
    try:
        win = client.fetch_win_odds(rid, source=src) or {}
    except Exception as e:
        print(f"  ⚠️ APIオッズ失敗 {rid}: {e}")
        win = {}
    n = len({k: v for k, v in win.items() if len(str(k)) <= 2 and v.get("単勝オッズ")})
    if n:
        return win
    # API未公開/空のとき出馬表の Popular 欄を使う（---.- は parser 側で空になる）
    try:
        entries = client.fetch_entries(rid, source=src) or []
    except Exception as e:
        print(f"  ⚠️ 出馬表フォールバック失敗 {rid}: {e}")
        return win
    for e in entries:
        ban = str(e.get("馬番") or "").strip()
        odds = str(e.get("単勝オッズ") or "").strip()
        if not ban or not odds:
            continue
        info = {
            "単勝オッズ": odds,
            "人気": str(e.get("人気") or "").strip(),
            "オッズ更新日時": "",
            "オッズ状態": "shutuba",
        }
        win[ban] = info
        if ban.isdigit():
            win[ban.zfill(2)] = info
    return win


def refresh_odds_for_dates(
    client: NetkeibaClient,
    runners: pd.DataFrame,
    dates: list[str],
    source: str | None = None,
) -> pd.DataFrame:
    """既存 runners の対象日だけオッズ列を更新（履歴再取得なし）。

    開催場ごとにレース番号順で全Rを取得する。途中の空オッズでも処理を止めない。
    """
    if runners.empty or not dates:
        return runners
    df = runners.copy()
    date_mask = parse_date(df["日付"]).dt.strftime("%Y-%m-%d").isin(dates)
    if source in ("jra", "nar"):
        date_mask = date_mask & (df["source"].astype(str) == source)
    target = df[date_mask].copy()
    if target.empty:
        print(f"⚠️  オッズ更新対象なし: {dates} source={source}")
        return runners

    race_ids = sorted(set(target["race_id"].astype(str)), key=_race_sort_key)
    print(f"💰 オッズ更新: {len(race_ids)}レース / 日={dates} / source={source or 'all'}")
    odds_by_race: dict[str, dict] = {}
    current_venue = None
    ok_races = 0
    empty_races = 0
    for i, rid in enumerate(race_ids, 1):
        # 旧JRA URL 行はスキップ
        if not str(rid).isdigit() or len(str(rid)) != 12:
            print(f"  [{i}/{len(race_ids)}] {rid}: skip (非netkeiba ID)")
            continue
        src = source if source in ("jra", "nar") else infer_source(rid)
        meta = client.parse_race_id(rid)
        venue = meta.get("venue") or "?"
        race_no = meta.get("race_no") or "?"
        if venue != current_venue:
            current_venue = venue
            print(f"—— {venue} ——")
        win = _fetch_win_odds_with_fallback(client, rid, src)
        odds_by_race[rid] = win
        n = len({k: v for k, v in win.items() if len(str(k)) <= 2 and v.get("単勝オッズ")})
        sample_status = ""
        for _k, info in win.items():
            if isinstance(info, dict) and info.get("オッズ状態"):
                sample_status = str(info.get("オッズ状態"))
                break
        if n:
            ok_races += 1
            try:
                _save_ticket_odds(client, rid, src)
            except Exception as e:
                print(f"  ⚠️ 券種保存スキップ {rid}: {e}")
        else:
            empty_races += 1
            # 4R以降を含む空応答を必ず残す（原因切り分け用）
            try:
                raw = client.fetch_odds_api(rid, 1, source=src)
                print(
                    f"  🔎 空オッズ詳細 {venue}{race_no}R {rid}: "
                    f"status={raw.get('status')!r} reason={raw.get('reason')!r} "
                    f"updated={raw.get('updated_at')!r} keys={len(raw.get('odds') or {})}"
                )
            except Exception as e:
                print(f"  🔎 空オッズ詳細取得失敗 {rid}: {e}")
        st = f" status={sample_status}" if sample_status else ""
        print(f"  [{i}/{len(race_ids)}] {venue}{race_no}R {rid}: オッズ{n}頭{st}")

    updated = 0
    for idx in target.index:
        rid = str(df.at[idx, "race_id"])
        ban = str(df.at[idx, "馬番"] or "").strip()
        info = (odds_by_race.get(rid) or {}).get(ban) or (odds_by_race.get(rid) or {}).get(ban.zfill(2))
        if not info or not info.get("単勝オッズ"):
            continue
        df.at[idx, "単勝オッズ"] = info["単勝オッズ"]
        if info.get("人気"):
            df.at[idx, "人気"] = info["人気"]
        if info.get("オッズ更新日時"):
            df.at[idx, "オッズ更新日時"] = info["オッズ更新日時"]
        updated += 1
    print(f"✅ オッズ反映: {updated}頭 / 取得成功{ok_races}R / 未公開{empty_races}R")
    return _normalize_runners(df)


def merge_runners(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """同一 race_id を差し替え（JRA/NAR 同日混在でも他ソースを消さない）。

    重要: 重複除去は (race_id, 馬番) ではなく (race_id, 馬名)。
    馬番未採番/空文字の出走を (race_id, 馬番='') でまとめると
    1レース1頭に潰れ、印・馬券候補が消える。
    """
    if new.empty:
        return base
    if base.empty:
        out = new
    else:
        # 既存より明らかに頭数が少ない取得結果で上書きしない（途中取得の破壊を防ぐ）
        base_n = base.groupby(base["race_id"].astype(str)).size().to_dict()
        new_n = new.groupby(new["race_id"].astype(str)).size().to_dict()
        safe_ids = set()
        skip_ids = set()
        for rid, n_new in new_n.items():
            n_old = int(base_n.get(rid, 0))
            n_new_i = int(n_new)
            # 既存が2頭以上あるのに新規が1頭以下 → 壊れた取得
            if n_old >= 2 and n_new_i <= 1:
                skip_ids.add(rid)
            # 既存が4頭以上で新規が半分未満 → 部分スクレイプとみなす
            elif n_old >= 4 and n_new_i < max(2, int(n_old * 0.5)):
                skip_ids.add(rid)
            else:
                safe_ids.add(rid)
        if skip_ids:
            print(f"⚠️  頭数不足のため上書きスキップ: {sorted(skip_ids)[:8]}{'...' if len(skip_ids)>8 else ''}")
            new = new[new["race_id"].astype(str).isin(safe_ids)].copy()
        if new.empty:
            return base
        drop_ids = set(new["race_id"].astype(str))
        keep = base[~base["race_id"].astype(str).isin(drop_ids)]
        out = pd.concat([keep, new], ignore_index=True)
    out = _normalize_runners(out)
    # 同一レース・馬名の重複を除去（後勝ち）。馬番空でも全頭を潰さない。
    if not out.empty:
        out = out.drop_duplicates(subset=["race_id", "馬名"], keep="last").reset_index(drop=True)
    return out


def save_runners(df: pd.DataFrame) -> None:
    df = _normalize_runners(df)
    df = df.sort_values(["日付", "source", "レース", "馬名"]).reset_index(drop=True)
    df.to_csv(RUNNERS, index=False, encoding="utf-8-sig")
    print(f"✅ runners.csv 保存: {len(df)}行 / 開催日 {available_dates(df)}")


def generate_predictions(dates: list[str] | None = None) -> None:
    if dates:
        runners = load_existing_runners()
        avail = set(available_dates(runners)) if runners is not None and not runners.empty else set()
        for d in dates:
            d = resolve_date_token(d)
            if d not in avail:
                print(f"⏭ predictions スキップ（出走なし）: {d}")
                continue
            print(f"🔮 predictions 生成: {d}")
            subprocess.run([sys.executable, "replay_predict.py", d], check=True, timeout=600)
    else:
        print("🔮 predictions 全開催日生成 (--all)")
        subprocess.run([sys.executable, "replay_predict.py", "--all"], check=True, timeout=1800)


def _sources_list(source: str) -> list[str]:
    if source == "all":
        return ["jra", "nar"]
    return [source]


def refresh(
    dates: list[str] | None = None,
    discover: bool = True,
    latest_only: bool = False,
    lookback: int = 28,
    lookahead: int = 14,
    skip_predict: bool = False,
    migrate_only: bool = False,
    odds_only: bool = False,
    include_odds: bool = True,
    source: str = "all",
    venues: list[str] | None = None,
) -> list[str]:
    runners = load_existing_runners()
    if migrate_only:
        save_runners(runners)
        if not skip_predict:
            generate_predictions()
        return available_dates(runners)

    client = NetkeibaClient()
    sources = _sources_list(source)
    all_target_dates: list[str] = []
    venue_list = [str(v).strip() for v in (venues or []) if str(v).strip()] or None
    dates = resolve_date_tokens(dates)

    try:
        for src in sources:
            target_dates: list[str] = []
            if dates:
                target_dates = dates
            elif discover:
                found = client.discover_kaisai_dates(
                    lookback=lookback, lookahead=lookahead, source=src
                )
                print(f"🗓️  {src.upper()} 自動検出開催日: {found}")
                if not found:
                    print(f"⚠️  {src.upper()} 開催日なし")
                    continue
                if latest_only:
                    # 未来カードではなく「本日優先・無ければ直近過去」を中心に ±2 日を取る（JST）
                    today_str = datetime.now(JST).date().isoformat()
                    past_or_today = [d for d in found if d <= today_str]
                    if today_str in found:
                        anchor = today_str
                    else:
                        anchor = max(past_or_today) if past_or_today else found[0]
                        print(
                            f"[pipeline] 開催取得 警告 date={anchor} source={src} "
                            f"reason=today_missing_in_discover today={today_str} "
                            f"found={found[:8]} 保存件数=0",
                            flush=True,
                        )
                    target_dates = [anchor]
                    for d in found:
                        if d == anchor:
                            continue
                        delta = abs((datetime.fromisoformat(anchor) - datetime.fromisoformat(d)).days)
                        if delta <= 2:
                            target_dates.append(d)
                    target_dates = sorted(set(target_dates))
                    print(
                        f"[pipeline] 開催取得 成功 date={anchor} source={src} "
                        f"anchor={anchor} targets={target_dates} 保存件数={len(target_dates)}",
                        flush=True,
                    )
                else:
                    existing = set(available_dates(runners, source=src))
                    recent = found[:4]
                    target_dates = sorted(set(recent) | (set(found) - existing))
            else:
                target_dates = available_dates(runners, source=src)

            print(f"🎯 {src.upper()} 更新対象: {target_dates}")
            all_target_dates.extend(target_dates)
            _log_mem(f"before {src} fetch")

            if odds_only:
                runners = refresh_odds_for_dates(client, runners, target_dates, source=src)
            else:
                for d in target_dates:
                    # 増分保存用の箱（タイムアウトでも途中まで残す）
                    state = {"df": runners}

                    def _persist(partial: pd.DataFrame, _state=state):
                        _state["df"] = merge_runners(_state["df"], partial)
                        save_runners(_state["df"])

                    built = build_date_runners(
                        client, d, source=src, include_results=True, include_odds=include_odds,
                        persist_cb=_persist,
                        venues=venue_list,
                    )
                    runners = merge_runners(state["df"], built)
                    _log_mem(f"after {src} {d}")
                    import gc
                    gc.collect()
    finally:
        try:
            client.session.close()
        except Exception:
            pass
        try:
            del client
        except Exception:
            pass
        import gc
        gc.collect()
        _log_mem("after client close")

    save_runners(runners)
    all_target_dates = sorted(set(all_target_dates))

    av = available_dates(runners)
    to_gen: list[str] = []
    if not skip_predict:
        missing = [d for d in av if not (PRED_DIR / f"predictions_{d}.csv").exists()]
        to_gen = sorted(set(all_target_dates) | set(missing))
        if not to_gen and all_target_dates:
            to_gen = list(all_target_dates)
    # 予想子プロセス起動前に大きな DataFrame を解放（RAM 二重化を緩和）
    try:
        del runners
    except Exception:
        pass
    import gc
    gc.collect()
    _log_mem("before predict subprocess")
    if to_gen:
        generate_predictions(to_gen)
    return av


def main():
    ap = argparse.ArgumentParser(
        description="ARERU PO-3/PO-7: JRA・NAR 開催日・オッズ取得 & runners/predictions 更新"
    )
    ap.add_argument("--dates", nargs="*", help="YYYY-MM-DD / today / yesterday / tomorrow")
    ap.add_argument("--latest-only", action="store_true", help="最新開催週末だけ更新")
    ap.add_argument("--no-discover", action="store_true", help="開催日自動検出をしない")
    ap.add_argument("--skip-predict", action="store_true", help="predictions 生成をスキップ")
    ap.add_argument("--migrate-only", action="store_true", help="既存CSVの移行のみ")
    ap.add_argument("--odds-only", action="store_true", help="オッズ列だけ再取得して再予想")
    ap.add_argument("--no-odds", action="store_true", help="オッズ取得をスキップ")
    ap.add_argument("--source", choices=["jra", "nar", "all"], default="all")
    ap.add_argument("--venue", nargs="*", help="開催場名で絞り込み（例: 帯広 盛岡）")
    ap.add_argument("--lookback", type=int, default=28)
    ap.add_argument("--lookahead", type=int, default=14)
    ap.add_argument("--list", action="store_true", help="検出開催日を表示して終了")
    args = ap.parse_args()

    if args.list:
        client = NetkeibaClient()
        for src in _sources_list(args.source):
            print(f"# {src}")
            print("\n".join(client.discover_kaisai_dates(
                lookback=args.lookback, lookahead=args.lookahead, source=src
            )))
        return

    av = refresh(
        dates=args.dates,
        discover=not args.no_discover,
        latest_only=args.latest_only,
        lookback=args.lookback,
        lookahead=args.lookahead,
        skip_predict=args.skip_predict,
        migrate_only=args.migrate_only,
        odds_only=args.odds_only,
        include_odds=not args.no_odds,
        source=args.source,
        venues=args.venue,
    )
    print()
    print("=" * 50)
    print("✅ PO-7 データ更新完了")
    print("source:", args.source)
    print("開催日:", ", ".join(av))
    print("runners:", RUNNERS)
    print("predictions:", PRED_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()
