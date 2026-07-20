from flask import Flask,render_template,request
import subprocess,sys,json,re,threading
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
import os
import pandas as pd
from areru_engine import parse_date
from ev_analysis import (
    apply_expected_value,
    build_ai_self_eval,
    day_performance,
    load_score_odds,
)

app=Flask(__name__)
BASE=Path(__file__).resolve().parent
DATA=BASE/'data'; ARCH=DATA/'predictions_by_date'; ARCH.mkdir(parents=True,exist_ok=True)
RUNNERS=DATA/'runners.csv'
LEGACY=DATA/'score_test_data.csv'
ANALYSIS_CSV=DATA/'analysis_result.csv'
JST=timezone(timedelta(hours=9))

# プロセス内キャッシュ / バックグラウンド生成（ページ表示をブロックしない）
_DATES_CACHE={}
_VERIFY_CACHE={}
_PRED_META_CACHE={'sig':None,'data':{}}
_PREDICT_JOBS={}
_PREDICT_JOBS_LOCK=threading.Lock()
# 重い取得・予想は1本ずつ（web + refresh + replay の三重起動を防ぐ）
_HEAVY_JOB_LOCK=threading.Lock()
_HEAVY_JOB_STATE={'name': ''}
_EMPTY_VERIFY={
    'has_data':False,'selected_date':'',
    'total_bets':0,'hit_rate':0.0,'recovery':0.0,'roi':0.0,
    'investment':0,'payout':0,'profit':0,'tone':'roi-bad',
    'daily':[],'by_type':[],'by_rank':[],'by_rank_type':[],'main':{},
    'recovery_series':[],'cum_profit':[],'recent_rows':[],
    'purchase_ranks_by_race':{},
}

# 地方の開催場一覧だけなら巨大JSON列を読まない
_NAR_VENUE_PICKER_COLS=(
    'race_id','source','開催地','レース','勝負ランク','期待値','投資判定',
    '本命','日付','荒れクラス',
)


@app.errorhandler(Exception)
def _unhandled_error(exc):
    """未処理例外でも真っ白にせず、必ず可視HTMLを返す。"""
    # Flask/Werkzeug の HTTPException はそのまま
    try:
        from werkzeug.exceptions import HTTPException
        if isinstance(exc, HTTPException):
            return exc
    except Exception:
        pass
    print(f'[unhandled] {type(exc).__name__}: {exc}')
    html=(
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>ARERU.CLOUD</title>'
        '<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
        'margin:40px auto;max-width:640px;padding:0 16px;color:#17212b;background:#f4f6f8}'
        'a{color:#176b45;font-weight:700}.box{background:#fff;border:1px solid #e1e7eb;'
        'border-radius:12px;padding:16px;margin:16px 0}</style></head><body>'
        '<h1>ARERU.CLOUD</h1>'
        '<div class="box"><p><b>通信エラー</b></p>'
        '<p>表示中にエラーが発生しました。再読み込みするか、トップへ戻ってください。</p>'
        f'<p style="color:#6f7b87;font-size:13px">{type(exc).__name__}</p>'
        '<p><a href="javascript:location.reload()">再読み込み</a> · <a href="/">トップへ戻る</a></p></div>'
        '</body></html>'
    )
    return html, 500


def _fs_sig(*paths):
    """キャッシュ無効化用の簡易シグネチャ（mtime/size）。"""
    parts=[]
    for p in paths:
        try:
            st=Path(p).stat()
            parts.append(f'{st.st_mtime_ns}:{st.st_size}')
        except Exception:
            parts.append('0')
    # predictions ディレクトリの件数変化も拾う
    try:
        parts.append(str(sum(1 for _ in ARCH.glob('predictions_*.csv'))))
        parts.append(str(int(ARCH.stat().st_mtime_ns)))
    except Exception:
        parts.append('0')
    return '|'.join(parts)


def _today_jst() -> str:
    """開催判定用の『本日』。Render(UTC)でも日本時間を使う。"""
    return datetime.now(JST).date().isoformat()

def _runner_path():
    if RUNNERS.exists(): return RUNNERS
    if LEGACY.exists(): return LEGACY
    return None

def dates(source='all'):
    """開催日一覧。runners.csv を正とし、生成済み predictions も合流する。"""
    rp=_runner_path()
    sig=_fs_sig(rp or Path('.'), ANALYSIS_CSV)
    key=(source,sig)
    hit=_DATES_CACHE.get(key)
    if hit is not None:
        return list(hit)
    found=set()
    if rp is not None:
        try:
            rdf=pd.read_csv(rp,encoding='utf-8-sig')
            if '日付' in rdf.columns:
                if source in ('jra','nar'):
                    if 'source' in rdf.columns:
                        rdf=rdf[rdf['source'].astype(str).str.lower()==source]
                    elif 'race_id' in rdf.columns:
                        from areru_engine import source_from_race_id
                        rdf=rdf[rdf['race_id'].map(source_from_race_id)==source]
                d=parse_date(rdf['日付']).dropna().dt.strftime('%Y-%m-%d')
                found.update(d.unique().tolist())
        except Exception:
            pass
    for f in ARCH.glob('predictions_*.csv'):
        m=re.fullmatch(r'predictions_(\d{4}-\d{2}-\d{2})\.csv', f.name)
        if not m:
            continue
        day=m.group(1)
        if source not in ('jra','nar'):
            found.add(day)
            continue
        # 高速化: 全行読まず source 列の先頭数千行相当だけ usecols
        try:
            pdf=pd.read_csv(f,encoding='utf-8-sig',usecols=lambda c: c in ('source','race_id'))
            if 'source' in pdf.columns:
                if (pdf['source'].astype(str).str.lower()==source).any():
                    found.add(day)
            elif 'race_id' in pdf.columns:
                from areru_engine import source_from_race_id
                if pdf['race_id'].map(source_from_race_id).eq(source).any():
                    found.add(day)
            else:
                found.add(day)
        except Exception:
            found.add(day)
    if ANALYSIS_CSV.exists():
        try:
            ad=pd.read_csv(ANALYSIS_CSV,encoding='utf-8-sig',usecols=lambda c: c in ('date','source')).fillna('')
            if source in ('jra','nar') and 'source' in ad.columns:
                ad=ad[ad['source'].astype(str).str.lower()==source]
            found.update([x for x in ad['date'].astype(str).tolist() if re.fullmatch(r'\d{4}-\d{2}-\d{2}', x)])
        except Exception:
            pass
    out=sorted(found, reverse=True)
    _DATES_CACHE[key]=list(out)
    # 古いキャッシュ肥大化防止
    if len(_DATES_CACHE)>24:
        _DATES_CACHE.clear(); _DATES_CACHE[key]=list(out)
    return out


def _predictions_has_source(pdf, source) -> bool:
    if source not in ('jra','nar'):
        return True
    from areru_engine import source_from_race_id
    if 'source' in pdf.columns:
        return bool((pdf['source'].astype(str).str.lower()==source).any())
    if 'race_id' in pdf.columns:
        return bool(pdf['race_id'].map(source_from_race_id).eq(source).any())
    return False


def _runners_need_source(d, source) -> bool:
    if source not in ('jra','nar'):
        return False
    rp=_runner_path()
    if rp is None:
        return False
    try:
        rdf=pd.read_csv(rp,encoding='utf-8-sig',usecols=lambda c: c in ('日付','source','race_id'))
        day=parse_date(rdf['日付']).dt.strftime('%Y-%m-%d')==d
        if not day.any():
            return False
        if 'source' in rdf.columns:
            return bool((day & (rdf['source'].astype(str).str.lower()==source)).any())
        from areru_engine import source_from_race_id
        return bool(rdf.loc[day,'race_id'].map(source_from_race_id).eq(source).any())
    except Exception:
        return False


def _need_regen(d, source='all') -> bool:
    f=ARCH/f'predictions_{d}.csv'
    if not f.exists():
        return True
    try:
        cols=pd.read_csv(f,encoding='utf-8-sig',nrows=0).columns.tolist()
        if '印データ' not in cols:
            return True
        if source in ('jra','nar'):
            pdf=pd.read_csv(f,encoding='utf-8-sig',usecols=lambda c: c in ('source','race_id'))
            if not _predictions_has_source(pdf, source) and _runners_need_source(d, source):
                return True
    except Exception:
        return True
    return False


def _run_predict_job(d, source='all'):
    """refresh + replay_predict を直列実行（同時多重を避ける）。"""
    key=f'{d}:{source}'
    def _body():
        need_refresh=False
        rp=_runner_path()
        if rp is None:
            need_refresh=True
        else:
            try:
                rdf=pd.read_csv(rp,encoding='utf-8-sig',usecols=lambda c: c in ('日付','source'))
                rd=parse_date(rdf['日付']).dt.strftime('%Y-%m-%d')
                day_mask=rd==d
                if source in ('jra','nar') and 'source' in rdf.columns:
                    need_refresh=not ((day_mask) & (rdf['source'].astype(str).str.lower()==source)).any()
                else:
                    need_refresh=d not in set(rd.dropna().tolist())
            except Exception:
                need_refresh=True
        if need_refresh:
            src=source if source in ('jra','nar','all') else 'all'
            _refresh_then_predict([d], src)
        else:
            subprocess.run([sys.executable,'replay_predict.py',d],check=False,timeout=600)
            _clear_runtime_caches()
        print(f'[predict-job] done {key}')

    try:
        _run_serialized_heavy(f'predict:{key}', _body, wait=True)
    except Exception as e:
        print(f'[predict-job] fail {key}: {e}')
    finally:
        with _PREDICT_JOBS_LOCK:
            _PREDICT_JOBS.pop(key, None)


def _start_predict_job(d, source='all'):
    key=f'{d}:{source}'
    with _PREDICT_JOBS_LOCK:
        if key in _PREDICT_JOBS:
            return False
        _PREDICT_JOBS[key]='running'
    threading.Thread(target=_run_predict_job, args=(d, source), daemon=True).start()
    print(f'[predict-job] start {key}')
    return True


def _read_predictions_for_venue_picker(pred_path, source: str) -> list:
    """開催場一覧用の軽量読み込み（巨大JSON列をスキップ）。"""
    try:
        cols=pd.read_csv(pred_path, encoding='utf-8-sig', nrows=0).columns.tolist()
        use=[c for c in _NAR_VENUE_PICKER_COLS if c in cols]
        if not use:
            return []
        df=pd.read_csv(pred_path, encoding='utf-8-sig', usecols=use).fillna('なし')
        if source in ('jra','nar') and 'source' in df.columns:
            df=df[df['source'].astype(str).str.lower()==source].copy()
        rows=df.to_dict('records')
        return _filter_records_by_source(rows, source)
    except Exception as e:
        print(f'[venue-picker] light read fail: {e}')
        return []


def _clear_runtime_caches():
    """runners / predictions 更新後に日付・検証キャッシュを捨てる。"""
    _DATES_CACHE.clear()
    _VERIFY_CACHE.clear()
    _PRED_META_CACHE['sig']=None
    _PRED_META_CACHE['data']={}


_NAR_JOB_STATUS=DATA/'.nar_job_status.json'
_NAR_JOB_STALE_SEC=45*60  # 取得中のまま放置しない上限


def _write_nar_job_status(
    state: str,
    stage: str = '',
    message: str = '',
    date_str: str = '',
    error: str = '',
) -> None:
    """地方取得ジョブの進捗を永続化（UIが取得中のまま固まらないようにする）。"""
    payload={
        'state': state,  # running | success | error | idle
        'stage': stage,
        'message': message,
        'date': date_str or '',
        'error': error or '',
        'updated_at': datetime.now(JST).isoformat(timespec='seconds'),
        'pid': os.getpid(),
    }
    try:
        _NAR_JOB_STATUS.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    except Exception as e:
        print(f'[nar-job] status write fail: {e}', flush=True)
    print(f'[nar-job] {state} | {stage} | {message or error}', flush=True)


def _read_nar_job_status() -> dict:
    try:
        if not _NAR_JOB_STATUS.exists():
            return {}
        data=json.loads(_NAR_JOB_STATUS.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _nar_job_age_sec(st: dict) -> float:
    raw=str(st.get('updated_at') or '')
    if not raw:
        return 1e9
    try:
        ts=datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts=ts.replace(tzinfo=JST)
        return max(0.0, (datetime.now(JST)-ts).total_seconds())
    except Exception:
        return 1e9


def _clear_stale_nar_locks() -> None:
    """死んだロックを掃除して取得中の永久表示を防ぐ。"""
    for p in DATA.glob('.nar_*.lock'):
        try:
            age=__import__('time').time()-p.stat().st_mtime
            if age > _NAR_JOB_STALE_SEC:
                p.unlink(missing_ok=True)
                print(f'[nar-job] stale lock removed: {p.name}', flush=True)
        except Exception:
            pass


def resolve_nar_fetch_status(selected_date: str = '', *, force_refresh: bool = False) -> tuple:
    """UI用: (data_status, message)。

    - running（新鮮）→ generating / データ取得中
    - success → ready（取得中を解除）
    - error → error / 取得失敗
    - running が古すぎる → error に落としてロック掃除
    """
    _clear_stale_nar_locks()
    st=_read_nar_job_status()
    state=str(st.get('state') or 'idle')
    age=_nar_job_age_sec(st)
    stage=str(st.get('stage') or '')
    msg=str(st.get('message') or '')
    err=str(st.get('error') or '')
    job_date=str(st.get('date') or '')

    # 取得中のまま放置されたジョブは失敗扱いに確定
    if state=='running' and age > _NAR_JOB_STALE_SEC:
        _write_nar_job_status(
            'error', stage='timeout', message='取得タイムアウト',
            date_str=job_date, error='取得がタイムアウトしました',
        )
        _clear_stale_nar_locks()
        return 'error', '取得失敗'

    if state=='running':
        label=msg or 'データ取得中'
        return 'generating', label

    if state=='error':
        # 強制再取得直後以外は失敗を表示
        if force_refresh and age < 5:
            return 'generating', 'データ取得中'
        return 'error', '取得失敗' + (f'（{err}）' if err and len(err) < 80 else '')

    if state=='success':
        # 成功後は取得中を必ず解除（force_refresh が URL に残っていても）
        if selected_date and job_date and selected_date != job_date:
            pass  # 別日の成功ステータス
        else:
            return 'ready', msg or '取得完了'

    # ロックが生きていれば取得中
    for name in ('.nar_today_pipeline.lock', '.nar_bootstrap.lock'):
        lp=DATA/name
        if lp.exists():
            try:
                if __import__('time').time()-lp.stat().st_mtime < _NAR_JOB_STALE_SEC:
                    return 'generating', 'データ取得中'
            except Exception:
                pass
    return 'ready', ''


def _run_serialized_heavy(name: str, fn, *, wait: bool = False) -> bool:
    """重いジョブを1本化。busy時はスキップ（wait=Trueなら完了待ち）。"""
    timeout=1800 if wait else 0
    acquired=_HEAVY_JOB_LOCK.acquire(timeout=timeout) if wait else _HEAVY_JOB_LOCK.acquire(blocking=False)
    if not acquired:
        busy=_HEAVY_JOB_STATE.get('name') or '?'
        print(f'[heavy] busy ({busy}), skip {name}', flush=True)
        return False
    _HEAVY_JOB_STATE['name']=name
    try:
        print(f'[heavy] start {name}', flush=True)
        fn()
        print(f'[heavy] done {name}', flush=True)
        return True
    except Exception as e:
        print(f'[heavy] fail {name}: {e}', flush=True)
        raise
    finally:
        _HEAVY_JOB_STATE['name']=''
        _HEAVY_JOB_LOCK.release()


def _refresh_then_predict(
    dates: list[str],
    source: str,
    extra_args: list[str] | None = None,
) -> bool:
    """開催取得 → レース取得 → AI予想。各段階をログ＆状態ファイルに残す。

    Returns: 成功なら True
    """
    days=[d for d in dates if d and re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(d))]
    if not days:
        _write_nar_job_status('error', stage='start', message='取得失敗', error='対象日なし')
        return False
    src=source if source in ('jra','nar','all') else 'all'
    primary=days[0]
    try:
        # 1) 取得開始
        _write_nar_job_status(
            'running', stage='start', message='データ取得中',
            date_str=primary,
        )
        print(f'[nar-job] 取得開始 dates={days} source={src}', flush=True)

        cmd=[
            sys.executable,'refresh_data.py',
            '--dates',*days,
            '--source',src,
            '--no-discover',
            '--skip-predict',
        ]
        if extra_args:
            cmd.extend(extra_args)
        print(f'[nar-job] 開催・レース取得開始: {" ".join(cmd)}', flush=True)
        rc=subprocess.run(cmd, check=False, timeout=1800)
        if rc.returncode != 0:
            raise RuntimeError(f'refresh_data 終了コード {rc.returncode}')

        # 2) 開催取得完了 / レース取得完了
        _write_nar_job_status(
            'running', stage='venues_done', message='開催取得完了',
            date_str=primary,
        )
        print('[nar-job] 開催取得完了', flush=True)
        _write_nar_job_status(
            'running', stage='races_done', message='レース取得完了',
            date_str=primary,
        )
        print('[nar-job] レース取得完了', flush=True)
        _clear_runtime_caches()

        # 3) AI予想
        for d in days:
            _write_nar_job_status(
                'running', stage='predict_start', message='AI予想生成中',
                date_str=d,
            )
            print(f'[nar-job] AI予想開始: {d}', flush=True)
            pr=subprocess.run([sys.executable,'replay_predict.py',d], check=False, timeout=600)
            if pr.returncode != 0:
                raise RuntimeError(f'replay_predict 終了コード {pr.returncode} ({d})')
            pred=ARCH/f'predictions_{d}.csv'
            if not pred.exists():
                raise RuntimeError(f'predictions_{d}.csv が生成されませんでした')
            print(f'[nar-job] AI予想完了: {d}', flush=True)
            _write_nar_job_status(
                'running', stage='predict_done', message='AI予想完了',
                date_str=d,
            )

        _clear_runtime_caches()
        _write_nar_job_status(
            'success', stage='done', message='取得完了',
            date_str=primary,
        )
        print('[nar-job] 取得完了（成功）', flush=True)
        return True
    except Exception as e:
        _clear_runtime_caches()
        _write_nar_job_status(
            'error', stage='failed', message='取得失敗',
            date_str=primary, error=str(e)[:200],
        )
        print(f'[nar-job] 取得失敗: {e}', flush=True)
        return False


def ensure_for_page(d, source='all'):
    """ページ表示用。同期再生成はしない。 (path|None, status)

    status: ready | updating | generating | error
    """
    f=ARCH/f'predictions_{d}.csv'
    try:
        if f.exists() and not _need_regen(d, source):
            return f, 'ready'
        if _need_regen(d, source):
            _start_predict_job(d, source)
            if f.exists():
                # 既存ファイルがあれば表示しつつ更新中
                return f, 'updating'
            return None, 'generating'
        return f, 'ready'
    except Exception as e:
        print(f'[ensure_for_page] {d}: {e}')
        if f.exists():
            return f, 'ready'
        return None, 'error'


def ensure(d, source='all'):
    """互換用。ページでは使わず、明示更新時のみ同期実行可。"""
    f=ARCH/f'predictions_{d}.csv'
    if not _need_regen(d, source):
        return f
    # 同期は重いのでジョブ起動＋既存があればそれを返す
    _start_predict_job(d, source)
    if f.exists():
        return f
    # ファイルが全く無いときだけ短時間待機（最大8秒）
    for _ in range(16):
        if f.exists():
            return f
        threading.Event().wait(0.5)
    raise FileNotFoundError(f'predictions_{d}.csv を生成中です。しばらくして再読み込みしてください。')

def _filter_records_by_source(records, source):
    if source not in ('jra','nar') or not records:
        return records
    from areru_engine import source_from_race_id
    out=[]
    for r in records:
        src=str(r.get('source') or '').strip().lower()
        if src not in ('jra','nar'):
            src=source_from_race_id(r.get('race_id',''))
        if src==source:
            out.append(r)
    return out


def _venue_meetings(records):
    """日付内の開催場一覧（レース数・S/A件数付き）。"""
    from netkeiba_client import normalize_venue_name
    buckets={}
    for r in records or []:
        venue=normalize_venue_name(str(r.get('開催地') or '').strip())
        if not venue:
            continue
        r['開催地']=venue
        buckets.setdefault(venue, []).append(r)
    meetings=[]
    for venue, rows in sorted(buckets.items(), key=lambda x: x[0]):
        try:
            race_nos=sorted({int(float(x.get('レース') or 0)) for x in rows if x.get('レース') not in ('',None)})
        except Exception:
            race_nos=[]
        ranks={}
        for x in rows:
            rk=str(x.get('勝負ランク') or '').upper()
            if rk:
                ranks[rk]=ranks.get(rk,0)+1
        meetings.append({
            'name':venue,
            'count':len(rows),
            'race_nos':race_nos,
            'race_label':f'{min(race_nos)}〜{max(race_nos)}R' if race_nos else f'{len(rows)}R',
            's':ranks.get('S',0),
            'a':ranks.get('A',0),
            'b':ranks.get('B',0),
            'c':ranks.get('C',0),
        })
    return meetings


def _pick_today_date(available, today_str=''):
    """本日開催日を解決。当日が無ければ直近の開催日へ。"""
    today_str=str(today_str or _today_jst())
    av=list(available or [])
    if not av:
        return ''
    if today_str in av:
        return today_str
    past=[d for d in av if d<=today_str]
    if past:
        return max(past)
    return min(av)


def _anchor_meeting_date(found, today_str=''):
    """検出開催日から『本日優先・無ければ直近過去』のアンカー日を返す。未来日は選ばない。"""
    today_str=str(today_str or _today_jst())
    days=[str(d) for d in (found or []) if re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(d))]
    if not days:
        return ''
    if today_str in days:
        return today_str
    past=[d for d in days if d<=today_str]
    if past:
        return max(past)
    # すべて未来なら最も近い未来日（カード先行取得用）
    return min(days)


def _result_available_dates(meeting_dates, result_days, today_str=''):
    """結果検証用の日付一覧。開催日(≦本日)と結果確定日を合流し、新しい順。"""
    today_str=str(today_str or _today_jst())
    meet=[d for d in (meeting_dates or []) if d<=today_str]
    res=list(result_days or [])
    return sorted(set(meet) | set(res), reverse=True)


def _prediction_race_ids(date_str: str, source: str = 'all') -> set[str]:
    """predictions CSV からその日の race_id 集合。"""
    path=ARCH/f'predictions_{date_str}.csv'
    if not path.exists():
        return set()
    try:
        pdf=pd.read_csv(path).fillna('')
    except Exception:
        return set()
    if 'race_id' not in pdf.columns:
        return set()
    if source in ('jra','nar') and 'source' in pdf.columns:
        pdf=pdf[pdf['source'].astype(str).str.lower()==source]
    out=set()
    for x in pdf['race_id'].tolist():
        rid=_norm_race_id(x)
        if rid.isdigit() and len(rid)==12:
            out.add(rid)
    return out


def _result_race_ids(date_str: str, source: str = 'all') -> set[str]:
    """results.csv に着順が入っている race_id 集合。"""
    rp=DATA/'results.csv'
    if not rp.exists():
        return set()
    try:
        rdf=pd.read_csv(rp,encoding='utf-8-sig').fillna('')
    except Exception:
        return set()
    if rdf.empty or 'race_id' not in rdf.columns:
        return set()
    if source in ('jra','nar') and 'source' in rdf.columns:
        rdf=rdf[rdf['source'].astype(str).str.lower()==source]
    if 'date' in rdf.columns:
        rdf=rdf[rdf['date'].astype(str)==str(date_str)]
    if '着順' in rdf.columns:
        rdf=rdf[rdf['着順'].astype(str).str.match(r'^\d')]
    out=set()
    for x in rdf['race_id'].tolist():
        rid=_norm_race_id(x)
        if rid:
            out.add(rid)
    return out


def date_needs_result_fetch(date_str: str, source: str = 'jra') -> bool:
    """予想レースに対して結果が欠けていれば True（部分取得済み日も再取得）。"""
    if not date_str:
        return False
    expected=_prediction_race_ids(date_str, source if source!='all' else 'all')
    have=_result_race_ids(date_str, source if source!='all' else 'all')
    if expected:
        missing=expected-have
        if missing:
            print(
                f'[bootstrap-results] incomplete {date_str}: '
                f'have={len(have)}/{len(expected)} missing={len(missing)}'
            )
            return True
        return False
    # 予想が無い日は「結果日に一度も無い」場合のみ
    return date_str not in set(dates_with_results(source))


def _local_runner_race_ids(date_str: str, source: str = 'nar') -> set[str]:
    """runners.csv から指定日・ソースの race_id 集合。"""
    rp=_runner_path()
    if rp is None or not date_str:
        return set()
    try:
        rdf=pd.read_csv(rp,encoding='utf-8-sig')
    except Exception:
        return set()
    if rdf.empty or 'race_id' not in rdf.columns or '日付' not in rdf.columns:
        return set()
    days=parse_date(rdf['日付']).dt.strftime('%Y-%m-%d')
    rdf=rdf[days==str(date_str)]
    if source in ('jra','nar'):
        if 'source' in rdf.columns:
            rdf=rdf[rdf['source'].astype(str).str.lower()==source]
        else:
            from areru_engine import source_from_race_id
            rdf=rdf[rdf['race_id'].map(source_from_race_id)==source]
    out=set()
    for x in rdf['race_id'].tolist():
        rid=_norm_race_id(x)
        if rid.isdigit() and len(rid)==12:
            out.add(rid)
    return out


def date_needs_runners_fetch(date_str: str, source: str = 'nar') -> bool:
    """リモート開催レースとローカル runners を比較し、欠けがあれば True。"""
    if not date_str or source not in ('jra','nar'):
        return False
    try:
        from netkeiba_client import NetkeibaClient
        remote=set(NetkeibaClient(sleep=0.08).list_race_ids(
            date_str.replace('-',''), source=source
        ))
    except Exception as e:
        print(f'[bootstrap] list_race_ids fail {date_str}: {e}', flush=True)
        return False
    if not remote:
        return False
    local=_local_runner_race_ids(date_str, source)
    missing=remote-local
    if missing:
        print(
            f'[bootstrap] incomplete card {source} {date_str}: '
            f'local={len(local)}/{len(remote)} missing={len(missing)}',
            flush=True,
        )
        return True
    # 予想未生成も再取得トリガ（カードはあるが predictions なし）
    pred=_prediction_race_ids(date_str, source)
    if local and not pred:
        print(f'[bootstrap] predictions missing {source} {date_str}', flush=True)
        return True
    return False

def bootstrap_missing_results(source: str = 'jra', prefer_dates: list | None = None) -> bool:
    """結果未取込・途中止まりの開催日をバックグラウンド取得。

    - 当日も含める（昼過ぎ以降の残りR更新のため）
    - 1レースでも結果があると完了扱いにしない（欠けがあれば再取得）
    """
    if source not in ('jra','nar','all'):
        return False
    today=_today_jst()
    meet=dates(source)
    candidates=[]
    # 明示日（結果タブで開いている日）を最優先
    for d in (prefer_dates or []):
        d=str(d or '').strip()
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d) and d<=today:
            candidates.append(d)
    for d in meet:
        if d<=today:
            candidates.append(d)
    # 新しい順・重複除去
    ordered=[]
    seen=set()
    for d in sorted(set(candidates), reverse=True):
        if d in seen:
            continue
        seen.add(d)
        ordered.append(d)
    missing=[d for d in ordered if date_needs_result_fetch(d, source)][:3]
    if not missing:
        return False
    lock=DATA/f'.results_bootstrap_{source}.lock'
    if lock.exists():
        try:
            age=(__import__('time').time()-lock.stat().st_mtime)
            # 当日の途中更新は短めクールダウン
            cooldown=300 if today in missing else 900
            if age < cooldown:
                return False
        except Exception:
            pass
    print(f'[bootstrap-results] source={source} missing={missing}', flush=True)
    try:
        lock.write_text(str(__import__('os').getpid()), encoding='utf-8')
        cmd=[sys.executable,'results.py','--source',source,'--dates',*missing]
        # タイムアウトしても results.py 側の増分保存分は残る
        subprocess.run(cmd,check=False,timeout=1800)
    except Exception as e:
        print(f'[bootstrap-results] fail: {e}', flush=True)
        return False
    finally:
        try: lock.unlink(missing_ok=True)
        except Exception: pass
    print(f'[bootstrap-results] done source={source} dates={missing}', flush=True)
    return True


def _norm_ban(x) -> str:
    """馬番を表示用の整数文字列へ。欠損は空文字。"""
    s=str(x or '').strip()
    if not s or s.lower() in ('nan','none','なし'):
        return ''
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _main_ban_map(selected_date: str) -> dict:
    """scores CSV から (race_id, 正規化馬名) → 馬番 を構築。"""
    p=ARCH/f'scores_{selected_date}.csv'
    if not p.exists():
        return {}
    try:
        sdf=pd.read_csv(p).fillna('')
    except Exception:
        return {}
    if 'race_id' not in sdf.columns or '馬名' not in sdf.columns or '馬番' not in sdf.columns:
        return {}
    m={}
    for _, row in sdf.iterrows():
        rid=_norm_race_id(row.get('race_id',''))
        name=clean_horse(row.get('馬名',''))
        ban=_norm_ban(row.get('馬番',''))
        if rid and name and ban:
            m[(rid, name)]=ban
    return m


def _json_field(raw, default=None):
    if default is None:
        default=[]
    if isinstance(raw, (dict, list)):
        return raw
    s=str(raw or '').strip()
    if not s or s.lower() in ('nan','none','なし'):
        return default
    try:
        # pandas 由来の NaN を JSON として壊さない
        s=s.replace('NaN','null').replace('Infinity','null')
        return json.loads(s)
    except Exception:
        return default


def apply_display_ranks(races: list, by_venue: bool = False) -> list:
    """表示用ランクを期待回収率から決定（相対順位は使わない）。

    S≥120 / A≥115 / B≥110 / C≥105 / D≥100 / 見送り<100
    by_venue は互換のため残すが、閾値は絶対値のため場内相対は行わない。
    """
    from ev_analysis import apply_ev_rank_and_labels, build_ai_buy_reasons
    for r in races or []:
        apply_ev_rank_and_labels(r)
        if not r.get('AI買い理由'):
            r['AI買い理由'] = build_ai_buy_reasons(r, limit=3)
    return races


def build_buy_candidates(races: list, limit: int = 12) -> list:
    """期待回収率100%以上を期待値順（本日の買い候補）。"""
    scored = []
    for r in races or []:
        try:
            ev = float(r.get('期待値') if r.get('期待値') is not None else str(r.get('レース期待回収率') or '').replace('%', '') or 0)
        except (TypeError, ValueError):
            continue
        if ev < 100:
            continue
        scored.append((ev, r))
    scored.sort(key=lambda x: (-x[0], str(x[1].get('開催地') or ''), int(float(x[1].get('レース') or 0) or 0)))
    return [r for _, r in scored[:limit]]


def build_today_ai_board(races: list, verification: dict | None = None) -> dict:
    """ヘッダー直下の本日AI成績カード。"""
    races = races or []
    total = len(races)
    buys = 0
    for r in races:
        try:
            ev = float(r.get('期待値') if r.get('期待値') is not None else 0)
        except (TypeError, ValueError):
            ev = 0
        if ev >= 100 or str(r.get('投資判定') or '').startswith('買い'):
            buys += 1
    verification = verification or {}
    recovery = verification.get('recovery') if verification.get('has_data') else None
    hit = verification.get('hit_rate') if verification.get('has_data') else None
    if verification.get('scope') != 'day' and verification.get('has_data'):
        day = str(verification.get('selected_date') or '')
        for row in verification.get('daily') or []:
            if str(row.get('date')) == day:
                recovery = row.get('recovery')
                hit = row.get('hit_rate', hit)
                break
    return {
        'has_data': total > 0,
        '回収率': recovery,
        '回収率表示': f'{float(recovery):.0f}%' if recovery is not None else '—',
        '的中率': hit,
        '的中率表示': f'{float(hit):.0f}%' if hit is not None else '—',
        '買いレース': f'{buys}/{total}' if total else '0/0',
        '買い数': buys,
        '総数': total,
        'tone': 'roi-good' if (recovery is not None and float(recovery) >= 100) else (
            'roi-mid' if recovery is not None else 'roi-bad'
        ),
    }


def prep(records, ban_map=None):
    from areru_engine import RANK_LABELS, RANK_CLASSES
    from race_sim import circle_ban
    ban_map=ban_map or {}
    for r in records:
        try: r['印一覧']=json.loads(str(r.get('印データ','[]')).replace('NaN','null'))
        except: r['印一覧']=[]
        if not isinstance(r['印一覧'], list):
            r['印一覧']=[]
        for k in ['ワイド買い目','馬連買い目','三連複買い目','馬単買い目','三連単買い目']:
            r[k+'一覧']=str(r.get(k,'見送り')).split('｜')
        cards=_json_field(r.get('ピックカード'), [])
        r['ピックカード一覧']=[c for c in cards if isinstance(c, dict)][:6]
        pace=_json_field(r.get('展開予想'), {})
        r['展開予想データ']=pace if isinstance(pace, dict) else {}
        tickets=_json_field(r.get('推奨馬券'), [])
        clean_tickets=[]
        for t in tickets if isinstance(tickets, list) else []:
            if not isinstance(t, dict):
                continue
            form=t.get('フォーメーション')
            if form is not None and not isinstance(form, dict):
                t=dict(t); t['フォーメーション']=None
            clean_tickets.append(t)
        r['推奨馬券一覧']=clean_tickets[:8]
        danger=_json_field(r.get('危険人気詳細'), {})
        r['危険人気カード']=danger if isinstance(danger, dict) else {}
        mainc=_json_field(r.get('本命詳細'), {})
        r['本命カード']=mainc if isinstance(mainc, dict) else {}
        for kind in ('ワイド','馬連','馬単','三連複','三連単'):
            plan=_json_field(r.get(kind+'詳細'), {})
            r[kind+'プラン']=plan if isinstance(plan, dict) else {}
        rank=str(r.get('勝負ランク','') or '').upper()
        if rank in RANK_LABELS:
            r['勝負ランク']=rank
            r['BET判定']=RANK_LABELS[rank]
            r['BETクラス']=RANK_CLASSES.get(rank, r.get('BETクラス',''))
        ban=_norm_ban(r.get('本命馬番',''))
        if not ban and ban_map:
            key=(_norm_race_id(r.get('race_id','')), clean_horse(r.get('本命','')))
            ban=ban_map.get(key, '')
        r['本命馬番']=ban
        r['本命馬番表示']=r.get('本命馬番表示') or (circle_ban(ban) if ban else '')
        horse=str(r.get('本命') or '').strip()
        if horse.lower() in ('nan','none','なし'):
            horse=''
        # 一覧・詳細共通: 馬番＋馬名
        if ban and horse:
            r['本命表示']=f'{ban}番 {horse}'
        elif ban:
            r['本命表示']=f'{ban}番'
        elif horse:
            r['本命表示']=horse
        else:
            r['本命表示']='—'
        r['レース名表示']=(
            f"{r.get('開催地','')} {int(float(r['レース'])):02d}R"
            if str(r.get('レース','')).replace('.','',1).isdigit()
            else str(r.get('開催地') or 'レース')
        )
        # 投資判定のフォールバック（apply_expected_value 後に EV ランクで上書き）
        if not r.get('投資判定'):
            try:
                ev=float(str(r.get('レース期待回収率') or '').replace('%',''))
                if ev>=100:
                    r['投資判定']='買い'; r['投資判定アイコン']='🟢'; r['投資判定トーン']='buy'
                else:
                    r['投資判定']='見送り'; r['投資判定アイコン']='🔴'; r['投資判定トーン']='skip'
            except Exception:
                r['投資判定']=r.get('投資判定') or '判定待ち'
                r['投資判定アイコン']=r.get('投資判定アイコン') or '⚪'
                r['投資判定トーン']=r.get('投資判定トーン') or 'wait'
        apply_expected_value(r)
        # ピックカードにも馬番＋馬名を付与（詳細は予想馬3頭のみ表示）
        for c in r.get('ピックカード一覧') or []:
            if not isinstance(c, dict):
                continue
            cb=str(c.get('馬番') or '').strip()
            cn=str(c.get('馬名') or '').strip()
            cban=c.get('馬番表示') or (circle_ban(cb) if cb else '')
            c['表示名']=f"{cb}番 {cn}".strip() if cb and cn else (cn or cban or '—')
            c['馬番表示']=cban or cb
        # 一覧・詳細の共通ラベルを保証
        if r.get('投資判定') in ('見送りレース','買いレース'):
            r['投資判定']='見送り' if '見送' in str(r.get('投資判定')) else '買い'
        if not r.get('投資判定表示'):
            r['投資判定表示']=r.get('投資判定') or '判定待ち'
        if not r.get('予想馬'):
            from pick_rationale import build_display_picks
            r['予想馬']=build_display_picks(r)
        if not r.get('本命短表示'):
            r['本命短表示']=(r.get('予想馬') or [{}])[0].get('表示行') or r.get('本命表示') or '—'
    return records

def clean_horse(x):
    """馬名正規化（areru_engine.clean_name と同等）。"""
    from areru_engine import clean_name
    return clean_name(x)


def _norm_race_id(x) -> str:
    """race_id を比較可能な文字列へ。float の .0 や空白を除去。"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ''
    s=str(x).strip()
    if not s or s.lower() in ('nan','none','なし'):
        return ''
    if s.endswith('.0') and s[:-2].replace('-','').isdigit():
        s=s[:-2]
    try:
        if re.fullmatch(r'\d+\.0+', s):
            s=str(int(float(s)))
    except Exception:
        pass
    return s


def _format_finish(raw) -> str:
    """着順を『1着』形式へ。未確定は空文字。"""
    s=str(raw or '').strip()
    if not s or s.lower() in ('nan','none','なし','結果待ち'):
        return ''
    if s.endswith('着'):
        return s
    try:
        n=int(float(s))
        if n>0:
            return f'{n}着'
    except Exception:
        pass
    # 除外・中止などはそのまま
    return s


_CIRCLED_FINISH='①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'


def _finish_circled(fin: str) -> str:
    """着順表示を一覧用の丸数字へ（例: 3着→③）。"""
    s=str(fin or '').strip()
    if not s or s=='結果待ち':
        return '－'
    if s in ('取消','除外','中止'):
        return '×'
    m=re.match(r'(\d+)', s)
    if not m:
        return '？'
    n=int(m.group(1))
    if 1<=n<=len(_CIRCLED_FINISH):
        return _CIRCLED_FINISH[n-1]
    return str(n)


def _race_date(record) -> str:
    for k in ('日付','_date','date'):
        v=str(record.get(k,'') or '').strip()
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', v):
            return v
    return ''


def _load_score_finishes(date_str: str) -> dict:
    """scores_{date}.csv の 実着順 → {(race_id, 馬名): 着順}"""
    if not date_str:
        return {}
    path=ARCH/f'scores_{date_str}.csv'
    if not path.exists():
        return {}
    try:
        sdf=pd.read_csv(path).fillna('')
    except Exception:
        return {}
    if '馬名' not in sdf.columns or '実着順' not in sdf.columns:
        return {}
    out={}
    for _,x in sdf.iterrows():
        fin=_format_finish(x.get('実着順',''))
        if not fin:
            continue
        rid=_norm_race_id(x.get('race_id',''))
        name=clean_horse(x.get('馬名',''))
        if rid and name:
            out[(rid,name)]=fin
        # 開催地+R フォールバック用キーは attach 側で日付付き辞書に載せる
        venue=str(x.get('開催地','') or '')
        try: rn=int(float(x.get('レース',0)))
        except Exception: rn=None
        if venue and rn is not None and name:
            out[(f'date:{date_str}',venue,rn,name)]=fin
    return out


def _load_analysis_by_race() -> dict:
    """analysis_result.csv → race_id ごとの的中サマリー。"""
    if not ANALYSIS_CSV.exists():
        return {}
    try:
        adf=pd.read_csv(ANALYSIS_CSV,encoding='utf-8-sig').fillna('')
    except Exception:
        return {}
    if adf.empty or 'race_id' not in adf.columns:
        return {}
    by_race={}
    for _,row in adf.iterrows():
        rid=_norm_race_id(row.get('race_id',''))
        if not rid:
            continue
        hit=int(pd.to_numeric(row.get('hit'), errors='coerce') or 0)
        by_race.setdefault(rid, []).append({
            'bet_type':str(row.get('bet_type','')),
            'hit':hit,
            'result':str(row.get('result','') or ''),
            'prediction':str(row.get('prediction','') or ''),
        })
    return by_race


def dates_with_results(source='all') -> list[str]:
    """results.csv / analysis_result.csv にある開催日（新しい順）。"""
    found=set()
    rp=DATA/'results.csv'
    if rp.exists():
        try:
            rdf=pd.read_csv(rp,encoding='utf-8-sig').fillna('')
            if source in ('jra','nar') and 'source' in rdf.columns:
                rdf=rdf[rdf['source'].astype(str).str.lower()==source]
            col='date' if 'date' in rdf.columns else ('日付' if '日付' in rdf.columns else None)
            if col:
                found.update([x for x in rdf[col].astype(str) if re.fullmatch(r'\d{4}-\d{2}-\d{2}', x)])
        except Exception:
            pass
    if ANALYSIS_CSV.exists():
        try:
            ad=pd.read_csv(ANALYSIS_CSV,encoding='utf-8-sig').fillna('')
            if source in ('jra','nar') and 'source' in ad.columns:
                ad=ad[ad['source'].astype(str).str.lower()==source]
            if 'date' in ad.columns:
                found.update([x for x in ad['date'].astype(str) if re.fullmatch(r'\d{4}-\d{2}-\d{2}', x)])
        except Exception:
            pass
    return sorted(found, reverse=True)


def _source_latest_in_runners(source: str) -> str:
    """runners.csv 上の指定ソース最新開催日。"""
    rp=_runner_path()
    if rp is None:
        return ''
    try:
        rdf=pd.read_csv(rp,encoding='utf-8-sig')
        if '日付' not in rdf.columns:
            return ''
        if source in ('jra','nar'):
            if 'source' in rdf.columns:
                rdf=rdf[rdf['source'].astype(str).str.lower()==source]
            elif 'race_id' in rdf.columns:
                from areru_engine import source_from_race_id
                rdf=rdf[rdf['race_id'].map(source_from_race_id)==source]
        days=parse_date(rdf['日付']).dropna().dt.strftime('%Y-%m-%d')
        vals=sorted(days.unique().tolist(), reverse=True)
        return vals[0] if vals else ''
    except Exception:
        return ''


def bootstrap_venue(date_str: str, venue: str, source: str = 'nar') -> bool:
    """失敗した開催場だけ再取得。他開催場の runners は merge で維持する。"""
    if source != 'nar' or not date_str or not venue:
        return False
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(date_str)):
        return False
    from netkeiba_client import normalize_venue_name
    venue=normalize_venue_name(str(venue).strip())
    if not venue:
        return False
    safe=re.sub(r'[^\w\-]+', '_', venue, flags=re.UNICODE)[:48] or 'venue'
    lock=DATA/f'.nar_venue_{date_str}_{safe}.lock'
    if lock.exists():
        try:
            age=(__import__('time').time()-lock.stat().st_mtime)
            if age < 180:
                print(f'[bootstrap-venue] already running {venue}, skip')
                return False
        except Exception:
            pass

    def _body():
        lock.write_text(str(os.getpid()), encoding='utf-8')
        try:
            print(f'[bootstrap-venue] {date_str} venue={venue}', flush=True)
            ok=_refresh_then_predict([date_str], 'nar', extra_args=['--venue', venue])
            if not ok:
                raise RuntimeError(f'{venue} の取得に失敗しました')
        finally:
            try: lock.unlink(missing_ok=True)
            except Exception: pass

    try:
        return _run_serialized_heavy(f'venue:{date_str}:{venue}', _body, wait=False)
    except Exception as e:
        _write_nar_job_status(
            'error', stage='failed', message='取得失敗',
            date_str=date_str, error=str(e)[:200],
        )
        return False


def run_nar_today_pipeline(force: bool = False) -> bool:
    """本日開催パイプライン: 開催取得 → レース取得 → AI予想生成（直列）。"""
    today=_today_jst()
    lock=DATA/'.nar_today_pipeline.lock'
    if lock.exists():
        try:
            age=(__import__('time').time()-lock.stat().st_mtime)
            cooldown=45 if force else 120
            if age < cooldown:
                st=_read_nar_job_status()
                if str(st.get('state'))=='running' and _nar_job_age_sec(st) < _NAR_JOB_STALE_SEC:
                    print('[nar-today] already running, skip')
                    return False
                # ロックだけ残っている → 掃除して続行可に
                try: lock.unlink(missing_ok=True)
                except Exception: pass
            elif age >= _NAR_JOB_STALE_SEC:
                try: lock.unlink(missing_ok=True)
                except Exception: pass
        except Exception:
            pass

    def _body():
        lock.write_text(str(os.getpid()), encoding='utf-8')
        try:
            print(f'[nar-today] pipeline start force={force} date={today}', flush=True)
            ok=_refresh_then_predict([today], 'nar')
            if not ok:
                raise RuntimeError('本日開催パイプライン失敗')
            print('[nar-today] pipeline done', flush=True)
        finally:
            try: lock.unlink(missing_ok=True)
            except Exception: pass

    try:
        started=_run_serialized_heavy(f'nar-today:{today}', _body, wait=bool(force))
        if not started:
            # 他ジョブ実行中なら取得中のまま。そうでなければ失敗確定（永久取得中を防ぐ）
            if _HEAVY_JOB_STATE.get('name'):
                st=_read_nar_job_status()
                if str(st.get('state'))!='running':
                    _write_nar_job_status(
                        'running', stage='queued', message='データ取得中',
                        date_str=today,
                    )
                return False
            _write_nar_job_status(
                'error', stage='failed', message='取得失敗',
                date_str=today, error='取得ジョブを開始できませんでした',
            )
            return False
        return True
    except Exception as e:
        try: lock.unlink(missing_ok=True)
        except Exception: pass
        _write_nar_job_status(
            'error', stage='failed', message='取得失敗',
            date_str=today, error=str(e)[:200],
        )
        return False


def bootstrap_source(source: str) -> bool:
    """地方タブでデータが古い/無い/不完全な場合に最新開催を自動取得する。

    Returns: 更新を実行したら True
    """
    if source != 'nar':
        return False
    today=_today_jst()
    try:
        from netkeiba_client import NetkeibaClient
        found=NetkeibaClient(sleep=0.12).discover_kaisai_dates(
            lookback=4, lookahead=2, source='nar'
        )
        remote=_anchor_meeting_date(found, today)
    except Exception:
        remote=''
        found=[]
    local=_source_latest_in_runners('nar')

    need_dates=[]
    check_days=[]
    if today in set(found):
        check_days.append(today)
    if remote and remote not in check_days:
        check_days.append(remote)
    future=[d for d in found if d > today]
    if future:
        nxt=min(future)
        if nxt not in check_days:
            check_days.append(nxt)
    for d in check_days:
        if date_needs_runners_fetch(d, 'nar'):
            need_dates.append(d)
    if today in set(found) and today not in dates('nar'):
        need_dates.append(today)
    if not found and (not local or local < today):
        need_dates.append(today)

    need_dates=sorted(set(need_dates), reverse=True)
    stale = bool(remote and local and local < remote)
    if not need_dates and not stale:
        return False
    if not need_dates and stale:
        need_dates=[remote] if remote else []

    lock=DATA/'.nar_bootstrap.lock'
    if lock.exists():
        try:
            age=(__import__('time').time()-lock.stat().st_mtime)
            cooldown=300 if need_dates else 1800
            if age < cooldown:
                print('[bootstrap] already running, skip')
                return False
        except Exception:
            pass

    target_dates=need_dates[:3]

    def _body():
        lock.write_text(str(os.getpid()), encoding='utf-8')
        try:
            print(
                f'[bootstrap] source=nar local={local or "-"} remote={remote or "-"} '
                f'need={target_dates} found={found[:5]}',
                flush=True,
            )
            if target_dates:
                ok=_refresh_then_predict(target_dates, 'nar')
                if not ok:
                    raise RuntimeError('bootstrap refresh/predict 失敗')
            else:
                _write_nar_job_status(
                    'running', stage='start', message='データ取得中',
                    date_str=today,
                )
                print('[nar-job] 取得開始 (latest-only)', flush=True)
                cmd=[
                    sys.executable,'refresh_data.py',
                    '--latest-only','--source','nar','--lookback','5','--lookahead','2',
                    '--skip-predict',
                ]
                rc=subprocess.run(cmd, check=False, timeout=1800)
                if rc.returncode != 0:
                    raise RuntimeError(f'latest-only 終了コード {rc.returncode}')
                print('[nar-job] 開催取得完了', flush=True)
                print('[nar-job] レース取得完了', flush=True)
                _write_nar_job_status(
                    'running', stage='races_done', message='レース取得完了',
                    date_str=today,
                )
                _clear_runtime_caches()
                pred_days=sorted({d for d in (today, remote) if d})
                for d in pred_days:
                    print(f'[nar-job] AI予想開始: {d}', flush=True)
                    _write_nar_job_status(
                        'running', stage='predict_start', message='AI予想生成中',
                        date_str=d,
                    )
                    pr=subprocess.run([sys.executable,'replay_predict.py',d], check=False, timeout=600)
                    if pr.returncode != 0:
                        raise RuntimeError(f'replay_predict 終了コード {pr.returncode}')
                    print(f'[nar-job] AI予想完了: {d}', flush=True)
                _clear_runtime_caches()
                _write_nar_job_status(
                    'success', stage='done', message='取得完了',
                    date_str=today,
                )
                print('[nar-job] 取得完了（成功）', flush=True)
        finally:
            try: lock.unlink(missing_ok=True)
            except Exception: pass

    try:
        return _run_serialized_heavy('nar-bootstrap', _body, wait=False)
    except Exception as e:
        try: lock.unlink(missing_ok=True)
        except Exception: pass
        _write_nar_job_status(
            'error', stage='failed', message='取得失敗',
            date_str=today, error=str(e)[:200],
        )
        return False

@app.route('/')
def index():
    source=request.args.get('source','jra')
    if source not in ('jra','nar','all'):
        source='jra'
    mode=request.args.get('mode','predict')
    if mode not in ('predict','result','analysis','ledger'):
        mode='predict'
    today=_today_jst()
    force_refresh=str(request.args.get('force_refresh') or '').strip() in ('1','true','yes')
    want_today=str(request.args.get('today') or '').strip() in ('1','true','yes') or force_refresh
    explicit_date=str(request.args.get('date') or '').strip()
    # 地方: 本日ボタンは開催→レース→予想のフルパイプライン。通常表示は不足分を自動補完
    try:
        if source=='nar':
            if force_refresh or want_today:
                threading.Thread(
                    target=run_nar_today_pipeline, kwargs={'force': True}, daemon=True
                ).start()
            else:
                threading.Thread(target=bootstrap_source, args=('nar',), daemon=True).start()
    except Exception as e:
        print(f'[bootstrap] skip: {e}')
        if source=='nar':
            _write_nar_job_status(
                'error', stage='failed', message='取得失敗',
                date_str=_today_jst(), error=str(e)[:200],
            )
    meeting_dates=dates(source)
    av=list(meeting_dates)
    selected=explicit_date
    # 【1】地方は日付未指定時・本日ボタン時はカレンダー当日へ自動切替（前日残存を防ぐ）
    if source=='nar' and (want_today or (not explicit_date and mode in ('predict','result'))):
        selected=today
        if today not in av:
            av=sorted(set(av)|{today}, reverse=True)
    # ソース切替で他開催の日付が残っていても、そのソースの開催日へ寄せる
    if not selected:
        selected=av[0] if av else (today if source=='nar' else '')
    elif selected not in av:
        if source=='nar' and selected==today:
            av=sorted(set(av)|{today}, reverse=True)
        else:
            selected=av[0] if av else ''
    # 結果検証タブ:
    # - プルダウンは「本日以前の開催日 + 結果確定日」（最新開催日も選択可）
    # - 明示指定日に予想があれば結果未取込でも寄せない（結果待ち表示＋バックグラウンド取得）
    # - 本日開催指定時は結果日へ強制しない
    result_days=dates_with_results(source)
    if mode=='result' and not want_today and explicit_date:
        av=_result_available_dates(meeting_dates, result_days, today)
        if today not in av and source=='nar' and selected==today:
            av=sorted(set(av)|{today}, reverse=True)
        pred_exists=(ARCH/f'predictions_{explicit_date}.csv').exists()
        if explicit_date and (explicit_date in av or pred_exists):
            selected=explicit_date
            if explicit_date not in av:
                av=sorted(set(av)|{explicit_date}, reverse=True)
        else:
            selected=(result_days[0] if result_days else (av[0] if av else ''))
        try:
            prefer=[selected] if selected else []
            threading.Thread(
                target=bootstrap_missing_results,
                kwargs={'source': source, 'prefer_dates': prefer},
                daemon=True,
            ).start()
        except Exception as e:
            print(f'[bootstrap-results] skip: {e}')
    elif mode=='result' and not want_today and not explicit_date and source!='nar':
        av=_result_available_dates(meeting_dates, result_days, today)
        selected=(result_days[0] if result_days else (av[0] if av else ''))
        try:
            prefer=[selected] if selected else []
            threading.Thread(
                target=bootstrap_missing_results,
                kwargs={'source': source, 'prefer_dates': prefer},
                daemon=True,
            ).start()
        except Exception as e:
            print(f'[bootstrap-results] skip: {e}')
    elif mode=='result' and source=='nar' and not want_today and not explicit_date:
        # 地方結果: 当日を優先表示しつつ結果補完
        av=_result_available_dates(meeting_dates, result_days, today)
        if today not in av:
            av=sorted(set(av)|{today}, reverse=True)
        selected=today
        try:
            threading.Thread(
                target=bootstrap_missing_results,
                kwargs={'source': source, 'prefer_dates': [today]},
                daemon=True,
            ).start()
        except Exception as e:
            print(f'[bootstrap-results] skip: {e}')
    # 本日開催では必ず開催場一覧からやり直す
    raw_venue='' if want_today else str(request.args.get('venue') or '').strip()
    selected_venue=''
    if raw_venue:
        try:
            from netkeiba_client import normalize_venue_name
            selected_venue=normalize_venue_name(raw_venue)
        except Exception:
            selected_venue=raw_venue
    races=[]; targets=[]; message='予想データがありません'; has_results=False
    venues=[]; show_venue_picker=False
    data_status='ready'
    buy_candidates=[]; today_ai_board={'has_data':False,'回収率表示':'—','的中率表示':'—','買いレース':'0/0','tone':'roi-bad'}
    data_updated_at=''
    label={'jra':'JRA中央','nar':'地方競馬','all':'全開催'}.get(source, source)

    # モード別に重い集計をスキップ（予想タブでは検証CSVを読まない）
    if mode=='ledger':
        verification=verification_data('', source=source)
    elif mode in ('result','analysis'):
        verification=verification_data(selected, source=source)
    else:
        verification=dict(_EMPTY_VERIFY)

    # 収支タブはレース詳細を組み立てない（高速化）
    if mode=='ledger':
        message=f'{label} / 収支分析'
        return render_template('index.html',races=[],targets=[],selected_date=selected,today=today,
            message=message,available_dates=av,source=source,mode=mode,has_results=False,
            analysis={'total':0,'verified':0,'ranks':[],'bands':[],'venues':[]},
            verification=verification,ledger=ledger_data(source=source, verification=verification),
            venues=[],selected_venue='',show_venue_picker=False,
            today_date=_pick_today_date(meeting_dates, today) if source=='nar' else today,
            day_stats=None,data_status='ready',
            buy_candidates=[],today_ai_board=today_ai_board,data_updated_at='')

    if selected in av:
        try:
            # 地方・当日: ジョブ状態は後段 resolve_nar_fetch_status で確定
            # （force_refresh だけで永久に generating にしない）
            pred_path, page_status = ensure_for_page(selected, source=source)
            data_status=page_status
            if pred_path is None:
                data_status='generating' if page_status!='error' else 'error'
                message='データ取得中' if source=='nar' and page_status!='error' else 'データ取得中です。完了後に再読み込みしてください。'
                if page_status=='error':
                    message='取得失敗' if source=='nar' else '通信エラー: 予想データの準備に失敗しました。再読み込みしてください。'
            else:
                # 地方・開催場未選択: 軽量列だけ読んで一覧を出す（メモリ削減）
                use_light=(
                    source=='nar'
                    and mode in ('predict','result')
                    and not selected_venue
                )
                if use_light:
                    light_rows=_read_predictions_for_venue_picker(pred_path, source)
                    light_rows=apply_display_ranks(light_rows, by_venue=True)
                    for row in light_rows:
                        if not _race_date(row):
                            row['日付']=selected
                    venues=_venue_meetings(light_rows)
                    show_venue_picker=True
                    races=[]
                    races_for_board=[]
                    buy_candidates=[]
                    today_ai_board={'has_data':False,'回収率表示':'—','的中率表示':'—','買いレース':'0/0','tone':'roi-bad'}
                    targets=[]
                    data_updated_at=''
                    try:
                        if pred_path and Path(pred_path).exists():
                            from datetime import datetime as _dt
                            data_updated_at=_dt.fromtimestamp(Path(pred_path).stat().st_mtime, JST).strftime('%m/%d %H:%M')
                    except Exception:
                        data_updated_at=''
                    if data_status=='updating':
                        message=f'{selected} / {label} / データ更新中（表示はキャッシュ）'
                        if data_updated_at:
                            message+=f' · 最終更新 {data_updated_at}'
                    elif data_status=='generating':
                        message='データ取得中'
                    elif venues:
                        if selected==today:
                            message=f'本日開催 {selected} / {label} / 開催場 {len(venues)}場'
                        else:
                            message=f'{selected} / {label} / 開催場 {len(venues)}場'
                    else:
                        if force_refresh or want_today or (DATA/'.nar_today_pipeline.lock').exists() or (DATA/'.nar_bootstrap.lock').exists():
                            # 仮置き。最終的に resolve_nar_fetch_status で上書き
                            data_status='generating'
                            message='データ取得中'
                        elif selected==today:
                            message='本日は地方競馬の開催はありません'
                        else:
                            message=f'{selected} の地方開催データがありません'
                else:
                    # 地方の開催場詳細 / 中央はフル読み
                    df=pd.read_csv(pred_path, encoding='utf-8-sig').fillna('なし')
                    # 先にソース絞り込み（中央タブで地方行まで prep しない）
                    if source in ('jra','nar') and 'source' in df.columns:
                        df=df[df['source'].astype(str).str.lower()==source].copy()
                    # 予想タブは表示に不要な巨大JSON列を落とす（HTML肥大・パース負荷対策）
                    if mode=='predict':
                        drop_cols=[c for c in (
                            'ワイド詳細','馬連詳細','馬単詳細','三連複詳細','三連単詳細','本命詳細'
                        ) if c in df.columns]
                        if drop_cols:
                            df=df.drop(columns=drop_cols, errors='ignore')
                    races=prep(df.to_dict('records'), ban_map=_main_ban_map(selected))
                    races=_filter_records_by_source(races, source)
                    races=apply_display_ranks(races, by_venue=(source=='nar'))
                    for row in races:
                        if not _race_date(row):
                            row['日付']=selected
                    # 結果照合は結果検証タブのみ（予想タブの表示遅延を避ける）
                    if mode=='result':
                        races,has_results=attach_results(races, selected_date=selected)
                        ranks_map=(verification or {}).get('purchase_ranks_by_race') or {}
                        tickets_by_race={}
                        for t in (verification or {}).get('recent_rows') or []:
                            tid=_norm_race_id(t.get('race_id',''))
                            if tid:
                                tickets_by_race.setdefault(tid, []).append(t)
                        for row in races:
                            rid=_norm_race_id(row.get('race_id',''))
                            row['purchase_ranks']=list(ranks_map.get(rid, []))
                            row['購入馬券一覧']=list(tickets_by_race.get(rid, []))
                    venues=_venue_meetings(races)
                    venue_names={v['name'] for v in venues}
                    # 地方: 日付→開催場一覧→レース一覧。開催場未選択なら場一覧を出す
                    races_for_board=list(races)
                    if source=='nar' and mode in ('predict','result'):
                        show_venue_picker=True
                        if selected_venue and selected_venue not in venue_names:
                            # 正規化後も不一致なら一覧へ戻す（空画面にしない）
                            selected_venue=''
                        if selected_venue:
                            from netkeiba_client import normalize_venue_name as _nv
                            races=[
                                r for r in races
                                if _nv(str(r.get('開催地') or '').strip())==selected_venue
                            ]
                            show_venue_picker=False
                            races_for_board=list(races)
                            # 【2】【3】【5】開場後にレースが空なら当該場だけ再取得＋取得中表示
                            if not races:
                                data_status='generating'
                                message='データ取得中'
                                try:
                                    threading.Thread(
                                        target=bootstrap_venue,
                                        args=(selected, selected_venue, 'nar'),
                                        daemon=True,
                                    ).start()
                                except Exception as e:
                                    print(f'[bootstrap-venue] skip: {e}')
                        else:
                            races=[]
                    else:
                        selected_venue=''
                    buy_candidates=build_buy_candidates(races_for_board)
                    # 予想タブは重い verification を避け、結果がある日だけ軽く成績を載せる
                    board_verify=verification if mode in ('result','analysis') else dict(_EMPTY_VERIFY)
                    today_ai_board=build_today_ai_board(races_for_board, board_verify)
                    data_updated_at=''
                    try:
                        if pred_path and Path(pred_path).exists():
                            from datetime import datetime as _dt
                            data_updated_at=_dt.fromtimestamp(Path(pred_path).stat().st_mtime, JST).strftime('%m/%d %H:%M')
                    except Exception:
                        data_updated_at=''
                    targets=buy_candidates[:8]
                    if data_status=='updating':
                        message=f'{selected} / {label} / データ更新中（表示はキャッシュ）'
                        if data_updated_at:
                            message+=f' · 最終更新 {data_updated_at}'
                    elif data_status=='generating':
                        message='データ取得中'
                    elif source=='nar' and show_venue_picker:
                        if venues:
                            if selected==today:
                                message=f'本日開催 {selected} / {label} / 開催場 {len(venues)}場'
                            else:
                                message=f'{selected} / {label} / 開催場 {len(venues)}場'
                        else:
                            if data_status=='error':
                                message='取得失敗'
                            elif selected==today:
                                message='本日は地方競馬の開催はありません'
                            else:
                                message=f'{selected} の地方開催データがありません'
                    elif not races:
                        if source=='nar' and selected_venue and data_status=='generating':
                            message='データ取得中'
                        else:
                            message=f'{selected} / {label} のレースがありません'
                    elif mode=='result':
                        if selected_venue:
                            message=f'{selected} / {selected_venue} / 結果検証'
                        else:
                            message=f'{selected} / {label} / 結果検証モード'
                    elif mode=='analysis':
                        message=f'{selected} / {label} / AI期待値分析'
                    else:
                        if selected_venue:
                            message=f'{selected} / {selected_venue} / 予想分析'
                        else:
                            message=f'{selected} / {label} / AI期待値分析'
        except FileNotFoundError as e:
            data_status='generating'
            message='データ取得中' if source=='nar' else (str(e) or 'データ取得中です。完了後に再読み込みしてください。')
        except Exception as e:
            data_status='error'
            message=f'通信エラー: {e}'
    elif selected:
        if source=='nar' and selected==today:
            data_status='generating'
            message='データ取得中'
            if today not in av:
                av=sorted(set(av)|{today}, reverse=True)
            try:
                threading.Thread(
                    target=run_nar_today_pipeline, kwargs={'force': force_refresh or want_today},
                    daemon=True,
                ).start()
            except Exception:
                pass
        else:
            data_status='error'
            message=f'{selected} は保存データにありません'
    elif source=='nar':
        data_status='generating'
        message='データ取得中'
        try:
            threading.Thread(target=run_nar_today_pipeline, kwargs={'force': True}, daemon=True).start()
        except Exception:
            pass
    day_stats=None
    if mode=='result' and races and not show_venue_picker:
        day_stats=day_performance(races, verification, safe_pct=_safe_pct)
        today_ai_board=build_today_ai_board(races, verification)
    ledger=ledger_data(source=source, verification=verification) if mode in ('analysis','ledger') else {
        'has_data':False,'investment':0,'payout':0,'recovery':0.0,'profit':0,
        'by_type':[],'monthly':[],'tone':'roi-bad',
    }

    # 地方: ジョブ状態で「取得中」を確定解除（成功→ready / 失敗→error）。永久取得中を禁止。
    status_refresh_url=''
    if source=='nar':
        job_status, job_msg = resolve_nar_fetch_status(selected, force_refresh=False)
        if (force_refresh or want_today) and job_status=='ready':
            # ボタン押下直後（状態ファイル更新前）は取得中を表示
            data_status='generating'
            message='データ取得中'
        elif job_status=='generating':
            data_status='generating'
            message=job_msg or 'データ取得中'
        elif job_status=='error':
            data_status='error'
            message=job_msg or '取得失敗'
        elif job_status=='ready':
            # 成功後は取得中を必ず解除。データがあれば ready/updating を維持。
            if data_status=='generating':
                if venues or races or (ARCH/f'predictions_{selected}.csv').exists():
                    data_status='ready'
                    if not message or message in ('データ取得中', '取得中'):
                        message=job_msg or (
                            f'本日開催 {selected} / {label} / 開催場 {len(venues)}場'
                            if venues else '取得完了'
                        )
                else:
                    data_status='ready'
                    message='本日は地方競馬の開催はありません' if selected==today else message
        from urllib.parse import urlencode
        q={'source':'nar','mode':mode}
        if selected:
            q['date']=selected
        if selected_venue:
            q['venue']=selected_venue
        status_refresh_url='/' + ('?' + urlencode(q) if q else '')

    return render_template('index.html',races=races,targets=targets,selected_date=selected,today=today,
        message=message,available_dates=av,source=source,mode=mode,has_results=has_results,
        analysis=analysis_data(races if not show_venue_picker else []),verification=verification,
        ledger=ledger,
        venues=venues,selected_venue=selected_venue,show_venue_picker=show_venue_picker,
        today_date=_pick_today_date(meeting_dates, today) if source=='nar' else today,
        day_stats=day_stats,data_status=data_status,
        buy_candidates=buy_candidates,today_ai_board=today_ai_board,data_updated_at=data_updated_at,
        status_refresh_url=status_refresh_url)


def attach_results(records, selected_date=''):
    """results.csv / scores / analysis_result を照合し、着順・的中・AI振り返りを付与。"""
    rp=DATA/'results.csv'
    rdf=None
    if rp.exists():
        try:
            rdf=pd.read_csv(rp).fillna('')
        except Exception:
            rdf=None
    if rdf is not None and not rdf.empty:
        if 'race_id' not in rdf.columns or '馬名' not in rdf.columns:
            rdf=None
        else:
            rdf=rdf[~rdf['race_id'].astype(str).str.startswith('http')].copy()
            rdf['race_id']=rdf['race_id'].map(_norm_race_id)
            if '着順' not in rdf.columns:
                rdf=None

    lookup={}          # (race_id, 馬名) -> 着順表示
    date_venue_lookup={}  # (date, 開催地, R, 馬名) -> 着順表示
    race_ids_with_result=set()
    resolve_map={}     # (date, 開催地, R) -> race_id

    if rdf is not None and not rdf.empty:
        for _,x in rdf.iterrows():
            fin=_format_finish(x.get('着順',''))
            if not fin:
                continue
            rid=_norm_race_id(x.get('race_id',''))
            name=clean_horse(x.get('馬名',''))
            if rid and name:
                lookup[(rid,name)]=fin
                race_ids_with_result.add(rid)
            d=str(x.get('date','') or '').strip()
            venue=str(x.get('開催地','') or '').strip()
            try: rn=int(float(x.get('レース',0)))
            except Exception: rn=None
            if d and venue and rn is not None and name:
                date_venue_lookup[(d,venue,rn,name)]=fin
                resolve_map.setdefault((d,venue,rn), rid)

    analysis_by_race=_load_analysis_by_race()
    score_cache={}
    odds_cache={}
    any_result=False

    for r in records:
        rid=_norm_race_id(r.get('race_id',''))
        r['race_id']=rid or str(r.get('race_id',''))
        try: race_no=int(float(r.get('レース',0)))
        except Exception: race_no=None
        venue=str(r.get('開催地','') or '').strip()
        d=_race_date(r) or str(selected_date or '').strip()

        # 旧JRA URL → netkeiba race_id 解決（同一日・開催地・R）
        if (not rid or rid.startswith('http') or not rid.isdigit()) and d and venue and race_no is not None:
            resolved=resolve_map.get((d,venue,race_no),'')
            if resolved:
                rid=resolved
                r['race_id']=rid

        if d and d not in score_cache:
            score_cache[d]=_load_score_finishes(d)
            odds_cache[d]=load_score_odds(ARCH, d, _norm_race_id, clean_horse)
        score_lu=score_cache.get(d,{})
        odds_lu=odds_cache.get(d,{})

        def lookup_finish(horse_name: str) -> str:
            hn=clean_horse(horse_name)
            if not hn:
                return ''
            fin=lookup.get((rid,hn),'') if rid else ''
            if not fin and d and venue and race_no is not None:
                fin=date_venue_lookup.get((d,venue,race_no,hn),'')
            if not fin and rid:
                fin=score_lu.get((rid,hn),'')
            if not fin and d and venue and race_no is not None:
                fin=score_lu.get((f'date:{d}',venue,race_no,hn),'')
            return fin

        race_has_result=(
            (rid and rid in race_ids_with_result)
            or rid in analysis_by_race
            or any(
                lookup_finish(n)
                for n in [r.get('本命','')] + [x.get('馬名','') for x in r.get('印一覧',[])]
                if n
            )
        )

        entries=[('◎',r.get('本命',''))]+[(x.get('印',''),x.get('馬名','')) for x in r.get('印一覧',[])]
        seen=set(); review=[]
        for mark,name in entries:
            if not name or name in seen:
                continue
            seen.add(name)
            finish=lookup_finish(name)
            if finish:
                any_result=True
                disp=finish
            elif race_has_result:
                # 結果確定レースで馬だけ見つからない（取消・除外など）
                disp='取消'
            else:
                disp='結果待ち'
            circled=_finish_circled(disp)
            review.append({'印':mark,'馬名':name,'着順':disp,'着順丸':circled})
        r['結果一覧']=review
        r['印着順要約']=' '.join(f"{x['印']}{x['着順丸']}" for x in review) if review else ''
        r['結果確定']=bool(race_has_result)

        # 的中 / 不的中（analysis_result 優先）
        hits=analysis_by_race.get(rid,[]) if rid else []
        if not hits and d and venue and race_no is not None:
            alt=resolve_map.get((d,venue,race_no),'')
            if alt:
                hits=analysis_by_race.get(alt,[])
        r['的中一覧']=hits
        if hits:
            parts=[]
            for h in hits:
                label='的中' if h['hit'] else '不的中'
                parts.append(f"{h['bet_type']}{label}")
            r['的中表示']=' / '.join(parts)
        elif race_has_result:
            r['的中表示']=''
        else:
            r['的中表示']='結果待ち'

        o_name=next((x.get('馬名','') for x in r.get('印一覧',[]) if x.get('印')=='○'), '')
        rival_odds=odds_lu.get((rid, clean_horse(o_name))) if (rid and o_name) else None
        ai_eval=build_ai_self_eval(r, review, clean_horse, rival_odds=rival_odds)
        r['AI評価']=ai_eval

        main_finish=lookup_finish(r.get('本命',''))
        if ai_eval.get('あり'):
            r['AI振り返り']=ai_eval.get('サマリー') or ''
        elif race_has_result and main_finish:
            parts=[f"◎{r.get('本命')}は{main_finish}"]
            if hits:
                main_hits=[h for h in hits if h['bet_type']=='本命']
                if main_hits:
                    parts.append('本命的中' if main_hits[0]['hit'] else '本命不的中')
                other=[h for h in hits if h['bet_type']!='本命']
                if other:
                    parts.append(' / '.join(
                        f"{h['bet_type']}{'的中' if h['hit'] else '不的中'}" for h in other[:4]
                    ))
            parts.append('軸評価を実着順と照合済み。印上位の着順を見て、次回の重み調整候補として蓄積します。')
            r['AI振り返り']='。'.join(parts)
        elif race_has_result:
            extra=f"的中状況: {r['的中表示']}。" if r.get('的中表示') else ''
            r['AI振り返り']=f'このレースの確定結果は保存済みです。{extra}印と実着順を照合してください。'
        else:
            r['AI振り返り']='このレースの確定結果はまだ保存されていません。結果取得後に自動照合します。'
    return records, any_result


def analysis_data(records):
    if not records:
        return {'total':0,'verified':0,'ranks':[],'venues':[],'bands':[]}
    from areru_engine import RANK_LABELS
    df=pd.DataFrame([{'rank':str(r.get('勝負ランク','')),'venue':str(r.get('開催地','')),
                      'score':float(r.get('BET期待値',0) or 0),
                      'verified':bool(r.get('結果確定')) or any(
                          str(x.get('着順','')) not in ('','結果待ち','取消') for x in r.get('結果一覧',[]))}
                     for r in records])
    ranks=[{'label':x,'name':RANK_LABELS.get(x,x),'count':int((df['rank']==x).sum())} for x in ['S','A','B','C','D']]
    venues=[{'label':str(k),'count':int(v)} for k,v in df['venue'].value_counts().items()]
    bands=[]
    for label,lo,hi in [('～69',0,70),('70～79',70,80),('80～89',80,90),('90～',90,101)]:
        bands.append({'label':label,'count':int(((df['score']>=lo)&(df['score']<hi)).sum())})
    return {'total':len(df),'verified':int(df['verified'].sum()),'ranks':ranks,'venues':venues,'bands':bands}


def _safe_pct(num, den):
    return round(float(num)/float(den)*100,1) if den else 0.0


def _roi_tone(recovery):
    """回収率の色区分: 100%以上緑 / 90〜99%黄 / 89%以下赤"""
    try:
        v=float(recovery)
    except (TypeError, ValueError):
        v=0.0
    if v>=100:
        return 'roi-good'
    if v>=90:
        return 'roi-mid'
    return 'roi-bad'


# analysis_result の bet_type → 画面表示名（本命＝単勝）
BET_TYPE_DISPLAY = {
    '本命': '単勝',
    '単勝': '単勝',
    'ワイド': 'ワイド',
    '馬連': '馬連',
    '三連複': '三連複',
    '三連単': '三連単',
}
RANK_TYPE_ORDER = ['単勝', 'ワイド', '馬連', '三連複', '三連単']


def _bet_type_label(bet_type):
    key=str(bet_type or '').strip()
    return BET_TYPE_DISPLAY.get(key, key or '—')


def _attach_rank_column(frame, pred_meta):
    """勝負ランク列を保証（欠損は予想メタから補完）。"""
    out=frame.copy()
    if out.empty:
        out['勝負ランク']=''
        return out
    if '勝負ランク' not in out.columns:
        out['勝負ランク']=''
    blank=out['勝負ランク'].astype(str).str.strip().eq('')
    if blank.any():
        out.loc[blank,'勝負ランク']=out.loc[blank].apply(
            lambda row: str((_pred_for_analysis_row(row, pred_meta) or {}).get('勝負ランク','') or ''),
            axis=1,
        )
    out['勝負ランク']=out['勝負ランク'].astype(str).str.upper().str.strip()
    return out


def _bar_width(recovery):
    try:
        v=float(recovery)
    except (TypeError, ValueError):
        v=0.0
    return round(min(max(v,0),100),1)


def parse_prediction_combos(prediction):
    """買い目文字列を [['馬A','馬B'], ...] に分解（横表示カード用）。"""
    text=str(prediction or '').strip()
    if not text or text in ('見送り','なし'):
        return []
    text=re.sub(r'[（(][^）)]*[）)]','',text)
    combos=[]
    for part in text.split('｜'):
        part=part.strip()
        if not part:
            continue
        horses=[h.strip() for h in re.split(r'\s*[－\-]\s*',part) if h.strip()]
        if horses:
            combos.append(horses)
    return combos


def _load_prediction_meta():
    """race_id / date+会場+R → 予想時メタデータ。旧JRA URL 形式にも対応。"""
    sig=_fs_sig(ANALYSIS_CSV)
    if _PRED_META_CACHE.get('sig')==sig and _PRED_META_CACHE.get('data') is not None:
        return _PRED_META_CACHE['data']
    meta={}
    # メタに必要な列だけ読む
    want={'race_id','開催地','レース','日付','勝負ランク','推奨券種','本命','本命馬番',
          'ワイド判定','馬連判定','三連複判定','印データ','BET判定','BET期待値','source'}
    for f in ARCH.glob('predictions_*.csv'):
        try:
            cols=pd.read_csv(f,encoding='utf-8-sig',nrows=0).columns.tolist()
            use=[c for c in cols if c in want]
            if not use:
                continue
            df=pd.read_csv(f,encoding='utf-8-sig',usecols=use).fillna('')
        except Exception:
            continue
        if df.empty:
            continue
        m=re.fullmatch(r'predictions_(\d{4}-\d{2}-\d{2})\.csv', f.name)
        file_date=m.group(1) if m else ''
        for _,row in df.iterrows():
            d=row.to_dict()
            rid=_norm_race_id(d.get('race_id',''))
            if rid and rid not in meta:
                meta[rid]=d
            venue=str(d.get('開催地','') or '').strip()
            try:
                race_i=int(float(d.get('レース',0) or 0))
            except (TypeError, ValueError):
                race_i=0
            day=str(d.get('日付','') or file_date or '').strip()
            if day and venue and race_i:
                alt=f'{day}|{venue}|{race_i}'
                if alt not in meta:
                    meta[alt]=d
    _PRED_META_CACHE['sig']=sig
    _PRED_META_CACHE['data']=meta
    return meta


def _pred_for_analysis_row(r, pred_meta):
    """analysis 行から予想メタを解決（race_id優先、date+会場+Rフォールバック）。"""
    rid=_norm_race_id(r.get('race_id',''))
    if rid and rid in pred_meta:
        return pred_meta[rid]
    venue=str(r.get('開催地','') or '').strip()
    day=str(r.get('date','') or '').strip()
    race_i=0
    race_label=str(r.get('race','') or '')
    m=re.search(r'(\d+)\s*R', race_label)
    if m:
        race_i=int(m.group(1))
    if not race_i:
        try: race_i=int(float(r.get('レース',0) or 0))
        except (TypeError, ValueError): race_i=0
    if day and venue and race_i:
        return pred_meta.get(f'{day}|{venue}|{race_i}') or {}
    return {}


def _ticket_judge_is_buy(pred, bet_type):
    """予想メタの券種判定が買い候補か。"""
    bt=str(bet_type or '').strip()
    if bt in ('本命','単勝'):
        rec=str(pred.get('推奨券種','') or '').strip()
        return rec in ('本命','単勝')
    col={'ワイド':'ワイド判定','馬連':'馬連判定','三連複':'三連複判定'}.get(bt)
    if not col or not pred:
        return False
    return str(pred.get(col,'') or '').strip()=='買い候補'


def _ensure_purchase_flags(df, pred_meta):
    """購入対象フラグを保証。推奨券種 or 券種判定「買い候補」を購入単位とする。"""
    out=df.copy()
    if '推奨券種' not in out.columns:
        out['推奨券種']=''
    if '購入対象' not in out.columns:
        out['購入対象']=0
    # 推奨券種が空の行だけメタから補完
    for idx in out.index[out['推奨券種'].astype(str).str.strip().eq('')]:
        pred=_pred_for_analysis_row(out.loc[idx], pred_meta)
        rec=str(pred.get('推奨券種','') or '').strip()
        if rec:
            out.at[idx,'推奨券種']=rec
    flags=[]
    for _,row in out.iterrows():
        bt=str(row.get('bet_type','') or '').strip()
        rec=str(row.get('推奨券種','') or '').strip()
        pred=_pred_for_analysis_row(row, pred_meta)
        is_buy=(rec!='' and bt==rec) or _ticket_judge_is_buy(pred, bt)
        flags.append(1 if is_buy else 0)
    out['購入対象']=flags
    return out


def _ticket_marks_label(prediction, pred):
    """買い目を ◎-○ / ◎-○-▲ 形式に短縮（表示用）。"""
    if not pred:
        return ''
    name_to_mark={}
    main=str(pred.get('本命','') or '').strip()
    if main:
        name_to_mark[main]='◎'
    try:
        marks=json.loads(pred.get('印データ','[]') or '[]')
    except Exception:
        marks=[]
    for x in marks:
        name=str(x.get('馬名','') or '').strip()
        mk=str(x.get('印','') or '').strip()
        if name and mk and name not in name_to_mark:
            name_to_mark[name]=mk
    combos=parse_prediction_combos(prediction)
    if not combos:
        return ''
    parts=[]
    for h in combos[0]:
        parts.append(name_to_mark.get(h, h))
    # 印に置換できたときだけ短縮表示
    if parts and all(p in ('◎','○','▲','△','☆') for p in parts):
        return '-'.join(parts)
    return ''


def _buy_reasons(pred):
    """予想時情報から購入理由リストを組み立てる。"""
    if not pred:
        return []
    reasons=[]
    ev_txt=str(pred.get('期待回収率','') or '')
    strategy=str(pred.get('馬券戦略理由','') or '')
    bet_reason=str(pred.get('BET理由','') or '')
    danger=str(pred.get('人気馬危険','') or '')
    rank=str(pred.get('勝負ランク','') or '')
    bet_judge=str(pred.get('BET判定','') or '')

    ev_num=None
    m=re.search(r'([\d.]+)',ev_txt)
    if m:
        try: ev_num=float(m.group(1))
        except ValueError: pass

    if '妙味' in strategy or '妙味' in bet_reason or (ev_num is not None and ev_num>=100):
        reasons.append('市場オッズ妙味あり')
    if ev_num is not None and ev_num>=100:
        reasons.append('期待値プラス')
    elif '期待値' in bet_reason:
        reasons.append('期待値プラス')
    if danger and danger not in ('なし','見送り',''):
        reasons.append('危険人気馬を除外')
    if rank in ('S','A','B','C','D'):
        reasons.append(f'AI評価{rank}')
    if bet_judge and bet_judge not in ('なし','見送り',''):
        reasons.append(f'買い判定：{bet_judge}')
    elif bet_judge=='見送り':
        reasons.append('判定は見送り（仮想検証）')

    seen=set(); out=[]
    for x in reasons:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def _rank_label(rank):
    from areru_engine import RANK_LABELS
    return RANK_LABELS.get(str(rank).upper(),'')


def _parse_race_no(race_label, pred=None):
    """開催ラベル / 予想メタからレース番号を抽出。"""
    text=str(race_label or '')
    m=re.search(r'(\d+)\s*R', text, re.I)
    if m:
        return f'{int(m.group(1)):02d}'
    if pred is not None:
        raw=pred.get('レース','')
        try:
            return f'{int(float(raw)):02d}'
        except (TypeError, ValueError):
            pass
    return ''


def _enrich_verify_row(r, pred_meta):
    rid=_norm_race_id(r.get('race_id',''))
    pred=_pred_for_analysis_row(r, pred_meta)
    prediction=str(r.get('prediction','') or '')
    combos=parse_prediction_combos(prediction)
    recovery=_safe_pct(float(r.get('payout') or 0), float(r.get('investment') or 0))
    if float(r.get('investment') or 0)==0 and 'roi' in r:
        try: recovery=float(r.get('roi') or 0)
        except (TypeError, ValueError): recovery=0.0
    # analysis_result に保存済みランクがあれば優先
    rank=str(r.get('勝負ランク') or pred.get('勝負ランク','') or '').upper()
    bet_judge=str(pred.get('BET判定','') or _rank_label(rank) or '')
    areru=str(pred.get('荒れ度','') or '')
    expect=str(pred.get('BET期待値','') or '')
    recommend=str(r.get('推奨券種') or pred.get('推奨券種','') or '')
    ai_comment=str(pred.get('馬券戦略理由','') or '')
    bet_type_raw=str(r.get('bet_type','') or '')
    try:
        is_purchase=int(float(r.get('購入対象') or 0))
    except (TypeError, ValueError):
        is_purchase=0
    if not is_purchase:
        if recommend and bet_type_raw==recommend:
            is_purchase=1
        elif _ticket_judge_is_buy(pred, bet_type_raw):
            is_purchase=1
    venue=str(r.get('開催地','') or pred.get('開催地','') or '')
    race_label=str(r.get('race','') or '')
    if not venue and race_label:
        venue=re.sub(r'\d+\s*R.*$','',race_label,flags=re.I).strip()
    race_no=_parse_race_no(race_label, pred)
    marks=_ticket_marks_label(prediction, pred)
    bet_label=_bet_type_label(bet_type_raw)
    ticket_short=f'{bet_label} {marks}'.strip() if marks else bet_label
    return {
        'date':str(r.get('date','')),
        'race':race_label,
        'race_id':rid,
        'venue':venue,
        'race_no':race_no,
        'bet_type':bet_type_raw,
        'ticket_marks':marks,
        'ticket_short':ticket_short,
        'prediction':prediction,
        'combos':combos,
        'result':str(r.get('result','') or ''),
        'hit':int(r.get('hit') or 0),
        'payout':int(r.get('payout') or 0),
        'investment':int(r.get('investment') or 0),
        'profit':int(r.get('profit') or 0),
        'roi':float(r.get('roi') or 0),
        'recovery':recovery,
        'tone':_roi_tone(recovery),
        'rank':rank,
        'rank_label':_rank_label(rank) or bet_judge,
        'areru':areru,
        'expect':expect,
        'recommend':recommend,
        'bet_judge':bet_judge,
        'ai_comment':ai_comment,
        'reasons':_buy_reasons(pred),
        'has_ai':bool(pred) or bool(rank),
        'is_purchase':is_purchase,
    }


def verification_data(selected_date='', source='all'):
    """analysis_result.csv から結果検証ダッシュボード用データを構築。"""
    empty_pack={
        'total_bets':0,'hits':0,'hit_rate':0.0,'recovery':0.0,'roi':0.0,
        'investment':0,'payout':0,'profit':0,'tone':'roi-bad','bar':0,
    }
    empty={
        'has_data':False,'selected_date':selected_date,
        'total_bets':0,'hit_rate':0.0,'recovery':0.0,'roi':0.0,
        'investment':0,'payout':0,'profit':0,'tone':'roi-bad',
        'daily':[],'by_type':[],'by_rank':[],'by_rank_type':[],'main':{},
        'recovery_series':[],'cum_profit':[],'recent_rows':[],
        'purchase_ranks_by_race':{},
    }
    sig=_fs_sig(ANALYSIS_CSV)
    cache_key=(str(selected_date or ''), source, sig)
    cached=_VERIFY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not ANALYSIS_CSV.exists():
        _VERIFY_CACHE[cache_key]=empty
        return empty
    try:
        df=pd.read_csv(ANALYSIS_CSV,encoding='utf-8-sig').fillna('')
    except Exception:
        _VERIFY_CACHE[cache_key]=empty
        return empty
    if df.empty or 'bet_type' not in df.columns:
        _VERIFY_CACHE[cache_key]=empty
        return empty
    if source in ('jra','nar'):
        if 'source' in df.columns:
            df=df[df['source'].astype(str).str.lower()==source].copy()
        elif 'race_id' in df.columns:
            from areru_engine import source_from_race_id
            df=df[df['race_id'].map(source_from_race_id)==source].copy()
        if df.empty:
            _VERIFY_CACHE[cache_key]=empty
            return empty
    for c in ['hit','payout','investment','profit','roi']:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
    pred_meta=_load_prediction_meta()
    df=_ensure_purchase_flags(df, pred_meta)
    all_df=df.copy()
    day_df=all_df[all_df['date'].astype(str)==str(selected_date)] if selected_date else all_df
    # 購入対象 = 推奨券種 or 買い候補の馬券。「Sだけ買えば勝てるか」の母集団（レース単位ではない）
    purchase_all=all_df[pd.to_numeric(all_df['購入対象'],errors='coerce').fillna(0).astype(int)==1].copy()

    def pack(frame):
        if frame is None or len(frame)==0:
            return dict(empty_pack)
        inv=float(frame['investment'].sum())
        pay=float(frame['payout'].sum())
        hits=int(frame['hit'].sum())
        n=len(frame)
        profit=pay-inv
        recovery=_safe_pct(pay,inv)
        return {
            'total_bets':n,
            'hits':hits,
            'hit_rate':_safe_pct(hits,n),
            'recovery':recovery,
            'roi':round(profit/inv*100,1) if inv else 0.0,
            'investment':int(inv),
            'payout':int(pay),
            'profit':int(profit),
            'tone':_roi_tone(recovery),
            'bar':_bar_width(recovery),
        }

    summary=pack(day_df if not day_df.empty else all_df)
    # 日別
    daily=[]
    for d,g in all_df.groupby('date',sort=True):
        s=pack(g)
        daily.append({'date':str(d),**s})
    # 累計収支・回収率系列
    cum=0; cum_inv=0; cum_pay=0
    recovery_series=[]; cum_profit=[]
    for row in daily:
        cum+=row['profit']; cum_inv+=row['investment']; cum_pay+=row['payout']
        rec=_safe_pct(cum_pay,cum_inv)
        recovery_series.append({'date':row['date'],'value':rec,'tone':_roi_tone(rec),'bar':_bar_width(rec)})
        cum_profit.append({'date':row['date'],'value':cum})
    # 券種別
    by_type=[]
    src=day_df if not day_df.empty else all_df
    for bt,g in src.groupby('bet_type'):
        s=pack(g)
        by_type.append({'bet_type':_bet_type_label(bt),'bet_key':str(bt),**s})
    by_type=sorted(by_type,key=lambda x:x['investment'],reverse=True)
    # 本命（単勝）成績
    main_df=src[src['bet_type'].astype(str).isin(['本命','単勝'])] if not src.empty else pd.DataFrame()
    main=pack(main_df) if not main_df.empty else dict(empty_pack)
    # AIランク別KPI（購入対象の馬券単位・全期間）※レース単位ではない
    ranked=_attach_rank_column(purchase_all, pred_meta)
    by_rank=[]
    for key, name in [('S','勝負'),('A','買い'),('B','様子見'),('C','警戒'),('D','見送り')]:
        g=ranked[ranked['勝負ランク']==key] if not ranked.empty else ranked
        s=pack(g)
        by_rank.append({'key':key,'name':name,**s})
    # ランク×券種（購入対象のみ）
    typed=ranked.copy()
    if not typed.empty:
        typed['券種表示']=typed['bet_type'].map(_bet_type_label)
    by_rank_type=[]
    for key, name in [('S','勝負'),('A','買い'),('B','様子見'),('C','警戒'),('D','見送り')]:
        g_rank=typed[typed['勝負ランク']==key] if not typed.empty else typed
        types=[]
        for label in RANK_TYPE_ORDER:
            g=g_rank[g_rank['券種表示']==label] if not g_rank.empty else g_rank
            types.append({'bet_type':label,**pack(g)})
        by_rank_type.append({'key':key,'name':name,'types':types,**pack(g_rank)})
    # 照合明細＝購入対象の馬券のみ（S押下でS購入分だけ）
    recent=[]
    if not ranked.empty:
        for _,r in ranked.sort_values(['date','race','bet_type'],ascending=[False,True,True]).iterrows():
            row=_enrich_verify_row(r, pred_meta)
            if not row.get('is_purchase'):
                continue
            row['bet_type']=_bet_type_label(row.get('bet_type'))
            recent.append(row)
    # レース一覧用: 購入馬券が存在するランク（レースの勝負ランクではない）
    purchase_ranks_by_race={}
    for row in recent:
        rid=_norm_race_id(row.get('race_id',''))
        rk=str(row.get('rank','') or '').upper()
        if not rid or rk not in ('S','A','B','C','D'):
            continue
        purchase_ranks_by_race.setdefault(rid,set()).add(rk)
    purchase_ranks_by_race={k:sorted(v) for k,v in purchase_ranks_by_race.items()}
    # グラフ用スケール
    max_abs=max([abs(x['value']) for x in cum_profit]+[1])
    for x in cum_profit:
        x['pct']=round(abs(x['value'])/max_abs*100,1)
        x['pos']=x['value']>=0
        x['tone']='roi-good' if x['pos'] else 'roi-bad'
    for x in recovery_series:
        x['pct']=x.get('bar',_bar_width(x['value']))

    out={
        'has_data':True,
        'selected_date':selected_date,
        'scope':'day' if selected_date and not day_df.empty else 'all',
        **summary,
        'daily':daily,
        'by_type':by_type,
        'by_rank':by_rank,
        'by_rank_type':by_rank_type,
        'main':main,
        'recovery_series':recovery_series,
        'cum_profit':cum_profit,
        'recent_rows':recent[:300],
        'purchase_count':len(recent),
        'purchase_ranks_by_race':purchase_ranks_by_race,
    }
    _VERIFY_CACHE[cache_key]=out
    if len(_VERIFY_CACHE)>16:
        # 古いエントリを間引く
        for k in list(_VERIFY_CACHE.keys())[:-8]:
            _VERIFY_CACHE.pop(k, None)
    return out

def ledger_data(source='all', verification=None):
    """AI推奨どおり購入した場合の収支分析（月別・券種別）。"""
    v=verification if verification is not None else verification_data('', source=source)
    if not v.get('has_data'):
        return {
            'has_data':False,'investment':0,'payout':0,'recovery':0.0,'profit':0,
            'by_type':[],'monthly':[],'tone':'roi-bad',
        }
    monthly=[]
    for row in v.get('daily') or []:
        ym=str(row.get('date') or '')[:7]
        if not ym:
            continue
        if not monthly or monthly[-1]['month']!=ym:
            monthly.append({'month':ym,'investment':0,'payout':0,'profit':0,'hits':0,'bets':0})
        m=monthly[-1]
        m['investment']+=int(row.get('investment') or 0)
        m['payout']+=int(row.get('payout') or 0)
        m['profit']+=int(row.get('profit') or 0)
        m['hits']+=int(row.get('hits') or 0)
        m['bets']+=int(row.get('total_bets') or 0)
    for m in monthly:
        inv=m['investment'] or 0
        m['recovery']=_safe_pct(m['payout'], inv)
        m['tone']=_roi_tone(m['recovery'])
    return {
        'has_data':True,
        'investment':v.get('investment',0),
        'payout':v.get('payout',0),
        'profit':v.get('profit',0),
        'recovery':v.get('recovery',0),
        'hit_rate':v.get('hit_rate',0),
        'tone':v.get('tone','roi-bad'),
        'by_type':v.get('by_type') or [],
        'monthly':monthly,
        'daily':v.get('daily') or [],
    }


@app.route('/cron/nar-daily', methods=['POST','GET'])
def cron_nar_daily():
    """外部cron向け: 地方の開催場→レース→結果をバックグラウンドで安定更新。"""
    token=str(request.args.get('token') or request.headers.get('X-Cron-Token') or '').strip()
    expected=str(os.environ.get('CRON_TOKEN') or '').strip()
    if expected and token != expected:
        return {'ok': False, 'error': 'unauthorized'}, 401

    def _run():
        try:
            print('[cron-nar] today pipeline (venues→races→predict)', flush=True)
            run_nar_today_pipeline(force=True)
            print('[cron-nar] bootstrap incomplete cards', flush=True)
            bootstrap_source('nar')
            print('[cron-nar] bootstrap results', flush=True)
            bootstrap_missing_results('nar')
            print('[cron-nar] done', flush=True)
        except Exception as e:
            print(f'[cron-nar] fail: {e}', flush=True)

    threading.Thread(target=_run, daemon=True).start()
    return {'ok': True, 'started': True, 'source': 'nar'}


@app.route('/refresh', methods=['POST','GET'])
def refresh_route():
    """最新開催日・オッズ・結果を取得して runners / predictions を更新。

    full / odds / results いずれも gunicorn タイムアウト回避のためバックグラウンド実行。
    重いジョブは直列化し、取得と予想を同時起動しない。
    """
    mode=request.args.get('mode','full')
    source=request.args.get('source','all')
    date=str(request.args.get('date') or '').strip()
    if source not in ('jra','nar','all'):
        source='all'
    try:
        def _run_refresh(_mode=mode, _source=source, _date=date):
            def _body():
                try:
                    if _mode=='odds':
                        cmd=[sys.executable,'refresh_data.py','--latest-only','--odds-only','--source',_source,'--skip-predict']
                        subprocess.run(cmd, check=False, timeout=1800)
                        # odds-only 後は直近日を再予想
                        today=_today_jst()
                        subprocess.run([sys.executable,'replay_predict.py',today], check=False, timeout=600)
                    elif _mode=='results':
                        if _date and re.fullmatch(r'\d{4}-\d{2}-\d{2}', _date):
                            cmd=[sys.executable,'results.py','--source',_source,'--dates',_date]
                        else:
                            cmd=[sys.executable,'results.py','--latest','--source',_source]
                        subprocess.run(cmd, check=False, timeout=1800)
                    else:
                        cmd=[
                            sys.executable,'refresh_data.py',
                            '--latest-only','--source',_source,'--skip-predict',
                        ]
                        subprocess.run(cmd, check=False, timeout=1800)
                        _clear_runtime_caches()
                        today=_today_jst()
                        pred_day=_date if _date and re.fullmatch(r'\d{4}-\d{2}-\d{2}', _date) else today
                        subprocess.run([sys.executable,'replay_predict.py',pred_day], check=False, timeout=600)
                    _clear_runtime_caches()
                    print(f'[refresh] finished mode={_mode}', flush=True)
                except Exception as e:
                    print(f'[refresh] fail mode={_mode}: {e}', flush=True)
            _run_serialized_heavy(f'refresh:{_mode}:{_source}', _body, wait=False)

        print(f'[refresh] start bg mode={mode} source={source}', flush=True)
        threading.Thread(target=_run_refresh, daemon=True).start()
        av=dates(source)
        return {
            'ok': True, 'started': True, 'background': True,
            'dates': av, 'latest': av[0] if av else None,
            'mode': mode, 'source': source, 'date': date or None,
        }
    except Exception as e:
        return {'ok':False,'error':str(e)}, 500
if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','5001')),debug=False)
