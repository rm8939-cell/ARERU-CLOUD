from flask import Flask,render_template,request
import subprocess,sys,json,re,threading
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
import os
import pandas as pd
from areru_engine import parse_date
from ev_analysis import (
    BUY_EV_FLOOR,
    apply_expected_value,
    build_ai_self_eval,
    day_performance,
    load_score_odds,
)
from pipeline_stages import log_orchestrator, log_pipeline_stage, predictions_label
from daily_ops import (
    PREDICT_READY_HOUR,
    past_predict_deadline,
    is_sealed,
    write_seal,
    auto_seal_if_ready,
    should_full_predict,
    should_odds_only_update,
    source_is_race_day,
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
# predictions_YYYY-MM-DD.csv に含まれる source。(path, mtime, size) キーなので
# 中身が変わったファイルだけ読み直せばよい。日付ファイルは増える一方なので、
# ここを毎回舐めると表示時間が日々伸びていく。
_PRED_SOURCE_CACHE={}
_VERIFY_CACHE={}
_PRED_META_CACHE={'sig':None,'data':{}}
_PREDICT_JOBS={}
_PREDICT_JOBS_LOCK=threading.Lock()
# 重い取得・予想は1本ずつ（web + refresh + replay の三重起動を防ぐ）
# RLock: odds-only / predict-only が refresh 経路から呼ばれてもデッドロックしない
_HEAVY_JOB_LOCK=threading.RLock()
_HEAVY_JOB_STATE={'name': ''}
# 予想サブプロセスの同時起動を明示的に禁止（キー違いの多重起動を防ぐ）
_PREDICT_GLOBAL_LOCK=threading.Lock()
_PREDICT_GLOBAL_STATE={'running': False, 'key': ''}
# Render Free: 起動時の NAR+JRA 二重シードは OOM の主因 → 既定スキップ
_ON_RENDER=str(os.environ.get('RENDER') or '').lower() in ('true', '1')
_SKIP_BOOT_DEFAULT='1' if _ON_RENDER else '0'


def _generation_enabled() -> bool:
    """このプロセスで取得・予想の重いジョブを走らせてよいか。

    Render Free は 0.1CPU / 512MB しかなく、本生成を始めると Web 応答が
    止まる（実測: healthz が 10 分以上タイムアウト）。データの正は
    GitHub Actions がコミットする CSV なので、既定では Render では生成せず
    配信のみ行う。ローカルや Actions では従来どおり生成する。
    """
    flag = str(os.environ.get('ARERU_ENABLE_GENERATION') or '').strip().lower()
    if flag in ('1', 'true', 'yes'):
        return True
    if flag in ('0', 'false', 'no'):
        return False
    return not _ON_RENDER
# ページからの復旧起動デバウンス（秒）
_RECOVERY_DEBOUNCE_SEC=120
_EMPTY_VERIFY={
    'has_data':False,'selected_date':'',
    'total_bets':0,'hit_rate':0.0,'recovery':0.0,'roi':0.0,
    'investment':0,'payout':0,'profit':0,'tone':'roi-bad',
    'daily':[],'by_type':[],'by_rank':[],'by_rank_type':[],'main':{},
    'recovery_series':[],'cum_profit':[],'recent_rows':[],
    'purchase_ranks_by_race':{},
}

# 地方の開催場一覧だけなら巨大JSON列を読まない
# ※S/買い厳選に必要な信頼度・オッズ・展開列は含める（相対ランクのまま表示しない）
_NAR_VENUE_PICKER_COLS=(
    'race_id','source','開催地','レース','勝負ランク','相対ランク','期待値','投資判定',
    '本命','日付','荒れクラス',
    '本命オッズ','AI適正オッズ','シミュレーション勝率','シミュレーション連対率',
    '本命データ件数','データ件数','ピックカード','展開予想','レース期待回収率',
    'AI信頼度スコア','レース信頼度スコア',
)

# 開催場詳細で使う列（全列読まずメモリを抑える）
# ※期待値再計算に必要な勝率・適正オッズ・データ件数は必ず含める
_NAR_VENUE_DETAIL_COLS=(
    'race_id','source','開催地','レース','日付','勝負ランク','相対ランク','期待値','投資判定',
    '本命','本命馬番','本命馬番表示','本命理由','本命詳細','本命オッズ','本命人気','本命AREru指数',
    'AI適正オッズ','シミュレーション勝率','シミュレーション連対率','シミュレーション3着内率',
    '本命データ件数','データ件数','ピックカード','推奨馬券','推奨券種','馬券戦略理由',
    '展開予想','印データ','AI買い理由','荒れクラス','ワイド判定','馬連判定',
    'ワイド買い目','馬連買い目','ワイド評価','馬連評価','レース期待回収率','期待回収率',
    'AI信頼度スコア','レース信頼度スコア','シミュレーション再現率','S降格','S降格理由',
    '補正勝率','市場暗示勝率','実エッジpp','BUY品質スコア','BUY品質判定','BUY品質理由',
    '近走指数順位',
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
    # predictions の件数・個別mtimeも拾う（CSV再生成で相対→厳格ランクが変わったとき用）
    try:
        pred_files=sorted(ARCH.glob('predictions_*.csv'))
        parts.append(str(len(pred_files)))
        parts.append(str(int(ARCH.stat().st_mtime_ns)))
        # 上位ファイルの mtime 合計（ディレクトリmtimeが更新されないFS対策）
        parts.append(str(sum(int(f.stat().st_mtime_ns) for f in pred_files[-14:])))
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

def _pred_file_sources(path: Path):
    """予想 CSV に含まれる source（jra/nar）。判別できないときは None。

    開催日一覧はキャッシュが切れるたびに predictions_*.csv を全部開き直して
    いた。ファイルは日ごとに増え続けるので 0.1CPU では日を追うごとに表示が
    遅くなる。中身が変わっていないファイルは読み直さない。
    """
    try:
        st=path.stat()
        key=(str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    hit=_PRED_SOURCE_CACHE.get(key)
    if hit is not None:
        return hit
    # 全行読まず source / race_id 列だけ usecols
    try:
        pdf=pd.read_csv(path,encoding='utf-8-sig',usecols=lambda c: c in ('source','race_id'))
        if 'source' in pdf.columns:
            srcs=frozenset(
                s for s in pdf['source'].astype(str).str.lower().unique()
                if s in ('jra','nar')
            )
        elif 'race_id' in pdf.columns:
            from areru_engine import source_from_race_id
            srcs=frozenset(
                s for s in pdf['race_id'].map(source_from_race_id).unique()
                if s in ('jra','nar')
            )
        else:
            srcs=None
    except Exception:
        srcs=None
    # 同じパスの古い世代を残さない（mtime が変わると別キーになるため）
    for k in [k for k in _PRED_SOURCE_CACHE if k[0]==str(path)]:
        _PRED_SOURCE_CACHE.pop(k, None)
    _PRED_SOURCE_CACHE[key]=srcs
    return srcs


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
        srcs=_pred_file_sources(f)
        if srcs is None or source in srcs:
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


def _ensure_pred_file_finalized(path) -> None:
    """読み込み前に未確定CSVを厳格確定へ昇格（日次後の相対ランク戻し防止）。"""
    if not path:
        return
    try:
        from ev_analysis import ensure_predictions_file_finalized
        if ensure_predictions_file_finalized(path):
            _clear_runtime_caches()
            print(f'[rank] finalized stale predictions: {Path(path).name}', flush=True)
    except Exception as e:
        print(f'[rank] finalize skip {path}: {e}', flush=True)


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
                del rdf
            except Exception:
                need_refresh=True
        if need_refresh:
            src=source if source in ('jra','nar','all') else 'all'
            _refresh_then_predict([d], src)
        else:
            _run_replay_predict_subprocess(d)
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
    """バックグラウンドで予想。重いジョブ実行中は新規起動しない。"""
    key=f'{d}:{source}'
    if _heavy_busy() or _predict_busy():
        print(f'[predict-job] skip busy key={key} heavy={_HEAVY_JOB_STATE.get("name")} pred={_PREDICT_GLOBAL_STATE.get("key")}', flush=True)
        return False
    with _PREDICT_JOBS_LOCK:
        if key in _PREDICT_JOBS or _PREDICT_JOBS:
            # 別日付の予想が走っていれば同時起動しない
            print(f'[predict-job] skip other-running key={key} active={list(_PREDICT_JOBS)}', flush=True)
            return False
        _PREDICT_JOBS[key]='running'
    threading.Thread(target=_run_predict_job, args=(d, source), daemon=True).start()
    print(f'[predict-job] start {key}')
    return True


def _heavy_busy() -> bool:
    """他ジョブが実行中か。自スレッドの再入は busy とみなさない。"""
    if hasattr(_HEAVY_JOB_LOCK, '_is_owned') and _HEAVY_JOB_LOCK._is_owned():
        return False
    return bool(_HEAVY_JOB_STATE.get('name')) or _HEAVY_JOB_LOCK.locked()


def _predict_busy() -> bool:
    return bool(_PREDICT_GLOBAL_STATE.get('running'))


def _run_replay_predict_subprocess(day: str, *, timeout: int = 360, env=None) -> subprocess.CompletedProcess:
    """replay_predict を1本だけ実行。同時実行をプロセス内で禁止。"""
    env = dict(env or os.environ)
    if _ON_RENDER:
        env.setdefault('ARERU_SIM_RUNS', '20000')
        env.setdefault('ARERU_ORDERS_KEEP_MAX', '20000')
    acquired = _PREDICT_GLOBAL_LOCK.acquire(blocking=False)
    if not acquired:
        busy = _PREDICT_GLOBAL_STATE.get('key') or '?'
        raise RuntimeError(f'predict already running ({busy})')
    _PREDICT_GLOBAL_STATE['running'] = True
    _PREDICT_GLOBAL_STATE['key'] = str(day)
    try:
        print(f'[predict] subprocess start date={day}', flush=True)
        return subprocess.run(
            [sys.executable, 'replay_predict.py', str(day)],
            check=False, timeout=timeout, env=env,
        )
    finally:
        _PREDICT_GLOBAL_STATE['running'] = False
        _PREDICT_GLOBAL_STATE['key'] = ''
        _PREDICT_GLOBAL_LOCK.release()
        try:
            import gc
            gc.collect()
        except Exception:
            pass


def _recovery_debounce_path(source: str) -> Path:
    src = source if source in ('jra', 'nar') else 'nar'
    return DATA / f'.{src}_recovery_debounce.lock'


def _maybe_schedule_recovery(source: str, reason: str, *, actor: str = 'page') -> bool:
    """ページ閲覧からの自動復旧は無効（データ取得は cron / 明示 refresh のみ）。"""
    log_orchestrator(actor, 'SKIP', source, reason='page_fetch_disabled', detail=reason)
    return False


def _latest_ready_pred_date(source: str, *, on_or_before: str = '') -> str:
    """ソースの最新完成予想日（キャッシュ即表示用）。"""
    src = source if source in ('jra', 'nar') else 'nar'
    cutoff = str(on_or_before or _today_jst())
    best = ''
    for f in ARCH.glob('predictions_*.csv'):
        m = re.fullmatch(r'predictions_(\d{4}-\d{2}-\d{2})\.csv', f.name)
        if not m:
            continue
        day = m.group(1)
        if day > cutoff:
            continue
        if _nar_pred_ready(day, src) and day > best:
            best = day
    return best


def _fillna_pred_df(df: pd.DataFrame) -> pd.DataFrame:
    """予想CSVの欠損埋め。数値列に『なし』を入れない（float変換クラッシュ防止）。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    numeric_hints = (
        '期待値', 'オッズ', '勝率', '連対', '人気', '馬番', 'レース', '信頼度',
        '件数', '回収', 'SIM', '指数', '評価', '着', '枠', '斤量', '再現',
    )
    for c in out.columns:
        name = str(c)
        dtype = str(out[c].dtype)
        if dtype.startswith(('float', 'int', 'Int', 'Float', 'uint')):
            continue
        if any(h in name for h in numeric_hints):
            # 欠損は空のまま（後段は safe_float で処理）
            continue
        if dtype == 'object' or dtype.startswith('string'):
            out[c] = out[c].where(out[c].notna(), '')
    return out


def _read_predictions_for_venue_picker(pred_path, source: str) -> list:
    """開催場一覧用の軽量読み込み（巨大JSON列をスキップ）。"""
    _ensure_pred_file_finalized(pred_path)
    try:
        cols=pd.read_csv(pred_path, encoding='utf-8-sig', nrows=0).columns.tolist()
        use=[c for c in _NAR_VENUE_PICKER_COLS if c in cols]
        if not use:
            return []
        df=_fillna_pred_df(pd.read_csv(pred_path, encoding='utf-8-sig', usecols=use))
        if source in ('jra','nar') and 'source' in df.columns:
            df=df[df['source'].astype(str).str.lower()==source].copy()
        rows=df.to_dict('records')
        return _filter_records_by_source(rows, source)
    except Exception as e:
        print(f'[venue-picker] light read fail: {e}')
        return []


def _read_predictions_for_venue_detail(pred_path, source: str, venue: str) -> list:
    """開催場詳細用。会場で絞り込んでから dict 化（Render のメモリ・タイムアウト対策）。"""
    _ensure_pred_file_finalized(pred_path)
    from netkeiba_client import normalize_venue_name
    venue=normalize_venue_name(str(venue or '').strip())
    if not venue:
        return []
    try:
        cols=pd.read_csv(pred_path, encoding='utf-8-sig', nrows=0).columns.tolist()
        # 必要列＋開催地。無い列は無視
        want=set(_NAR_VENUE_DETAIL_COLS) | {'開催地','source','race_id'}
        use=[c for c in cols if c in want]
        if '開催地' not in use:
            # フォールバック: 全列（古いCSV）
            df=_fillna_pred_df(pd.read_csv(pred_path, encoding='utf-8-sig'))
        else:
            df=_fillna_pred_df(pd.read_csv(pred_path, encoding='utf-8-sig', usecols=use))
        if source in ('jra','nar') and 'source' in df.columns:
            df=df[df['source'].astype(str).str.lower()==source].copy()
        if df.empty:
            return []
        mask=df['開催地'].astype(str).map(lambda x: normalize_venue_name(str(x).strip())==venue)
        df=df.loc[mask].copy()
        if df.empty:
            return []
        # 予測タブでは超巨大JSON列を落とす（既に usecols で制限済みだが保険）
        drop_cols=[c for c in (
            'ワイド詳細','馬連詳細','馬単詳細','三連複詳細','三連単詳細',
        ) if c in df.columns]
        if drop_cols:
            df=df.drop(columns=drop_cols, errors='ignore')
        rows=df.to_dict('records')
        return _filter_records_by_source(rows, source)
    except Exception as e:
        print(f'[venue-detail] light read fail venue={venue}: {e}', flush=True)
        return []


def _simple_nar_tickets(race: dict) -> list:
    """地方向けのシンプル推奨馬券（本命必須の単勝・馬連・ワイド）。"""
    picks=[p for p in (race.get('予想馬') or []) if isinstance(p, dict)]
    if not picks:
        # ピックカードから本命だけでも組む
        cards=[c for c in (race.get('ピックカード一覧') or []) if isinstance(c, dict)]
        main_c=next((c for c in cards if c.get('役割')=='本命'), None)
        if main_c:
            picks=[{
                '役割': '本命',
                '馬番表示': main_c.get('馬番表示') or main_c.get('馬番'),
                '馬番': main_c.get('馬番'),
            }]
            rival=next((c for c in cards if c.get('役割')=='対抗'), None)
            dark=next((c for c in cards if c.get('役割') in ('注目馬','穴馬')), None)
            if rival:
                picks.append({'役割':'対抗','馬番表示':rival.get('馬番表示') or rival.get('馬番')})
            if dark:
                picks.append({'役割':'注目馬','馬番表示':dark.get('馬番表示') or dark.get('馬番')})
    if not picks:
        return []

    def _ban(p):
        b=str(p.get('馬番表示') or p.get('馬番') or '').strip()
        return b or '—'

    main=picks[0]
    rival=picks[1] if len(picks) > 1 else None
    dark=picks[2] if len(picks) > 2 else None
    mb=_ban(main)
    if mb == '—':
        return []
    tickets=[]

    try:
        ev=float(race.get('期待値') or race.get('レース期待回収率') or 100)
    except (TypeError, ValueError):
        ev=100.0
    is_buy=str(race.get('投資判定') or '').startswith('買い')
    star_main='★★★★★' if is_buy else '★★★☆☆'
    star_sub='★★★★☆' if is_buy else '★★★☆☆'

    # 単勝: 本命1点（必須）
    tickets.append({
        '券種': '単勝',
        '馬番表示': mb,
        '馬番': [mb],
        '的中率': None,
        '期待値': round(ev, 0) if ev else None,
        '推定回収率': round(ev, 0) if ev else None,
        '推奨度': star_main,
        'フォーメーション': None,
        '説明': f'◎本命 {mb} を単勝で素直に買う',
    })

    # 馬連: 本命－対抗（本命必須）
    if rival:
        rb=_ban(rival)
        if rb != '—':
            tickets.append({
                '券種': '馬連',
                '馬番表示': f'{mb}-{rb}',
                '馬番': [mb, rb],
                '的中率': None,
                '期待値': round(max(ev * 0.95, 90), 0),
                '推定回収率': round(max(ev * 0.95, 90), 0),
                '推奨度': star_sub,
                'フォーメーション': None,
                '説明': '◎－○ の軸で馬連1点',
            })

    # ワイド: 本命軸（対抗/注目）
    wide_parts=[]
    if rival and _ban(rival) != '—':
        wide_parts.append(f'{mb}-{_ban(rival)}')
    if dark and _ban(dark) != '—':
        wide_parts.append(f'{mb}-{_ban(dark)}')
    if wide_parts:
        tickets.append({
            '券種': 'ワイド',
            '馬番表示': ' / '.join(wide_parts[:2]),
            '馬番': wide_parts[:2],
            '的中率': None,
            '期待値': round(max(ev * 0.9, 85), 0),
            '推定回収率': round(max(ev * 0.9, 85), 0),
            '推奨度': star_sub,
            'フォーメーション': None,
            '説明': '本命軸のワイドで堅く拾う',
        })
    return tickets[:3]


def _jra_main_tickets(race: dict) -> list:
    """JRA向け: 本命を必ず含む単勝・馬連・ワイド（既存フォーメーションは維持しつつ先頭に追加）。"""
    picks=[p for p in (race.get('予想馬') or []) if isinstance(p, dict)]
    cards=[c for c in (race.get('ピックカード一覧') or []) if isinstance(c, dict)]
    main = next((p for p in picks if p.get('役割')=='本命'), picks[0] if picks else None)
    if not main:
        main_c=next((c for c in cards if c.get('役割')=='本命'), None)
        if main_c:
            main={'馬番表示': main_c.get('馬番表示') or main_c.get('馬番'), '馬番': main_c.get('馬番')}
    if not main:
        return []
    mb=str(main.get('馬番表示') or main.get('馬番') or '').strip()
    if not mb:
        return []
    rival=next((p for p in picks if p.get('役割')=='対抗'), None)
    dark=next((p for p in picks if p.get('役割') in ('注目馬','穴馬')), None)
    if not rival:
        rival_c=next((c for c in cards if c.get('役割')=='対抗'), None)
        if rival_c:
            rival={'馬番表示': rival_c.get('馬番表示') or rival_c.get('馬番')}
    if not dark:
        dark_c=next((c for c in cards if c.get('役割') in ('注目馬','穴馬')), None)
        if dark_c:
            dark={'馬番表示': dark_c.get('馬番表示') or dark_c.get('馬番')}

    def _b(p):
        return str((p or {}).get('馬番表示') or (p or {}).get('馬番') or '').strip()

    try:
        ev=float(race.get('期待値') or race.get('レース期待回収率') or 100)
    except (TypeError, ValueError):
        ev=100.0
    is_buy=str(race.get('投資判定') or '').startswith('買い')
    out=[]
    out.append({
        '券種': '単勝', '馬番表示': mb, '馬番': [mb],
        '的中率': None, '期待値': round(ev, 0), '推定回収率': round(ev, 0),
        '推奨度': '★★★★★' if is_buy else '★★★☆☆',
        'フォーメーション': None,
        '説明': f'◎本命 {mb} 単勝',
    })
    rb=_b(rival)
    if rb:
        # 馬連フォーメーション可: 本命軸で対抗＋注目
        partners=[x for x in [rb, _b(dark)] if x and x != mb]
        partners=list(dict.fromkeys(partners))
        if len(partners) >= 2:
            out.append({
                '券種': '馬連',
                '馬番表示': f'{mb}－{"・".join(partners)}',
                '馬番': [mb] + partners,
                '的中率': None,
                '期待値': round(max(ev * 0.95, 90), 0),
                '推定回収率': round(max(ev * 0.95, 90), 0),
                '推奨度': '★★★★☆' if is_buy else '★★★☆☆',
                'フォーメーション': {'軸': [mb], '相手': partners, '点数': len(partners)},
                '説明': '本命軸の馬連フォーメーション',
            })
        else:
            out.append({
                '券種': '馬連', '馬番表示': f'{mb}-{rb}', '馬番': [mb, rb],
                '的中率': None, '期待値': round(max(ev * 0.95, 90), 0),
                '推定回収率': round(max(ev * 0.95, 90), 0),
                '推奨度': '★★★★☆' if is_buy else '★★★☆☆',
                'フォーメーション': None,
                '説明': '◎－○ 馬連',
            })
        wide_parts=[f'{mb}-{rb}']
        db=_b(dark)
        if db and db != rb:
            wide_parts.append(f'{mb}-{db}')
        out.append({
            '券種': 'ワイド',
            '馬番表示': ' / '.join(wide_parts[:2]),
            '馬番': wide_parts[:2],
            '的中率': None,
            '期待値': round(max(ev * 0.9, 85), 0),
            '推定回収率': round(max(ev * 0.9, 85), 0),
            '推奨度': '★★★★☆' if is_buy else '★★★☆☆',
            'フォーメーション': {'軸': [mb], '相手': [x for x in [rb, db] if x], '点数': len(wide_parts)},
            '説明': '本命軸ワイド',
        })
    return out[:3]


def _clear_runtime_caches():
    """runners / predictions 更新後に日付・検証キャッシュを捨てる。"""
    _DATES_CACHE.clear()
    _VERIFY_CACHE.clear()
    _PRED_META_CACHE['sig']=None
    _PRED_META_CACHE['data']={}


def _clear_runtime_caches_logged(source: str, date_str: str = '', *, actor: str = 'pipeline') -> None:
    try:
        _clear_runtime_caches()
        log_pipeline_stage(source, 6, True, date_str, actor=actor)
    except Exception as e:
        log_pipeline_stage(source, 6, False, date_str, error=str(e)[:120], actor=actor)


_NAR_JOB_STATUS=DATA/'.nar_job_status.json'  # 互換: 地方は従来パス
_JOB_STALE_SEC=10*60  # 10分超の running は自動失敗（updating 解除）
_NAR_JOB_STALE_SEC=_JOB_STALE_SEC
_NAR_BOOT_SEEDED=False
_JRA_BOOT_SEEDED=False
_NAR_BOOT_LOCK=threading.Lock()
_JRA_BOOT_LOCK=threading.Lock()


def _job_status_path(source: str = 'nar') -> Path:
    src = source if source in ('jra', 'nar') else 'nar'
    if src == 'nar':
        return _NAR_JOB_STATUS
    return DATA / f'.{src}_job_status.json'


def _write_job_status(
    source: str = 'nar',
    state: str = 'idle',
    stage: str = '',
    message: str = '',
    date_str: str = '',
    error: str = '',
) -> None:
    """ソース別ジョブ進捗を永続化（UIが取得中のまま固まらないようにする）。"""
    src = source if source in ('jra', 'nar') else 'nar'
    today = _today_jst()
    # 当日予想が揃っているのに、前日ジョブ成功で status.date を巻き戻さない
    if state == 'success' and date_str and date_str < today and _nar_pred_ready(today, src):
        print(
            f'[pipeline] 保存 スキップ status巻き戻し防止 '
            f'source={src} date={date_str}→today={today} (当日ready維持)',
            flush=True,
        )
        date_str = today
        message = message or '取得完了'
    now_iso = datetime.now(JST).isoformat(timespec='seconds')
    prev = _read_job_status(src)
    started_at = str(prev.get('started_at') or '')
    if state == 'running':
        # 進捗書き込みでタイマーをリセットしない（開始時刻を維持）
        if str(prev.get('state') or '') != 'running' or not started_at:
            started_at = now_iso
    else:
        started_at = ''
    payload = {
        'state': state,  # running | success | error | idle
        'stage': stage,
        'message': message,
        'date': date_str or '',
        'source': src,
        'error': error or '',
        'updated_at': now_iso,
        'started_at': started_at,
        'pid': os.getpid(),
        # UI用: updating | success | failed | timeout
        'outcome': (
            'updating' if state == 'running'
            else ('timeout' if stage in ('timeout', 'timeout_ready') or 'タイムアウト' in (message or '')
                  else ('success' if state == 'success' else ('failed' if state == 'error' else 'idle')))
        ),
    }
    path = _job_status_path(src)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    except Exception as e:
        print(f'[{src}-job] status write fail: {e}', flush=True)
    print(f'[{src}-job] {state} | {stage} | {message or error} | date={date_str or "-"}', flush=True)


def _write_nar_job_status(
    state: str,
    stage: str = '',
    message: str = '',
    date_str: str = '',
    error: str = '',
) -> None:
    """互換ラッパー（地方）。"""
    _write_job_status(
        'nar', state=state, stage=stage, message=message,
        date_str=date_str, error=error,
    )


def _nar_day_counts(date_str: str, source: str = 'nar') -> dict:
    """工程ログ用: runners / predictions / odds の当日件数。"""
    out={
        'runners_races': 0, 'runners_venues': 0, 'runners_rows': 0,
        'pred_races': 0, 'pred_venues': 0, 'venues': [], 'odds_json': 0,
    }
    if not date_str:
        return out
    try:
        if RUNNERS.exists():
            rdf=pd.read_csv(
                RUNNERS, encoding='utf-8-sig',
                usecols=lambda c: c in ('race_id','日付','source'),
            )
            if not rdf.empty and '日付' in rdf.columns:
                # 会場一覧と同じく先頭10桁で突合（時刻付き日付の取りこぼし防止）
                day=rdf[rdf['日付'].astype(str).str[:10]==str(date_str)[:10]]
                if source in ('jra','nar') and 'source' in day.columns:
                    day=day[day['source'].astype(str).str.lower()==source]
                out['runners_rows']=int(len(day))
                if 'race_id' in day.columns and not day.empty:
                    rids=day['race_id'].astype(str)
                    out['runners_races']=int(rids.nunique())
                    codes=sorted({rid[4:6] for rid in rids if len(rid) >= 6})
                    out['runners_venues']=len(codes)
                    # odds_tickets: race_id.json
                    odds_dir=DATA/'odds_tickets'
                    if odds_dir.exists():
                        out['odds_json']=sum(
                            1 for rid in rids.unique().tolist()
                            if (odds_dir/f'{rid}.json').exists()
                        )
    except Exception as e:
        print(f'[pipeline] runners件数取得失敗: {e}', flush=True)
    try:
        pred=ARCH/f'predictions_{date_str}.csv'
        if pred.exists():
            pdf=pd.read_csv(
                pred, encoding='utf-8-sig',
                usecols=lambda c: c in ('race_id','source','開催地'),
            )
            if source in ('jra','nar') and 'source' in pdf.columns:
                pdf=pdf[pdf['source'].astype(str).str.lower()==source]
            if not pdf.empty:
                out['pred_races']=int(pdf['race_id'].astype(str).nunique()) if 'race_id' in pdf.columns else int(len(pdf))
                if '開催地' in pdf.columns:
                    venues=sorted({str(v) for v in pdf['開催地'].dropna().astype(str) if v and v != 'なし'})
                    out['pred_venues']=len(venues)
                    out['venues']=venues
    except Exception as e:
        print(f'[pipeline] predictions件数取得失敗: {e}', flush=True)
    return out


def _pipeline_log(stage: str, result: str, date_str: str = '', **counts) -> None:
    """各工程の 成功/失敗/保存件数 を統一ログ。"""
    bits=[f'[pipeline] {stage} {result}']
    if date_str:
        bits.append(f'date={date_str}')
    for k, v in counts.items():
        if v is None or v == '' or v == []:
            continue
        bits.append(f'{k}={v}')
    print(' '.join(bits), flush=True)


def _read_job_status(source: str = 'nar') -> dict:
    path = _job_status_path(source)
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_nar_job_status() -> dict:
    return _read_job_status('nar')


def _nar_job_age_sec(st: dict) -> float:
    """running 中は started_at 基準（進捗更新でタイマーが延びない）。"""
    state = str((st or {}).get('state') or '')
    if state == 'running':
        raw = str((st or {}).get('started_at') or (st or {}).get('updated_at') or '')
    else:
        raw = str((st or {}).get('updated_at') or (st or {}).get('started_at') or '')
    if not raw:
        return 1e9
    try:
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=JST)
        return max(0.0, (datetime.now(JST) - ts).total_seconds())
    except Exception:
        return 1e9


def _expire_stale_job(source: str = 'nar') -> dict:
    """10分超の running を確定終了し、updating を必ず解除する。"""
    src = source if source in ('jra', 'nar') else 'nar'
    st = _read_job_status(src)
    state = str(st.get('state') or '')
    date_str = str(st.get('date') or _today_jst())
    ready = _nar_pred_ready(date_str, src) or _nar_pred_ready(_today_jst(), src)

    # 予想が完成しているのに error のまま → success に修復（表示と整合）
    if state == 'error' and ready:
        day = _today_jst() if _nar_pred_ready(_today_jst(), src) else date_str
        _write_job_status(
            src, state='success', stage='done',
            message='更新成功', date_str=day,
        )
        print(f'[{src}-job] HEAL error→success (predictions ready)', flush=True)
        return _read_job_status(src)

    if state != 'running':
        return st
    age = _nar_job_age_sec(st)
    if age <= _JOB_STALE_SEC:
        return st
    date_str = str(st.get('date') or _today_jst())
    ready = _nar_pred_ready(date_str, src)
    _clear_stale_source_locks(src)
    mins = max(1, int(_JOB_STALE_SEC // 60))
    if ready:
        _write_job_status(
            src, state='success', stage='timeout_ready',
            message='更新タイムアウト（表示データは維持）',
            date_str=date_str,
            error=f'running>{_JOB_STALE_SEC}s',
        )
    else:
        _write_job_status(
            src, state='error', stage='timeout',
            message='更新タイムアウト',
            date_str=date_str,
            error=f'{mins}分以内に完了しませんでした（Render切断の可能性）',
        )
    print(
        f'[{src}-job] AUTO-EXPIRE age={age:.0f}s ready={int(ready)} date={date_str}',
        flush=True,
    )
    return _read_job_status(src)


def _job_in_failure_cooldown(source: str = 'nar', sec: float = 180.0) -> bool:
    """失敗直後の連続再起動を防ぐ（Render Free の無限ループ対策）。"""
    st = _read_job_status(source)
    if str(st.get('state') or '') != 'error':
        return False
    age = _nar_job_age_sec(st)
    return age < sec


def _job_is_active(source: str = 'nar') -> bool:
    """有効な更新ジョブが走っているか（期限切れは先に解除）。"""
    st = _expire_stale_job(source)
    return (
        str(st.get('state') or '') == 'running'
        and _nar_job_age_sec(st) < _JOB_STALE_SEC
    )


def _finalize_job_if_running(
    source: str,
    *,
    ok: bool,
    date_str: str = '',
    message: str = '',
    error: str = '',
) -> None:
    """例外・途中切断でも running を残さない（finally 用）。"""
    src = source if source in ('jra', 'nar') else 'nar'
    st = _read_job_status(src)
    if str(st.get('state') or '') != 'running':
        return
    day = date_str or str(st.get('date') or _today_jst())
    if ok:
        _write_job_status(
            src, state='success', stage='done',
            message=message or '取得完了', date_str=day,
        )
    else:
        _write_job_status(
            src, state='error', stage='failed',
            message=message or '取得失敗', date_str=day,
            error=error or 'ジョブが running のまま終了',
        )


def _clear_stale_source_locks(source: str = 'nar') -> None:
    """死んだロックを掃除して取得中の永久表示を防ぐ。"""
    src = source if source in ('jra', 'nar') else 'nar'
    for p in DATA.glob(f'.{src}_*.lock'):
        try:
            age = __import__('time').time() - p.stat().st_mtime
            if age > _JOB_STALE_SEC:
                p.unlink(missing_ok=True)
                print(f'[{src}-job] stale lock removed: {p.name}', flush=True)
        except Exception:
            pass


def _clear_stale_nar_locks() -> None:
    _clear_stale_source_locks('nar')


def resolve_fetch_status(
    selected_date: str = '',
    *,
    source: str = 'nar',
    force_refresh: bool = False,
) -> tuple:
    """UI用: (data_status, message)。地方・JRA共通。

    キャッシュ優先: 予想が実在すれば絶対に generating にしない。
    generating はキャッシュが無い初回のみ。更新中は updating（表示は維持）。
    """
    src = source if source in ('jra', 'nar') else 'nar'
    _clear_stale_source_locks(src)
    st = _expire_stale_job(src)
    state = str(st.get('state') or 'idle')
    age = _nar_job_age_sec(st)
    msg = str(st.get('message') or '')
    err = str(st.get('error') or '')
    today = _today_jst()
    selected_date = str(selected_date or '')
    today_ready = _nar_pred_ready(today, src)
    selected_ready = (
        _nar_pred_ready(selected_date, src) if selected_date else False
    )
    after_deadline = past_predict_deadline()
    job_running = state == 'running'
    if not job_running:
        for name in (f'.{src}_today_pipeline.lock', f'.{src}_bootstrap.lock'):
            lp = DATA / name
            if lp.exists():
                try:
                    if (__import__('time').time() - lp.stat().st_mtime) < _JOB_STALE_SEC:
                        job_running = True
                        break
                except Exception:
                    pass

    # 完成キャッシュあり → 常に表示可能（裏更新中のみ updating）
    if selected_ready or (selected_date == today and today_ready):
        if selected_date == today and today_ready:
            auto_seal_if_ready(today, src, True)
        if job_running:
            return 'updating', (msg or 'データ更新中')
        if state == 'error':
            stage = str(st.get('stage') or '')
            if stage in ('timeout', 'timeout_ready') or 'タイムアウト' in msg:
                return 'timeout', msg or '更新タイムアウト（表示データは維持）'
            return 'ready', '取得完了'
        return 'ready', (msg or '取得完了') if state == 'success' else ''

    # キャッシュなし
    if state == 'error':
        stage = str(st.get('stage') or '')
        if stage in ('timeout', 'timeout_ready') or 'タイムアウト' in msg:
            return 'timeout', msg or '更新タイムアウト'
        return 'error', '更新失敗' + (f'（{err}）' if err and len(err) < 80 else '')

    if selected_date == today and not today_ready:
        if _today_source_has_no_meeting(src, today) or (
            after_deadline and not source_is_race_day(src, today)
        ):
            return 'ready', (
                '本日はJRAの開催はありません' if src == 'jra'
                else '本日は地方競馬の開催はありません'
            )
        # 初回のみ取得中。ページからは起動しない（cron 待ち）
        if job_running or force_refresh:
            return 'generating', (msg or 'データ取得中')
        return 'generating', 'データ取得中'

    if selected_date and not selected_ready:
        return 'error', '取得失敗'
    return 'ready', ''


def resolve_nar_fetch_status(selected_date: str = '', *, force_refresh: bool = False) -> tuple:
    return resolve_fetch_status(selected_date, source='nar', force_refresh=force_refresh)


def _nar_pred_ready(date_str: str, source: str = 'nar') -> bool:
    """指定日の地方予想が実在するか（ファイル有無だけでなく source 行を確認）。"""
    if not date_str:
        return False
    f=ARCH/f'predictions_{date_str}.csv'
    if not f.exists() or f.stat().st_size < 32:
        return False
    try:
        pdf=pd.read_csv(f, encoding='utf-8-sig', usecols=lambda c: c in ('source','race_id','開催地'))
        if pdf.empty:
            return False
        if source in ('jra','nar') and 'source' in pdf.columns:
            return bool((pdf['source'].astype(str).str.lower()==source).any())
        return True
    except Exception:
        return False


def _stay_on_selected_calendar_day(selected: str | None, source: str = '') -> bool:
    """JRAのカレンダー当日は、別日の完成カードへフォールバックしない。

    地方は当日未完成なら最新の NAR 完成日を出してよい。
    JRA非開催日に日曜カードを出すのは禁止。
    """
    if str(source or '').lower() != 'jra':
        return False
    return bool(selected) and selected == _today_jst()


def _today_source_has_no_meeting(source: str, today: str) -> bool:
    """当日この source の開催カードが無いと確定できるか。

    出走も予想も無い「あとで生成される土曜朝」は False（取得中）。
    朝8時以降かつ runners にも行が無いときだけ開催なし。
    """
    if source not in ('jra', 'nar'):
        return False
    if _nar_pred_ready(today, source):
        return False
    if _runners_need_source(today, source):
        return False
    return past_predict_deadline()


def _nar_venues_from_runners(date_str: str) -> list:
    """予想CSV前でも runners から開催場一覧を出す（地方は毎日開催のため）。"""
    if not date_str or not RUNNERS.exists():
        return []
    try:
        from areru_engine import NAR_VENUE_CODES, source_from_race_id
        from netkeiba_client import normalize_venue_name
        rdf=pd.read_csv(
            RUNNERS, encoding='utf-8-sig',
            usecols=lambda c: c in ('race_id', '日付', 'source', '開催地', 'レース'),
        )
        if rdf.empty or '日付' not in rdf.columns or 'race_id' not in rdf.columns:
            return []
        day=rdf[rdf['日付'].astype(str).str[:10] == str(date_str)]
        if day.empty:
            return []
        if 'source' in day.columns:
            day=day[day['source'].astype(str).str.lower() == 'nar']
        else:
            day=day[day['race_id'].map(source_from_race_id) == 'nar']
        if day.empty:
            return []
        seen=set()
        records=[]
        for _, row in day.iterrows():
            rid=str(row.get('race_id') or '')
            if not rid or rid in seen:
                continue
            seen.add(rid)
            venue=''
            if '開催地' in day.columns:
                venue=normalize_venue_name(str(row.get('開催地') or '').strip())
            if not venue:
                code=rid[4:6] if len(rid) >= 6 else ''
                venue=NAR_VENUE_CODES.get(code, '')
            if not venue:
                continue
            race_no=row.get('レース') if 'レース' in day.columns else None
            if race_no in (None, ''):
                try:
                    race_no=int(rid[-2:]) if len(rid) >= 2 else 0
                except Exception:
                    race_no=0
            records.append({'開催地': venue, 'レース': race_no, '勝負ランク': ''})
        return _venue_meetings(records)
    except Exception as e:
        print(f'[nar-venues] runners fallback fail: {e}', flush=True)
        return []


def _ensure_source_today_seeded(source: str = 'nar') -> None:
    """起動時シード。完成済みなら触らない。未完成のみ朝の本生成/復旧。"""
    global _NAR_BOOT_SEEDED, _JRA_BOOT_SEEDED
    src = source if source in ('jra', 'nar') else 'nar'
    lock = _NAR_BOOT_LOCK if src == 'nar' else _JRA_BOOT_LOCK
    with lock:
        if src == 'nar':
            if _NAR_BOOT_SEEDED:
                return
            _NAR_BOOT_SEEDED = True
        else:
            if _JRA_BOOT_SEEDED:
                return
            _JRA_BOOT_SEEDED = True
    if str(os.environ.get('ARERU_SKIP_NAR_BOOT') or '').strip() == '1':
        return
    # Render では既定で起動シードをスキップ（NAR+JRA 同時起動→OOM→白画面の主因）
    skip_boot = str(os.environ.get('ARERU_SKIP_BOOT') or _SKIP_BOOT_DEFAULT).strip()
    if skip_boot == '1':
        print(f'[{src}-boot] skipped (ARERU_SKIP_BOOT={skip_boot})', flush=True)
        return
    today = _today_jst()
    if _nar_pred_ready(today, src):
        auto_seal_if_ready(today, src, True)
        print(f'[{src}-boot] today ready date={today} sealed={is_sealed(today, src)}', flush=True)
        return
    if past_predict_deadline() and not source_is_race_day(src, today):
        _write_job_status(
            src, state='success', stage='done', message='本日は開催なし', date_str=today,
        )
        return

    def _run():
        import time
        time.sleep(2 if src == 'nar' else 4)
        try:
            print(f'[{src}-boot] seeding today pipeline date={today}', flush=True)
            _write_job_status(
                src, state='running', stage='boot', message='データ取得中', date_str=today,
            )
            # 締切前=本生成、締切後未完成=復旧本生成
            run_today_pipeline(src, force=True, force_full=True)
        except Exception as e:
            print(f'[{src}-boot] fail: {e}', flush=True)

    try:
        threading.Thread(target=_run, daemon=True, name=f'{src}-boot-seed').start()
    except Exception as e:
        print(f'[{src}-boot] start fail: {e}', flush=True)


def _ensure_nar_today_seeded() -> None:
    _ensure_source_today_seeded('nar')


def _force_calendar_today(explicit_date: str, want_today: bool, mode: str, allow_past: bool) -> bool:
    """中央・地方とも原則カレンダー当日。履歴閲覧は history=1 のときだけ前日を許可。"""
    today = _today_jst()
    if want_today:
        return True
    if allow_past:
        return False
    if mode not in ('predict', 'result'):
        return False
    if not explicit_date:
        return True
    if explicit_date < today:
        return True
    return False


def _nar_force_calendar_today(explicit_date: str, want_today: bool, mode: str, allow_past: bool) -> bool:
    return _force_calendar_today(explicit_date, want_today, mode, allow_past)


def _run_serialized_heavy(name: str, fn, *, wait: bool = False) -> bool:
    """重いジョブを1本化。busy時はスキップ（wait=Trueなら完了待ち）。

    同一スレッドの再入は RLock で許可（odds-only が refresh 内から呼ばれる経路用）。
    """
    holding_before = hasattr(_HEAVY_JOB_LOCK, '_is_owned') and _HEAVY_JOB_LOCK._is_owned()
    if holding_before:
        prev = _HEAVY_JOB_STATE.get('name') or ''
        _HEAVY_JOB_STATE['name'] = name
        try:
            print(f'[heavy] nested start {name} (under {prev})', flush=True)
            fn()
            print(f'[heavy] nested done {name}', flush=True)
            return True
        except Exception as e:
            print(f'[heavy] nested fail {name}: {e}', flush=True)
            raise
        finally:
            _HEAVY_JOB_STATE['name'] = prev

    timeout = 1800 if wait else 0
    acquired = _HEAVY_JOB_LOCK.acquire(timeout=timeout) if wait else _HEAVY_JOB_LOCK.acquire(blocking=False)
    if not acquired:
        busy = _HEAVY_JOB_STATE.get('name') or '?'
        log_orchestrator('heavy', 'SKIP', '', reason='lock_busy', name=name, busy=busy, wait=int(wait))
        print(f'[heavy] busy ({busy}), skip {name}', flush=True)
        return False
    _HEAVY_JOB_STATE['name'] = name
    try:
        print(f'[heavy] start {name}', flush=True)
        fn()
        print(f'[heavy] done {name}', flush=True)
        return True
    except Exception as e:
        print(f'[heavy] fail {name}: {e}', flush=True)
        raise
    finally:
        _HEAVY_JOB_STATE['name'] = ''
        _HEAVY_JOB_LOCK.release()


def _refresh_then_predict(
    dates: list[str],
    source: str,
    extra_args: list[str] | None = None,
) -> bool:
    """開催取得 → レース取得 → AI予想。各段階をログ＆状態ファイルに残す。

    Returns: 成功なら True
    """
    import time as _time
    days=[d for d in dates if d and re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(d))]
    if not days:
        _pipeline_log('開始', '失敗', error='対象日なし')
        status_src=source if source in ('jra','nar') else 'nar'
        _write_job_status(status_src, state='error', stage='failed', message='取得失敗', error='対象日なし')
        return False
    src=source if source in ('jra','nar','all') else 'all'
    count_src=src if src in ('jra','nar') else 'nar'
    status_src=count_src
    today=_today_jst()
    # 当日を含む場合は当日を primary にし、status が前日へ寄らないようにする
    primary=today if today in days else days[0]
    terminal=False
    t_all=_time.perf_counter()
    print(f'[Local Update] START source={status_src} dates={days}', flush=True)
    try:
        # 1) 取得開始
        _write_job_status(
            status_src, state='running', stage='start', message='データ取得中',
            date_str=primary,
        )
        log_orchestrator('pipeline', 'START', status_src, dates=','.join(days))
        _pipeline_log('開始', '成功', primary, dates=','.join(days), source=src)
        print('[Local Update] 取得開始', flush=True)

        cmd=[
            sys.executable,'refresh_data.py',
            '--dates',*days,
            '--source',src,
            '--no-discover',
            '--skip-predict',
        ]
        if extra_args:
            cmd.extend(extra_args)
        env=os.environ.copy()
        env['ARERU_PIPELINE_SOURCE'] = status_src
        # Render Free: NAR は高速モード（履歴ネット取得抑制）
        if status_src == 'nar' and (
            str(env.get('RENDER') or '').lower() in ('true', '1')
            or str(env.get('ARERU_FAST_NAR') or '') in ('1', 'true', 'yes')
        ):
            env['ARERU_FAST_NAR']='1'
            env.setdefault('ARERU_FETCH_BUDGET_SEC', '240')
            env.setdefault('ARERU_REFRESH_TIMEOUT_SEC', '360')
            print('[Local Update] FAST_NAR=1 (Render予算内完了)', flush=True)
        print(f'[{status_src}-job] 開催・レース取得開始: {" ".join(cmd)}', flush=True)
        t_fetch=_time.perf_counter()
        fetch_timeout=int(env.get('ARERU_REFRESH_TIMEOUT_SEC') or (360 if status_src=='nar' else 480))
        refresh_ok = True
        try:
            # Render Free: サブプロセスを短めに打ち切り、部分 runners があれば予想へ進む
            rc=subprocess.run(cmd, check=False, timeout=fetch_timeout, env=env)
            fetch_sec=_time.perf_counter()-t_fetch
            print(f'[Local Update] 開催・レース取得 subprocess終了 sec={fetch_sec:.1f} rc={rc.returncode}', flush=True)
            if rc.returncode != 0:
                refresh_ok = False
                _pipeline_log('開催取得', '失敗', primary, 保存件数=0, error=f'refresh_data rc={rc.returncode}')
                # 部分保存があれば予想へ（会場スキップ後の途中成果を活かす）
                partial = any(_nar_day_counts(d, count_src)['runners_races'] > 0 for d in days)
                if not partial:
                    log_pipeline_stage(status_src, 1, False, primary, error=f'refresh_rc={rc.returncode}')
                    log_pipeline_stage(status_src, 2, False, primary, error=f'refresh_rc={rc.returncode}')
                    log_pipeline_stage(status_src, 3, False, primary, error=f'refresh_rc={rc.returncode}')
                    _pipeline_log('レース取得', '失敗', primary, 保存件数=0)
                    raise RuntimeError(f'refresh_data 終了コード {rc.returncode}')
                print('[Local Update] refresh_data非0だが runners あり → 予想へ継続', flush=True)
        except subprocess.TimeoutExpired as e:
            refresh_ok = False
            fetch_sec=_time.perf_counter()-t_fetch
            print(f'[Local Update] 取得TIMEOUT sec={fetch_sec:.1f}: {e}', flush=True)
            _clear_runtime_caches_logged(status_src, primary)
            partial = any(_nar_day_counts(d, count_src)['runners_races'] > 0 for d in days)
            if not partial:
                log_pipeline_stage(status_src, 1, False, primary, error='refresh_timeout')
                log_pipeline_stage(status_src, 2, False, primary, error='refresh_timeout')
                log_pipeline_stage(status_src, 3, False, primary, error='refresh_timeout')
                raise RuntimeError(
                    f'refresh_data タイムアウト({fetch_timeout}s / Render制限の可能性)'
                ) from e
            print(
                f'[Local Update] TIMEOUTだが runners 部分あり → AI予想へ継続',
                flush=True,
            )
            _write_job_status(
                status_src, state='running', stage='races_partial',
                message='レース部分取得（タイムアウト継続）', date_str=primary,
            )

        # 2) 開催取得 / レース取得 / 出馬表 — runners 件数で検証
        for d in days:
            c=_nar_day_counts(d, count_src)
            if c['runners_races'] <= 0:
                # 中央の平日など「開催なし」は失敗ではなく完了扱い
                if len(days) == 1 and d == today:
                    log_pipeline_stage(status_src, 1, True, d, reason='no_meeting', races=0)
                    log_pipeline_stage(status_src, 2, True, d, reason='no_meeting', races=0)
                    log_pipeline_stage(status_src, 3, True, d, reason='no_meeting', horses=0)
                    _pipeline_log('開催取得', '成功', d, 開催場数=0, レース数=0, 保存件数=0, reason='no_meeting')
                    _write_job_status(
                        status_src, state='success', stage='done',
                        message='本日は開催なし', date_str=d,
                    )
                    print('[Local Update] END no_meeting', flush=True)
                    terminal=True
                    return True
                log_pipeline_stage(status_src, 1, False, d, venues=0, races=0)
                log_pipeline_stage(status_src, 2, False, d, races=0)
                log_pipeline_stage(status_src, 3, False, d, horses=c['runners_rows'])
                _pipeline_log('開催取得', '失敗', d, 開催場数=0, レース数=0, 保存件数=0)
                _pipeline_log('レース取得', '失敗', d, 保存件数=0)
                raise RuntimeError(f'{d}: 開催・レースが runners に保存されませんでした')
            log_pipeline_stage(
                status_src, 1, True, d,
                venues=c['runners_venues'], races=c['runners_races'],
            )
            log_pipeline_stage(
                status_src, 2, True, d,
                races=c['runners_races'], venues=c['runners_venues'],
            )
            log_pipeline_stage(
                status_src, 3, True, d,
                horses=c['runners_rows'], races=c['runners_races'],
            )
            _pipeline_log(
                '開催取得', '成功', d,
                開催場数=c['runners_venues'], レース数=c['runners_races'], 保存件数=c['runners_races'],
            )
            _pipeline_log(
                'レース取得', '成功', d,
                レース数=c['runners_races'], 出走頭数=c['runners_rows'], 保存件数=c['runners_races'],
            )
            print(
                f'[Local Update] 開催場取得完了 venues={c["runners_venues"]} '
                f'レース取得完了 races={c["runners_races"]}',
                flush=True,
            )
        _write_job_status(
            status_src, state='running', stage='venues_done', message='開催取得完了',
            date_str=primary,
        )
        _write_job_status(
            status_src, state='running', stage='races_done', message='レース取得完了',
            date_str=primary,
        )

        # 3) AI予想
        for d in days:
            _write_job_status(
                status_src, state='running', stage='predict_start', message='AI予想生成中',
                date_str=d,
            )
            _pipeline_log('AI予想生成', '開始', d)
            print(f'[Local Update] AI予想生成開始 date={d}', flush=True)
            t_pred=_time.perf_counter()
            try:
                pr=_run_replay_predict_subprocess(d, timeout=360, env=env)
            except subprocess.TimeoutExpired as e:
                log_pipeline_stage(status_src, 4, False, d, error='predict_timeout')
                print(f'[Local Update] AI予想TIMEOUT date={d}: {e}', flush=True)
                raise RuntimeError(f'replay_predict タイムアウト ({d})') from e
            except RuntimeError as e:
                log_pipeline_stage(status_src, 4, False, d, error=str(e)[:120])
                print(f'[Local Update] AI予想スキップ date={d}: {e}', flush=True)
                raise
            pred_sec=_time.perf_counter()-t_pred
            print(f'[Local Update] AI予想生成終了 date={d} sec={pred_sec:.1f} rc={pr.returncode}', flush=True)
            if pr.returncode != 0:
                log_pipeline_stage(status_src, 4, False, d, error=f'rc={pr.returncode}')
                _pipeline_log('AI予想生成', '失敗', d, 保存件数=0, error=f'rc={pr.returncode}')
                raise RuntimeError(f'replay_predict 終了コード {pr.returncode} ({d})')
            pred=ARCH/f'predictions_{d}.csv'
            if not pred.exists():
                log_pipeline_stage(status_src, 4, False, d, error='predictions_missing')
                log_pipeline_stage(status_src, 5, False, d, file=predictions_label(status_src, d))
                _pipeline_log('AI予想生成', '失敗', d, 保存件数=0, error='predictions未生成')
                raise RuntimeError(f'predictions_{d}.csv が生成されませんでした')
            c=_nar_day_counts(d, count_src)
            if c['pred_races'] <= 0:
                if len(days) == 1 and d == today:
                    log_pipeline_stage(status_src, 4, True, d, reason='no_meeting', races=0)
                    log_pipeline_stage(status_src, 5, True, d, file=predictions_label(status_src, d), races=0)
                    _pipeline_log('AI予想生成', '成功', d, 保存件数=0, reason='no_meeting')
                    _write_job_status(
                        status_src, state='success', stage='done',
                        message='本日は開催なし', date_str=d,
                    )
                    terminal=True
                    return True
                log_pipeline_stage(status_src, 4, False, d, error='pred_zero')
                log_pipeline_stage(status_src, 5, False, d, file=predictions_label(status_src, d))
                _pipeline_log('AI予想生成', '失敗', d, 保存件数=0, error='予想0件')
                raise RuntimeError(f'{d}: 予想が0件です')
            log_pipeline_stage(status_src, 4, True, d, races=c['pred_races'], venues=c['pred_venues'])
            log_pipeline_stage(
                status_src, 5, True, d,
                file=predictions_label(status_src, d),
                races=c['pred_races'], path=str(pred.name),
            )
            _pipeline_log(
                'AI予想生成', '成功', d,
                レース数=c['pred_races'], 開催場数=c['pred_venues'], 保存件数=c['pred_races'],
                開催場=','.join(c['venues'][:12]),
            )
            print(
                f'[Local Update] DB保存完了 predictions={c["pred_races"]} '
                f'venues={c["pred_venues"]}',
                flush=True,
            )
            _write_job_status(
                status_src, state='running', stage='predict_done', message='AI予想完了',
                date_str=d,
            )

        # 成功は「取得できた日の予想が実在するとき」だけ。失敗時に前日へ切り替えない。
        ok_days=[d for d in days if _nar_pred_ready(d, count_src)]
        if not ok_days:
            log_pipeline_stage(status_src, 5, False, primary, file=predictions_label(status_src, primary))
            _pipeline_log('保存', '失敗', primary, 保存件数=0, error='予想readyなし')
            raise RuntimeError('予想ファイルが生成されませんでした')
        # status.date は当日優先（前日成功で表示が巻き戻るのを防ぐ）
        if today in ok_days:
            primary_ok=today
        elif primary in ok_days:
            primary_ok=primary
        else:
            primary_ok=max(ok_days)
        _clear_runtime_caches_logged(status_src, primary_ok)
        for d in ok_days:
            c=_nar_day_counts(d, count_src)
            _pipeline_log(
                '保存', '成功', d,
                predictions=c['pred_races'], runners=c['runners_races'],
                odds_json=c['odds_json'], 保存件数=c['pred_races'],
            )
        _write_job_status(
            status_src, state='success', stage='done', message='更新成功',
            date_str=primary_ok,
        )
        total_sec=_time.perf_counter()-t_all
        _pipeline_log('全体', '成功', primary_ok, ok_days=','.join(ok_days), 保存件数=len(ok_days))
        print(f'[Local Update] END success total_sec={total_sec:.1f}', flush=True)
        terminal=True
        return True
    except Exception as e:
        _clear_runtime_caches_logged(status_src, primary, actor='pipeline_error')
        err=str(e)[:200]
        is_timeout=('タイムアウト' in err) or ('Timeout' in err) or ('timeout' in err.lower())
        _write_job_status(
            status_src,
            state='error',
            stage='timeout' if is_timeout else 'failed',
            message='更新タイムアウト' if is_timeout else '更新失敗',
            date_str=primary, error=err,
        )
        _pipeline_log('全体', '失敗', primary, 保存件数=0, error=err)
        print(
            f'[Local Update] END {"timeout" if is_timeout else "failed"} '
            f'total_sec={_time.perf_counter()-t_all:.1f} err={err}',
            flush=True,
        )
        terminal=True
        return False
    finally:
        # Render SIGKILL 以外では running を残さない
        if not terminal:
            _finalize_job_if_running(
                status_src, ok=False, date_str=primary,
                message='更新失敗', error='パイプラインが予期せず終了',
            )
            print('[Local Update] END unexpected (finally cleared running)', flush=True)



def ensure_for_page(d, source='all'):
    """ページ表示用。同期再生成も裏ジョブ起動もしない。 (path|None, status)

    キャッシュ優先: ファイルがあれば即 ready。無い初回のみ generating。
    データ取得は cron / 明示 refresh のみ。
    """
    f=ARCH/f'predictions_{d}.csv'
    try:
        if f.exists():
            _ensure_pred_file_finalized(f)
        src = source if source in ('jra', 'nar') else 'all'
        today = _today_jst()
        if src in ('jra', 'nar') and _nar_pred_ready(str(d), src):
            if (
                str(d) == today
                and (past_predict_deadline() or is_sealed(str(d), src))
            ):
                auto_seal_if_ready(str(d), src, True)
            return f, 'ready'
        if (
            src in ('jra', 'nar')
            and not _nar_pred_ready(str(d), src)
            and not _runners_need_source(str(d), src)
        ):
            # 他sourceのCSVがあっても、このsourceの開催カードではない
            return None, 'ready'
        if f.exists() and not _need_regen(d, source):
            return f, 'ready'
        # キャッシュ無し → ページからは取得しない（cron 待ち）
        if f.exists():
            # ファイルはあるが表示ソース行が無い等
            return None, 'generating'
        return None, 'generating'
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


AREru_PREDICTIONS = DATA / 'predictions.csv'


def build_areru_pipeline_board(date_str: str = '', source: str = '') -> dict:
    """表示中の開催日の predictions_YYYY-MM-DD.csv から本命一覧を読む。

    表示専用で、既存の予想順位・BUY・EV には関与しない。
    data/predictions.csv（単発パイプラインの残り）は使わない。
    """
    empty = {'あり': False, '行': [], '開催日': '', '更新': '', '件数': 0}
    day = str(date_str or '').strip()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day):
        return empty
    path = ARCH / f'predictions_{day}.csv'
    if not path.exists() or path.stat().st_size < 32:
        return empty
    try:
        df = pd.read_csv(path, encoding='utf-8-sig').fillna('')
    except Exception as e:
        print(f'[areru-pipeline] read fail: {e}', flush=True)
        return empty
    src = source if source in ('jra', 'nar') else ''
    if src and 'source' in df.columns:
        df = df[df['source'].astype(str).str.lower() == src]
    need = {'レース', '本命', '本命オッズ', '本命AREru指数', '判定'}
    if df.empty or (need - set(df.columns)):
        return empty

    from ev_analysis import safe_float, safe_int

    def cell(v) -> str:
        s = str(v if v is not None else '').strip()
        return '' if s.lower() in ('nan', 'none', '—', '-') else s

    rows: list[dict] = []
    for raw in df.to_dict('records'):
        rno = safe_int(raw.get('レース'), None)
        if rno is None:
            continue
        venue = cell(raw.get('開催地'))
        judge = cell(raw.get('判定')) or 'データなし'
        rows.append({
            'レース': rno,
            '本命': cell(raw.get('本命')) or 'データなし',
            '本命オッズ': safe_float(raw.get('本命オッズ'), None),
            'AREru指数': safe_float(raw.get('本命AREru指数'), None),
            '判定': f'{venue} · {judge}' if venue else judge,
            '荒れ度': safe_float(raw.get('荒れ度'), None),
        })
    if not rows:
        return empty
    rows.sort(key=lambda x: (str(x.get('判定') or ''), x['レース']))

    try:
        updated = datetime.fromtimestamp(path.stat().st_mtime, JST).strftime('%m/%d %H:%M')
    except Exception:
        updated = ''

    return {
        'あり': True,
        '行': rows,
        '開催日': day,
        '更新': updated,
        '件数': len(rows),
    }


def _venue_meetings(records):
    """日付内の開催場一覧（レース数・S/A件数付き）。"""
    from netkeiba_client import normalize_venue_name
    from ev_analysis import safe_int
    buckets={}
    for r in records or []:
        venue=normalize_venue_name(str(r.get('開催地') or '').strip())
        if not venue:
            continue
        r['開催地']=venue
        buckets.setdefault(venue, []).append(r)
    meetings=[]
    for venue, rows in sorted(buckets.items(), key=lambda x: x[0]):
        race_nos=sorted({
            n for n in (safe_int(x.get('レース'), None) for x in rows)
            if n is not None and n > 0
        })
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


def _day_venues_for_nav(pred_path, source, fallback=None):
    """表示専用: 当日の開催場チップ用。予想・BUYは再計算しない。"""
    try:
        light = _read_predictions_for_venue_picker(pred_path, source)
        meetings = _venue_meetings(light)
        if meetings:
            return meetings
    except Exception as e:
        print(f'[venue-nav] skip: {e}', flush=True)
    return fallback or []


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


def _fmt_display_num(v, *, kind: str = '') -> str:
    """表示専用。欠損は空文字（推測しない）。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    s = str(v).strip()
    if not s or s.lower() in ('nan', 'none', 'なし', '—', '-'):
        return ''
    if kind == '枠':
        try:
            return str(int(float(s)))
        except (TypeError, ValueError):
            return s
    if kind == '斤量':
        try:
            x = float(s)
            return f'{x:g}kg'
        except (TypeError, ValueError):
            return s
    return s


def _horse_display_meta_for_records(records: list) -> dict:
    """表示専用: scores/runners から枠・騎手・斤量・日付を引く。スコア計算には使わない。"""
    rids = {_norm_race_id(r.get('race_id')) for r in (records or [])}
    rids.discard('')
    if not rids:
        return {}
    want = ('race_id', '日付', '馬名', '馬番', '枠', '騎手', '斤量', '単勝オッズ', '人気')
    frames = []
    dates = {
        str(r.get('日付') or r.get('開催日') or '').strip()
        for r in (records or [])
        if str(r.get('日付') or r.get('開催日') or '').strip()
    }
    for d in sorted(dates):
        p = ARCH / f'scores_{d}.csv'
        if p.exists():
            frames.append(p)
    frames.append(RUNNERS)
    out = {}
    for path in frames:
        if not Path(path).exists():
            continue
        try:
            df = pd.read_csv(
                path, encoding='utf-8-sig',
                usecols=lambda c: c in want,
            )
        except Exception:
            continue
        if df is None or df.empty or 'race_id' not in df.columns or '馬名' not in df.columns:
            continue
        df = df.copy()
        df['_rid'] = df['race_id'].map(_norm_race_id)
        df = df[df['_rid'].isin(rids)]
        for _, row in df.iterrows():
            rid = str(row.get('_rid') or '')
            name = clean_horse(row.get('馬名', ''))
            if not rid or not name:
                continue
            cur = out.setdefault((rid, name), {})
            for dst, src, kind in (
                ('枠番', '枠', '枠'),
                ('騎手', '騎手', ''),
                ('斤量', '斤量', '斤量'),
                ('馬番', '馬番', '枠'),
                ('日付', '日付', ''),
                ('単勝オッズ', '単勝オッズ', ''),
                ('人気', '人気', '枠'),
            ):
                if cur.get(dst):
                    continue
                txt = _fmt_display_num(row.get(src), kind=kind)
                if txt:
                    cur[dst] = txt
    return out


_EMPTY_MINUS = {
    '特記すべき大きな不安は少ない',
    '特記なし',
    'なし',
    '—',
    '-',
}


def _display_factor_lists(card: dict) -> tuple[list, list, list]:
    """既存カードのプラス/マイナス/判断根拠だけを返す。寄与度は作らない。"""
    plus, minus = [], []
    for src in (card.get('プラス材料一覧') or [], str(card.get('プラス材料') or '').split(' / ')):
        for p in src:
            s = str(p or '').strip()
            if s and s not in plus and s not in _EMPTY_MINUS:
                plus.append(s)
    for src in (card.get('不安材料一覧') or [], str(card.get('不安材料') or '').split(' / ')):
        for m in src:
            s = str(m or '').strip()
            if s and s not in minus and s not in _EMPTY_MINUS:
                minus.append(s)
    why = []
    skip_eval = {'', '—', '-', 'なし', '対象外', 'データ不足'}
    for row in card.get('判断根拠') or []:
        if not isinstance(row, dict):
            continue
        item = str(row.get('項目') or '').strip()
        ev = str(row.get('評価') or '').strip()
        desc = str(row.get('説明') or '').strip()
        if not item:
            continue
        if any(tok in desc for tok in ('データ不足', '未取得', '対象外')):
            continue
        if ev in skip_eval:
            continue
        why.append({'項目': item, '評価': ev or desc})
    return plus[:6], minus[:6], why[:8]


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
    """レース信頼度で S〜D を付け、開催単位で買いを厳選する。

    by_venue=True: 地方（開催場ごと） / False: JRA（日次全体）
    CSV には build_predictions 時点で同じ基準が焼き込まれているが、
    表示時にも再適用して古いCSVやキャッシュ漏れを防ぐ。
    """
    from ev_analysis import apply_ev_rank_and_labels, build_ai_buy_reasons, build_ai_risks, build_buy_rationale, tighten_buy_selection
    cleaned = []
    for r in races or []:
        try:
            apply_ev_rank_and_labels(r)
            if not r.get('AI買い理由'):
                r['AI買い理由'] = build_ai_buy_reasons(r, limit=4)
            if not r.get('BUY根拠'):
                r['BUY根拠'] = build_buy_rationale(r)
            if not r.get('AIリスク'):
                r['AIリスク'] = build_ai_risks(r, limit=3)
            cleaned.append(r)
        except Exception as e:
            print(f'[rank] skip race={r.get("race_id")}: {e}', flush=True)
            cleaned.append(r)
    try:
        # JRA と NAR を同じ日次枠で厳選すると、全開催タブで地方の買いが消える。
        # source ごとに既存ロジックを適用し、オブジェクトは元のリスト順のまま返す。
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in cleaned:
            src = str(r.get('source') or '').strip().lower()
            if src not in ('jra', 'nar'):
                try:
                    from areru_engine import source_from_race_id
                    src = source_from_race_id(r.get('race_id', ''))
                except Exception:
                    src = ''
            buckets[src if src in ('jra', 'nar') else '_'].append(r)
        for src, chunk in buckets.items():
            if src == 'nar':
                venue_scope = True
            elif src == 'jra':
                venue_scope = False
            else:
                venue_scope = by_venue
            tighten_buy_selection(chunk, by_venue=venue_scope)
        out = cleaned
    except Exception as e:
        print(f'[rank] tighten fail: {e}', flush=True)
        out = cleaned
    # 厳選後の投資判定に合わせて本命軸馬券を再生成
    for r in out:
        try:
            src = str(r.get('source') or '').lower()
            if not src:
                try:
                    from areru_engine import source_from_race_id
                    src = source_from_race_id(r.get('race_id', ''))
                except Exception:
                    src = ''
            if src == 'nar':
                simple = _simple_nar_tickets(r)
                if simple:
                    r['推奨馬券一覧'] = simple
            elif src == 'jra':
                main_tix = _jra_main_tickets(r)
                if main_tix:
                    rest = [
                        t for t in (r.get('推奨馬券一覧') or [])
                        if isinstance(t, dict) and str(t.get('券種')) not in ('単勝', '馬連', 'ワイド')
                    ]
                    r['推奨馬券一覧'] = (main_tix + rest)[:8]
            if not r.get('AI買い理由'):
                r['AI買い理由'] = build_ai_buy_reasons(r, limit=4)
            if not r.get('BUY根拠'):
                from ev_analysis import build_buy_rationale
                r['BUY根拠'] = build_buy_rationale(r)
            if not r.get('AIリスク'):
                r['AIリスク'] = build_ai_risks(r, limit=3)
        except Exception as e:
            print(f'[rank] ticket rebuild skip: {e}', flush=True)
    _stamp_buy_display(out)
    return out


def _stamp_buy_display(races: list) -> None:
    """確定済みの投資判定を馬カードへ転写するだけ。再計算しない。"""
    for r in races or []:
        race_buy = str(r.get('投資判定') or '').startswith('買い')
        honmei = str(r.get('本命') or '').strip()
        for p in r.get('予想馬') or []:
            if not isinstance(p, dict):
                continue
            is_honmei = (str(p.get('役割') or '') == '本命') or (str(p.get('馬名') or '') == honmei)
            p['BUY表示'] = bool(race_buy and is_honmei)


def build_buy_candidates(races: list, limit: int = 12) -> list:
    """厳選後の買いレースを期待値順（本日の買い候補）。"""
    from ev_analysis import safe_float, safe_int
    scored = []
    for r in races or []:
        if not str(r.get('投資判定') or '').startswith('買い'):
            continue
        ev = safe_float(r.get('期待値'), None)
        if ev is None:
            ev = safe_float(str(r.get('レース期待回収率') or '').replace('%', ''), None)
        if ev is None:
            continue
        scored.append((ev, r))
    scored.sort(key=lambda x: (
        -x[0],
        str(x[1].get('開催地') or ''),
        safe_int(x[1].get('レース'), 0),
    ))
    return [r for _, r in scored[:limit]]


def build_today_ai_board(races: list, verification: dict | None = None) -> dict:
    """ヘッダー直下の本日AI成績カード。"""
    races = races or []
    total = len(races)
    buys = sum(1 for r in races if str(r.get('投資判定') or '').startswith('買い'))
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
    from ev_analysis import safe_int
    ban_map=ban_map or {}
    horse_meta=_horse_display_meta_for_records(records)
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
            f"{r.get('開催地','')} {safe_int(r.get('レース'), 0):02d}R"
            if safe_int(r.get('レース'), None)
            else str(r.get('開催地') or 'レース')
        )
        # 投資判定のフォールバック（apply_expected_value 後に EV ランクで上書き）
        if not r.get('投資判定'):
            try:
                ev=float(str(r.get('レース期待回収率') or '').replace('%',''))
                if ev>=BUY_EV_FLOOR:
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
        rid=_norm_race_id(r.get('race_id',''))
        race_date=''
        for p in r.get('予想馬') or []:
            if not isinstance(p, dict):
                continue
            card=p.get('カード') if isinstance(p.get('カード'), dict) else {}
            meta=horse_meta.get((rid, clean_horse(p.get('馬名') or card.get('馬名') or ''))) or {}
            if not race_date:
                race_date=str(meta.get('日付') or '')
            p['枠番']=p.get('枠番') or card.get('枠番') or meta.get('枠番') or ''
            p['騎手']=p.get('騎手') or card.get('騎手') or meta.get('騎手') or ''
            p['斤量']=p.get('斤量') or card.get('斤量') or meta.get('斤量') or ''
            if not p.get('馬番表示') and meta.get('馬番'):
                p['馬番表示']=meta.get('馬番')
            if not p.get('人気') and meta.get('人気'):
                p['人気']=meta.get('人気')
            if p.get('単勝オッズ表示') in (None, '', '—') and meta.get('単勝オッズ'):
                p['単勝オッズ表示']=meta.get('単勝オッズ')
            plus, minus, why=_display_factor_lists(card)
            if not plus and p.get('要点'):
                plus=[str(x).strip() for x in (p.get('要点') or []) if str(x).strip()]
            p['プラス要因']=plus
            p['マイナス要因']=minus
            p['評価内訳']=why
            score=card.get('AI評価')
            try:
                p['予想スコア']=round(float(score), 1) if score not in (None, '', '—') else None
            except (TypeError, ValueError):
                p['予想スコア']=None
            p['単勝オッズ表示']=p.get('単勝オッズ表示') or card.get('単勝オッズ')
            p['カード期待値']=card.get('期待値')
        picks=[p for p in (r.get('予想馬') or []) if isinstance(p, dict)]
        known={clean_horse(p.get('馬名') or '') for p in picks}
        extras=[]
        for (mrid, name), meta in horse_meta.items():
            if mrid != rid or not name or name in known:
                continue
            extras.append({
                '役割': '',
                '馬名': name,
                '馬番表示': meta.get('馬番') or '',
                '馬番': meta.get('馬番') or '',
                '枠番': meta.get('枠番') or '',
                '騎手': meta.get('騎手') or '',
                '斤量': meta.get('斤量') or '',
                '人気': meta.get('人気') or '',
                '単勝オッズ表示': meta.get('単勝オッズ') or '',
                'BUY表示': False,
                'プラス要因': [],
                'マイナス要因': [],
            })
        def _ban_key(p):
            try:
                return int(float(str(p.get('馬番') or p.get('馬番表示') or 99)))
            except (TypeError, ValueError):
                return 99
        extras.sort(key=_ban_key)
        r['AI一覧']=picks + extras
        n=len(picks)+len(extras)
        if n:
            r['表示頭数']=n
        if race_date and not r.get('開催日'):
            r['開催日']=race_date
        # 地方: 単勝・馬連・ワイドのシンプル買い目を優先表示
        # 中央: 本命軸の単勝・馬連・ワイドを先頭に保証（既存フォーメーションは後ろに残す）
        src=str(r.get('source') or '').lower()
        if not src:
            try:
                from areru_engine import source_from_race_id
                src=source_from_race_id(r.get('race_id',''))
            except Exception:
                src=''
        if src == 'nar':
            simple=_simple_nar_tickets(r)
            if simple:
                r['推奨馬券一覧']=simple
                if not r.get('推奨券種') or str(r.get('推奨券種')) in ('三連単','三連複','馬単'):
                    r['推奨券種']='単勝'
                if not r.get('馬券戦略理由'):
                    r['馬券戦略理由']='地方は本命軸の単勝・馬連・ワイドを優先'
        elif src == 'jra':
            main_tix=_jra_main_tickets(r)
            if main_tix:
                existing=[t for t in (r.get('推奨馬券一覧') or []) if isinstance(t, dict)]
                # 既存から単勝/馬連/ワイドを外し、本命軸を先頭へ
                rest=[t for t in existing if str(t.get('券種')) not in ('単勝','馬連','ワイド')]
                r['推奨馬券一覧']=(main_tix + rest)[:8]
                if not r.get('推奨券種') or str(r.get('推奨券種')) in ('三連単','三連複'):
                    r['推奨券種']='単勝'
                if not r.get('馬券戦略理由'):
                    r['馬券戦略理由']='本命を軸にした単勝・馬連・ワイドを優先'
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


def _patch_predictions_odds_from_runners(date_str: str, source: str) -> int:
    """runners の最新オッズを predictions へ反映（勝負ランク・投資判定は触らない）。"""
    src = source if source in ('jra', 'nar') else 'nar'
    day = str(date_str or '')
    pred_path = ARCH / f'predictions_{day}.csv'
    rp = _runner_path()
    if not day or not pred_path.exists() or rp is None:
        return 0
    try:
        runners = pd.read_csv(rp, encoding='utf-8-sig')
        pred = pd.read_csv(pred_path, encoding='utf-8-sig')
        if runners.empty or pred.empty:
            return 0
        runners['日付'] = parse_date(runners['日付']).dt.strftime('%Y-%m-%d')
        day_r = runners[runners['日付'] == day].copy()
        if 'source' in day_r.columns:
            day_r = day_r[day_r['source'].astype(str).str.lower() == src]
        if day_r.empty:
            return 0
        # race_id + 本命馬番 → 単勝
        odds_map = {}
        for _, row in day_r.iterrows():
            rid = str(row.get('race_id') or '').strip()
            ban = str(row.get('馬番') or '').strip()
            if not rid or not ban:
                continue
            try:
                win = float(row.get('単勝オッズ') or 0)
            except (TypeError, ValueError):
                win = 0.0
            if win <= 0:
                continue
            pop = row.get('人気')
            odds_map[(rid, ban)] = (win, pop)
            odds_map[(rid, ban.zfill(2))] = (win, pop)
        if not odds_map:
            return 0
        n = 0
        for idx in pred.index:
            if 'source' in pred.columns:
                if str(pred.at[idx, 'source'] or '').strip().lower() != src:
                    continue
            rid = str(pred.at[idx, 'race_id'] or '').strip()
            ban = str(pred.at[idx, '本命馬番'] or '').strip()
            info = odds_map.get((rid, ban)) or odds_map.get((rid, ban.zfill(2) if ban.isdigit() else ban))
            if not info:
                continue
            win, pop = info
            pred.at[idx, '本命オッズ'] = win
            if pop not in (None, '', 'なし') and '本命人気' in pred.columns:
                pred.at[idx, '本命人気'] = pop
            n += 1
        if n:
            pred.to_csv(pred_path, index=False, encoding='utf-8-sig')
            _clear_runtime_caches()
            print(f'[{src}-odds] patched predictions odds n={n} date={day}', flush=True)
        try:
            del runners, pred, day_r, odds_map
        except Exception:
            pass
        import gc
        gc.collect()
        return n
    except Exception as e:
        print(f'[{src}-odds] patch fail: {e}', flush=True)
        return 0


def _runners_ready(date_str: str, source: str = 'nar') -> bool:
    """当日の runners が source 分あるか。"""
    c = _nar_day_counts(date_str, source)
    return int(c.get('runners_races') or 0) > 0


def _predict_only_from_runners(source: str, date_str: str, *, wait: bool = False) -> bool:
    """runners 済みで予想だけ作る（Render Free 向け軽量復旧）。"""
    import time as _time
    src = source if source in ('jra', 'nar') else 'nar'
    day = date_str or _today_jst()

    def _body():
        nonlocal_terminal = {'ok': False}
        t0 = _time.perf_counter()
        print(f'[Local Update] START predict-only source={src} date={day}', flush=True)
        try:
            _write_job_status(
                src, state='running', stage='predict_start', message='AI予想生成中',
                date_str=day,
            )
            if not _runners_ready(day, src):
                raise RuntimeError(f'{day}: runners が無いため予想不可')
            print('[Local Update] AI予想生成開始 (predict-only)', flush=True)
            pr = _run_replay_predict_subprocess(day, timeout=360)
            print(
                f'[Local Update] AI予想生成終了 sec={_time.perf_counter()-t0:.1f} rc={pr.returncode}',
                flush=True,
            )
            if pr.returncode != 0:
                raise RuntimeError(f'replay_predict rc={pr.returncode}')
            if not _nar_pred_ready(day, src):
                raise RuntimeError('predictions 未生成')
            c = _nar_day_counts(day, src)
            print(
                f'[Local Update] DB保存完了 predictions={c["pred_races"]} venues={c["pred_venues"]}',
                flush=True,
            )
            _clear_runtime_caches()
            _write_job_status(
                src, state='success', stage='done', message='更新成功', date_str=day,
            )
            auto_seal_if_ready(day, src, True)
            print(f'[Local Update] END success predict-only total_sec={_time.perf_counter()-t0:.1f}', flush=True)
            nonlocal_terminal['ok'] = True
            return True
        except Exception as e:
            err = str(e)[:200]
            is_timeout = 'タイムアウト' in err or 'Timeout' in err or 'timeout' in err.lower()
            _write_job_status(
                src, state='error',
                stage='timeout' if is_timeout else 'failed',
                message='更新タイムアウト' if is_timeout else '更新失敗',
                date_str=day, error=err,
            )
            print(f'[Local Update] END failed predict-only err={err}', flush=True)
            raise

    try:
        # 必ず heavy lock 経由（ロック抜け同時予想が OOM の主因だった）
        ok = _run_serialized_heavy(f'predict-only:{src}:{day}', _body, wait=wait)
        return bool(ok)
    except Exception:
        return False


def _run_odds_only_update(source: str, date_str: str, *, wait: bool = False) -> bool:
    """締切後の最小更新: 出走・オッズのみ。ランク/買い判定は再計算しない。"""
    src = source if source in ('jra', 'nar') else 'nar'
    day = date_str or _today_jst()

    def _body():
        terminal = False
        try:
            _write_job_status(
                src, state='running', stage='odds', message='オッズ更新中',
                date_str=day,
            )
            cmd = [
                sys.executable, 'refresh_data.py',
                '--dates', day, '--source', src,
                '--odds-only', '--skip-predict',
            ]
            print(f'[{src}-odds] {" ".join(cmd)}', flush=True)
            try:
                rc = subprocess.run(cmd, check=False, timeout=900)
            except subprocess.TimeoutExpired as e:
                _write_job_status(
                    src, state='error', stage='odds_timeout', message='オッズ更新タイムアウト',
                    date_str=day, error='odds refresh timeout',
                )
                raise RuntimeError('odds refresh timeout') from e
            if rc.returncode != 0:
                raise RuntimeError(f'odds refresh rc={rc.returncode}')
            _clear_runtime_caches()
            patched = _patch_predictions_odds_from_runners(day, src)
            if _nar_pred_ready(day, src):
                write_seal(day, sources=[src], note='post_deadline_odds', mode='odds')
            _write_job_status(
                src, state='success', stage='odds_done',
                message=f'オッズ更新完了({patched})', date_str=day,
            )
            terminal = True
            return True
        except Exception as e:
            err = str(e)[:200]
            is_timeout = 'タイムアウト' in err or 'Timeout' in err or 'timeout' in err.lower()
            _write_job_status(
                src, state='error',
                stage='timeout' if is_timeout else 'failed',
                message='更新タイムアウト' if is_timeout else '更新失敗',
                date_str=day, error=err,
            )
            terminal = True
            raise
        finally:
            if not terminal:
                _finalize_job_if_running(
                    src, ok=False, date_str=day,
                    message='更新失敗', error='odds-only ended while running',
                )

    try:
        return bool(_run_serialized_heavy(f'odds-only:{src}:{day}', _body, wait=wait))
    except Exception:
        return False


def run_today_pipeline(
    source: str = 'nar',
    force: bool = False,
    *,
    force_full: bool = False,
) -> bool:
    """本日パイプライン。

    - 朝8時前 / 未完成: 開催取得→AI予想（本生成）
    - 朝8時以降かつ完成済み: オッズ・取消のみ（ロジック固定）
    - JRA は開催日のみ本生成
    - Render + NAR: runners 済みなら予想のみ（重取得で Free が固まらない）
    """
    src = source if source in ('jra', 'nar') else 'nar'
    today = _today_jst()
    if not _generation_enabled():
        # Render 配信専用モード。生成は GitHub Actions 側で行いコミットされる
        log_orchestrator('pipeline', 'SKIP', src, reason='generation_disabled', date=today)
        print(f'[{src}-today] skip: generation disabled on this host', flush=True)
        _write_job_status(
            src, state='success', stage='done',
            message='データは自動更新されます', date_str=today,
        )
        return True
    # 前回ジョブが running のまま残っていたら先に解除
    _expire_stale_job(src)
    ready = _nar_pred_ready(today, src)
    do_full = should_full_predict(
        src, date_str=today, ready=ready, force_full=force_full,
    )
    # 締切後の通常更新・本日ボタンは odds-only
    if not do_full and should_odds_only_update(src, date_str=today, ready=ready):
        print(f'[{src}-today] post-{PREDICT_READY_HOUR}:00 odds-only date={today}', flush=True)
        return _run_odds_only_update(src, today, wait=bool(force))
    if not do_full and not ready and past_predict_deadline():
        # 朝の生成に失敗 → 復旧として本生成を許可
        do_full = True
        print(f'[{src}-today] recovery full predict (not ready after deadline)', flush=True)
    if not do_full and not source_is_race_day(src, today):
        _write_job_status(
            src, state='success', stage='done', message='本日は開催なし',
            date_str=today,
        )
        print(f'[{src}-today] no meeting date={today}', flush=True)
        return True

    # 根本対策: Render Free で NAR が runners まで来ているなら予想だけ走る
    on_render = str(os.environ.get('RENDER') or '').lower() in ('true', '1')
    if (
        src == 'nar'
        and not ready
        and _runners_ready(today, src)
        and (on_render or str(os.environ.get('ARERU_FAST_NAR') or '') in ('1', 'true', 'yes'))
        and not (force and force_full and str(os.environ.get('ARERU_FORCE_FULL_FETCH') or '') in ('1', 'true'))
    ):
        print(
            f'[Local Update] runners済み → predict-only に切替 '
            f'(Render Free 固まり防止) date={today}',
            flush=True,
        )
        return _predict_only_from_runners(src, today, wait=bool(force))

    lock = DATA / f'.{src}_today_pipeline.lock'
    if lock.exists():
        try:
            age = (__import__('time').time() - lock.stat().st_mtime)
            cooldown = 45 if force else 120
            if age < cooldown:
                st = _read_job_status(src)
                if str(st.get('state')) == 'running' and _nar_job_age_sec(st) < _JOB_STALE_SEC:
                    print(f'[{src}-today] already running, skip')
                    return False
                try:
                    lock.unlink(missing_ok=True)
                except Exception:
                    pass
            elif age >= _JOB_STALE_SEC:
                try:
                    lock.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

    def _body():
        lock.write_text(str(os.getpid()), encoding='utf-8')
        terminal = False
        try:
            print(
                f'[{src}-today] FULL pipeline start force={force} '
                f'force_full={force_full} date={today}',
                flush=True,
            )
            ok = _refresh_then_predict([today], src)
            if not ok:
                raise RuntimeError(f'{src} 本日開催パイプライン失敗')
            if _nar_pred_ready(today, src):
                auto_seal_if_ready(today, src, True)
            print(f'[{src}-today] pipeline done (sealed={is_sealed(today, src)})', flush=True)
            terminal = True
        except Exception as e:
            _finalize_job_if_running(
                src, ok=False, date_str=today,
                message='取得失敗', error=str(e)[:200],
            )
            terminal = True
            raise
        finally:
            try:
                lock.unlink(missing_ok=True)
            except Exception:
                pass
            if not terminal:
                _finalize_job_if_running(
                    src, ok=False, date_str=today,
                    message='取得失敗', error='today pipeline ended while running',
                )

    try:
        started = _run_serialized_heavy(f'{src}-today:{today}', _body, wait=bool(force))
        if not started:
            busy_name = _HEAVY_JOB_STATE.get('name') or ''
            log_orchestrator('bg', 'SKIP', src, reason='heavy_busy', busy=busy_name)
            return False
        return True
    except Exception as e:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass
        _write_job_status(
            src, state='error', stage='failed', message='取得失敗',
            date_str=today, error=str(e)[:200],
        )
        return False


def run_nar_today_pipeline(force: bool = False) -> bool:
    return run_today_pipeline('nar', force=force)


def run_jra_today_pipeline(force: bool = False) -> bool:
    return run_today_pipeline('jra', force=force)


def bootstrap_source(source: str) -> bool:
    """タブでデータが古い/無い/不完全な場合に最新開催を自動取得する（地方・JRA共通）。

    Returns: 更新を実行したら True

    重要: 当日カードが存在する／当日予想が ready のときは、前日以前を
    再取得しない（status.date が前日へ寄って表示が巻き戻るのを防ぐ）。
    """
    if source not in ('jra', 'nar'):
        return False
    src = source
    today = _today_jst()
    today_ready = _nar_pred_ready(today, src)
    try:
        from netkeiba_client import NetkeibaClient
        found = NetkeibaClient(sleep=0.12).discover_kaisai_dates(
            lookback=4, lookahead=2, source=src,
        )
        remote = _anchor_meeting_date(found, today)
    except Exception:
        remote = ''
        found = []
    local = _source_latest_in_runners(src)
    found_set = set(found)

    need_dates = []
    check_days = []
    if today in found_set:
        check_days.append(today)
    if remote and remote not in check_days and not (remote < today and (today in found_set or today_ready)):
        check_days.append(remote)
    future = [d for d in found if d > today]
    if future:
        nxt = min(future)
        if nxt not in check_days:
            check_days.append(nxt)
    for d in check_days:
        if date_needs_runners_fetch(d, src):
            need_dates.append(d)
    if today in found_set and today not in dates(src):
        need_dates.append(today)
    if not found and (not local or local < today):
        need_dates.append(today)

    need_dates = sorted(set(need_dates), reverse=True)
    if today in found_set or today_ready:
        before = list(need_dates)
        need_dates = [d for d in need_dates if d >= today]
        dropped = [d for d in before if d not in need_dates]
        if dropped:
            _pipeline_log(
                '開催取得', 'スキップ', today,
                reason='past_bootstrap_blocked', source=src,
                dropped=','.join(dropped), 保存件数=0,
            )
    stale = bool(remote and local and local < remote)
    if stale and remote and remote < today and (today in found_set or today_ready):
        _pipeline_log(
            '開催取得', 'スキップ', today,
            reason='stale_past_ignored', source=src,
            remote=remote, local=local or '-', 保存件数=0,
        )
        stale = False
    if not need_dates and not stale:
        return False
    if not need_dates and stale:
        need_dates = [remote] if remote and remote >= today else ([today] if today in found_set else [])
        if not need_dates:
            return False

    lock = DATA / f'.{src}_bootstrap.lock'
    if lock.exists():
        try:
            age = (__import__('time').time() - lock.stat().st_mtime)
            cooldown = 300 if need_dates else 1800
            if age < cooldown:
                print(f'[bootstrap] {src} already running, skip')
                return False
        except Exception:
            pass

    target_dates = need_dates[:3]

    def _body():
        lock.write_text(str(os.getpid()), encoding='utf-8')
        try:
            print(
                f'[bootstrap] source={src} local={local or "-"} remote={remote or "-"} '
                f'need={target_dates} found={found[:5]}',
                flush=True,
            )
            if target_dates:
                ok = _refresh_then_predict(target_dates, src)
                if not ok:
                    raise RuntimeError('bootstrap refresh/predict 失敗')
            else:
                _write_job_status(
                    src, state='running', stage='start', message='データ取得中',
                    date_str=today,
                )
                cmd = [
                    sys.executable, 'refresh_data.py',
                    '--latest-only', '--source', src, '--lookback', '5', '--lookahead', '2',
                    '--skip-predict',
                ]
                rc = subprocess.run(cmd, check=False, timeout=1800)
                if rc.returncode != 0:
                    raise RuntimeError(f'latest-only 終了コード {rc.returncode}')
                _write_job_status(
                    src, state='running', stage='races_done', message='レース取得完了',
                    date_str=today,
                )
                _clear_runtime_caches()
                pred_days = sorted({d for d in (today, remote) if d})
                for d in pred_days:
                    _write_job_status(
                        src, state='running', stage='predict_start', message='AI予想生成中',
                        date_str=d,
                    )
                    pr = _run_replay_predict_subprocess(d, timeout=600)
                    if pr.returncode != 0:
                        raise RuntimeError(f'replay_predict 終了コード {pr.returncode}')
                _clear_runtime_caches()
                _write_job_status(
                    src, state='success', stage='done', message='取得完了',
                    date_str=today,
                )
        finally:
            try:
                lock.unlink(missing_ok=True)
            except Exception:
                pass

    try:
        return _run_serialized_heavy(f'{src}-bootstrap', _body, wait=False)
    except Exception as e:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass
        _write_job_status(
            src, state='error', stage='failed', message='取得失敗',
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
    # history=1 は「前日以前を明示閲覧」のときだけ有効（当日URLに残っていても無視）
    raw_history=str(request.args.get('history') or '').strip() in ('1','true','yes')
    allow_past=bool(
        raw_history
        and explicit_date
        and re.fullmatch(r'\d{4}-\d{2}-\d{2}', explicit_date)
        and explicit_date < today
    )
    # 地方のみ: 当日未完成で、URL日付に完成予想があるならその日を閲覧許可
    # （会場リンクからレース一覧へ入れる。force_cal で venue を消さない）
    # JRAは history=1 以外で前開催へ寄せない（非開催日に日曜カードが出るのを防ぐ）
    if (
        source == 'nar'
        and not allow_past
        and explicit_date
        and re.fullmatch(r'\d{4}-\d{2}-\d{2}', explicit_date)
        and explicit_date < today
        and not _nar_pred_ready(today, source)
        and _nar_pred_ready(explicit_date, source)
    ):
        allow_past = True
        print(
            f'[{source}-date] allow cache-day view date={explicit_date} '
            f'(today={today} not ready)',
            flush=True,
        )
    # 地方・中央: 本日ボタン／日付未指定／URLに前日残存 → カレンダー当日へ
    # ※完成キャッシュ日の閲覧（allow_past）は矯正しない
    force_cal_today=(
        source in ('nar', 'jra')
        and _force_calendar_today(explicit_date, want_today, mode, allow_past)
    )
    # 地方の当日トップは開催場一覧ではなくカードを出す。JRAタブの source は書き換えない。
    today_live_card = bool(
        source == 'nar'
        and not allow_past
        and _nar_pred_ready(today, source)
        and (
            want_today
            or force_cal_today
            or not explicit_date
            or explicit_date == today
        )
    )
    try:
        if source in ('nar', 'jra'):
            _expire_stale_job(source)
            today_ready=_nar_pred_ready(today, source)
            # ページ閲覧ではデータ取得を起動しない（cron / /refresh のみ）
            if today_ready and (past_predict_deadline() or is_sealed(today, source)):
                auto_seal_if_ready(today, source, True)
            _pipeline_log(
                '表示', '成功', today,
                source=source,
                reason='cache_first_no_page_fetch',
                today_ready=int(today_ready),
                force_refresh=int(force_refresh),
                保存件数='-',
            )
    except Exception as e:
        print(f'[bootstrap] skip: {e}')
        if source in ('nar', 'jra'):
            _write_job_status(
                source, state='error', stage='failed', message='取得失敗',
                date_str=_today_jst(), error=str(e)[:200],
            )
    # 日付キャッシュを当日判定前に捨て、古い開催日一覧を使わない
    if source in ('nar', 'jra') and (force_cal_today or force_refresh or want_today):
        _clear_runtime_caches()
    meeting_dates=dates(source)
    av=list(meeting_dates)
    selected=explicit_date
    if force_cal_today:
        selected=today
        if today not in av:
            av=sorted(set(av)|{today}, reverse=True)
        print(f'[{source}-date] force calendar today={today} (was explicit={explicit_date or "-"})', flush=True)
    # ソース切替で他開催の日付が残っていても、そのソースの開催日へ寄せる
    if not selected:
        selected=today if source in ('nar', 'jra') else (av[0] if av else '')
        if source in ('nar', 'jra') and today not in av:
            av=sorted(set(av)|{today}, reverse=True)
    elif selected not in av:
        if source in ('nar', 'jra') and selected==today:
            av=sorted(set(av)|{today}, reverse=True)
        elif source in ('nar', 'jra') and force_cal_today:
            selected=today
            av=sorted(set(av)|{today}, reverse=True)
        else:
            # 当日意図のときに av[0]（前日）へ落とさない
            if source in ('nar', 'jra') and (selected==today or not allow_past):
                selected=today
                av=sorted(set(av)|{today}, reverse=True)
            else:
                selected=av[0] if av else ''
                if source in ('nar', 'jra'):
                    _pipeline_log(
                        '表示', '失敗', selected,
                        source=source,
                        reason='fallback_av0', explicit=explicit_date or '-', 保存件数=0,
                    )
    # キャッシュ優先: 当日未完成なら最新完成予想日を即表示（履歴明示時以外）
    # ただしカレンダー当日は別日へ飛ばさない（JRA非開催日は空状態）
    if source in ('nar', 'jra') and not allow_past and mode in ('predict', 'result', 'analysis'):
        if (
            not _nar_pred_ready(selected or '', source)
            and not _stay_on_selected_calendar_day(selected, source)
        ):
            latest = _latest_ready_pred_date(source, on_or_before=today)
            if latest and latest != selected:
                print(
                    f'[{source}-date] cache-first {selected or "-"} -> {latest}',
                    flush=True,
                )
                selected = latest
                if latest not in av:
                    av = sorted(set(av) | {latest}, reverse=True)
        elif (
            not _nar_pred_ready(selected or '', source)
            and _stay_on_selected_calendar_day(selected, source)
        ):
            print(
                f'[{source}-date] keep calendar day {selected} empty (no cross-day fallback)',
                flush=True,
            )
    # 当日完成時のみ「前日へ落とさない」。未完成は上で最新キャッシュへ寄せ済み
    block_stale_today=(
        source in ('nar', 'jra')
        and selected == today
        and not allow_past
        and _nar_pred_ready(today, source)
    )
    block_stale_nar=(source=='nar' and selected==today and not allow_past and _nar_pred_ready(today, source))
    # 結果検証タブ:
    # - プルダウンは「本日以前の開催日 + 結果確定日」（最新開催日も選択可）
    # - 明示指定日に予想があれば結果未取込でも寄せない（結果待ち表示＋バックグラウンド取得）
    # - 本日開催指定時は結果日へ強制しない
    result_days=dates_with_results(source)
    if mode=='result' and not want_today and explicit_date and not force_cal_today:
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
    elif mode=='result' and force_cal_today:
        # 当日矯正中は explicit の前日で selected を上書きしない
        av=_result_available_dates(meeting_dates, result_days, today)
        if today not in av:
            av=sorted(set(av)|{today}, reverse=True)
        selected=today
    elif mode=='result' and not want_today and not explicit_date and source!='nar':
        av=_result_available_dates(meeting_dates, result_days, today)
        selected=(result_days[0] if result_days else (av[0] if av else ''))
    elif mode=='result' and source=='nar' and not want_today and not explicit_date:
        # 地方結果: 当日を優先表示（未完成なら後段で最新キャッシュへ）
        av=_result_available_dates(meeting_dates, result_days, today)
        if today not in av:
            av=sorted(set(av)|{today}, reverse=True)
        selected=today
    # 結果タブで当日未完成へ戻された場合も、最新キャッシュへ再寄せ（当日ビューは除く）
    if source in ('nar', 'jra') and not allow_past and mode in ('predict', 'result', 'analysis'):
        if (
            not _nar_pred_ready(selected or '', source)
            and not _stay_on_selected_calendar_day(selected, source)
        ):
            latest = _latest_ready_pred_date(source, on_or_before=today)
            if latest and latest != selected:
                print(
                    f'[{source}-date] cache-first(post) {selected or "-"} -> {latest}',
                    flush=True,
                )
                selected = latest
                if latest not in av:
                    av = sorted(set(av) | {latest}, reverse=True)
    # 開催場パラメータ:
    # - 「本日開催」ボタン（venue無し）→ 一覧からやり直す
    # - 当日矯正で URL が前日のとき → 前日会場を捨てる
    # - キャッシュ日閲覧（allow_past / 完成予想日）→ 会場を保持（レース一覧の生命線）
    # - それ以外（開くリンクの venue=大井 等）→ 必ず保持
    raw_venue=str(request.args.get('venue') or '').strip()
    if want_today and not raw_venue:
        raw_venue=''
    elif force_cal_today and explicit_date and explicit_date < today and not allow_past:
        # 前日URL矯正時のみ会場を捨てる（キャッシュ日の「開く」は残す）
        raw_venue=''
    selected_venue=''
    if raw_venue:
        try:
            from netkeiba_client import normalize_venue_name
            # 二重エンコード対策
            from urllib.parse import unquote
            decoded=unquote(unquote(raw_venue))
            selected_venue=normalize_venue_name(decoded)
        except Exception:
            selected_venue=raw_venue
    if selected_venue:
        print(f'[nar-venue] keep venue={selected_venue} force_cal={int(force_cal_today)} want_today={int(want_today)}', flush=True)
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
            # キャッシュ優先: 完成予想があれば即表示（ページからは取得しない）
            pred_path, page_status = ensure_for_page(selected, source=source)
            data_status=page_status
            if pred_path is None:
                # 初回のみ（キャッシュ無し）。ジョブは cron 側。
                # 当日ビューは別日カードへ飛ばさない（JRA非開催日は空状態）
                latest = ''
                if source in ('nar', 'jra') and not _stay_on_selected_calendar_day(selected, source):
                    latest = _latest_ready_pred_date(source, on_or_before=today)
                if latest and latest != selected:
                    selected = latest
                    if latest not in av:
                        av = sorted(set(av) | {latest}, reverse=True)
                    pred_path, page_status = ensure_for_page(selected, source=source)
                    data_status = page_status if pred_path else 'generating'
                if pred_path is None:
                    no_meet = (
                        source in ('nar', 'jra')
                        and _stay_on_selected_calendar_day(selected, source)
                        and _today_source_has_no_meeting(source, selected or today)
                    )
                    if no_meet:
                        data_status = 'ready'
                        message = (
                            '本日はJRAの開催はありません' if source == 'jra'
                            else '本日は地方競馬の開催はありません'
                        )
                        races=[]; venues=[]; show_venue_picker=False
                        races_for_board=[]; buy_candidates=[]; targets=[]; data_updated_at=''
                    else:
                        data_status='generating' if page_status!='error' else 'error'
                        message='データ取得中' if page_status!='error' else (
                            '取得失敗' if source=='nar' else '通信エラー: 予想データの準備に失敗しました。再読み込みしてください。'
                        )
                        races=[]; venues=[]; show_venue_picker=(source=='nar')
                        races_for_board=[]; buy_candidates=[]; targets=[]; data_updated_at=''
                else:
                    data_status = 'ready'
            if pred_path is not None:
                # 読み込んだファイル日付と selected が一致すること（取り違え防止）
                if selected and pred_path and f'predictions_{selected}.csv' not in str(pred_path):
                    print(f'[nar-date] refuse stale file {pred_path} for selected={selected}', flush=True)
                    pred_path=None
                    data_status='error'
                    message='取得失敗'
                    races=[]; venues=[]; show_venue_picker=(source=='nar')
                else:
                    # 地方・開催場未選択: 軽量列だけ読んで一覧を出す（メモリ削減）
                    use_light=(
                        source=='nar'
                        and mode in ('predict','result')
                        and not selected_venue
                        and not today_live_card
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
                            if selected==today:
                                message='本日は地方競馬の開催はありません'
                            else:
                                message=f'{selected} の地方開催データがありません'
                    else:
                        # 地方の開催場詳細: 会場で先に絞ってから prep（Render OOM/タイムアウト防止）
                        # 中央は従来どおりフル読み
                        if source=='nar' and selected_venue and mode in ('predict','result'):
                            venue_rows=_read_predictions_for_venue_detail(
                                pred_path, source, selected_venue,
                            )
                            if not venue_rows:
                                # 会場名不一致のフォールバック: 一覧だけ出して再選択を促す
                                light_rows=_read_predictions_for_venue_picker(pred_path, source)
                                light_rows=apply_display_ranks(light_rows, by_venue=True)
                                venues=_venue_meetings(light_rows)
                                venue_names={v['name'] for v in venues}
                                print(
                                    f'[nar-venue] miss venue={selected_venue} '
                                    f'known={sorted(venue_names)} → picker',
                                    flush=True,
                                )
                                if selected_venue in venue_names:
                                    # キャッシュはあるが会場詳細が空 → 一覧へ戻す（ページから再取得しない）
                                    selected_venue=''
                                    show_venue_picker=True
                                    races=[]; races_for_board=[]; buy_candidates=[]; targets=[]
                                    data_status='ready'
                                    message=f'{selected} / {label} / 開催場 {len(venues)}場'
                                else:
                                    selected_venue=''
                                    show_venue_picker=True
                                    races=[]; races_for_board=[]
                                    buy_candidates=[]; targets=[]
                                    data_status='ready'
                                    message=f'{selected} / {label} / 開催場 {len(venues)}場'
                                today_ai_board={'has_data':False,'回収率表示':'—','的中率表示':'—','買いレース':'0/0','tone':'roi-bad'}
                                data_updated_at=''
                                try:
                                    if pred_path and Path(pred_path).exists():
                                        from datetime import datetime as _dt
                                        data_updated_at=_dt.fromtimestamp(Path(pred_path).stat().st_mtime, JST).strftime('%m/%d %H:%M')
                                except Exception:
                                    pass
                            else:
                                races=prep(venue_rows, ban_map=_main_ban_map(selected))
                                races=_filter_records_by_source(races, source)
                                races=apply_display_ranks(races, by_venue=True)
                                for row in races:
                                    if not _race_date(row):
                                        row['日付']=selected
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
                                venues=_day_venues_for_nav(pred_path, source, fallback=[{
                                    'name': selected_venue,
                                    'count': len(races),
                                    'race_label': f'1-{len(races)}R' if races else '—',
                                    's': sum(1 for r in races if str(r.get('勝負ランク'))=='S'),
                                    'a': sum(1 for r in races if str(r.get('勝負ランク'))=='A'),
                                }])
                                show_venue_picker=False
                                races_for_board=list(races)
                                buy_candidates=build_buy_candidates(races_for_board)
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
                                    message=f'{selected} / {selected_venue} / データ更新中'
                                elif mode=='result':
                                    message=f'{selected} / {selected_venue} / 結果検証'
                                else:
                                    message=f'{selected} / {selected_venue} / 予想分析'
                                print(
                                    f'[nar-venue] detail ok venue={selected_venue} '
                                    f'races={len(races)} 保存件数={len(races)}',
                                    flush=True,
                                )
                        else:
                            # 全列読みは禁止（Render OOM）。会場詳細と同じ列制限で読む。
                            venue_rows = _read_predictions_for_venue_detail(
                                pred_path, source, selected_venue
                            ) if selected_venue else _read_predictions_for_venue_picker(pred_path, source)
                            if selected_venue and venue_rows:
                                races = prep(venue_rows, ban_map=_main_ban_map(selected))
                                races = _filter_records_by_source(races, source)
                                races = apply_display_ranks(races, by_venue=(source == 'nar'))
                                for row in races:
                                    if not _race_date(row):
                                        row['日付'] = selected
                                venues = _day_venues_for_nav(
                                    pred_path, source, fallback=_venue_meetings(races)
                                )
                                show_venue_picker = False
                                races_for_board = list(races)
                                buy_candidates = build_buy_candidates(races_for_board)
                                board_verify = verification if mode in ('result', 'analysis') else dict(_EMPTY_VERIFY)
                                today_ai_board = build_today_ai_board(races_for_board, board_verify)
                                data_updated_at = ''
                                try:
                                    if pred_path and Path(pred_path).exists():
                                        from datetime import datetime as _dt
                                        data_updated_at = _dt.fromtimestamp(Path(pred_path).stat().st_mtime, JST).strftime('%m/%d %H:%M')
                                except Exception:
                                    data_updated_at = ''
                                targets = buy_candidates[:8]
                                if data_status == 'updating':
                                    message = f'{selected} / {selected_venue} / データ更新中'
                                elif mode == 'result':
                                    message = f'{selected} / {selected_venue} / 結果検証'
                                else:
                                    message = f'{selected} / {selected_venue} / 予想分析'
                            else:
                                df = _fillna_pred_df(pd.read_csv(
                                    pred_path, encoding='utf-8-sig',
                                    usecols=lambda c: c in set(_NAR_VENUE_DETAIL_COLS) | {'開催地', 'source', 'race_id', '日付'},
                                ))
                                if source in ('jra', 'nar') and 'source' in df.columns:
                                    df = df[df['source'].astype(str).str.lower() == source].copy()
                                if mode == 'predict':
                                    drop_cols = [c for c in (
                                        'ワイド詳細', '馬連詳細', '馬単詳細', '三連複詳細', '三連単詳細', '本命詳細'
                                    ) if c in df.columns]
                                    if drop_cols:
                                        df = df.drop(columns=drop_cols, errors='ignore')
                                races = prep(df.to_dict('records'), ban_map=_main_ban_map(selected))
                                del df
                                races = _filter_records_by_source(races, source)
                                races = apply_display_ranks(races, by_venue=(source == 'nar'))
                                for row in races:
                                    if not _race_date(row):
                                        row['日付'] = selected
                                if mode == 'result':
                                    races, has_results = attach_results(races, selected_date=selected)
                                    ranks_map = (verification or {}).get('purchase_ranks_by_race') or {}
                                    tickets_by_race = {}
                                    for t in (verification or {}).get('recent_rows') or []:
                                        tid = _norm_race_id(t.get('race_id', ''))
                                        if tid:
                                            tickets_by_race.setdefault(tid, []).append(t)
                                    for row in races:
                                        rid = _norm_race_id(row.get('race_id', ''))
                                        row['purchase_ranks'] = list(ranks_map.get(rid, []))
                                        row['購入馬券一覧'] = list(tickets_by_race.get(rid, []))
                                venues = _venue_meetings(races)
                                venue_names = {v['name'] for v in venues}
                                races_for_board = list(races)
                                if source == 'nar' and mode in ('predict', 'result') and not today_live_card:
                                    show_venue_picker = True
                                    if selected_venue and selected_venue not in venue_names:
                                        selected_venue = ''
                                    if selected_venue:
                                        from netkeiba_client import normalize_venue_name as _nv
                                        races = [
                                            r for r in races
                                            if _nv(str(r.get('開催地') or '').strip()) == selected_venue
                                        ]
                                        show_venue_picker = False
                                        races_for_board = list(races)
                                        if not races:
                                            # キャッシュ一覧へ戻す（ページから会場再取得しない）
                                            selected_venue = ''
                                            show_venue_picker = True
                                            data_status = 'ready'
                                            message = f'{selected} / {label} / 開催場 {len(venues)}場'
                                    else:
                                        races = []
                                else:
                                    selected_venue = ''
                                buy_candidates = build_buy_candidates(races_for_board)
                                board_verify = verification if mode in ('result', 'analysis') else dict(_EMPTY_VERIFY)
                                today_ai_board = build_today_ai_board(races_for_board, board_verify)
                                data_updated_at = ''
                                try:
                                    if pred_path and Path(pred_path).exists():
                                        from datetime import datetime as _dt
                                        data_updated_at = _dt.fromtimestamp(Path(pred_path).stat().st_mtime, JST).strftime('%m/%d %H:%M')
                                except Exception:
                                    data_updated_at = ''
                                targets = buy_candidates[:8]
                                if data_status == 'updating':
                                    message = f'{selected} / {label} / データ更新中（表示はキャッシュ）'
                                    if data_updated_at:
                                        message += f' · 最終更新 {data_updated_at}'
                                elif data_status == 'generating':
                                    message = 'データ取得中'
                                elif source == 'nar' and show_venue_picker:
                                    if venues:
                                        if selected == today:
                                            message = f'本日開催 {selected} / {label} / 開催場 {len(venues)}場'
                                        else:
                                            message = f'{selected} / {label} / 開催場 {len(venues)}場'
                                    else:
                                        if data_status == 'error':
                                            message = '取得失敗'
                                        elif selected == today:
                                            message = '本日は地方競馬の開催はありません'
                                        else:
                                            message = f'{selected} の地方開催データがありません'
                                elif not races:
                                    if source == 'nar' and selected_venue and data_status == 'generating':
                                        message = 'データ取得中'
                                    else:
                                        message = f'{selected} / {label} のレースがありません'
                                elif mode == 'result':
                                    if selected_venue:
                                        message = f'{selected} / {selected_venue} / 結果検証'
                                    else:
                                        message = f'{selected} / {label} / 結果検証モード'
                                elif mode == 'analysis':
                                    message = f'{selected} / {label} / AI期待値分析'
                                else:
                                    if selected_venue:
                                        message = f'{selected} / {selected_venue} / 予想分析'
                                    else:
                                        message = f'{selected} / {label} / AI期待値分析'
        except FileNotFoundError as e:
            if (
                source in ('nar', 'jra')
                and _stay_on_selected_calendar_day(selected, source)
                and _today_source_has_no_meeting(source, today)
            ):
                data_status = 'ready'
                message = (
                    '本日はJRAの開催はありません' if source == 'jra'
                    else '本日は地方競馬の開催はありません'
                )
            else:
                latest = _latest_ready_pred_date(source, on_or_before=today) if source in ('nar', 'jra') else ''
                if latest:
                    selected = latest
                    data_status = 'ready'
                    message = f'{selected} / {label}'
                else:
                    data_status='generating'
                    message='データ取得中' if source=='nar' else (str(e) or 'データ取得中です。完了後に再読み込みしてください。')
        except Exception as e:
            # 安定稼働: 表示クラッシュでもキャッシュを落とさない
            print(f'[index] render fail source={source} selected={selected}: {e}', flush=True)
            if (
                source in ('nar', 'jra')
                and _stay_on_selected_calendar_day(selected, source)
                and _today_source_has_no_meeting(source, today)
            ):
                data_status = 'ready'
                message = (
                    '本日はJRAの開催はありません' if source == 'jra'
                    else '本日は地方競馬の開催はありません'
                )
            else:
                latest = _latest_ready_pred_date(source, on_or_before=today) if source in ('nar', 'jra') else ''
                if latest:
                    selected = latest
                    try:
                        pred = ARCH / f'predictions_{latest}.csv'
                        if pred.exists() and source == 'nar':
                            light_rows = _read_predictions_for_venue_picker(pred, source)
                            light_rows = apply_display_ranks(light_rows, by_venue=True)
                            venues = _venue_meetings(light_rows)
                            show_venue_picker = True
                            races = []
                            data_status = 'ready'
                            message = f'{selected} / {label} / 開催場 {len(venues)}場'
                            if latest not in av:
                                av = sorted(set(av) | {latest}, reverse=True)
                        else:
                            data_status = 'ready'
                            message = f'{selected} / {label}'
                    except Exception as e2:
                        print(f'[index] cache fallback fail: {e2}', flush=True)
                        data_status = 'ready'
                        message = f'{selected} / {label}'
                else:
                    data_status = 'error'
                    message = f'通信エラー: {e}'
    elif selected:
        if source in ('nar', 'jra'):
            latest = _latest_ready_pred_date(source, on_or_before=today)
            if _nar_pred_ready(selected, source):
                data_status='ready'
                message=f'{selected} / {label}'
                if selected not in av:
                    av=sorted(set(av)|{selected}, reverse=True)
            elif _stay_on_selected_calendar_day(selected, source) or _today_source_has_no_meeting(source, today):
                data_status='ready'
                message=(
                    '本日はJRAの開催はありません' if source=='jra'
                    else '本日は地方競馬の開催はありません'
                ) if (selected == today) else f'{selected} / {label} のレースがありません'
            elif latest:
                selected = latest
                data_status='ready'
                message=f'{selected} / {label}'
                if latest not in av:
                    av=sorted(set(av)|{latest}, reverse=True)
            elif selected == today:
                data_status='ready'
                message='本日はJRAの開催はありません' if source=='jra' else '本日は地方競馬の開催はありません'
            else:
                data_status='generating'
                message='データ取得中'
                if today not in av:
                    av=sorted(set(av)|{today}, reverse=True)
        else:
            data_status='error'
            message=f'{selected} は保存データにありません'
    elif source in ('nar', 'jra'):
        latest = _latest_ready_pred_date(source, on_or_before=today)
        if _nar_pred_ready(today, source):
            data_status='ready'
            message=f'{today} / {label}'
        elif _today_source_has_no_meeting(source, today):
            selected = today
            data_status='ready'
            message=(
                '本日はJRAの開催はありません' if source=='jra'
                else '本日は地方競馬の開催はありません'
            )
            if today not in av:
                av=sorted(set(av)|{today}, reverse=True)
        elif latest:
            selected = latest
            data_status='ready'
            message=f'{selected} / {label}'
            if latest not in av:
                av=sorted(set(av)|{latest}, reverse=True)
        else:
            data_status='generating'
            message='データ取得中'
    day_stats=None
    if mode=='result' and races and not show_venue_picker:
        day_stats=day_performance(races, verification, safe_pct=_safe_pct)
        today_ai_board=build_today_ai_board(races, verification)
    ledger=ledger_data(source=source, verification=verification) if mode in ('analysis','ledger') else {
        'has_data':False,'investment':0,'payout':0,'recovery':0.0,'profit':0,
        'by_type':[],'monthly':[],'tone':'roi-bad',
    }

    # 地方・中央: ジョブ状態で「取得中」を確定解除。
    # 重要: 表示中データは消さない。裏更新中は updating で維持し、成功後の再読込でのみ切替。
    # 朝8時以降・完成済みは必ず ready（generating は初回失敗時のみ）。
    status_refresh_url=''
    data_file_mtime=''
    if source in ('nar', 'jra'):
        job_status, job_msg = resolve_fetch_status(selected, source=source, force_refresh=False)
        has_content=bool(races) or (bool(venues) and show_venue_picker)
        selected_ok=_nar_pred_ready(selected, source) if selected else False
        sealed_ready=(
            selected == today
            and selected_ok
            and (past_predict_deadline() or is_sealed(today, source))
        )
        bg_running=(job_status in ('generating', 'updating')) or force_refresh
        no_meet_label=(
            '本日は地方競馬の開催はありません' if source=='nar'
            else '本日はJRAの開催はありません'
        )

        # キャッシュ優先: 完成データ/表示ありなら絶対に generating にしない
        if selected_ok or has_content:
            if job_status == 'updating' or (bg_running and force_refresh):
                data_status='updating'
                if selected_venue and races:
                    message=f'{selected} / {selected_venue} / 予想分析（更新中）'
                elif venues and show_venue_picker:
                    if selected == today:
                        message=f'本日開催 {selected} / {label} / 開催場 {len(venues)}場'
                    else:
                        message=f'{selected} / {label} / 開催場 {len(venues)}場'
                elif not message or message in ('データ取得中', '取得中', '取得失敗', 'オッズ更新中'):
                    message=job_msg or 'データ更新中（表示は維持）'
            else:
                data_status='ready'
                if not message or message in ('データ取得中', '取得中', '取得失敗', 'オッズ更新中', 'データ更新中（表示は維持）'):
                    if selected_venue and races:
                        message=f'{selected} / {selected_venue} / 予想分析'
                    elif venues and show_venue_picker:
                        if selected == today:
                            message=f'本日開催 {selected} / {label} / 開催場 {len(venues)}場'
                        else:
                            message=f'{selected} / {label} / 開催場 {len(venues)}場'
                    else:
                        message=job_msg or f'{selected} / {label} / AI期待値分析'
            # 完成予想があるのに会場が空 → 表示パス取りこぼしの再読込
            if (not has_content) and selected_ok and source == 'nar' and selected and not selected_venue:
                try:
                    pred = ARCH / f'predictions_{selected}.csv'
                    if pred.exists():
                        light_rows = _read_predictions_for_venue_picker(pred, source)
                        light_rows = apply_display_ranks(light_rows, by_venue=True)
                        venues = _venue_meetings(light_rows)
                        show_venue_picker = True
                        has_content = bool(venues)
                        if venues:
                            message = (
                                f'本日開催 {selected} / {label} / 開催場 {len(venues)}場'
                                if selected == today
                                else f'{selected} / {label} / 開催場 {len(venues)}場'
                            )
                except Exception as e:
                    print(f'[nar-venue] cache reload fail: {e}', flush=True)
        elif selected == today and not selected_ok and (
            _today_source_has_no_meeting(source, today)
            or (past_predict_deadline() and not source_is_race_day(source, today))
        ):
            data_status='ready'
            message=no_meet_label
        elif job_status=='timeout':
            data_status='timeout'
            message=job_msg or '更新タイムアウト'
        elif job_status=='error':
            data_status='error'
            message=job_msg or '更新失敗'
        elif job_status=='generating':
            # 初回（キャッシュ無し）のみ
            data_status='generating'
            message=job_msg or 'データ取得中'
        else:
            if data_status == 'generating' and not selected_ok and not has_content:
                pass
            elif not message:
                message = f'{selected} / {label}' if selected else message

        # 最終ガード: キャッシュ表示中は generating に落とさない
        has_content=bool(races) or (bool(venues) and show_venue_picker)
        selected_ok=_nar_pred_ready(selected, source) if selected else False
        if (selected_ok or has_content) and data_status == 'generating':
            data_status = 'updating' if job_status == 'updating' else 'ready'
            if not message or message == 'データ取得中':
                message = 'データ更新中（表示は維持）' if data_status == 'updating' else (
                    f'{selected} / {label} / 開催場 {len(venues)}場' if venues and show_venue_picker
                    else f'{selected} / {label}'
                )

        try:
            pf=ARCH/f'predictions_{selected}.csv' if selected else None
            if pf and pf.exists():
                data_file_mtime=str(int(pf.stat().st_mtime))
        except Exception:
            data_file_mtime=''

        from urllib.parse import urlencode
        q={'source':source,'mode':mode}
        if allow_past and selected and selected < today:
            q['date']=selected
            q['history']='1'
        else:
            q['date']=today
        if selected_venue:
            q['venue']=selected_venue
        status_refresh_url='/' + ('?' + urlencode(q) if q else '')
        _pipeline_log(
            '表示', '成功' if (selected==today or allow_past) else '失敗',
            selected,
            source=source,
            today=today,
            force_cal=int(force_cal_today),
            allow_past=int(allow_past),
            explicit=explicit_date or '-',
            status=data_status,
            venues=len(venues),
            venue=selected_venue or '-',
            races=len(races),
            pred_ready=int(_nar_pred_ready(selected, source)),
            job_date=str((_read_job_status(source) or {}).get('date') or '-'),
            refresh_date=q.get('date') or '-',
            保存件数=len(venues) if show_venue_picker else len(races),
        )

    if source in ('nar', 'jra'):
        disp_date = selected or today
        if data_status in ('ready', 'success', 'updating') and (
            _nar_pred_ready(disp_date, source) or races or venues
        ):
            log_pipeline_stage(
                source, 7, True, disp_date,
                status=data_status, races=len(races), venues=len(venues),
            )
        elif data_status == 'generating' and not _nar_pred_ready(disp_date, source):
            log_pipeline_stage(
                source, 7, False, disp_date,
                status=data_status, message=(message or '')[:80],
            )

    return render_template('index.html',races=races,targets=targets,selected_date=selected,today=today,
        message=message,available_dates=av,source=source,mode=mode,has_results=has_results,
        analysis=analysis_data(races if not show_venue_picker else []),verification=verification,
        ledger=ledger,
        venues=venues,selected_venue=selected_venue,show_venue_picker=show_venue_picker,
        today_date=today,
        day_stats=day_stats,data_status=data_status,
        buy_candidates=buy_candidates,today_ai_board=today_ai_board,data_updated_at=data_updated_at,
        status_refresh_url=status_refresh_url,data_file_mtime=data_file_mtime,
        areru_pipeline=build_areru_pipeline_board(selected, source))


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
    # 読み込み前に未確定CSVを昇格（検証画面の相対S漏洩を防ぐ）
    try:
        for f in ARCH.glob('predictions_*.csv'):
            _ensure_pred_file_finalized(f)
    except Exception:
        pass
    sig=_fs_sig(ANALYSIS_CSV)
    if _PRED_META_CACHE.get('sig')==sig and _PRED_META_CACHE.get('data') is not None:
        return _PRED_META_CACHE['data']
    meta={}
    # メタに必要な列だけ読む
    want={'race_id','開催地','レース','日付','勝負ランク','相対ランク','推奨券種','本命','本命馬番',
          'ワイド判定','馬連判定','三連複判定','印データ','BET判定','BET期待値','source',
          '投資判定','期待値','S降格'}
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


@app.route('/api/refresh-status', methods=['GET'])
@app.route('/api/nar-refresh-status', methods=['GET'])
def api_refresh_status():
    """バックグラウンド更新の完了監視用（画面は消さず、成功後にだけ再読込する）。"""
    today=_today_jst()
    date_str=str(request.args.get('date') or today).strip() or today
    source=str(request.args.get('source') or 'nar').strip().lower()
    if source not in ('jra', 'nar'):
        # 旧 /api/nar-refresh-status 互換
        source='nar'
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date_str):
        date_str=today
    # 10分超 running はここで必ず解除（updating 永久化を防ぐ）
    st=_expire_stale_job(source)
    state=str(st.get('state') or 'idle')
    age=_nar_job_age_sec(st)
    ready=_nar_pred_ready(date_str, source)
    mtime=''
    try:
        pf=ARCH/f'predictions_{date_str}.csv'
        if pf.exists():
            mtime=str(int(pf.stat().st_mtime))
    except Exception:
        mtime=''
    # 開催なし完了も finished
    no_meet = str(st.get('message') or '') == '本日は開催なし' and state == 'success'
    # success/error/idle は finished。running だけ継続。ready でも running 中は未完了。
    finished = (state != 'running') or no_meet
    updating = state == 'running'
    outcome = str(st.get('outcome') or '')
    if not outcome:
        stage = str(st.get('stage') or '')
        if state == 'running':
            outcome = 'updating'
        elif stage in ('timeout', 'timeout_ready') or 'タイムアウト' in str(st.get('message') or ''):
            outcome = 'timeout'
        elif state == 'success':
            outcome = 'success'
        elif state == 'error':
            outcome = 'failed'
        else:
            outcome = 'idle'
    return {
        'ok': True,
        'date': date_str,
        'source': source,
        'state': state,
        'outcome': outcome,  # updating | success | failed | timeout
        'ready': ready or no_meet,
        'finished': finished,
        'updating': updating,
        'age_sec': int(age) if age < 1e8 else None,
        'stale_sec': _JOB_STALE_SEC,
        'mtime': mtime,
        'message': str(st.get('message') or st.get('error') or ''),
        'error': str(st.get('error') or ''),
        'stage': str(st.get('stage') or ''),
    }


@app.route('/healthz', methods=['GET'])
def healthz():
    """軽量ヘルスチェック。ジョブ起動なし（Render wake / GHA 用）。

    pandas で CSV 本文を開かない。重い / 描画と独立して 200 を返す必要がある。
    """
    today = _today_jst()
    latest_date = ''
    latest_mtime = ''
    try:
        files = sorted(ARCH.glob('predictions_????-??-??.csv'))
        if files:
            newest = files[-1]
            latest_date = newest.stem.replace('predictions_', '')
            latest_mtime = datetime.fromtimestamp(
                newest.stat().st_mtime, JST,
            ).strftime('%Y-%m-%d %H:%M:%S%z')
    except Exception:
        pass
    today_pred = ARCH / f'predictions_{today}.csv'
    try:
        today_present = today_pred.exists() and today_pred.stat().st_size >= 32
    except OSError:
        today_present = False
    return {
        'ok': True,
        'today': today,
        'heavy': _HEAVY_JOB_STATE.get('name') or '',
        'predict': _PREDICT_GLOBAL_STATE.get('key') or '',
        'pid': os.getpid(),
        'latest_pred_date': latest_date,
        'latest_pred_mtime': latest_mtime,
        'today_ready': {
            'nar': today_present,
            'jra': today_present,
        },
        'commit': (
            os.environ.get('RENDER_GIT_COMMIT')
            or os.environ.get('GIT_COMMIT')
            or ''
        )[:12],
    }


@app.route('/api/job-status', methods=['GET'])
def api_job_status():
    """ジョブ状態デバッグ（生成パイプライン停止箇所の特定用）。"""
    source = str(request.args.get('source') or 'nar').strip().lower()
    if source not in ('nar', 'jra'):
        source = 'nar'
    today = _today_jst()
    st = _expire_stale_job(source)
    c = _nar_day_counts(today, source)
    return {
        'ok': True,
        'source': source,
        'today': today,
        'heavy': _HEAVY_JOB_STATE.get('name') or '',
        'predict': _PREDICT_GLOBAL_STATE.get('key') or '',
        'predict_running': _predict_busy(),
        'heavy_busy': _heavy_busy(),
        'job_active': _job_is_active(source),
        'ready': _nar_pred_ready(today, source),
        'counts': c,
        'status': st,
    }


def _cron_token_ok() -> bool:
    token=str(request.args.get('token') or request.headers.get('X-Cron-Token') or '').strip()
    expected=str(os.environ.get('CRON_TOKEN') or '').strip()
    if expected and token != expected:
        return False
    return True


def _cron_pipeline_mode(source: str = '') -> str:
    """cron クエリ mode=morning|odds|auto。

    auto はデータ主導（時刻では決めない）。
    GitHub の schedule は数時間遅延するため、時刻で odds に落とすと
    当日予想が一度も生成されないまま「成功」してしまう。
    """
    mode = str(request.args.get('mode') or '').strip().lower()
    if mode in ('morning', 'full'):
        return 'morning'
    if mode == 'odds':
        return 'odds'
    today = _today_jst()
    srcs = [source] if source in ('nar', 'jra') else ['nar', 'jra']
    # 当日予想が未完成のソースが1つでもあれば本生成
    if any(not _nar_pred_ready(today, s) for s in srcs):
        return 'morning'
    return 'odds'


def _run_cron_source(src: str, mode: str) -> None:
    today = _today_jst()
    force_full = mode == 'morning'
    log_orchestrator('cron', 'START', src, mode=mode, date=today)
    print(
        f'[cron-{src}] mode={mode} force_full={force_full} '
        f'deadline_passed={past_predict_deadline()} date={today}',
        flush=True,
    )
    if mode == 'morning' and not source_is_race_day(src, today):
        _write_job_status(
            src, state='success', stage='done', message='本日は開催なし', date_str=today,
        )
        print(f'[cron-{src}] skip: no meeting', flush=True)
        return
    run_today_pipeline(src, force=True, force_full=force_full)
    if mode == 'morning':
        # 朝の本生成後のみカード補完。締切後 odds では触らない
        print(f'[cron-{src}] bootstrap incomplete cards', flush=True)
        bootstrap_source(src)
    print(f'[cron-{src}] bootstrap results', flush=True)
    bootstrap_missing_results(src, prefer_dates=[today])
    ready = _nar_pred_ready(today, src)
    if ready:
        auto_seal_if_ready(today, src, True)
    c = _nar_day_counts(today, src)
    ok = ready or (src == 'jra' and c['runners_races'] == 0) or (
        mode == 'morning' and not source_is_race_day(src, today)
    )
    _pipeline_log(
        '全体', '成功' if ok else '失敗',
        today,
        source=src,
        mode=mode,
        runners=c['runners_races'], predictions=c['pred_races'],
        保存件数=c['pred_races'],
        sealed=int(is_sealed(today, src)),
    )
    print(f'[cron-{src}] done ready={int(ready)} sealed={int(is_sealed(today, src))}', flush=True)
    log_orchestrator(
        'cron', 'OK' if ok else 'ERROR', src,
        mode=mode, ready=int(ready), predictions=c['pred_races'],
    )


@app.route('/cron/nar-daily', methods=['POST','GET'])
def cron_nar_daily():
    """外部cron向け: 地方の開催場→レース→結果をバックグラウンドで安定更新。

    mode=morning … 本生成
    mode=odds … オッズ・取消のみ
    mode=auto … 当日予想の有無で自動切替（時刻では決めない）
    """
    if not _cron_token_ok():
        return {'ok': False, 'error': 'unauthorized'}, 401
    if not _generation_enabled():
        log_orchestrator('cron', 'SKIP', 'nar', reason='generation_disabled')
        return {'ok': True, 'started': False, 'source': 'nar', 'skipped': 'generation disabled'}
    mode = _cron_pipeline_mode('nar')

    def _run():
        try:
            _run_cron_source('nar', mode)
        except Exception as e:
            log_orchestrator('cron', 'ERROR', 'nar', error=str(e)[:120])
            print(f'[cron-nar] fail: {e}', flush=True)

    threading.Thread(target=_run, daemon=True).start()
    return {'ok': True, 'started': True, 'source': 'nar', 'mode': mode}


@app.route('/cron/jra-daily', methods=['POST','GET'])
def cron_jra_daily():
    """外部cron向け: JRAの開催→レース→予想。開催日のみ本生成。"""
    if not _cron_token_ok():
        return {'ok': False, 'error': 'unauthorized'}, 401
    if not _generation_enabled():
        log_orchestrator('cron', 'SKIP', 'jra', reason='generation_disabled')
        return {'ok': True, 'started': False, 'source': 'jra', 'skipped': 'generation disabled'}
    mode = _cron_pipeline_mode('jra')

    def _run():
        try:
            _run_cron_source('jra', mode)
        except Exception as e:
            log_orchestrator('cron', 'ERROR', 'jra', error=str(e)[:120])
            print(f'[cron-jra] fail: {e}', flush=True)

    threading.Thread(target=_run, daemon=True).start()
    return {'ok': True, 'started': True, 'source': 'jra', 'mode': mode}


@app.route('/cron/daily', methods=['POST','GET'])
def cron_daily_both():
    """外部cron向け: 地方→JRA の順で日次更新（直列）。推奨エンドポイント。"""
    if not _cron_token_ok():
        return {'ok': False, 'error': 'unauthorized'}, 401
    if not _generation_enabled():
        log_orchestrator('cron', 'SKIP', 'all', reason='generation_disabled')
        return {
            'ok': True, 'started': False, 'source': 'all',
            'skipped': 'generation disabled; data is built by GitHub Actions',
        }
    mode = _cron_pipeline_mode()
    log_orchestrator('cron', 'QUEUED', 'all', mode=mode)

    def _run():
        for src in ('nar', 'jra'):
            try:
                # ソース単位で morning/odds を再判定（片方だけ未生成のケース）
                _run_cron_source(src, 'morning' if not _nar_pred_ready(_today_jst(), src) else mode)
            except Exception as e:
                log_orchestrator('cron', 'ERROR', src, error=str(e)[:120])
                print(f'[cron-daily] {src} fail: {e}', flush=True)
            # ソース切替前にメモリ回収
            try:
                import gc
                gc.collect()
            except Exception:
                pass
        log_orchestrator('cron', 'DONE', 'all', mode=mode)

    threading.Thread(target=_run, daemon=True).start()
    return {'ok': True, 'started': True, 'source': 'all', 'mode': mode}


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
    if not _generation_enabled():
        # 配信専用ホストでは重いジョブを受け付けない（Free の固まり防止）
        log_orchestrator('refresh', 'SKIP', source, reason='generation_disabled', mode=mode)
        return {
            'ok': True, 'started': False, 'mode': mode, 'source': source,
            'skipped': 'generation disabled; data is built by GitHub Actions',
        }
    try:
        def _run_refresh(_mode=mode, _source=source, _date=date):
            def _body():
                try:
                    if _mode=='odds':
                        today=_today_jst()
                        # 締切後はランク再計算しない（パッチのみ）
                        if past_predict_deadline():
                            srcs = (
                                [_source] if _source in ('jra', 'nar')
                                else ['nar', 'jra']
                            )
                            for s in srcs:
                                _run_odds_only_update(s, today)
                        else:
                            cmd=[sys.executable,'refresh_data.py','--latest-only','--odds-only','--source',_source,'--skip-predict']
                            subprocess.run(cmd, check=False, timeout=1800)
                            _run_replay_predict_subprocess(today, timeout=600)
                    elif _mode=='results':
                        if _date and re.fullmatch(r'\d{4}-\d{2}-\d{2}', _date):
                            cmd=[sys.executable,'results.py','--source',_source,'--dates',_date]
                        else:
                            cmd=[sys.executable,'results.py','--latest','--source',_source]
                        subprocess.run(cmd, check=False, timeout=1800)
                    else:
                        today=_today_jst()
                        pred_day=_date if _date and re.fullmatch(r'\d{4}-\d{2}-\d{2}', _date) else today
                        # 地方は --latest-only（前日アンカー）を使わず、カレンダー当日を明示指定
                        if _date and re.fullmatch(r'\d{4}-\d{2}-\d{2}', _date):
                            cmd=[
                                sys.executable,'refresh_data.py',
                                '--dates', _date, '--source', _source,
                                '--no-discover', '--skip-predict',
                            ]
                        elif _source == 'nar':
                            cmd=[
                                sys.executable,'refresh_data.py',
                                '--dates', today, '--source', 'nar',
                                '--no-discover', '--skip-predict',
                            ]
                            _pipeline_log('開催取得', '開始', today, mode='refresh_full', 保存件数='-')
                        else:
                            cmd=[
                                sys.executable,'refresh_data.py',
                                '--latest-only','--source',_source,'--skip-predict',
                            ]
                        subprocess.run(cmd, check=False, timeout=1800)
                        _clear_runtime_caches()
                        _run_replay_predict_subprocess(pred_day, timeout=600)
                        if _source in ('nar','all'):
                            c=_nar_day_counts(pred_day, 'nar')
                            _pipeline_log(
                                '保存', '成功' if c['pred_races'] else '失敗',
                                pred_day,
                                predictions=c['pred_races'], runners=c['runners_races'],
                                odds_json=c.get('odds_json', 0), 保存件数=c['pred_races'],
                            )
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


# gunicorn / flask 起動時に当日の地方・JRAを自動シード（リクエスト待ちで空のままにしない）
try:
    _ensure_source_today_seeded('nar')
    _ensure_source_today_seeded('jra')
except Exception as _boot_e:
    print(f'[boot] init skip: {_boot_e}', flush=True)


if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','5001')),debug=False)
