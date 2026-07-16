from flask import Flask,render_template,request
import subprocess,sys,json,re
from pathlib import Path
from datetime import date
import os
import pandas as pd
from areru_engine import parse_date

app=Flask(__name__)
BASE=Path(__file__).resolve().parent
DATA=BASE/'data'; ARCH=DATA/'predictions_by_date'; ARCH.mkdir(parents=True,exist_ok=True)
RUNNERS=DATA/'runners.csv'
LEGACY=DATA/'score_test_data.csv'
ANALYSIS_CSV=DATA/'analysis_result.csv'

def _runner_path():
    if RUNNERS.exists(): return RUNNERS
    if LEGACY.exists(): return LEGACY
    return None

def dates(source='all'):
    """開催日一覧。runners.csv を正とし、生成済み predictions も合流する。"""
    found=set()
    p=_runner_path()
    if p is not None:
        try:
            rdf=pd.read_csv(p,encoding='utf-8-sig')
            if '日付' not in rdf.columns:
                pass
            else:
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
        if source in ('jra','nar'):
            try:
                pdf=pd.read_csv(f,encoding='utf-8-sig')
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
        else:
            found.add(day)
    if ANALYSIS_CSV.exists():
        try:
            ad=pd.read_csv(ANALYSIS_CSV,encoding='utf-8-sig').fillna('')
            if source in ('jra','nar') and 'source' in ad.columns:
                ad=ad[ad['source'].astype(str).str.lower()==source]
            found.update([x for x in ad['date'].astype(str).tolist() if re.fullmatch(r'\d{4}-\d{2}-\d{2}', x)])
        except Exception:
            pass
    return sorted(found, reverse=True)

def ensure(d, source='all'):
    f=ARCH/f'predictions_{d}.csv'; regen=True
    if f.exists():
        try:
            pdf=pd.read_csv(f,encoding='utf-8-sig')
            regen='印データ' not in pdf.columns
            # 指定ソースのレースが predictions に無いが runners にある場合は再生成
            if not regen and source in ('jra','nar'):
                from areru_engine import source_from_race_id
                if 'source' in pdf.columns:
                    has_src=(pdf['source'].astype(str).str.lower()==source).any()
                else:
                    has_src=pdf['race_id'].map(source_from_race_id).eq(source).any() if 'race_id' in pdf.columns else False
                if not has_src:
                    rp=_runner_path()
                    if rp is not None:
                        rdf=pd.read_csv(rp,encoding='utf-8-sig')
                        day=parse_date(rdf['日付']).dt.strftime('%Y-%m-%d')==d
                        if 'source' in rdf.columns:
                            need=(rdf[day]['source'].astype(str).str.lower()==source).any()
                        else:
                            need=rdf.loc[day,'race_id'].map(source_from_race_id).eq(source).any()
                        if need:
                            regen=True
        except Exception:
            regen=True
    if regen:
        # runners が無い/対象日が無い場合は refresh で取得を試みる
        need_refresh=False
        rp=_runner_path()
        if rp is None:
            need_refresh=True
        else:
            try:
                rdf=pd.read_csv(rp,encoding='utf-8-sig')
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
            subprocess.run(
                [sys.executable,'refresh_data.py','--dates',d,'--skip-predict','--source',src],
                check=True,timeout=900,
            )
        subprocess.run([sys.executable,'replay_predict.py',d],check=True,timeout=600)
    return f

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
def prep(records):
    from areru_engine import RANK_LABELS, RANK_CLASSES
    for r in records:
        try: r['印一覧']=json.loads(r.get('印データ','[]'))
        except: r['印一覧']=[]
        for k in ['ワイド買い目','馬連買い目','三連複買い目']:
            r[k+'一覧']=str(r.get(k,'見送り')).split('｜')
        rank=str(r.get('勝負ランク','') or '').upper()
        if rank in RANK_LABELS:
            r['勝負ランク']=rank
            r['BET判定']=RANK_LABELS[rank]
            r['BETクラス']=RANK_CLASSES.get(rank, r.get('BETクラス',''))
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


def bootstrap_source(source: str) -> bool:
    """地方タブでデータが古い/無い場合に最新開催を自動取得する。

    Returns: 更新を実行したら True
    """
    if source != 'nar':
        return False
    # 直近数日だけ走査（毎回28日走査しない）
    try:
        from netkeiba_client import NetkeibaClient
        found=NetkeibaClient(sleep=0.12).discover_kaisai_dates(
            lookback=4, lookahead=1, source='nar'
        )
        remote=found[0] if found else ''
    except Exception:
        remote=''
    local=_source_latest_in_runners('nar')
    if remote and local and local>=remote:
        return False
    if not remote and local:
        return False
    lock=DATA/'.nar_bootstrap.lock'
    if lock.exists():
        try:
            age=(__import__('time').time()-lock.stat().st_mtime)
            if age < 1800:
                print('[bootstrap] already running, skip')
                return False
        except Exception:
            pass
    print(f'[bootstrap] source=nar local={local or "-"} remote={remote or "-"}')
    try:
        lock.write_text(str(__import__('os').getpid()), encoding='utf-8')
        subprocess.run(
            [sys.executable,'refresh_data.py','--latest-only','--source','nar','--lookback','5','--lookahead','1'],
            check=True,timeout=1800,
        )
    finally:
        try: lock.unlink(missing_ok=True)
        except Exception: pass
    return True


@app.route('/')
def index():
    source=request.args.get('source','jra')
    if source not in ('jra','nar','all'):
        source='jra'
    mode=request.args.get('mode','predict')
    # 地方タブ初回/鮮度切れ時は実データを自動取得
    try:
        if source=='nar':
            bootstrap_source(source)
    except Exception as e:
        print(f'[bootstrap] skip: {e}')
    av=dates(source)
    selected=request.args.get('date','').strip()
    # ソース切替で他開催の日付が残っていても、そのソースの開催日へ寄せる
    if not selected or selected not in av:
        selected=av[0] if av else ''
    # 結果検証タブ: 選択日に結果が無い場合は最新の結果日へ寄せる（結果待ちの誤表示を防ぐ）
    result_days=dates_with_results(source)
    if mode=='result' and result_days:
        if not selected or selected not in result_days:
            selected=result_days[0]
    races=[]; targets=[]; message='予想データがありません'; has_results=False
    verification=verification_data(selected, source=source)

    if selected in av:
        try:
            pred_path=ARCH/f'predictions_{selected}.csv'
            if pred_path.exists() or mode!='result':
                df=pd.read_csv(ensure(selected, source=source)).fillna('なし')
                races=prep(df.to_dict('records'))
                races=_filter_records_by_source(races, source)
                for row in races:
                    if not _race_date(row):
                        row['日付']=selected
                races,has_results=attach_results(races, selected_date=selected)
            targets=sorted([r for r in races if r.get('勝負ランク') in ['S','A']],key=lambda x:float(x.get('BET期待値',0)),reverse=True)[:5]
            label={'jra':'JRA中央','nar':'地方競馬','all':'全開催'}.get(source, source)
            if not races:
                message=f'{selected} / {label} のレースがありません'
            elif mode=='result':
                message=f'{selected} / {label} / 結果検証モード'
            elif mode=='analysis':
                message=f'{selected} / {label} / AI仮想レース分析 β版'
            else:
                message=f'{selected} / {label} / AI仮想レース分析 β版'
        except Exception as e: message=f'生成エラー: {e}'
    elif selected: message=f'{selected} は保存データにありません'
    elif source=='nar':
        message='地方開催データがありません。/refresh?source=nar で取得できます。'
    return render_template('index.html',races=races,targets=targets,selected_date=selected,today=date.today().isoformat(),
        message=message,available_dates=av,source=source,mode=mode,has_results=has_results,
        analysis=analysis_data(races),verification=verification)


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
        score_lu=score_cache.get(d,{})

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
            review.append({'印':mark,'馬名':name,'着順':disp})
        r['結果一覧']=review
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

        main_finish=lookup_finish(r.get('本命',''))
        if race_has_result and main_finish:
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
    ranks=[{'label':x,'name':RANK_LABELS.get(x,x),'count':int((df['rank']==x).sum())} for x in ['S','A','B','C']]
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
    meta={}
    for f in ARCH.glob('predictions_*.csv'):
        try:
            df=pd.read_csv(f,encoding='utf-8-sig').fillna('')
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
    if rank in ('S','A','B','C'):
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
    }
    if not ANALYSIS_CSV.exists():
        return empty
    try:
        df=pd.read_csv(ANALYSIS_CSV,encoding='utf-8-sig').fillna('')
    except Exception:
        return empty
    if df.empty or 'bet_type' not in df.columns:
        return empty
    if source in ('jra','nar'):
        if 'source' in df.columns:
            df=df[df['source'].astype(str).str.lower()==source].copy()
        elif 'race_id' in df.columns:
            from areru_engine import source_from_race_id
            df=df[df['race_id'].map(source_from_race_id)==source].copy()
        if df.empty:
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
    for key, name in [('S','勝負'),('A','買い'),('B','様子見'),('C','見送り')]:
        g=ranked[ranked['勝負ランク']==key] if not ranked.empty else ranked
        s=pack(g)
        by_rank.append({'key':key,'name':name,**s})
    # ランク×券種（購入対象のみ）
    typed=ranked.copy()
    if not typed.empty:
        typed['券種表示']=typed['bet_type'].map(_bet_type_label)
    by_rank_type=[]
    for key, name in [('S','勝負'),('A','買い'),('B','様子見'),('C','見送り')]:
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
    # グラフ用スケール
    max_abs=max([abs(x['value']) for x in cum_profit]+[1])
    for x in cum_profit:
        x['pct']=round(abs(x['value'])/max_abs*100,1)
        x['pos']=x['value']>=0
        x['tone']='roi-good' if x['pos'] else 'roi-bad'
    for x in recovery_series:
        x['pct']=x.get('bar',_bar_width(x['value']))

    return {
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
        'recent_rows':recent,
        'purchase_count':len(recent),
    }

@app.route('/refresh', methods=['POST','GET'])
def refresh_route():
    """最新開催日・オッズを取得して runners / predictions を更新。"""
    mode=request.args.get('mode','full')
    source=request.args.get('source','all')
    if source not in ('jra','nar','all'):
        source='all'
    try:
        if mode=='odds':
            cmd=[sys.executable,'refresh_data.py','--latest-only','--odds-only','--source',source]
        elif mode=='results':
            cmd=[sys.executable,'results.py','--latest','--source',source]
        else:
            cmd=[sys.executable,'refresh_data.py','--latest-only','--source',source]
        subprocess.run(cmd,check=True,timeout=1800)
        av=dates(source)
        return {'ok':True,'dates':av,'latest':av[0] if av else None,'mode':mode,'source':source}
    except Exception as e:
        return {'ok':False,'error':str(e)}, 500

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','5001')),debug=False)
