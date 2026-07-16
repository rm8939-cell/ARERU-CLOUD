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

def dates():
    """開催日一覧。runners.csv を正とし、生成済み predictions も合流する。"""
    found=set()
    p=_runner_path()
    if p is not None:
        try:
            d=parse_date(pd.read_csv(p,usecols=['日付'])['日付']).dropna().dt.strftime('%Y-%m-%d')
            found.update(d.unique().tolist())
        except Exception:
            pass
    for f in ARCH.glob('predictions_*.csv'):
        m=re.fullmatch(r'predictions_(\d{4}-\d{2}-\d{2})\.csv', f.name)
        if m: found.add(m.group(1))
    if ANALYSIS_CSV.exists():
        try:
            ad=pd.read_csv(ANALYSIS_CSV,usecols=['date']).fillna('')
            found.update([x for x in ad['date'].astype(str).tolist() if re.fullmatch(r'\d{4}-\d{2}-\d{2}', x)])
        except Exception:
            pass
    return sorted(found, reverse=True)

def ensure(d):
    f=ARCH/f'predictions_{d}.csv'; regen=True
    if f.exists():
        try: regen='印データ' not in pd.read_csv(f,nrows=1).columns
        except: regen=True
    if regen:
        # runners が無い/対象日が無い場合は refresh で取得を試みる
        need_refresh=False
        rp=_runner_path()
        if rp is None:
            need_refresh=True
        else:
            try:
                rd=parse_date(pd.read_csv(rp,usecols=['日付'])['日付']).dt.strftime('%Y-%m-%d')
                need_refresh=d not in set(rd.dropna().tolist())
            except Exception:
                need_refresh=True
        if need_refresh:
            subprocess.run([sys.executable,'refresh_data.py','--dates',d,'--skip-predict'],check=True,timeout=900)
        subprocess.run([sys.executable,'replay_predict.py',d],check=True,timeout=240)
    return f

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


def dates_with_results() -> list[str]:
    """results.csv / analysis_result.csv にある開催日（新しい順）。"""
    found=set()
    rp=DATA/'results.csv'
    if rp.exists():
        try:
            rdf=pd.read_csv(rp,usecols=lambda c: c in ('date','日付')).fillna('')
            col='date' if 'date' in rdf.columns else ('日付' if '日付' in rdf.columns else None)
            if col:
                found.update([x for x in rdf[col].astype(str) if re.fullmatch(r'\d{4}-\d{2}-\d{2}', x)])
        except Exception:
            pass
    if ANALYSIS_CSV.exists():
        try:
            ad=pd.read_csv(ANALYSIS_CSV,usecols=['date'],encoding='utf-8-sig').fillna('')
            found.update([x for x in ad['date'].astype(str) if re.fullmatch(r'\d{4}-\d{2}-\d{2}', x)])
        except Exception:
            pass
    return sorted(found, reverse=True)


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
    """回収率の色区分: 100%以上緑 / 80〜99%黄 / 79%以下赤"""
    try:
        v=float(recovery)
    except (TypeError, ValueError):
        v=0.0
    if v>=100:
        return 'roi-good'
    if v>=80:
        return 'roi-mid'
    return 'roi-bad'


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
    """race_id → 予想時メタデータ。"""
    meta={}
    for f in ARCH.glob('predictions_*.csv'):
        try:
            df=pd.read_csv(f,encoding='utf-8-sig').fillna('')
        except Exception:
            continue
        if 'race_id' not in df.columns:
            continue
        for _,row in df.iterrows():
            rid=_norm_race_id(row.get('race_id',''))
            if not rid or rid in meta:
                continue
            meta[rid]=row.to_dict()
    return meta


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


def _enrich_verify_row(r, pred_meta):
    rid=_norm_race_id(r.get('race_id',''))
    pred=pred_meta.get(rid) or {}
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
    recommend=str(pred.get('推奨券種','') or '')
    ai_comment=str(pred.get('馬券戦略理由','') or '')
    return {
        'date':str(r.get('date','')),
        'race':str(r.get('race','')),
        'race_id':rid,
        'venue':str(r.get('開催地','') or pred.get('開催地','') or ''),
        'bet_type':str(r.get('bet_type','')),
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
    }


def verification_data(selected_date=''):
    """analysis_result.csv から結果検証ダッシュボード用データを構築。"""
    empty_pack={
        'total_bets':0,'hits':0,'hit_rate':0.0,'recovery':0.0,'roi':0.0,
        'investment':0,'payout':0,'profit':0,'tone':'roi-bad','bar':0,
    }
    empty={
        'has_data':False,'selected_date':selected_date,
        'total_bets':0,'hit_rate':0.0,'recovery':0.0,'roi':0.0,
        'investment':0,'payout':0,'profit':0,'tone':'roi-bad',
        'daily':[],'by_type':[],'by_rank':[],'main':{},
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
    for c in ['hit','payout','investment','profit','roi']:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
    all_df=df.copy()
    day_df=all_df[all_df['date'].astype(str)==str(selected_date)] if selected_date else all_df

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
        by_type.append({'bet_type':str(bt),**s})
    by_type=sorted(by_type,key=lambda x:x['investment'],reverse=True)
    # 本命成績
    main_df=src[src['bet_type']=='本命'] if '本命' in set(src['bet_type'].astype(str)) else pd.DataFrame()
    main=pack(main_df) if not main_df.empty else dict(empty_pack)
    # AIランク結合 → ランク別KPI（件数・的中率・回収率・ROI・収支）
    pred_meta=_load_prediction_meta()
    ranked=src.copy()
    if '勝負ランク' not in ranked.columns or ranked['勝負ランク'].astype(str).str.strip().eq('').all():
        ranked['勝負ランク']=ranked['race_id'].map(
            lambda rid: str((pred_meta.get(_norm_race_id(rid)) or {}).get('勝負ランク','') or '').upper()
        )
    else:
        ranked['勝負ランク']=ranked['勝負ランク'].astype(str).str.upper().str.strip()
    by_rank=[]
    for key, ranks, name in [
        ('S',['S'],'勝負'),('A',['A'],'買い'),('B',['B'],'様子見'),
        ('C',['C'],'見送り'),('S+A',['S','A'],'勝負+買い'),
    ]:
        g=ranked[ranked['勝負ランク'].isin(ranks)]
        s=pack(g)
        by_rank.append({'key':key,'name':name,**s})
    # カード表示用明細（予想メタ結合）
    show=day_df if not day_df.empty else all_df
    recent=[]
    for _,r in show.tail(120).iloc[::-1].iterrows():
        recent.append(_enrich_verify_row(r, pred_meta))
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
        'main':main,
        'recovery_series':recovery_series,
        'cum_profit':cum_profit,
        'recent_rows':recent,
    }

@app.route('/')
def index():
    source=request.args.get('source','jra')
    mode=request.args.get('mode','predict')
    av=dates()
    selected=request.args.get('date','').strip() or (av[0] if av else '')
    # 結果検証タブ: 選択日に結果が無い場合は最新の結果日へ寄せる（結果待ちの誤表示を防ぐ）
    result_days=dates_with_results()
    if mode=='result' and result_days:
        if not selected or selected not in result_days:
            selected=result_days[0]
    races=[]; targets=[]; message='予想データがありません'; has_results=False
    verification=verification_data(selected)

    if source=='nar':
        message='地方競馬エンジンは接続準備中です。JRA予想ロジックとは分離して実装します。'
        return render_template('index.html',races=[],targets=[],selected_date=selected,today=date.today().isoformat(),
            message=message,available_dates=av,source=source,mode=mode,has_results=False,
            analysis=analysis_data([]),verification=verification)

    if selected in av:
        try:
            pred_path=ARCH/f'predictions_{selected}.csv'
            if pred_path.exists() or mode!='result':
                df=pd.read_csv(ensure(selected)).fillna('なし')
                races=prep(df.to_dict('records'))
                for row in races:
                    if not _race_date(row):
                        row['日付']=selected
                races,has_results=attach_results(races, selected_date=selected)
            targets=sorted([r for r in races if r.get('勝負ランク') in ['S','A']],key=lambda x:float(x.get('BET期待値',0)),reverse=True)[:5]
            if mode=='result':
                message=f'{selected} / 結果検証モード'
            elif mode=='analysis':
                message=f'{selected} / AI仮想レース分析 β版'
            else:
                message=f'{selected} / AI仮想レース分析 β版'
        except Exception as e: message=f'生成エラー: {e}'
    elif selected: message=f'{selected} は保存データにありません'
    return render_template('index.html',races=races,targets=targets,selected_date=selected,today=date.today().isoformat(),
        message=message,available_dates=av,source=source,mode=mode,has_results=has_results,
        analysis=analysis_data(races),verification=verification)

@app.route('/refresh', methods=['POST','GET'])
def refresh_route():
    """最新開催日・オッズを取得して runners / predictions を更新。"""
    mode=request.args.get('mode','full')
    try:
        if mode=='odds':
            cmd=[sys.executable,'refresh_data.py','--latest-only','--odds-only']
        elif mode=='results':
            cmd=[sys.executable,'results.py','--latest']
        else:
            cmd=[sys.executable,'refresh_data.py','--latest-only']
        subprocess.run(cmd,check=True,timeout=1800)
        av=dates()
        return {'ok':True,'dates':av,'latest':av[0] if av else None,'mode':mode}
    except Exception as e:
        return {'ok':False,'error':str(e)}, 500

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','5001')),debug=False)
