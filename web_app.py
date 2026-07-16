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
    for r in records:
        try: r['印一覧']=json.loads(r.get('印データ','[]'))
        except: r['印一覧']=[]
        for k in ['ワイド買い目','馬連買い目','三連複買い目']:
            r[k+'一覧']=str(r.get(k,'見送り')).split('｜')
    return records

def clean_horse(x):
    return re.sub(r'\s+','',str(x)).strip()

def attach_results(records):
    rp=DATA/'results.csv'
    if not rp.exists(): return records, False
    try:
        rdf=pd.read_csv(rp).fillna('')
    except: return records, False
    # 旧JRA URL行は無視
    if 'race_id' not in rdf.columns or '馬名' not in rdf.columns:
        return records, False
    rdf=rdf[~rdf['race_id'].astype(str).str.startswith('http')].copy()
    finish_col='着順' if '着順' in rdf.columns else None
    if finish_col is None: return records, False
    lookup={(str(x['race_id']),clean_horse(x['馬名'])):str(x[finish_col]) for _,x in rdf.iterrows() if str(x[finish_col]).strip()}
    # 旧JRA URL予想向け: (開催地, レース, 馬名) フォールバック
    venue_lookup={}
    if '開催地' in rdf.columns and 'レース' in rdf.columns:
        for _,x in rdf.iterrows():
            try: rn=int(float(x['レース']))
            except Exception: continue
            venue_lookup[(str(x['開催地']),rn,clean_horse(x['馬名']))]=str(x[finish_col])
    any_result=False
    for r in records:
        rid=str(r['race_id'])
        try: race_no=int(float(r.get('レース',0)))
        except Exception: race_no=None
        venue=str(r.get('開催地',''))
        entries=[('◎',r.get('本命',''))]+[(x.get('印',''),x.get('馬名','')) for x in r.get('印一覧',[])]
        seen=set(); review=[]
        for mark,name in entries:
            key=(rid,clean_horse(name))
            if name and name not in seen:
                seen.add(name)
                finish=lookup.get(key,'')
                if not finish and race_no is not None:
                    finish=venue_lookup.get((venue,race_no,clean_horse(name)),'')
                if finish: any_result=True
                review.append({'印':mark,'馬名':name,'着順':finish or '結果待ち'})
        r['結果一覧']=review
        main_finish=lookup.get((rid,clean_horse(r.get('本命',''))),'')
        if not main_finish and race_no is not None:
            main_finish=venue_lookup.get((venue,race_no,clean_horse(r.get('本命',''))),'')
        if main_finish:
            r['AI振り返り']=f"◎{r.get('本命')}は{main_finish}着。軸評価を実着順と照合済み。印上位の着順を見て、次回の重み調整候補として蓄積します。"
        else:
            r['AI振り返り']='このレースの確定結果はまだ保存されていません。結果取得後に自動照合します。'
    return records, any_result


def analysis_data(records):
    if not records:
        return {'total':0,'verified':0,'ranks':[],'venues':[],'bands':[]}
    df=pd.DataFrame([{'rank':str(r.get('勝負ランク','')),'venue':str(r.get('開催地','')),
                      'score':float(r.get('BET期待値',0) or 0),
                      'verified':any(str(x.get('着順','')) not in ('','結果待ち') for x in r.get('結果一覧',[]))}
                     for r in records])
    ranks=[{'label':x,'count':int((df['rank']==x).sum())} for x in ['S','A','B','C']]
    venues=[{'label':str(k),'count':int(v)} for k,v in df['venue'].value_counts().items()]
    bands=[]
    for label,lo,hi in [('～69',0,70),('70～79',70,80),('80～89',80,90),('90～',90,101)]:
        bands.append({'label':label,'count':int(((df['score']>=lo)&(df['score']<hi)).sum())})
    return {'total':len(df),'verified':int(df['verified'].sum()),'ranks':ranks,'venues':venues,'bands':bands}


def _safe_pct(num, den):
    return round(float(num)/float(den)*100,1) if den else 0.0


def verification_data(selected_date=''):
    """analysis_result.csv から結果検証ダッシュボード用データを構築。"""
    empty={
        'has_data':False,'selected_date':selected_date,
        'total_bets':0,'hit_rate':0.0,'recovery':0.0,'roi':0.0,
        'investment':0,'payout':0,'profit':0,
        'daily':[],'by_type':[],'main':{},'recovery_series':[],'cum_profit':[],
        'recent_rows':[],
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
        inv=float(frame['investment'].sum())
        pay=float(frame['payout'].sum())
        hits=int(frame['hit'].sum())
        n=len(frame)
        profit=pay-inv
        return {
            'total_bets':n,
            'hits':hits,
            'hit_rate':_safe_pct(hits,n),
            'recovery':_safe_pct(pay,inv),
            'roi':round(profit/inv*100,1) if inv else 0.0,
            'investment':int(inv),
            'payout':int(pay),
            'profit':int(profit),
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
        recovery_series.append({'date':row['date'],'value':_safe_pct(cum_pay,cum_inv)})
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
    main=pack(main_df) if not main_df.empty else {'total_bets':0,'hits':0,'hit_rate':0.0,'recovery':0.0,'roi':0.0,'investment':0,'payout':0,'profit':0}
    # 表示用テーブル
    show=day_df if not day_df.empty else all_df
    recent=[]
    for _,r in show.tail(80).iloc[::-1].iterrows():
        recent.append({
            'date':str(r.get('date','')),
            'race':str(r.get('race','')),
            'bet_type':str(r.get('bet_type','')),
            'prediction':str(r.get('prediction',''))[:60],
            'result':str(r.get('result',''))[:40],
            'hit':int(r.get('hit') or 0),
            'payout':int(r.get('payout') or 0),
            'investment':int(r.get('investment') or 0),
            'profit':int(r.get('profit') or 0),
            'roi':float(r.get('roi') or 0),
        })
    # グラフ用スケール
    max_abs=max([abs(x['value']) for x in cum_profit]+[1])
    for x in cum_profit:
        x['pct']=round(abs(x['value'])/max_abs*100,1)
        x['pos']=x['value']>=0
    max_rec=max([x['value'] for x in recovery_series]+[100])
    for x in recovery_series:
        x['pct']=round(min(x['value']/max_rec*100,100),1)

    return {
        'has_data':True,
        'selected_date':selected_date,
        'scope':'day' if selected_date and not day_df.empty else 'all',
        **summary,
        'daily':daily,
        'by_type':by_type,
        'main':main,
        'recovery_series':recovery_series,
        'cum_profit':cum_profit,
        'recent_rows':recent,
    }

@app.route('/')
def index():
    source=request.args.get('source','jra')
    mode=request.args.get('mode','predict')
    av=dates(); selected=request.args.get('date','').strip() or (av[0] if av else '')
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
                races,has_results=attach_results(races)
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
