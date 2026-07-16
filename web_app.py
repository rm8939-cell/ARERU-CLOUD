from flask import Flask,render_template,request,jsonify
import subprocess,sys,json,re
from pathlib import Path
from datetime import date
import os
import pandas as pd
from areru_engine import parse_date

app=Flask(__name__)
BASE=Path(__file__).resolve().parent
DATA=BASE/'data'; ARCH=DATA/'predictions_by_date'; ARCH.mkdir(parents=True,exist_ok=True)

def dates():
    p=DATA/'score_test_data.csv'
    if not p.exists(): return []
    d=parse_date(pd.read_csv(p,usecols=['日付'])['日付']).dropna().dt.strftime('%Y-%m-%d').unique().tolist()
    # predictions_by_date 側にだけある日も候補に含める
    for f in ARCH.glob('predictions_*.csv'):
        ds=f.stem.replace('predictions_','')
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', ds) and ds not in d:
            d.append(ds)
    return sorted(d,reverse=True)

def latest_meta():
    p=DATA/'refresh_meta.json'
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}

def ensure(d):
    f=ARCH/f'predictions_{d}.csv'; regen=True
    if f.exists():
        try: regen='印データ' not in pd.read_csv(f,nrows=1).columns
        except: regen=True
    if regen: subprocess.run([sys.executable,'replay_predict.py',d],check=True,timeout=240,cwd=str(BASE))
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
    lookup={(str(x['race_id']),clean_horse(x['馬名'])):str(x['着順']) for _,x in rdf.iterrows()}
    any_result=False
    for r in records:
        rid=str(r['race_id'])
        entries=[('◎',r.get('本命',''))]+[(x.get('印',''),x.get('馬名','')) for x in r.get('印一覧',[])]
        seen=set(); review=[]
        for mark,name in entries:
            key=(rid,clean_horse(name))
            if name and name not in seen:
                seen.add(name)
                finish=lookup.get(key,'')
                if finish: any_result=True
                review.append({'印':mark,'馬名':name,'着順':finish or '結果待ち'})
        r['結果一覧']=review
        main_finish=lookup.get((rid,clean_horse(r.get('本命',''))),'')
        if main_finish:
            r['AI振り返り']=f"◎{r.get('本命')}は{main_finish}着。軸評価を実着順と照合済み。印上位の着順を見て、次回の重み調整候補として蓄積します。"
        else:
            r['AI振り返り']='このレースの確定結果はまだ保存されていません。結果取得後に自動照合します。'
    return records, any_result


def analysis_data(records, selected_date=''):
    if not records:
        base={'total':0,'verified':0,'ranks':[],'venues':[],'bands':[],'roi':None}
    else:
        df=pd.DataFrame([{'rank':str(r.get('勝負ランク','')),'venue':str(r.get('開催地','')),
                          'score':float(r.get('BET期待値',0) or 0),
                          'verified':any(str(x.get('着順','')) not in ('','結果待ち') for x in r.get('結果一覧',[]))}
                         for r in records])
        ranks=[{'label':x,'count':int((df['rank']==x).sum())} for x in ['S','A','B','C']]
        venues=[{'label':str(k),'count':int(v)} for k,v in df['venue'].value_counts().items()]
        bands=[]
        for label,lo,hi in [('～69',0,70),('70～79',70,80),('80～89',80,90),('90～',90,101)]:
            bands.append({'label':label,'count':int(((df['score']>=lo)&(df['score']<hi)).sum())})
        base={'total':len(df),'verified':int(df['verified'].sum()),'ranks':ranks,'venues':venues,'bands':bands,'roi':None}
    # 払戻ベース実回収率（取得済み日だけ）
    if selected_date:
        try:
            from roi_analyzer import analyze_date
            roi=analyze_date(selected_date)
            if roi.get('evaluated'):
                base['roi']=roi
        except Exception:
            pass
    return base

@app.route('/health')
def health():
    av=dates(); meta=latest_meta()
    return jsonify({
        'status':'ok',
        'today':date.today().isoformat(),
        'available_dates':av,
        'latest_refresh':meta,
        'version':'ARERU.CLOUD β',
    })

@app.route('/')
def index():
    source=request.args.get('source','jra')
    mode=request.args.get('mode','predict')
    av=dates(); selected=request.args.get('date','').strip() or (av[0] if av else '')
    races=[]; targets=[]; message='予想データがありません'; has_results=False
    meta=latest_meta()

    if source=='nar':
        message='地方競馬エンジンは接続準備中です。JRA予想ロジックとは分離して実装します。'
        return render_template('index.html',races=[],targets=[],selected_date=selected,today=date.today().isoformat(),
            message=message,available_dates=av,source=source,mode=mode,has_results=False,analysis=analysis_data([],selected),refresh_meta=meta)

    if selected in av:
        try:
            df=pd.read_csv(ensure(selected)).fillna('なし')
            races=prep(df.to_dict('records'))
            races,has_results=attach_results(races)
            targets=sorted([r for r in races if r.get('勝負ランク') in ['S','A']],key=lambda x:float(x.get('BET期待値',0)),reverse=True)[:5]
            mode_label='結果検証モード' if mode=='result' else ('分析ダッシュボード' if mode=='analysis' else 'AI仮想レース分析 β版')
            refreshed=meta.get('updated_at','')
            message=f'{selected} / {mode_label}' + (f' / データ更新 {refreshed}' if refreshed else '')
        except Exception as e: message=f'生成エラー: {e}'
    elif selected: message=f'{selected} は保存データにありません'
    return render_template('index.html',races=races,targets=targets,selected_date=selected,today=date.today().isoformat(),
        message=message,available_dates=av,source=source,mode=mode,has_results=has_results,analysis=analysis_data(races,selected),refresh_meta=meta)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT','5001')),debug=False)
