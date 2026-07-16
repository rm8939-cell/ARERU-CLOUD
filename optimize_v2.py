from pathlib import Path
import json, random
import numpy as np, pandas as pd
from areru_engine import build_predictions, parse_date, DEFAULT_WEIGHTS
DATA=Path('data'); random.seed(42)
runners_path=DATA/'runners.csv' if (DATA/'runners.csv').exists() else DATA/'score_test_data.csv'
r=pd.read_csv(runners_path); h=pd.read_csv(DATA/'all_history.csv')
dates=sorted(parse_date(r['日付']).dropna().dt.strftime('%Y-%m-%d').unique())
if len(dates)<2: raise SystemExit('検証できる開催日が不足')
train=dates[:-1]; holdout=dates[-1:]
keys=list(DEFAULT_WEIGHTS)
def norm(v):
 s=sum(v); return {k:x/s for k,x in zip(keys,v)}
def metric(weights,ds):
 vals=[]
 for d in ds:
  _,scores=build_predictions(d,r,h,weights)
  for _,g in scores.groupby('race_id'):
   top=g.sort_values('AREru指数',ascending=False).iloc[0]
   vals.append(float(pd.to_numeric(top.get('実着順'),errors='coerce')<=3))
 return np.mean(vals) if vals else 0
cands=[DEFAULT_WEIGHTS]
for _ in range(24): cands.append(norm([random.uniform(.05,.45) for _ in keys]))
rows=[]
for i,w in enumerate(cands): rows.append({'id':i,**w,'train_top3':metric(w,train)})
res=pd.DataFrame(rows).sort_values('train_top3',ascending=False)
best={k:float(res.iloc[0][k]) for k in keys}; hold=metric(best,holdout)
prev={}
cfg_path=DATA/'areru_v2_config.json'
if cfg_path.exists():
 try: prev=json.loads(cfg_path.read_text(encoding='utf-8'))
 except Exception: prev={}
config={'version':'2.0','weights':best,'train_dates':train,'holdout_dates':holdout,'train_top3':float(res.iloc[0]['train_top3']),'holdout_top3':float(hold),'note':'実着順は評価専用。指数入力には未使用。'}
if prev.get('rank_thresholds'):
 config['rank_thresholds']=prev['rank_thresholds']
cfg_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
res.to_csv(DATA/'areru_v2_optimizer.csv',index=False,encoding='utf-8-sig')
print('✅ 最適化完了',json.dumps(config,ensure_ascii=False,indent=2))
