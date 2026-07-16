from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR=Path('data'); CONFIG_FILE=DATA_DIR/'areru_v2_config.json'
DEFAULT_WEIGHTS={'performance':0.28,'upset':0.24,'consistency':0.12,'trend':0.12,'value':0.14,'context':0.10}
RECENCY=np.array([1.0,.82,.65,.48,.34])

def clean_name(x): return re.sub(r'[\s\u3000]+','',str(x)).strip()
def parse_date(s):
    s=s.astype(str).str.strip().str.replace('年','-',regex=False).str.replace('月','-',regex=False).str.replace('日','',regex=False).str.replace('/','-',regex=False)
    return pd.to_datetime(s,errors='coerce')
def num(x): return pd.to_numeric(x,errors='coerce')
def clamp(x,a=0,b=100): return float(max(a,min(b,x)))

def load_weights():
    if CONFIG_FILE.exists():
        try: return {**DEFAULT_WEIGHTS,**json.loads(CONFIG_FILE.read_text(encoding='utf-8')).get('weights',{})}
        except Exception: pass
    return DEFAULT_WEIGHTS.copy()

def weighted(vals):
    vals=np.asarray(vals,dtype=float); ok=~np.isnan(vals)
    if not ok.any(): return np.nan
    w=RECENCY[:len(vals)][ok]; return float(np.sum(vals[ok]*w)/np.sum(w))

def class_level(name):
    s=str(name)
    if 'G1' in s or 'Ｇ１' in s: return 9
    if 'G2' in s or 'Ｇ２' in s: return 8
    if 'G3' in s or 'Ｇ３' in s: return 7
    if 'オープン' in s or 'OP' in s: return 6
    if '3勝' in s: return 5
    if '2勝' in s: return 4
    if '1勝' in s: return 3
    if '新馬' in s: return 1
    if '未勝利' in s: return 2
    return 2

def context_features(history, horse, target):
    if history is None or history.empty: return {'context':50,'context_reason':[]}
    h=history[(history['_horse']==clean_name(horse)) & (history['_date']<target)].sort_values('_date',ascending=False).head(8).copy()
    if h.empty: return {'context':45,'context_reason':['履歴少']}
    finish=num(h['着順']); pop=num(h['人気']); reasons=[]; score=50
    # Same-condition aptitude inferred from the latest race profile available before target.
    latest=h.iloc[0]; dist=str(latest.get('距離','')); venue=str(latest.get('場','')); surface=dist[:1]
    same_surface=h[h['距離'].astype(str).str.startswith(surface)] if surface in ('芝','ダ') else h
    if len(same_surface)>=2:
        sf=num(same_surface['着順']); sr=(sf<=5).mean(); score += (sr-.35)*22
        if sr>=.6: reasons.append(f'{surface}適性')
    same_venue=h[h['場'].astype(str)==venue]
    if len(same_venue)>=2 and (num(same_venue['着順'])<=5).mean()>=.5: score+=7; reasons.append(f'{venue}実績')
    dnum=pd.to_numeric(pd.Series([re.sub(r'\D','',dist)]),errors='coerce').iloc[0]
    if pd.notna(dnum):
        hd=pd.to_numeric(h['距離'].astype(str).str.extract(r'(\d+)')[0],errors='coerce')
        near=h[(hd-dnum).abs()<=200]
        if len(near)>=2 and (num(near['着順'])<=5).mean()>=.5: score+=8; reasons.append('距離適性')
    if len(h)>=2:
        cls=h['レース名'].map(class_level)
        if cls.iloc[0] < cls.iloc[1]: score+=4; reasons.append('クラス条件好転')
        heads=num(h['頭数'])
        if pd.notna(heads.iloc[0]) and len(heads.dropna())>1 and heads.iloc[0] < heads.iloc[1]: score+=2; reasons.append('頭数減経験')
    if pd.notna(finish.iloc[0]) and pd.notna(pop.iloc[0]) and finish.iloc[0]+3<=pop.iloc[0]: reasons.append('前走人気以上')
    return {'context':clamp(score),'context_reason':reasons}

def score_runner(row, history, target, weights):
    finishes=np.array([num(row.get(f'着順{i}')) for i in range(1,6)],dtype=float)
    pops=np.array([num(row.get(f'人気{i}')) for i in range(1,6)],dtype=float)
    valid=~np.isnan(finishes)
    if not valid.any():
        perf=upset=cons=trend=value=35.0
    else:
        # finish quality, normalized so 1st=100, 10th~=28, 16th=0
        q=np.clip(108-finishes*8,0,100); perf=weighted(q)
        gaps=pops-finishes; upset=clamp(50+weighted(gaps)*7) if (~np.isnan(gaps)).any() else 50
        top5=(finishes<=5).astype(float)*100; cons=weighted(top5)
        fv=finishes[valid]
        trend=50 if len(fv)<2 else clamp(50+(np.mean(fv[1:])-fv[0])*6)
        pv=pops[valid]; value=clamp(50+np.nanmean(pv-fv)*5) if len(pv) else 50
    # PO-3: 当日単勝オッズ/人気があれば value 因子を市場との乖離で補正
    market_odds=num(row.get('単勝オッズ')); market_pop=num(row.get('人気'))
    market_reason=None
    if pd.notna(market_odds) and market_odds>0:
        # 高オッズほど「市場過小評価」寄り。極端な大穴は頭打ち。
        market_boost=clamp(np.log10(max(market_odds,1.0))*28, 0, 35)
        if pd.notna(market_pop) and market_pop>=8:
            market_boost=min(40, market_boost+6)
        value=clamp(0.65*value + 0.35*(50+market_boost))
        if market_odds>=12: market_reason='市場オッズ妙味'
    ctx=context_features(history,row['馬名'],target)
    factors={'performance':perf,'upset':upset,'consistency':cons,'trend':trend,'value':value,'context':ctx['context']}
    score=sum(factors[k]*weights[k] for k in weights)
    reasons=list(ctx['context_reason'])
    if upset>=65: reasons.append('人気以上に走る傾向')
    if trend>=65: reasons.append('近走上向き')
    if cons>=60: reasons.append('安定感')
    if value>=65: reasons.append('過小評価傾向')
    if market_reason: reasons.append(market_reason)
    return clamp(score),factors,reasons

VENUE_CODES={"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}
def venue_from_race_id(race_id):
    s=str(race_id)
    # netkeiba: YYYY + venue(2) + kai(2) + day(2) + race(2)
    m=re.fullmatch(r"\d{4}(\d{2})\d{6}",s)
    if m: return VENUE_CODES.get(m.group(1),"開催地不明")
    # 旧JRA URL (accessS / accessD)
    m=re.search(r"pw01[sd]de\d{2}(\d{2})\d{4}",s)
    return VENUE_CODES.get(m.group(1),"開催地不明") if m else "開催地不明"

def simulate_race(g, runs=20000):
    g=g.copy().reset_index(drop=True)
    base=g["AREru指数"].astype(float).to_numpy()
    cons=g["因子_consistency"].astype(float).to_numpy()
    sigma=np.clip(16-(cons*.08),7,17)
    seed=int(abs(hash(str(g.iloc[0]["race_id"])))%(2**32-1))
    rng=np.random.default_rng(seed)
    n_h=len(g)
    finish_counts=np.zeros((n_h,n_h),dtype=int)
    orders=[]
    for start in range(0,runs,2000):
        n=min(2000,runs-start)
        latent=rng.normal(base,sigma,size=(n,n_h))
        order=np.argsort(-latent,axis=1)
        orders.append(order)
        for pos in range(n_h):
            finish_counts[:,pos] += np.bincount(order[:,pos],minlength=n_h)
    order_all=np.vstack(orders)
    g["SIM勝率"]=finish_counts[:,0]/runs*100
    g["SIM2着内率"]=finish_counts[:,:min(2,n_h)].sum(axis=1)/runs*100
    g["SIM3着内率"]=finish_counts[:,:min(3,n_h)].sum(axis=1)/runs*100
    g["AI適正オッズ"]=np.where(g["SIM勝率"]>0,100/g["SIM勝率"],999)
    return g, order_all

def _ticket_candidates(g, orders):
    names=g["馬名"].astype(str).tolist()
    n=len(names); runs=len(orders)
    pos=np.empty_like(orders)
    rows=np.arange(runs)[:,None]
    pos[rows,orders]=np.arange(n)[None,:]
    in2=pos<2; in3=pos<3
    wide=[]; quinella=[]; trio=[]
    for i in range(n):
        for j in range(i+1,n):
            p_w=np.mean(in3[:,i] & in3[:,j])*100
            p_q=np.mean(in2[:,i] & in2[:,j])*100
            wide.append((float(p_w),(i,j)))
            quinella.append((float(p_q),(i,j)))
            for k in range(j+1,n):
                p_t=np.mean(in3[:,i] & in3[:,j] & in3[:,k])*100
                trio.append((float(p_t),(i,j,k)))
    return {
        "ワイド": sorted(wide,reverse=True),
        "馬連": sorted(quinella,reverse=True),
        "三連複": sorted(trio,reverse=True),
    }

def _ban_key(g, idxs):
    bans=[]
    for i in idxs:
        b=str(g.iloc[i].get("馬番","")).strip()
        try:
            bans.append(int(float(b)))
        except Exception:
            return None
    return "".join(f"{b:02d}" for b in sorted(bans))


def _lookup_combo_odds(kind, g, idxs, ticket_odds):
    if not ticket_odds:
        return None
    table=ticket_odds.get(kind) or {}
    key=_ban_key(g, idxs)
    if not key:
        return None
    raw=table.get(key)
    if raw in (None, ""):
        return None
    try:
        return float(str(raw).replace(",", ""))
    except Exception:
        return None


def _optimize_ticket(kind, candidates, g, max_points, ticket_odds=None):
    """組み合わせ確率の集中度で点数を圧縮。
    実オッズがある場合は合成オッズ・期待回収率も算出する。"""
    if not candidates:
        return {"買い目":[],"的中期待":0.0,"候補数":0,"圧縮理由":"候補なし",
                "合成オッズ":None,"期待回収率":None}
    best=candidates[0][0]
    floor={"ワイド":0.42,"馬連":0.34,"三連複":0.28}[kind]
    chosen=[]
    for p,idxs in candidates:
        if len(chosen)>=max_points: break
        if p < best*floor: break
        chosen.append((p,idxs))
    if not chosen:
        chosen=[candidates[0]]
    # 同一レース内で、確率の薄い点を広げ過ぎないための暫定圧縮。
    # 複数買い目全体の真の的中確率はイベント重複があるため単純加算しない。
    strength=float(np.mean([x[0] for x in chosen]))
    rows=[]; odds_vals=[]; ev_vals=[]
    for p,idxs in chosen:
        horses=[str(g.iloc[i]["馬名"]) for i in idxs]
        o=_lookup_combo_odds(kind,g,idxs,ticket_odds)
        item={"馬名":" － ".join(horses),"仮想的中率":round(float(p),1)}
        if o is not None and o>0:
            # 的中率(%)×オッズ → 期待回収率(%). 例: 20%×5.0倍=100%
            item["実オッズ"]=round(o,1)
            item["期待回収率"]=round(float(p)*o,1)
            odds_vals.append(o); ev_vals.append(float(p)*o)
        rows.append(item)
    synth=round(float(np.mean(odds_vals)),1) if odds_vals else None
    # 均等買い想定の期待回収率(%)
    ev=round(float(np.mean(ev_vals)),1) if ev_vals else None
    return {
        "買い目":rows,
        "的中期待":round(strength,1),
        "候補数":len(candidates),
        "圧縮理由":f"{len(candidates)}候補から上位{len(rows)}点へ圧縮",
        "合成オッズ":synth,
        "期待回収率":ev,
    }


ODDS_TICKETS_DIR=DATA_DIR/'odds_tickets'


def load_ticket_odds(race_id, fetch_if_missing=True):
    """data/odds_tickets/{race_id}.json を読み込む。無ければ netkeiba から取得。"""
    path=ODDS_TICKETS_DIR/f'{race_id}.json'
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            pass
    if not fetch_if_missing:
        return {}
    rid=str(race_id)
    if not (rid.isdigit() and len(rid)==12):
        return {}
    try:
        from netkeiba_client import NetkeibaClient
        maps=NetkeibaClient(sleep=0.15).fetch_ticket_odds_maps(rid)
        ODDS_TICKETS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(maps, ensure_ascii=False), encoding='utf-8')
        return maps
    except Exception:
        return {}


def build_predictions(target_str, runners, history=None, weights=None, fetch_ticket_odds=True):
    target=pd.Timestamp(target_str); weights=weights or load_weights(); r=runners.copy()
    r['_date']=parse_date(r['日付']); r=r[r['_date'].dt.normalize()==target.normalize()].copy()
    if r.empty: raise ValueError(f'{target_str} の出走データがありません')
    if history is not None:
        history=history.copy(); history['_date']=parse_date(history['年月日']); history['_horse']=history['馬名'].map(clean_name)
    scored=[]
    for _,row in r.iterrows():
        s,f,why=score_runner(row,history,target,weights)
        x=row.to_dict(); x.update({'AREru指数':round(s,2),**{f'因子_{k}':round(v,1) for k,v in f.items()},'理由':' / '.join(dict.fromkeys(why[:4])) or '総合評価'})
        scored.append(x)
    sd=pd.DataFrame(scored); out=[]
    for race_id,g0 in sd.groupby('race_id',sort=False):
        g, orders=simulate_race(g0.sort_values('AREru指数',ascending=False),20000); n=len(g)
        top=g['AREru指数'].iloc[0]; spread=top-g['AREru指数'].iloc[min(4,n-1)]; upset_share=(g['因子_upset']>=65).mean()
        chaos=clamp(35+(75-top)*.8+(18-spread)*1.3+upset_share*28)
        p1=num(g['人気1']) if '人気1' in g else pd.Series(np.nan,index=g.index)
        danger_score=pd.Series(50,index=g.index,dtype=float)+(g['AREru指数'].max()-g['AREru指数'])*.8+np.where(p1<=3,12,0)-(g['因子_upset']-50)*.35
        danger_idx=danger_score.idxmax(); danger=g.loc[danger_idx]
        hole_score=g['AREru指数']*.40+g['因子_upset']*.20+g['因子_value']*.15+g['SIM3着内率']*.25+np.where(p1>=6,8,0)
        main=g.sort_values(['SIM3着内率','AREru指数'],ascending=False).iloc[0]
        rest=g[g['馬名']!=main['馬名']].copy().assign(_hole=hole_score[g['馬名']!=main['馬名']])
        ranked=rest.sort_values(['_hole','SIM3着内率'],ascending=False).head(4)
        marks=['○','▲','△','☆']; mark_rows=[]
        for mark,(_,x) in zip(marks,ranked.iterrows()):
            mark_rows.append({'印':mark,'馬名':str(x['馬名']),'3着内率':round(float(x['SIM3着内率']),1),'理由':str(x['理由'])})
        main_place=float(main['SIM3着内率']); alt_place=float(ranked['SIM3着内率'].max()) if len(ranked) else 0
        clarity=max(0,float(main['SIM3着内率'])-float(g['SIM3着内率'].median()))
        bet=clamp(main_place*.38+alt_place*.22+clarity*.75+chaos*.16+float(danger_score.max())*.08)
        if bet>=80: bet_label='勝負'; bet_class='battle'
        elif bet>=65: bet_label='狙い目'; bet_class='target'
        elif bet>=50: bet_label='監視'; bet_class='watch'
        else: bet_label='見送り'; bet_class='skip'
        bet_reason=[]
        if main_place>=45: bet_reason.append('軸候補の3着内率が高い')
        if alt_place>=25: bet_reason.append('相手候補に上位進出余地')
        if chaos>=60: bet_reason.append('波乱シグナル')
        if clarity<8: bet_reason.append('軸が絞りにくい')
        if bet<50: bet_reason.append('買い条件不足')
        judge='大荒れ警戒' if chaos>=80 else ('波乱' if chaos>=60 else ('注意' if chaos>=40 else '平穏'))
        names={x['印']:x['馬名'] for x in mark_rows}; main_name=str(main['馬名'])

        # 券種別実オッズ（キャッシュ優先）。単勝が無いレースは取得を省略。
        main_win=num(main.get('単勝オッズ'))
        has_win=pd.notna(main_win) and float(main_win)>0
        ticket_odds=load_ticket_odds(race_id, fetch_if_missing=fetch_ticket_odds and has_win) if has_win else {}

        # 20,000回の全着順から、券種別の組み合わせ確率を直接集計。
        candidates=_ticket_candidates(g,orders)
        wide_plan=_optimize_ticket("ワイド",candidates["ワイド"],g,3,ticket_odds)
        quinella_plan=_optimize_ticket("馬連",candidates["馬連"],g,2,ticket_odds)
        trio_plan=_optimize_ticket("三連複",candidates["三連複"],g,6,ticket_odds)

        wide_score=clamp(wide_plan["的中期待"]*2.0 + main_place*.45)
        quinella_score=clamp(quinella_plan["的中期待"]*3.0 + float(main["SIM勝率"])*.8)
        trio_score=clamp(trio_plan["的中期待"]*5.0 + chaos*.25)

        def go_label(v):
            return '買い候補' if v>=70 else ('条件付き' if v>=55 else '見送り')

        plans=[
            ("ワイド",wide_score,wide_plan),
            ("馬連",quinella_score,quinella_plan),
            ("三連複",trio_score,trio_plan),
        ]
        plans_sorted=sorted(plans,key=lambda x:x[1],reverse=True)
        best_kind=plans_sorted[0][0]
        best_score=plans_sorted[0][1]
        best_plan=plans_sorted[0][2]

        # その日の全レース内でのS/A/B/Cは後段で相対順位化するため、ここでは基礎値を保存。
        ticket_reason=(
            f"{best_kind}型。20,000回の仮想レースで組み合わせ同時好走率を比較し、"
            f"{best_plan['圧縮理由']}。"
        )
        if best_plan.get('期待回収率') is not None:
            ticket_reason += f" 実オッズ接続済・期待回収率 {best_plan['期待回収率']}%。"

        def plan_text(plan):
            parts=[]
            for x in plan["買い目"]:
                s=f"{x['馬名']}（仮想的中 {x['仮想的中率']}%"
                if x.get('実オッズ') is not None:
                    s+=f" / 実{x['実オッズ']}倍"
                if x.get('期待回収率') is not None:
                    s+=f" / EV{x['期待回収率']}%"
                s+="）"
                parts.append(s)
            return '｜'.join(parts) if parts else '見送り'

        main_odds=num(main.get('単勝オッズ'))
        main_odds_disp=round(float(main_odds),1) if pd.notna(main_odds) else ''
        main_pop=num(main.get('人気'))
        main_pop_disp=int(float(main_pop)) if pd.notna(main_pop) else ''
        synth=best_plan.get('合成オッズ')
        ev=best_plan.get('期待回収率')
        synth_disp=f"{synth}倍" if synth is not None else '券種別オッズ待ち'
        ev_disp=f"{ev}%" if ev is not None else 'オッズ接続後に算出'

        out.append({
          'race_id':race_id,'開催地':venue_from_race_id(race_id),'レース':int(float(main['レース'])),'荒れ度':round(chaos,1),'判定':judge,
          '荒れクラス':'storm' if chaos>=80 else ('wave' if chaos>=60 else ('caution' if chaos>=40 else 'calm')),
          'BET期待値':round(bet,1),'BET判定':bet_label,'BETクラス':bet_class,'BET理由':' / '.join(bet_reason),
          'シミュレーション回数':20000,'本命':main_name,'本命AREru指数':main['AREru指数'],'シミュレーション勝率':round(main['SIM勝率'],1),'シミュレーション3着内率':round(main['SIM3着内率'],1),'AI適正オッズ':round(main['AI適正オッズ'],1),'本命理由':main['理由'],
          '本命オッズ':main_odds_disp,'本命人気':main_pop_disp,
          '人気馬危険':danger['馬名'],'危険度':round(clamp(danger_score.loc[danger_idx]),1),'危険理由':'近走評価と人気履歴のズレを検出',
          '印データ':json.dumps(mark_rows,ensure_ascii=False),
          '推奨券種':best_kind,'馬券戦略理由':ticket_reason,
          'ワイド評価':round(wide_score,1),'ワイド判定':go_label(wide_score),'ワイド買い目':plan_text(wide_plan),'ワイド圧縮':wide_plan['圧縮理由'],
          '馬連評価':round(quinella_score,1),'馬連判定':go_label(quinella_score),'馬連買い目':plan_text(quinella_plan),'馬連圧縮':quinella_plan['圧縮理由'],
          '三連複評価':round(trio_score,1),'三連複判定':go_label(trio_score),'三連複買い目':plan_text(trio_plan),'三連複圧縮':trio_plan['圧縮理由'],
          '合成オッズ':synth_disp,'期待回収率':ev_disp,
          'データ頭数':n})

    result=pd.DataFrame(out).sort_values(['開催地','レース']).reset_index(drop=True)

    # 開催日内の相対順位でS/A/B/Cを付与。
    # 生の基礎値は上限100に張り付きやすいため、表示用の買い期待度は
    # 順位パーセンタイル + 基礎値の差を使って再スケールする。
    raw_bet=result['BET期待値'].astype(float).copy()
    order=raw_bet.rank(method='first',ascending=False).astype(int)
    total=len(result)
    pct=(total-order)/(max(total-1,1))
    raw_min=float(raw_bet.min()); raw_max=float(raw_bet.max())
    raw_norm=(raw_bet-raw_min)/(raw_max-raw_min) if raw_max>raw_min else pd.Series(0.5,index=result.index)
    display_score=(38 + pct*52 + raw_norm*8).clip(0,98.7)
    result['買い期待度基礎値']=raw_bet.round(1)
    result['BET期待値']=display_score.round(1)
    def grade(rank):
        if rank<=min(2,total): return 'S'
        if rank<=min(5,total): return 'A'
        if rank<=max(8,int(total*.35)): return 'B'
        return 'C'
    result['勝負ランク']=order.map(grade)
    result['BET判定']=result['勝負ランク'].map({'S':'今日の勝負','A':'買い候補','B':'オッズ次第','C':'見送り'})
    result['BETクラス']=result['勝負ランク'].map({'S':'battle','A':'target','B':'watch','C':'skip'})
    return result,sd

