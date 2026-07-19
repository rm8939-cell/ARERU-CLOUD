from __future__ import annotations
import json, logging, re
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR=Path('data'); CONFIG_FILE=DATA_DIR/'areru_v2_config.json'
HORSE_CACHE_DIR=DATA_DIR/'cache'/'horse_results'
DEFAULT_WEIGHTS={'performance':0.28,'upset':0.24,'consistency':0.12,'trend':0.12,'value':0.14,'context':0.10}
RECENCY=np.array([1.0,.82,.65,.48,.34])
log=logging.getLogger('areru')
# 地方→中央転入時の着順品質スケール（1.0=中央と同格）
NAR_TO_JRA_SCALE_DEFAULT=0.55
NAR_TO_JRA_SCALE_STAKES=0.62  # 地方重賞 / Jpn
NAR_TO_JRA_SCALE_A=0.58
NAR_TO_JRA_SCALE_C=0.45
EV_EXTREME_PCT=1000

# AIランク選別（PO-5/PO-6）。ラベルと閾値は将来ROI学習で自動調整可能な構造。
RANK_LABELS={'S':'S','A':'A','B':'B','C':'C','D':'D'}
RANK_CLASSES={'S':'battle','A':'target','B':'watch','C':'caution','D':'skip'}
DEFAULT_RANK_THRESHOLDS={
    'mode':'ev',  # ev | percentile | absolute | hybrid
    's_ev_min':120,
    'a_ev_min':115,
    'b_ev_min':110,
    'c_ev_min':105,
    'd_ev_min':100,
    's_top_n':2,
    'a_top_n':5,
    'b_percentile':0.35,
    'b_min_n':8,
    'c_percentile':0.70,
    'c_min_n':12,
    's_min_score':0.0,
    'a_min_score':0.0,
    'b_min_score':0.0,
    'c_min_score':0.0,
}

def clean_name(x): return re.sub(r'[\s\u3000]+','',str(x)).strip()
def parse_date(s):
    s=s.astype(str).str.strip().str.replace('年','-',regex=False).str.replace('月','-',regex=False).str.replace('日','',regex=False).str.replace('/','-',regex=False)
    return pd.to_datetime(s,errors='coerce')
def num(x): return pd.to_numeric(x,errors='coerce')
def clamp(x,a=0,b=100): return float(max(a,min(b,x)))

def _load_config():
    if CONFIG_FILE.exists():
        try: return json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
        except Exception: pass
    return {}

def load_weights():
    cfg=_load_config()
    return {**DEFAULT_WEIGHTS,**cfg.get('weights',{})} if cfg else DEFAULT_WEIGHTS.copy()

def load_rank_thresholds():
    """ランク閾値をconfigから読み込む。未設定時はDEFAULTを返す。"""
    cfg=_load_config()
    raw=cfg.get('rank_thresholds') or {}
    out={**DEFAULT_RANK_THRESHOLDS,**raw}
    for k in ('s_top_n','a_top_n','b_min_n','c_min_n'):
        try: out[k]=int(out[k])
        except (TypeError,ValueError): out[k]=DEFAULT_RANK_THRESHOLDS[k]
    for k in ('b_percentile','c_percentile','s_min_score','a_min_score','b_min_score','c_min_score'):
        try: out[k]=float(out[k])
        except (TypeError,ValueError): out[k]=DEFAULT_RANK_THRESHOLDS[k]
    mode=str(out.get('mode') or 'percentile').lower()
    out['mode']=mode if mode in ('percentile','absolute','hybrid') else 'percentile'
    return out

def rank_label(rank):
    return RANK_LABELS.get(str(rank).upper(),'')

def assign_ai_ranks(result, thresholds=None):
    """開催日内の相対順位（または絶対スコア）で S/A/B/C を付与。

    thresholds は load_rank_thresholds() 形式。将来 rank_optimizer が
    各ランクの実ROIを見て s_top_n / a_top_n / *_min_score を自動更新する。
    """
    if result is None or len(result)==0:
        return result
    th=thresholds or load_rank_thresholds()
    raw_bet=result['BET期待値'].astype(float).copy()
    order=raw_bet.rank(method='first',ascending=False).astype(int)
    total=len(result)
    pct=(total-order)/(max(total-1,1))
    raw_min=float(raw_bet.min()); raw_max=float(raw_bet.max())
    raw_norm=(raw_bet-raw_min)/(raw_max-raw_min) if raw_max>raw_min else pd.Series(0.5,index=result.index)
    display_score=(38 + pct*52 + raw_norm*8).clip(0,98.7)
    result=result.copy()
    result['買い期待度基礎値']=raw_bet.round(1)
    result['BET期待値']=display_score.round(1)

    s_n=max(1,min(int(th['s_top_n']),total))
    a_n=max(s_n,min(int(th['a_top_n']),total))
    b_cut=max(int(th['b_min_n']),int(total*float(th['b_percentile'])))
    b_n=max(a_n,min(b_cut,total))
    c_cut=max(int(th.get('c_min_n') or 12),int(total*float(th.get('c_percentile') or 0.70)))
    c_n=max(b_n,min(c_cut,total))
    mode=th.get('mode','percentile')
    s_min=float(th.get('s_min_score') or 0)
    a_min=float(th.get('a_min_score') or 0)
    b_min=float(th.get('b_min_score') or 0)
    c_min=float(th.get('c_min_score') or 0)

    def grade(rank_i, score):
        # absolute: 基礎値のみ / hybrid: 相対順位かつ基礎値下限 / percentile: 相対のみ
        # 5段階: S / A / B / C / D
        if mode=='absolute':
            if score>=s_min and s_min>0: return 'S'
            if score>=a_min and a_min>0: return 'A'
            if score>=b_min and b_min>0: return 'B'
            if score>=c_min and c_min>0: return 'C'
            if s_min<=0 and a_min<=0 and b_min<=0 and c_min<=0:
                pass
            else:
                return 'D'
        if rank_i<=s_n:
            if mode=='hybrid' and s_min>0 and score<s_min: return 'A' if rank_i<=a_n else 'B'
            return 'S'
        if rank_i<=a_n:
            if mode=='hybrid' and a_min>0 and score<a_min: return 'B'
            return 'A'
        if rank_i<=b_n:
            if mode=='hybrid' and b_min>0 and score<b_min: return 'C'
            return 'B'
        if rank_i<=c_n:
            if mode=='hybrid' and c_min>0 and score<c_min: return 'D'
            return 'C'
        return 'D'

    ranks=[grade(int(order.loc[i]), float(raw_bet.loc[i])) for i in result.index]
    result['勝負ランク']=ranks
    result['BET判定']=result['勝負ランク'].map(RANK_LABELS)
    result['BETクラス']=result['勝負ランク'].map(RANK_CLASSES)
    return result

def weighted(vals):
    vals=np.asarray(vals,dtype=float); ok=~np.isnan(vals)
    if not ok.any(): return np.nan
    w=RECENCY[:len(vals)][ok]; return float(np.sum(vals[ok]*w)/np.sum(w))

JRA_VENUE_CODES={"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}
NAR_VENUE_CODES={
    "30":"門別","35":"盛岡","36":"水沢",
    "42":"浦和","43":"船橋","44":"大井","45":"川崎",
    "46":"金沢","47":"笠松","48":"名古屋",
    "50":"園田","51":"姫路","54":"高知","55":"佐賀","65":"帯広",
}
VENUE_CODES={**JRA_VENUE_CODES,**NAR_VENUE_CODES}
NAR_VENUE_NAMES=set(NAR_VENUE_CODES.values())
JRA_VENUE_NAMES=set(JRA_VENUE_CODES.values())


def is_nar_venue(venue) -> bool:
    """開催場文字列が地方か。'3京都9' のような中央表記は False。"""
    s=str(venue or '').strip()
    if not s:
        return False
    if s in NAR_VENUE_NAMES:
        return True
    if s in JRA_VENUE_NAMES:
        return False
    # 中央: 「2阪神7」「1小倉4」など
    if re.match(r'^\d+[札幌函館福島新潟東京中山中京京都阪神小倉]', s):
        return False
    for name in JRA_VENUE_NAMES:
        if name in s:
            return False
    for name in NAR_VENUE_NAMES:
        if name in s:
            return True
    return False


def class_level(name, venue=None):
    """レース格付け。地方重賞は中央G3相当にしない。"""
    s=str(name or '')
    nar=is_nar_venue(venue) if venue is not None else False
    # Jpn / ダートグレード（地方開催でも中央馬混合）
    if re.search(r'Jpn\s*I\b|Jpn1|JPN1|Ｊｐｎ１', s, re.I): return 7 if nar else 9
    if re.search(r'Jpn\s*II\b|Jpn2|JPN2|Ｊｐｎ２', s, re.I): return 6 if nar else 8
    if re.search(r'Jpn\s*III\b|Jpn3|JPN3|Ｊｐｎ３', s, re.I): return 5 if nar else 7
    if 'G1' in s or 'Ｇ１' in s: return 9
    if 'G2' in s or 'Ｇ２' in s: return 8
    if 'G3' in s or 'Ｇ３' in s: return 7
    if '重賞' in s or '準重賞' in s:
        return 5 if nar else 7
    if 'オープン' in s or re.search(r'\bOP\b', s) or '特別' in s:
        return 4 if nar else 6
    if '3勝' in s: return 5
    if '2勝' in s: return 4
    if '1勝' in s: return 3
    # 地方クラス表記（A1 > B1 > C1 / 〇〇組）
    if re.search(r'A[1-4]', s, re.I) or 'A級' in s: return 4
    if re.search(r'B[1-4]', s, re.I) or 'B級' in s: return 3
    if re.search(r'C[1-4]', s, re.I) or 'C級' in s: return 2
    if re.search(r'[一二三四五六七八九十]+組', s): return 2
    if '新馬' in s: return 1
    if '未勝利' in s or '未受賞' in s: return 2
    return 2


def nar_to_jra_scale(venue, race_name) -> float:
    """地方過去走を中央レース評価へ落とす倍率。"""
    if not is_nar_venue(venue):
        return 1.0
    s=str(race_name or '')
    lvl=class_level(s, venue)
    if re.search(r'Jpn|重賞', s, re.I) or lvl >= 5:
        return NAR_TO_JRA_SCALE_STAKES
    if lvl >= 4 or re.search(r'A[1-4]|A級', s, re.I):
        return NAR_TO_JRA_SCALE_A
    if lvl <= 2 or re.search(r'C[1-4]|C級|組', s, re.I):
        return NAR_TO_JRA_SCALE_C
    return NAR_TO_JRA_SCALE_DEFAULT


def _past_meta_for_row(row, history, target):
    """直近5走の (場, レース名) 。runners列 → history → 馬キャッシュの順。"""
    metas=[]
    for i in range(1,6):
        venue=str(row.get(f'場{i}') or '').strip()
        rname=str(row.get(f'レース名{i}') or '').strip()
        metas.append((venue, rname))
    if any(v for v,_ in metas):
        return metas
    if history is not None and not getattr(history, 'empty', True):
        try:
            h=history[(history['_horse']==clean_name(row.get('馬名'))) & (history['_date']<target)]
            h=h.sort_values('_date', ascending=False).head(5)
            out=[]
            for _,x in h.iterrows():
                out.append((str(x.get('場') or '').strip(), str(x.get('レース名') or '').strip()))
            while len(out)<5:
                out.append(('',''))
            if any(v for v,_ in out):
                return out
        except Exception:
            pass
    return metas


_CACHE_FINISH_INDEX=None

def _horse_cache_finish_index():
    """着順・人気パターン → 場/レース名 の簡易索引（既存runners補完用）。"""
    global _CACHE_FINISH_INDEX
    if _CACHE_FINISH_INDEX is not None:
        return _CACHE_FINISH_INDEX
    idx={}
    if not HORSE_CACHE_DIR.exists():
        _CACHE_FINISH_INDEX=idx
        return idx
    for path in HORSE_CACHE_DIR.glob('*.json'):
        try:
            hist=json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(hist, list) or not hist:
            continue
        fins=[]; pops=[]; venues=[]; names=[]
        for h in hist[:5]:
            f=_safe_int(h.get('着順')); p=_safe_int(h.get('人気'))
            fins.append(f); pops.append(p)
            venues.append(str(h.get('場') or '').strip())
            names.append(str(h.get('レース名') or '').strip())
        key=(tuple(fins), tuple(pops))
        if any(fins) and key not in idx:
            idx[key]=list(zip(venues, names))
    _CACHE_FINISH_INDEX=idx
    return idx


def _safe_int(v):
    s=str(v or '').strip()
    m=re.search(r'\d+', s)
    return int(m.group(0)) if m else None


def enrich_runners_past_venues(runners: pd.DataFrame) -> pd.DataFrame:
    """場1..5 が空の runners を馬キャッシュから補完。"""
    df=runners.copy()
    for i in range(1,6):
        if f'場{i}' not in df.columns:
            df[f'場{i}']=''
        if f'レース名{i}' not in df.columns:
            df[f'レース名{i}']=''
    need=df['場1'].astype(str).str.strip().eq('') & df['着順1'].astype(str).str.strip().ne('')
    if not need.any():
        return df
    index=_horse_cache_finish_index()
    if not index:
        return df
    filled=0
    for i in df.index[need]:
        fins=tuple(_safe_int(df.at[i, f'着順{k}']) for k in range(1,6))
        pops=tuple(_safe_int(df.at[i, f'人気{k}']) for k in range(1,6))
        meta=index.get((fins, pops))
        if not meta:
            continue
        for k,(venue, rname) in enumerate(meta, 1):
            df.at[i, f'場{k}']=venue
            df.at[i, f'レース名{k}']=rname
        filled+=1
    if filled:
        log.info('past venue backfill: %s horses from cache', filled)
    return df


def context_features(history, horse, target, target_source='jra'):
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
    # 中央対象日に地方場実績を適性加点しない
    if not (str(target_source)=='jra' and is_nar_venue(venue)):
        same_venue=h[h['場'].astype(str)==venue]
        if len(same_venue)>=2 and (num(same_venue['着順'])<=5).mean()>=.5:
            score+=7; reasons.append(f'{venue}実績')
    dnum=pd.to_numeric(pd.Series([re.sub(r'\D','',dist)]),errors='coerce').iloc[0]
    if pd.notna(dnum):
        hd=pd.to_numeric(h['距離'].astype(str).str.extract(r'(\d+)')[0],errors='coerce')
        near=h[(hd-dnum).abs()<=200]
        if len(near)>=2 and (num(near['着順'])<=5).mean()>=.5: score+=8; reasons.append('距離適性')
    if len(h)>=2:
        cls=h.apply(lambda r: class_level(r.get('レース名'), r.get('場')), axis=1)
        if cls.iloc[0] < cls.iloc[1]: score+=4; reasons.append('クラス条件好転')
        heads=num(h['頭数'])
        if pd.notna(heads.iloc[0]) and len(heads.dropna())>1 and heads.iloc[0] < heads.iloc[1]: score+=2; reasons.append('頭数減経験')
    # 中央戦で直近が地方のみなら文脈を抑制
    if str(target_source)=='jra':
        nar_share=sum(1 for v in h['場'].astype(str) if is_nar_venue(v))/max(len(h),1)
        if nar_share>=0.6:
            score-=10; reasons.append('地方実績中心')
    if pd.notna(finish.iloc[0]) and pd.notna(pop.iloc[0]) and finish.iloc[0]+3<=pop.iloc[0]: reasons.append('前走人気以上')
    return {'context':clamp(score),'context_reason':reasons}

def score_runner(row, history, target, weights):
    finishes=np.array([num(row.get(f'着順{i}')) for i in range(1,6)],dtype=float)
    pops=np.array([num(row.get(f'人気{i}')) for i in range(1,6)],dtype=float)
    target_source=str(row.get('source') or source_from_race_id(row.get('race_id',''))).lower()
    metas=_past_meta_for_row(row, history, target)
    # キャッシュ補完（場列が空で finish パターン一致時）
    if target_source=='jra' and not any(v for v,_ in metas):
        fins=tuple(_safe_int(row.get(f'着順{i}')) for i in range(1,6))
        pp=tuple(_safe_int(row.get(f'人気{i}')) for i in range(1,6))
        cached=_horse_cache_finish_index().get((fins, pp))
        if cached:
            metas=list(cached)
            while len(metas)<5:
                metas.append(('',''))
    scales=np.ones(5, dtype=float)
    nar_n=0
    if target_source=='jra':
        for i in range(5):
            venue, rname = metas[i] if i < len(metas) else ('','')
            sc=nar_to_jra_scale(venue, rname)
            scales[i]=sc
            if sc < 1.0:
                nar_n+=1
    valid=~np.isnan(finishes)
    n_valid=int(valid.sum())
    transfer_reason=None
    if not valid.any():
        perf=upset=cons=trend=value=35.0
    else:
        # finish quality, normalized so 1st=100, 10th~=28, 16th=0
        q=np.clip(108-finishes*8,0,100)*scales
        perf=weighted(q)
        gaps=(pops-finishes)*scales
        upset=clamp(50+weighted(gaps)*7) if (~np.isnan(gaps)).any() else 50
        top5=(finishes<=5).astype(float)*100*scales
        cons=weighted(top5)
        fv=finishes[valid]
        trend=50 if len(fv)<2 else clamp(50+(np.mean(fv[1:])-fv[0])*6)
        # 地方走が多い場合、トレンドも控えめ
        if nar_n>=3 and target_source=='jra':
            trend=clamp(40+0.5*(trend-40))
        pv=pops[valid]; value=clamp(50+np.nanmean(pv-fv)*5) if len(pv) else 50
        if nar_n>=2 and target_source=='jra':
            transfer_reason='地方実績を中央換算'
            # 地方走が多いほど performance/consistency をさらに抑制
            damp=0.85 if nar_n>=4 else 0.92
            perf=clamp(perf*damp); cons=clamp(cons*damp)
    # サンプル少: 事前分布へ強く縮小（1走だけの本命化を防ぐ）
    if n_valid == 1:
        prior, shrink = 38.0, 0.22
    elif n_valid == 2:
        prior, shrink = 40.0, 0.55
    else:
        prior, shrink = None, None
    if prior is not None:
        perf=prior+(perf-prior)*shrink
        upset=prior+(upset-prior)*shrink
        cons=prior+(cons-prior)*shrink
        trend=prior+(trend-prior)*shrink
        value=prior+(value-prior)*shrink
    # PO-3: 当日単勝オッズ/人気があれば value 因子を市場との乖離で補正
    # ※極端な大穴・サンプル不足では「妙味」加点を抑える
    market_odds=num(row.get('単勝オッズ')); market_pop=num(row.get('人気'))
    market_reason=None
    if pd.notna(market_odds) and market_odds>0 and n_valid >= 2:
        if market_odds>=50 or (pd.notna(market_pop) and market_pop>=12):
            market_boost=clamp(np.log10(max(market_odds,1.0))*8, 0, 10)
        elif market_odds>=20:
            market_boost=clamp(np.log10(max(market_odds,1.0))*16, 0, 18)
        else:
            market_boost=clamp(np.log10(max(market_odds,1.0))*28, 0, 35)
            if pd.notna(market_pop) and market_pop>=8:
                market_boost=min(40, market_boost+6)
        value=clamp(0.65*value + 0.35*(50+market_boost))
        if 12<=market_odds<50 and not (pd.notna(market_pop) and market_pop>=12):
            market_reason='市場オッズ妙味'
    elif n_valid < 2:
        value=clamp(min(value, 42.0))
    ctx=context_features(history,row['馬名'],target, target_source=target_source)
    factors={'performance':perf,'upset':upset,'consistency':cons,'trend':trend,'value':value,'context':ctx['context']}
    score=sum(factors[k]*weights[k] for k in weights)
    reasons=list(ctx['context_reason'])
    if transfer_reason: reasons.insert(0, transfer_reason)
    if n_valid and n_valid<3: reasons.append('サンプル少')
    if upset>=65: reasons.append('人気以上に走る傾向')
    if trend>=65: reasons.append('近走上向き')
    if cons>=60: reasons.append('安定感')
    if value>=65: reasons.append('過小評価傾向')
    if market_reason: reasons.append(market_reason)
    return clamp(score),factors,reasons

def source_from_race_id(race_id):
    s=str(race_id)
    m=re.fullmatch(r"\d{4}(\d{2})\d{6}",s)
    if not m: return "jra"
    code=m.group(1)
    if code in NAR_VENUE_CODES: return "nar"
    if code in JRA_VENUE_CODES: return "jra"
    try:
        return "nar" if int(code)>=30 else "jra"
    except Exception:
        return "jra"

def venue_from_race_id(race_id):
    s=str(race_id)
    # netkeiba: YYYY + venue(2) + ... + race(2)  ※JRAは回次・日次、NARはMMDD
    m=re.fullmatch(r"\d{4}(\d{2})\d{6}",s)
    if m:
        name=VENUE_CODES.get(m.group(1),"開催地不明")
        try:
            from netkeiba_client import normalize_venue_name
            return normalize_venue_name(name)
        except Exception:
            return name
    # 旧JRA URL (accessS / accessD)
    m=re.search(r"pw01[sd]de\d{2}(\d{2})\d{4}",s)
    if not m:
        return "開催地不明"
    name=VENUE_CODES.get(m.group(1),"開催地不明")
    try:
        from netkeiba_client import normalize_venue_name
        return normalize_venue_name(name)
    except Exception:
        return name

# 仮想勝率の上限（99%以上は非現実的）と適正オッズ下限（1.0倍固定を防ぐ）
SIM_WIN_MAX_PCT = 98.0
AI_FAIR_ODDS_MIN = 1.1


def _cap_sim_win_rates(win_pct, max_win=SIM_WIN_MAX_PCT, min_fair=AI_FAIR_ODDS_MIN):
    """勝率を上限でクリップし、余りを他馬へ再配分。適正オッズ下限とも整合させる。"""
    # 1.1倍 ⇔ 約90.91%。1桁表示でも 1.0 に丸まらないよう両方で抑える
    cap = float(min(max_win, 100.0 / min_fair))
    win = np.asarray(win_pct, dtype=float).copy()
    if win.size == 0:
        return win
    excess = float(np.maximum(win - cap, 0).sum())
    win = np.minimum(win, cap)
    if excess > 1e-9:
        room = np.maximum(cap - win, 0)
        total_room = float(room.sum())
        if total_room > 1e-9:
            win = win + excess * (room / total_room)
            win = np.minimum(win, cap)
    return win


def simulate_race(g, runs=None, profiles=None, pace=None):
    """段階シミュレーション（デフォルト10万回）。profiles/pace が無い場合は指数ガウスにフォールバック。"""
    from race_sim import SIM_RUNS, build_profiles, predict_pace, simulate_race_stages
    g=g.copy().reset_index(drop=True)
    runs=int(runs or SIM_RUNS)
    if profiles is None or pace is None:
        # 後方互換: 旧ガウス latent
        base=g["AREru指数"].astype(float).to_numpy()
        cons=g["因子_consistency"].astype(float).to_numpy() if "因子_consistency" in g.columns else np.full(len(g),50.0)
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
    else:
        order_all, finish_counts, _corner_avg = simulate_race_stages(g, profiles, pace, runs=runs)
        g["_平均4角順位"] = _corner_avg
    n_h=len(g)
    raw_win=finish_counts[:,0]/runs*100
    win=_cap_sim_win_rates(raw_win)
    # 連対・複勝は過信しやすいので、一様分布へ縮小して校正
    raw2=finish_counts[:,:min(2,n_h)].sum(axis=1)/runs*100
    raw3=finish_counts[:,:min(3,n_h)].sum(axis=1)/runs*100
    target2=200.0/max(n_h,1)
    target3=300.0/max(n_h,1)
    shrink=0.62  # モデル寄与。残りはフィールド平均へ
    place2=shrink*raw2+(1-shrink)*target2
    place3=shrink*raw3+(1-shrink)*target3
    # 合計が理論値に近づくよう再スケール
    if place2.sum()>0: place2=place2*(200.0/place2.sum())
    if place3.sum()>0: place3=place3*(300.0/place3.sum())
    # 単勝オッズがある場合は市場暗示確率へ部分収縮（大穴の過大勝率→極端EVを抑制）
    if "単勝オッズ" in g.columns:
        market_arr=pd.to_numeric(g["単勝オッズ"], errors="coerce").to_numpy(dtype=float)
        valid=np.isfinite(market_arr) & (market_arr > 1.01)
        if int(valid.sum()) >= max(3, n_h // 2):
            impl=np.zeros(n_h, dtype=float)
            impl[valid]=1.0/market_arr[valid]
            # オッズ欠損馬は現状SIM比率で埋める
            if (~valid).any():
                fallback=np.maximum(win, 0.01)
                impl[~valid]=fallback[~valid]/max(float(fallback[~valid].sum()), 1e-9)
            impl=impl/max(float(impl.sum()), 1e-9)*100.0
            # SIMが市場より強いほど市場寄りへ（長穴の37%勝ち等を潰す）
            ratio=np.maximum(win, 0.01)/np.maximum(impl, 0.05)
            sim_w=np.clip(0.58 - 0.12*np.log1p(np.maximum(ratio-1.0, 0.0)), 0.28, 0.62)
            win=sim_w*win+(1.0-sim_w)*impl
            win=_cap_sim_win_rates(win)
            # 合計100%へ再正規化
            if float(win.sum()) > 0:
                win=win*(100.0/float(win.sum()))
                win=_cap_sim_win_rates(win)
    g["SIM勝率"]=win
    g["SIM2着内率"]=place2
    g["SIM3着内率"]=place3
    safe_win=np.maximum(win, 0.01)
    fair=np.where(win>0,100.0/safe_win,999.0)
    g["AI適正オッズ"]=np.maximum(fair, AI_FAIR_ODDS_MIN)
    market=pd.to_numeric(g["単勝オッズ"], errors="coerce") if "単勝オッズ" in g.columns else pd.Series(np.nan, index=g.index)
    # 単勝期待値も信頼できる帯にソフトクリップ（表示・ログ用）
    raw_ticket_ev=np.where((market>0)&(win>0), market*(win/100.0)*100.0, np.nan)
    g["単勝期待値"]=np.where(np.isfinite(raw_ticket_ev), np.clip(raw_ticket_ev, 70, 130), np.nan)
    return g, order_all

def _ticket_candidates(g, orders):
    """全組み合わせの仮想的中率。印に依存しない。確率は一様事前へ縮小して校正。"""
    n=len(g); runs=len(orders)
    pos=np.empty_like(orders)
    rows=np.arange(runs)[:,None]
    pos[rows,orders]=np.arange(n)[None,:]
    in2=pos<2; in3=pos<3
    # 一様事前（%）
    prior_w=600.0/max(n*(n-1),1)          # 特定2頭が両方3着内
    prior_q=200.0/max(n*(n-1),1)          # 特定2頭が1-2着
    prior_e=100.0/max(n*(n-1),1)          # 特定順序の馬単
    prior_t=600.0/max(n*(n-1)*(n-2),1)    # 特定3頭が3着内（近似）
    prior_tf=100.0/max(n*(n-1)*(n-2),1)   # 特定三連単
    cal=0.55
    def mix(p, prior): return cal*float(p)+(1-cal)*float(prior)
    wide=[]; quinella=[]; exacta=[]; trio=[]; trifecta=[]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            p_e=mix(np.mean((orders[:,0]==i)&(orders[:,1]==j))*100, prior_e)
            if p_e>0:
                exacta.append((p_e,(i,j)))
            if i<j:
                p_w=mix(np.mean(in3[:,i] & in3[:,j])*100, prior_w)
                p_q=mix(np.mean(in2[:,i] & in2[:,j])*100, prior_q)
                wide.append((p_w,(i,j)))
                quinella.append((p_q,(i,j)))
                for k in range(j+1,n):
                    p_t=mix(np.mean(in3[:,i] & in3[:,j] & in3[:,k])*100, prior_t)
                    trio.append((p_t,(i,j,k)))
    top_idx=list(np.argsort(-g["SIM勝率"].to_numpy())[:min(8,n)])
    for i in top_idx:
        for j in top_idx:
            if j==i: continue
            for k in top_idx:
                if k==i or k==j: continue
                p_tf=mix(np.mean((orders[:,0]==i)&(orders[:,1]==j)&(orders[:,2]==k))*100, prior_tf)
                if p_tf>0:
                    trifecta.append((p_tf,(i,j,k)))
    return {
        "ワイド": sorted(wide,reverse=True),
        "馬連": sorted(quinella,reverse=True),
        "馬単": sorted(exacta,reverse=True),
        "三連複": sorted(trio,reverse=True),
        "三連単": sorted(trifecta,reverse=True),
    }

def _ban_list(g, idxs, ordered=False):
    bans=[]
    for i in idxs:
        b=str(g.iloc[i].get("馬番","")).strip()
        try:
            bans.append(int(float(b)))
        except Exception:
            return None
    if ordered:
        return bans
    return sorted(bans)


def _ban_key(g, idxs, ordered=False):
    bans=_ban_list(g, idxs, ordered=ordered)
    if not bans:
        return None
    return "".join(f"{b:02d}" for b in bans)


def _lookup_combo_odds(kind, g, idxs, ticket_odds):
    if not ticket_odds:
        return None
    ordered = kind in ("馬単", "三連単")
    table=ticket_odds.get(kind) or {}
    key=_ban_key(g, idxs, ordered=ordered)
    if not key:
        return None
    raw=table.get(key)
    if raw in (None, ""):
        return None
    try:
        return float(str(raw).replace(",", ""))
    except Exception:
        return None


def _circled_pair(bans, sep="－"):
    from race_sim import circle_ban
    return sep.join(circle_ban(b) for b in bans)


def _formation_from_combos(chosen, g, ordered=False):
    """選択組み合わせからフォーメーション表示を生成。"""
    from race_sim import circle_ban
    if not chosen:
        return None
    if not ordered:
        # 三連複: 出現頻度で軸→相手
        from collections import Counter
        cnt=Counter()
        for _,idxs in chosen:
            for i in idxs:
                b=_ban_list(g,(i,),ordered=True)
                if b: cnt[b[0]]+=1
        ranked=[b for b,_ in cnt.most_common()]
        if not ranked:
            return None
        axis=ranked[:1]
        second=ranked[:max(2,min(4,len(ranked)))]
        third=ranked[:max(3,min(8,len(ranked)))]
        return {
            "1頭目":[circle_ban(x) for x in axis],
            "2頭目":[circle_ban(x) for x in second],
            "3頭目":[circle_ban(x) for x in third],
            "点数":len(chosen),
        }
    # 三連単/馬単: 着順位置ごとの集合
    from collections import defaultdict
    slots=[defaultdict(int) for _ in range(3)]
    for _,idxs in chosen:
        for pos,i in enumerate(idxs[:3]):
            b=_ban_list(g,(i,),ordered=True)
            if b: slots[pos][b[0]]+=1
    def top_slot(c, n):
        return [circle_ban(b) for b,_ in sorted(c.items(), key=lambda x:-x[1])[:n]]
    return {
        "1頭目":top_slot(slots[0], max(1,min(3,len(slots[0]) or 1))),
        "2頭目":top_slot(slots[1], max(2,min(5,len(slots[1]) or 2))),
        "3頭目":top_slot(slots[2], max(2,min(8,len(slots[2]) or 2))) if len(chosen[0][1])>2 else [],
        "点数":len(chosen),
    }


def _optimize_ticket(kind, candidates, g, max_points, ticket_odds=None, min_ev=100.0):
    """期待値（的中率%×実オッズ）優先で買い目を採用。印は使わない。

    超高配当の一点突破を避けるため、券種別に的中率下限とオッズ上限（評価用）を設ける。
    """
    from race_sim import stars_from_ev
    if not candidates:
        return {"買い目":[],"的中期待":0.0,"候補数":0,"圧縮理由":"候補なし",
                "合成オッズ":None,"期待回収率":None,"フォーメーション":None,"推奨度":"☆☆☆☆☆"}
    ordered = kind in ("馬単", "三連単")
    min_hit={"ワイド":7.0,"馬連":4.0,"馬単":2.5,"三連複":1.5,"三連単":0.6}.get(kind,2.0)
    odds_cap={"ワイド":25.0,"馬連":40.0,"馬単":60.0,"三連複":80.0,"三連単":150.0}.get(kind,50.0)
    scored=[]
    for p,idxs in candidates:
        p=float(p)
        if p < min_hit*0.5:
            continue
        o=_lookup_combo_odds(kind,g,idxs,ticket_odds)
        if o is not None and o>0:
            ev=p*float(o)
            # ランキング用: 極端な高オッズは頭打ち（宝くじ化防止）
            utility=p*min(float(o), odds_cap)
        else:
            ev=p*2.5
            utility=p
        scored.append((utility, ev, p, idxs, o))
    scored.sort(key=lambda x:(-x[0], -x[2]))
    has_odds=any(x[4] is not None and x[4]>0 for x in scored[:80])
    chosen=[]
    for utility,ev,p,idxs,o in scored:
        if len(chosen)>=max_points: break
        if has_odds:
            if p < min_hit:
                continue
            # 評価用オッズ上限を超える高配当は推奨から除外（的中率優先）
            if o is not None and float(o) > odds_cap:
                continue
            if ev < min_ev and chosen:
                continue
            if ev < min_ev * 0.9 and not chosen:
                # 閾値未満でも、上限内の最上位は候補として残す
                chosen.append((ev,p,idxs,o)); break
            if ev > 180:
                continue
            chosen.append((ev,p,idxs,o))
        else:
            if p < min_hit*0.8 and chosen:
                break
            if not chosen:
                chosen.append((ev,p,idxs,o)); continue
            if p < chosen[0][1]*({"ワイド":0.42,"馬連":0.34,"馬単":0.30,"三連複":0.28,"三連単":0.22}.get(kind,0.3)):
                break
            chosen.append((ev,p,idxs,o))
    if not chosen and scored:
        for _,ev,p,idxs,o in scored:
            if o is None or float(o)<=odds_cap*1.15:
                chosen=[(ev,p,idxs,o)]
                break
        if not chosen:
            _,ev,p,idxs,o=scored[0]
            chosen=[(ev,p,idxs,o)]

    strength=float(np.mean([x[1] for x in chosen])) if chosen else 0.0
    rows=[]; odds_vals=[]; ev_vals=[]
    for ev,p,idxs,o in chosen:
        bans=_ban_list(g,idxs,ordered=ordered) or []
        real_ev=float(p)*float(o) if (o is not None and o>0) else None
        item={
            "馬番表示":_circled_pair(bans) if bans else "",
            "馬番":bans,
            "馬名":" － ".join(str(g.iloc[i]["馬名"]) for i in idxs),
            "仮想的中率":round(float(p),1),
            "期待値":round(real_ev,1) if real_ev is not None else None,
            "推奨度":stars_from_ev(real_ev, p),
        }
        if o is not None and o>0:
            item["実オッズ"]=round(float(o),1)
            item["期待回収率"]=round(real_ev,1)
            odds_vals.append(float(o)); ev_vals.append(real_ev)
        rows.append(item)
    synth=round(float(np.mean(odds_vals)),1) if odds_vals else None
    ev=round(float(np.mean(ev_vals)),1) if ev_vals else None
    form=_formation_from_combos([(p,idxs) for _,p,idxs,_ in chosen], g, ordered=ordered) if kind in ("三連複","三連単") else None
    reason=f"全組み合わせを期待値順に評価し上位{len(rows)}点を採用（印非依存）"
    return {
        "買い目":rows,
        "的中期待":round(strength,1),
        "候補数":len(candidates),
        "圧縮理由":reason,
        "合成オッズ":synth,
        "期待回収率":ev,
        "フォーメーション":form,
        "推奨度":stars_from_ev(ev, strength),
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
        from netkeiba_client import NetkeibaClient, infer_source
        maps=NetkeibaClient(sleep=0.15).fetch_ticket_odds_maps(rid, source=infer_source(rid))
        ODDS_TICKETS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(maps, ensure_ascii=False), encoding='utf-8')
        return maps
    except Exception:
        return {}


def build_predictions(target_str, runners, history=None, weights=None, fetch_ticket_odds=True):
    target=pd.Timestamp(target_str); weights=weights or load_weights(); r=runners.copy()
    r['_date']=parse_date(r['日付']); r=r[r['_date'].dt.normalize()==target.normalize()].copy()
    if r.empty: raise ValueError(f'{target_str} の出走データがありません')
    r=enrich_runners_past_venues(r)
    if history is not None:
        history=history.copy(); history['_date']=parse_date(history['年月日']); history['_horse']=history['馬名'].map(clean_name)
    from race_sim import (
        SIM_RUNS, build_profiles, predict_pace, lap_aptitude, style_label,
        circle_ban, stars_from_ev,
    )
    scored=[]
    for _,row in r.iterrows():
        s,f,why=score_runner(row,history,target,weights)
        x=row.to_dict(); x.update({'AREru指数':round(s,2),**{f'因子_{k}':round(v,1) for k,v in f.items()},'理由':' / '.join(dict.fromkeys(why[:4])) or '総合評価'})
        scored.append(x)
    sd=pd.DataFrame(scored); out=[]
    for race_id,g0 in sd.groupby('race_id',sort=False):
        venue=venue_from_race_id(race_id)
        g_base=g0.sort_values('AREru指数',ascending=False).reset_index(drop=True)
        profiles=build_profiles(g_base, history, target, venue)
        pace=predict_pace(profiles)
        # ラップ適性を個別に付与
        for i,p in enumerate(profiles):
            fit, lab=lap_aptitude(p['style'], pace['想定ペース'])
            p['lap_fit']=fit; p['lap_label']=lab
            if fit>=60: p['plus'].append(lab)
            elif fit<=40: p['minus'].append(lab)

        g, orders=simulate_race(g_base, runs=SIM_RUNS, profiles=profiles, pace=pace); n=len(g)
        # プロファイルを馬名で引けるように
        prof_by_name={str(g_base.iloc[i]['馬名']):profiles[i] for i in range(len(profiles))}

        top=g['AREru指数'].iloc[0]; spread=top-g['AREru指数'].iloc[min(4,n-1)]; upset_share=(g['因子_upset']>=65).mean()
        chaos=clamp(0.55*float(pace.get('荒れ指数',50))+0.45*(35+(75-top)*.8+(18-spread)*1.3+upset_share*28))
        p1=num(g['人気1']) if '人気1' in g else pd.Series(np.nan,index=g.index)
        market_pop=num(g['人気']) if '人気' in g else pd.Series(np.nan,index=g.index)

        def _n_valid_row(rr):
            return int(sum(pd.notna(num(rr.get(f'着順{i}'))) for i in range(1,6)))

        main_order=g.sort_values(['SIM3着内率','AREru指数'],ascending=False)
        main=main_order.iloc[0]
        for _,cand in main_order.iterrows():
            nv=_n_valid_row(cand)
            mo=num(cand.get('単勝オッズ'))
            if nv>=2 or pd.isna(mo) or float(mo)<40:
                main=cand
                break

        # 対抗=勝率2位、穴=期待値/穴スコア上位
        hole_score=(
            g['AREru指数']*.30+g['因子_upset']*.20+g['因子_value']*.20+g['SIM3着内率']*.20
            +np.where(market_pop>=6,10,0)+np.where(num(g.get('単勝期待値')).fillna(0)>=120,8,0)
        )
        rest=g[g['馬名']!=main['馬名']].copy().assign(_hole=hole_score.loc[g['馬名']!=main['馬名']])
        rival=rest.sort_values(['SIM勝率','SIM3着内率'],ascending=False).head(1)
        hole_pool=rest.sort_values(['_hole','SIM3着内率'],ascending=False)
        # 印行（後方互換）+ 構造化カード
        pick_cards=[]
        mark_rows=[]

        def _ban_str(row):
            b=str(row.get('馬番','') or '').strip()
            try: return str(int(float(b))) if b else ''
            except Exception: return ''

        # 上がり順位（フィールド内）
        last3_vals={str(g_base.iloc[i]['馬名']): float(profiles[i].get('last3f') or 50) for i in range(len(profiles))}
        last3_rank_map={}
        for rk,(hn,_) in enumerate(sorted(last3_vals.items(), key=lambda x: -x[1]), 1):
            last3_rank_map[hn]=rk

        def _detail_for(row, role):
            from pick_rationale import build_pick_rationale
            name=str(row['馬名']); pr=dict(prof_by_name.get(name,{}) or {})
            fo=num(row.get('AI適正オッズ')); mo=num(row.get('単勝オッズ')); pop=num(row.get('人気'))
            idx_rank=int(g['AREru指数'].rank(ascending=False).loc[row.name])
            pace_fit,_=lap_aptitude(pr.get('style',0.5), pace['想定ペース'])
            style=style_label(pr.get('style',0.5))
            pr['pace_fit']=pace_fit
            pr['style_label']=style
            plus=list(pr.get('plus') or [])
            minus=list(pr.get('minus') or [])
            if float(row['SIM3着内率'])>=40: plus.append('複勝圏安定')
            if pd.notna(mo) and pd.notna(fo) and float(mo)>float(fo)*1.15: plus.append('市場より割安')
            n_sample=_n_valid_row(row)
            if n_sample<3: minus.append('サンプル少')
            win_ev=None
            if pd.notna(mo) and pd.notna(fo) and float(fo)>0:
                # 市場比1.30倍超の割安はCSV段階でも認めない（1000%級を防止）
                fo_cap=max(float(fo), float(mo)/1.30, AI_FAIR_ODDS_MIN)
                win_ev=int(round(min(130.0, max(70.0, float(mo)/fo_cap*100.0))))
            pack=build_pick_rationale(
                role=role,
                horse=name,
                row=row.to_dict() if hasattr(row,'to_dict') else dict(row),
                profile=pr,
                pace=pace,
                idx_rank=idx_rank,
                field_n=int(n),
                last3_rank=last3_rank_map.get(name),
                n_sample=n_sample,
                win_pct=round(float(row['SIM勝率']),1),
                quinella_pct=round(float(row['SIM2着内率']),1),
                place_pct=round(float(row['SIM3着内率']),1),
                market_odds=float(mo) if pd.notna(mo) else None,
                fair_odds=float(fo) if pd.notna(fo) else None,
                reason_text=str(row.get('理由') or ''),
                existing_plus=plus,
                existing_minus=minus,
                lap_label=str(pr.get('lap_label') or '平均ペース適性'),
                lap_fit=float(pr.get('lap_fit') or pace_fit or 50),
                pace_fit=float(pace_fit),
                style=style,
            )
            detail={
                '役割':role,
                '馬名':name,
                '馬番':_ban_str(row),
                '馬番表示':circle_ban(_ban_str(row)),
                'AI評価':round(float(row['AREru指数']),1),
                '近走指数順位':idx_rank,
                '勝率':round(float(row['SIM勝率']),1),
                '連対率':round(float(row['SIM2着内率']),1),
                '複勝率':round(float(row['SIM3着内率']),1),
                'AI適正オッズ':round(max(float(fo),AI_FAIR_ODDS_MIN),1) if pd.notna(fo) else None,
                '単勝オッズ':round(float(mo),1) if pd.notna(mo) else None,
                '人気':int(float(pop)) if pd.notna(pop) else None,
                '期待値':win_ev,
                **pack,
            }
            if role=='穴馬' or role=='注目馬':
                detail['期待値が高い理由']=f"単勝期待値{win_ev}%" if win_ev else '仮想複勝率に対し人気が薄い'
                detail['人気以上に評価した理由']=str(row.get('理由') or '人気薄の上昇余地')
            return detail

        main_detail=_detail_for(main,'本命')
        pick_cards.append(main_detail)
        if len(rival):
            rival_detail=_detail_for(rival.iloc[0],'対抗')
            pick_cards.append(rival_detail)
            mark_rows.append({
                '印':'○','馬名':rival_detail['馬名'],'馬番':rival_detail['馬番'],'馬番表示':rival_detail['馬番表示'],
                '3着内率':rival_detail['複勝率'],'理由':rival_detail['プラス材料'],
                'AI適正オッズ':rival_detail['AI適正オッズ'],'単勝オッズ':rival_detail['単勝オッズ'],
                '詳細':rival_detail,
            })
        # 穴馬/注目馬: 本命・対抗以外で穴スコア上位。1頭目は☆注目馬
        used={main_detail['馬名']}
        if len(rival): used.add(str(rival.iloc[0]['馬名']))
        hole_marks=['☆','▲','△']; hi=0
        for _,x in hole_pool.iterrows():
            if str(x['馬名']) in used: continue
            role='注目馬' if hi==0 else '穴馬'
            hd=_detail_for(x, role)
            pick_cards.append(hd)
            mark_rows.append({
                '印':hole_marks[min(hi,len(hole_marks)-1)],'馬名':hd['馬名'],'馬番':hd['馬番'],'馬番表示':hd['馬番表示'],
                '3着内率':hd['複勝率'],'理由':hd.get('人気以上に評価した理由') or hd['プラス材料'],
                'AI適正オッズ':hd['AI適正オッズ'],'単勝オッズ':hd['単勝オッズ'],'詳細':hd,
            })
            used.add(hd['馬名']); hi+=1
            if hi>=3: break

        # 危険人気馬: 1〜5番人気のみ、AI順位低・EV<80・売れ過ぎ
        danger_card=None
        danger_name=''; danger_reason='該当なし'; danger_score_val=0.0
        cand_danger=[]
        ai_rank_map=g['AREru指数'].rank(ascending=False)
        for idx,row in g.iterrows():
            pop=num(row.get('人気'))
            if pd.isna(pop) or not (1<=float(pop)<=5): continue
            fo=num(row.get('AI適正オッズ')); mo=num(row.get('単勝オッズ'))
            if pd.isna(fo) or pd.isna(mo) or float(fo)<=0: continue
            win_ev=float(mo)/float(fo)*100
            ai_r=int(ai_rank_map.loc[idx])
            overbet=float(mo)<float(fo)*0.85  # 適正より安い＝売れ過ぎ
            if ai_r>=max(5,int(n*0.45)) and win_ev<80 and overbet:
                score=(5-float(pop))*8+(ai_r)+max(0,80-win_ev)*0.5+(float(fo)-float(mo))*2
                cand_danger.append((score,row,win_ev,ai_r,overbet))
        if cand_danger:
            cand_danger.sort(key=lambda x:-x[0])
            sc,drow,win_ev,ai_r,_=cand_danger[0]
            danger_score_val=clamp(sc)
            danger_name=str(drow['馬名'])
            dban=_ban_str(drow)
            reasons=[]
            reasons.append(f"AI順位{ai_r}位と人気の乖離")
            reasons.append(f"期待値{win_ev:.0f}%未満")
            reasons.append(f"現在{float(num(drow['単勝オッズ'])):.1f}倍 < 適正{float(num(drow['AI適正オッズ'])):.1f}倍で売れ過ぎ")
            danger_reason=' / '.join(reasons)
            danger_card={
                '馬名':danger_name,'馬番':dban,'馬番表示':circle_ban(dban),
                '人気':int(float(num(drow['人気']))),
                'AI順位':ai_r,'期待値':round(win_ev),'危険理由':danger_reason,
                '単勝オッズ':round(float(num(drow['単勝オッズ'])),1),
                'AI適正オッズ':round(float(num(drow['AI適正オッズ'])),1),
            }

        main_place=float(main['SIM3着内率']); alt_place=float(rival['SIM3着内率'].max()) if len(rival) else 0
        clarity=max(0,float(main['SIM3着内率'])-float(g['SIM3着内率'].median()))
        bet=clamp(main_place*.38+alt_place*.22+clarity*.75+chaos*.16)
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
        main_name=str(main['馬名']); main_ban=_ban_str(main)

        main_win=num(main.get('単勝オッズ'))
        has_win=pd.notna(main_win) and float(main_win)>0
        ticket_odds=load_ticket_odds(race_id, fetch_if_missing=fetch_ticket_odds and has_win) if has_win else {}

        candidates=_ticket_candidates(g,orders)
        wide_plan=_optimize_ticket("ワイド",candidates["ワイド"],g,4,ticket_odds,min_ev=105)
        quinella_plan=_optimize_ticket("馬連",candidates["馬連"],g,3,ticket_odds,min_ev=105)
        exacta_plan=_optimize_ticket("馬単",candidates["馬単"],g,4,ticket_odds,min_ev=110)
        trio_plan=_optimize_ticket("三連複",candidates["三連複"],g,8,ticket_odds,min_ev=110)
        trifecta_plan=_optimize_ticket("三連単",candidates["三連単"],g,6,ticket_odds,min_ev=120)

        def plan_score(plan, base):
            ev=plan.get('期待回収率')
            hit=plan.get('的中期待') or 0
            if ev is not None:
                return clamp(float(ev)*0.45 + float(hit)*1.2 + base)
            return clamp(float(hit)*2.2 + base)

        wide_score=plan_score(wide_plan, main_place*.2)
        quinella_score=plan_score(quinella_plan, float(main['SIM勝率'])*.3)
        exacta_score=plan_score(exacta_plan, float(main['SIM勝率'])*.25)
        trio_score=plan_score(trio_plan, chaos*.15)
        trifecta_score=plan_score(trifecta_plan, chaos*.1)

        def go_label(v, plan):
            ev=plan.get('期待回収率')
            if ev is not None:
                if ev>=115 and v>=60: return '買い候補'
                if ev>=100: return '条件付き'
                return '見送り'
            return '買い候補' if v>=70 else ('条件付き' if v>=55 else '見送り')

        plans=[
            ("ワイド",wide_score,wide_plan),
            ("馬連",quinella_score,quinella_plan),
            ("馬単",exacta_score,exacta_plan),
            ("三連複",trio_score,trio_plan),
            ("三連単",trifecta_score,trifecta_plan),
        ]
        # 主戦券種: 現実的EV（ワイド/馬連寄り）を優先。三連単の過大EVで押し上げない
        kind_priority={"ワイド":5,"馬連":4,"三連複":3,"馬単":2,"三連単":1}
        def plan_rank_key(item):
            kind,score,plan=item
            ev=plan.get('期待回収率')
            capped=min(float(ev), 160.0) if ev is not None else 0.0
            return (ev is not None, capped + kind_priority.get(kind,0)*3, score)
        plans_sorted=sorted(plans,key=plan_rank_key,reverse=True)
        best_kind,best_score,best_plan=plans_sorted[0]

        ticket_reason=(
            f"{best_kind}型。{SIM_RUNS:,}回の段階シミュレーションで全組み合わせ期待値を比較。"
            f"{best_plan['圧縮理由']}。"
        )
        if best_plan.get('期待回収率') is not None:
            ticket_reason += f" 期待回収率 {best_plan['期待回収率']}%。"

        def plan_text(plan):
            parts=[]
            for x in plan["買い目"]:
                label=x.get('馬番表示') or x.get('馬名')
                s=f"{label}（的中 {x['仮想的中率']}%"
                if x.get('期待値') is not None:
                    s+=f" / 期待値{x['期待値']}%"
                elif x.get('期待回収率') is not None:
                    s+=f" / EV{x['期待回収率']}%"
                if x.get('推奨度'):
                    s+=f" / {x['推奨度']}"
                s+="）"
                parts.append(s)
            return '｜'.join(parts) if parts else '見送り'

        # 投資判定は表示時に ev_analysis が信頼度込みで再計算する（ここでは暫定）
        main_odds=num(main.get('単勝オッズ'))
        core_evs=[]
        for kind,_,plan in plans:
            if kind not in ('ワイド','馬連'):
                continue
            ev=plan.get('期待回収率')
            if ev is not None:
                core_evs.append(min(float(ev), 130.0))
        race_ev=round(float(np.median(core_evs)),1) if core_evs else None
        invest_label='判定待ち'; invest_icon='⚪'; invest_tone='wait'
        main_n_valid=_n_valid_row(main)

        # 推奨馬券JSON（UI・資金配分用）
        recommend_tickets=[]
        for kind,score,plan in plans_sorted:
            if go_label(score,plan)=='見送り' and plan.get('期待回収率') is not None and plan['期待回収率']<100:
                continue
            for item in plan.get('買い目') or []:
                ev_v=item.get('期待値') or item.get('期待回収率')
                hit_v=item.get('仮想的中率') or 0
                # UI推奨からは異常EV（宝くじ）を除外
                if ev_v is not None and (float(ev_v)>350 or (float(ev_v)>220 and float(hit_v)<2)):
                    continue
                recommend_tickets.append({
                    '券種':kind,
                    '馬番表示':item.get('馬番表示'),
                    '馬番':item.get('馬番'),
                    '的中率':hit_v,
                    '期待値':ev_v,
                    '推定回収率':item.get('期待回収率') or ev_v,
                    '推奨度':item.get('推奨度') or plan.get('推奨度'),
                    'フォーメーション':plan.get('フォーメーション') if kind in ('三連複','三連単') else None,
                })
        # 期待値が高く、かつ的中率も見込める馬券へ多く配分
        ev_weights=[]
        for t in recommend_tickets:
            ev=float(t.get('期待値') or 100)
            hit=float(t.get('的中率') or 1)
            w=max(min(ev,200)-95, 1.0)* (0.5+min(hit,20)/20)
            ev_weights.append(w)
        total_w=sum(ev_weights) or 1.0
        for t,w in zip(recommend_tickets, ev_weights):
            t['配分比率']=round(w/total_w,4)

        main_odds_disp=round(float(main_odds),1) if pd.notna(main_odds) else ''
        main_pop=num(main.get('人気'))
        main_pop_disp=int(float(main_pop)) if pd.notna(main_pop) else ''
        synth=best_plan.get('合成オッズ')
        ev=best_plan.get('期待回収率')
        synth_disp=f"{synth}倍" if synth is not None else '券種別オッズ待ち'
        ev_disp=f"{ev}%" if ev is not None else 'オッズ接続後に算出'
        fair_odds=round(max(float(main['AI適正オッズ']), AI_FAIR_ODDS_MIN),1)
        if pd.notna(main_odds) and float(main_odds)>0 and fair_odds>0:
            # 市場から極端に離れた適正はCSV保存値でも抑制（表示は ev_analysis が再計算）
            fair_odds=round(max(fair_odds, float(main_odds)/1.30, AI_FAIR_ODDS_MIN),1)
            ui_ev=round(float(main_odds)/fair_odds*100)
            if ui_ev>=EV_EXTREME_PCT:
                log.warning(
                    'EV_EXTREME race_id=%s venue=%s R=%s horse=%s market=%.1f fair=%.1f ev=%s%% '
                    'sim_win=%.1f score=%.2f reasons=%s',
                    race_id, venue, int(float(main['レース'])), main_name,
                    float(main_odds), fair_odds, ui_ev,
                    float(main['SIM勝率']), float(main['AREru指数']), main.get('理由'),
                )

        src=str(main.get('source') or source_from_race_id(race_id))
        out.append({
          'race_id':race_id,'source':src,'開催地':venue,'レース':int(float(main['レース'])),'荒れ度':round(chaos,1),'判定':judge,
          '荒れクラス':'storm' if chaos>=80 else ('wave' if chaos>=60 else ('caution' if chaos>=40 else 'calm')),
          'BET期待値':round(bet,1),'BET判定':bet_label,'BETクラス':bet_class,'BET理由':' / '.join(bet_reason),
          'シミュレーション回数':SIM_RUNS,
          '本命':main_name,'本命馬番':main_ban,'本命馬番表示':circle_ban(main_ban),
          '本命AREru指数':round(float(main['AREru指数']),2),
          'シミュレーション勝率':round(float(main['SIM勝率']),1),
          'シミュレーション連対率':round(float(main['SIM2着内率']),1),
          'シミュレーション3着内率':round(float(main['SIM3着内率']),1),
          'AI適正オッズ':fair_odds,'本命理由':main_detail['プラス材料'],
          '本命詳細':json.dumps(main_detail,ensure_ascii=False),
          '本命オッズ':main_odds_disp,'本命人気':main_pop_disp,
          '人気馬危険':danger_name or '該当なし',
          '危険度':round(danger_score_val,1),
          '危険理由':danger_reason,
          '危険人気詳細':json.dumps(danger_card,ensure_ascii=False) if danger_card else '',
          'ピックカード':json.dumps(pick_cards,ensure_ascii=False),
          '展開予想':json.dumps(pace,ensure_ascii=False),
          '印データ':json.dumps(mark_rows,ensure_ascii=False),
          '推奨券種':best_kind,'馬券戦略理由':ticket_reason,
          '推奨馬券':json.dumps(recommend_tickets,ensure_ascii=False),
          '投資判定':invest_label,'投資判定アイコン':invest_icon,'投資判定トーン':invest_tone,
          'レース期待回収率':race_ev if race_ev is not None else '',
          '本命データ件数':int(main_n_valid),
          'ワイド評価':round(wide_score,1),'ワイド判定':go_label(wide_score,wide_plan),'ワイド買い目':plan_text(wide_plan),'ワイド圧縮':wide_plan['圧縮理由'],
          'ワイド詳細':json.dumps(wide_plan,ensure_ascii=False),
          '馬連評価':round(quinella_score,1),'馬連判定':go_label(quinella_score,quinella_plan),'馬連買い目':plan_text(quinella_plan),'馬連圧縮':quinella_plan['圧縮理由'],
          '馬連詳細':json.dumps(quinella_plan,ensure_ascii=False),
          '馬単評価':round(exacta_score,1),'馬単判定':go_label(exacta_score,exacta_plan),'馬単買い目':plan_text(exacta_plan),'馬単圧縮':exacta_plan['圧縮理由'],
          '馬単詳細':json.dumps(exacta_plan,ensure_ascii=False),
          '三連複評価':round(trio_score,1),'三連複判定':go_label(trio_score,trio_plan),'三連複買い目':plan_text(trio_plan),'三連複圧縮':trio_plan['圧縮理由'],
          '三連複詳細':json.dumps(trio_plan,ensure_ascii=False),
          '三連単評価':round(trifecta_score,1),'三連単判定':go_label(trifecta_score,trifecta_plan),'三連単買い目':plan_text(trifecta_plan),'三連単圧縮':trifecta_plan['圧縮理由'],
          '三連単詳細':json.dumps(trifecta_plan,ensure_ascii=False),
          '合成オッズ':synth_disp,'期待回収率':ev_disp,
          'データ頭数':n})

    result=pd.DataFrame(out).sort_values(['開催地','レース']).reset_index(drop=True)
    # source 別に相対順位で S/A/B/C を付与（JRA/NAR混在日でもプールを分けて評価）
    ranked_parts=[]
    if 'source' in result.columns and result['source'].nunique()>1:
        for _,g in result.groupby('source',sort=False):
            ranked_parts.append(assign_ai_ranks(g))
        result=pd.concat(ranked_parts,ignore_index=True).sort_values(['開催地','レース']).reset_index(drop=True)
    else:
        result=assign_ai_ranks(result)
    return result,sd

