#!/usr/bin/env python3
"""能力差シグナル単体の再現性検証。ランク条件・閾値の再調整は禁止。

出力: data/ability_signal_reproducibility_report.json
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_END = "2026-07-28"
BANDS = [("50-59",50,59),("60-69",60,69),("70-79",70,79),("80-89",80,89),("90-99",90,99),("100+",100,10**9)]

def fnum(s, default=np.nan):
    try:
        t=str(s).strip().replace("%","").replace("倍","").replace(",","")
        if t in ("","nan","None","なし","—","-"): return default
        return float(t)
    except Exception:
        return default

def metrics(df):
    n=len(df)
    if n==0:
        return {"n":0,"hits":0,"hit_rate":None,"recovery":None,"avg_odds":None,"avg_ev":None,"profit":None}
    hits=int(df["hit"].sum()); inv=n*100.0; pay=float(df["payout"].sum())
    return {"n":n,"hits":hits,"hit_rate":round(hits/n*100,1),"recovery":round(pay/inv*100,1),
            "avg_odds":round(float(df["本命オッズ"].mean()),2),
            "avg_ev":round(float(df["期待値"].mean()),1) if df["期待値"].notna().any() else None,
            "profit":int(pay-inv)}

def main():
    rows=[]
    for path in sorted((ROOT/"data/predictions_by_date").glob("predictions_*.csv")):
        day=re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
        df=pd.read_csv(path, dtype=str); df["date"]=day.group(1) if day else ""; rows.append(df)
    P=pd.concat(rows, ignore_index=True)
    for c in ["能力差スコア","期待値","本命オッズ"]:
        P[c]=P[c].map(fnum)
    P["source"]=P["source"].astype(str).str.lower()
    P["本命馬番_k"]=P["本命馬番"].map(lambda x: str(int(float(x))) if fnum(x)==fnum(x) else str(x).strip())
    R=pd.read_csv(ROOT/"data/results.csv", dtype=str)
    R["着順_n"]=pd.to_numeric(R["着順"], errors="coerce")
    W=R[R["着順_n"]==1][["race_id","馬番","確定オッズ"]].rename(columns={"馬番":"win_umaban","確定オッズ":"win_odds"})
    W["win_umaban"]=W["win_umaban"].astype(str).str.strip(); W["win_odds"]=pd.to_numeric(W["win_odds"], errors="coerce")
    P=P.merge(W, on="race_id", how="inner")
    P=P[P["本命オッズ"].notna()&(P["本命オッズ"]>0)].copy()
    P["hit"]=(P["本命馬番_k"]==P["win_umaban"]).astype(int)
    P["payout"]=np.where(P["hit"]==1, P["win_odds"].fillna(P["本命オッズ"]).fillna(0)*100, 0.0)
    def band_mask(df,lo,hi):
        a=df["能力差スコア"]; return (a>=lo) if hi>=10**9 else ((a>=lo)&(a<=hi))
    scopes={"全体":P,"JRA":P[P.source=="jra"],"地方":P[P.source=="nar"]}
    periods={"all":lambda d:d,"train":lambda d:d[d.date<=TRAIN_END],"test":lambda d:d[d.date>TRAIN_END]}
    report={"purpose":"能力差シグナル単体の再現性検証","no_retuning":True,"tables":{},"baseline":{},
            "ability_discrete_values":{str(float(k)):int(v) for k,v in P["能力差スコア"].value_counts().sort_index().items()},
            "band_note":"実スコアは 32/45/48/62/75/88 のみ。50-59・90-99・100+ は空帯。"}
    for scope,sdf in scopes.items():
        report["tables"][scope]={}; report["baseline"][scope]={}
        for period,pfn in periods.items():
            part=pfn(sdf)
            report["baseline"][scope][period]=metrics(part)
            report["tables"][scope][period]={band: metrics(part[band_mask(part,lo,hi)]) for band,lo,hi in BANDS}
            print(f"【{scope}/{period}】 baseline n={report['baseline'][scope][period]['n']} rec={report['baseline'][scope][period]['recovery']}")
            for band,_,_ in BANDS:
                m=report["tables"][scope][period][band]
                print(f"  {band}: n={m['n']} hit={m['hit_rate']} rec={m['recovery']} odds={m['avg_odds']} EV={m['avg_ev']}")
    out=ROOT/"data/ability_signal_reproducibility_report.json"
    # preserve prior conclusion block if regenerating numbers only
    prior=json.loads(out.read_text()) if out.exists() else {}
    if "conclusion" in prior: report["conclusion"]=prior["conclusion"]
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
