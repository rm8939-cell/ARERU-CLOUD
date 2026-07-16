import argparse
from pathlib import Path
import pandas as pd
from areru_engine import build_predictions, parse_date
from ticket_value import enrich_predictions, load_ticket_odds

DATA=Path('data'); OUT=DATA/'predictions_by_date'; OUT.mkdir(parents=True,exist_ok=True)

def available_dates(runners):
    d=parse_date(runners['日付']).dropna().dt.strftime('%Y-%m-%d').unique().tolist()
    return sorted(d)

def run_date(target,runners,history):
    result,scores=build_predictions(target,runners,history)
    try:
        result=enrich_predictions(result,scores,load_ticket_odds())
    except Exception as e:
        print(f'⚠️ オッズ接続スキップ: {e}')
    result.to_csv(OUT/f'predictions_{target}.csv',index=False,encoding='utf-8-sig')
    scores.to_csv(OUT/f'scores_{target}.csv',index=False,encoding='utf-8-sig')
    print(f'✅ {target}: {len(result)}レース → {OUT/f"predictions_{target}.csv"}', flush=True)
    return result

def main():
    ap=argparse.ArgumentParser(description='ARERU.EXE v2 過去日再現')
    ap.add_argument('date',nargs='?',help='YYYY-MM-DD')
    ap.add_argument('--all',action='store_true',help='利用可能な全開催日を一括生成')
    ap.add_argument('--list',action='store_true',help='利用可能日を表示')
    a=ap.parse_args()
    runners=pd.read_csv(DATA/'score_test_data.csv'); history=pd.read_csv(DATA/'all_history.csv')
    dates=available_dates(runners)
    if a.list: print('\n'.join(dates)); return
    if a.all:
        for d in dates: run_date(d,runners,history)
        print(f'🔥 全{len(dates)}開催日 一括再現完了'); return
    if not a.date: ap.error('日付または --all を指定してください')
    run_date(a.date,runners,history)
if __name__=='__main__': main()
