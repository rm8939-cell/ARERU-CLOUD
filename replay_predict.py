import argparse
from pathlib import Path
import pandas as pd
from areru_engine import build_predictions, parse_date

DATA=Path('data'); OUT=DATA/'predictions_by_date'; OUT.mkdir(parents=True,exist_ok=True)
RUNNERS=DATA/'runners.csv'
LEGACY=DATA/'score_test_data.csv'


def load_runners():
    if RUNNERS.exists():
        return pd.read_csv(RUNNERS, encoding='utf-8-sig')
    if LEGACY.exists():
        # 移行過渡期のみ許容。新規処理は runners.csv を使う。
        return pd.read_csv(LEGACY, encoding='utf-8-sig')
    raise FileNotFoundError('data/runners.csv がありません。先に python3 refresh_data.py を実行してください')


def available_dates(runners):
    d=parse_date(runners['日付']).dropna().dt.strftime('%Y-%m-%d').unique().tolist()
    return sorted(d)

def run_date(target,runners,history):
    result,scores=build_predictions(target,runners,history)
    result.to_csv(OUT/f'predictions_{target}.csv',index=False,encoding='utf-8-sig')
    scores.to_csv(OUT/f'scores_{target}.csv',index=False,encoding='utf-8-sig')
    print(f'✅ {target}: {len(result)}レース → {OUT/f"predictions_{target}.csv"}')
    return result

def main():
    ap=argparse.ArgumentParser(description='ARERU.EXE v2 過去日再現')
    ap.add_argument('date',nargs='?',help='YYYY-MM-DD')
    ap.add_argument('--all',action='store_true',help='利用可能な全開催日を一括生成')
    ap.add_argument('--list',action='store_true',help='利用可能日を表示')
    a=ap.parse_args()
    runners=load_runners(); history=pd.read_csv(DATA/'all_history.csv')
    dates=available_dates(runners)
    if a.list: print('\n'.join(dates)); return
    if a.all:
        for d in dates: run_date(d,runners,history)
        print(f'🔥 全{len(dates)}開催日 一括再現完了'); return
    if not a.date: ap.error('日付または --all を指定してください')
    run_date(a.date,runners,history)
if __name__=='__main__': main()
