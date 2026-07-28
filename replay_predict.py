import argparse
import gc
import os
import resource
import sys
from pathlib import Path

import pandas as pd
from areru_engine import build_predictions, parse_date

_BASE = Path(__file__).resolve().parent
DATA = _BASE / 'data'
OUT = DATA / 'predictions_by_date'
OUT.mkdir(parents=True, exist_ok=True)
RUNNERS = DATA / 'runners.csv'
LEGACY = DATA / 'score_test_data.csv'


def _rss_mb() -> float:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return float(ru) / (1024 * 1024)
        return float(ru) / 1024
    except Exception:
        return -1.0


def _log_mem(tag: str) -> None:
    mb = _rss_mb()
    print(f'[predict] MEM {mb:.1f} MB | {tag}', flush=True)


def load_runners():
    if RUNNERS.exists():
        return pd.read_csv(RUNNERS, encoding='utf-8-sig')
    if LEGACY.exists():
        # 移行過渡期のみ許容。新規処理は runners.csv を使う。
        return pd.read_csv(LEGACY, encoding='utf-8-sig')
    raise FileNotFoundError('data/runners.csv がありません。先に python3 refresh_data.py を実行してください')


def available_dates(runners):
    d = parse_date(runners['日付']).dropna().dt.strftime('%Y-%m-%d').unique().tolist()
    return sorted(d)


def run_date(target, runners, history):
    from ev_analysis import assert_predictions_finalized, ensure_predictions_file_finalized
    from race_sim import SIM_RUNS

    _log_mem(f'start {target} SIM_RUNS={SIM_RUNS}')
    # 対象日だけに絞ってから予想（全 runners をレースループ内で持ち回さない）
    day_mask = parse_date(runners['日付']).dt.strftime('%Y-%m-%d') == target
    day_runners = runners.loc[day_mask].copy()
    if day_runners.empty:
        raise ValueError(f'{target} の出走データがありません')
    result, scores = build_predictions(target, day_runners, history)
    assert_predictions_finalized(result, label=target)
    out_path = OUT / f'predictions_{target}.csv'
    result.to_csv(out_path, index=False, encoding='utf-8-sig')
    scores.to_csv(OUT / f'scores_{target}.csv', index=False, encoding='utf-8-sig')
    # 書き込み後も未確定ならその場で確定（途中失敗・旧ロジック混入の保険）
    if ensure_predictions_file_finalized(out_path):
        print(f'⚠ {target}: 未確定ランクを検出したため再確定して保存')
    print(f'✅ {target}: {len(result)}レース → {out_path}')
    _log_mem(f'done {target} races={len(result)}')
    del result, scores, day_runners
    gc.collect()
    return out_path


def main():
    ap = argparse.ArgumentParser(description='ARERU.EXE v2 過去日再現')
    ap.add_argument('date', nargs='?', help='YYYY-MM-DD')
    ap.add_argument('--all', action='store_true', help='利用可能な全開催日を一括生成')
    ap.add_argument('--list', action='store_true', help='利用可能日を表示')
    a = ap.parse_args()
    _log_mem('boot')
    runners = load_runners()
    _log_mem(f'runners loaded rows={len(runners)}')
    history = pd.read_csv(DATA / 'all_history.csv', encoding='utf-8-sig')
    _log_mem(f'history loaded rows={len(history)}')
    dates = available_dates(runners)
    if a.list:
        print('\n'.join(dates))
        return
    if a.all:
        for d in dates:
            run_date(d, runners, history)
        print(f'🔥 全{len(dates)}開催日 一括再現完了')
        return
    if not a.date:
        ap.error('日付または --all を指定してください')
    run_date(a.date, runners, history)
    # 子プロセス終了前に大きな DF を解放（親Webとの合算ピーク低減）
    try:
        del runners, history
    except Exception:
        pass
    gc.collect()
    _log_mem('exit')


if __name__ == '__main__':
    main()
