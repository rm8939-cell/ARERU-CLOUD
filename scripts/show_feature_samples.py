#!/usr/bin/env python3
"""生成された過去走特徴量を、実レース単位で目視確認するための出力。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

SHOW = [
    ("馬名", "馬名", 16),
    ("過去走数", "過去走", 6),
    ("前走内容", "前走評価", 9),
    ("近5走内容", "近5走評価", 10),
    ("上がり性能", "上がり評価", 10),
    ("同距離適性", "距離適性", 9),
    ("同コース適性", "コース適性", 10),
    ("馬場適性", "馬場適性", 9),
    ("ペース適性差", "ペース適性", 10),
]


def fmt(v, width: int) -> str:
    if pd.isna(v):
        return "―".rjust(width)
    if isinstance(v, float) and float(v).is_integer():
        return f"{int(v)}".rjust(width)
    if isinstance(v, (int, float)):
        return f"{v:+.3f}".rjust(width)
    return str(v)[:width].ljust(width)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/past_features/panel.csv")
    ap.add_argument("--races", type=int, default=10)
    ap.add_argument("--min-runners", type=int, default=8)
    args = ap.parse_args()

    d = pd.read_csv(args.panel, low_memory=False)
    d["race_id"] = d["race_id"].astype(str)
    size = d.groupby("race_id").size()
    cov = d.groupby("race_id")["過去走数"].apply(lambda s: (s > 0).mean())
    # 出走頭数が揃っていて、過去走が取れている率が高いレースから見る。
    ok = size[(size >= args.min_runners)].index.intersection(cov[cov >= 0.8].index)
    picks = sorted(ok)[:: max(1, len(ok) // args.races)][: args.races]

    for rid in picks:
        r = d[d["race_id"] == rid]
        head = r.iloc[0]
        surf = head.get("当日芝ダ") or "?"
        dist = head.get("当日距離")
        going = head.get("当日馬場")
        going_s = {0: "良", 1: "稍", 2: "重", 3: "不"}.get(
            int(going) if pd.notna(going) else -1, "?")
        print(f"\n=== {rid}  {str(head['date'])[:10]} {head.get('当日会場')} "
              f"{surf}{int(dist) if pd.notna(dist) else '?'}m {going_s} "
              f"{len(r)}頭 ===")
        print("".join(lbl.rjust(w) if i else lbl.ljust(w)
                      for i, (_, lbl, w) in enumerate(SHOW)))
        for _, row in r.iterrows():
            print("".join(
                fmt(row.get(col), w) if i else str(row.get(col))[:w].ljust(w)
                for i, (col, _, w) in enumerate(SHOW)
            ))

    print(f"\n表示 {len(picks)} レース / パネル全体 {d['race_id'].nunique()}R {len(d)}頭")
    have = (d["過去走数"] > 0).mean() * 100
    print(f"過去走を1走以上持つ馬の割合 {have:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
