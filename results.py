"""P0-4: JRA/NAR レース結果取得・予想照合・回収率集計。

使い方:
  python3 results.py                  # 最新開催日（結果確定済み）を自動取得
  python3 results.py --date 2026-07-12
  python3 results.py --dates 2026-07-11 2026-07-12
  python3 results.py --source jra     # jra / nar / all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from areru_engine import clean_name
from netkeiba_client import NetkeibaClient

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

DATA = Path("data")
PRED_DIR = DATA / "predictions_by_date"
RESULTS_CSV = DATA / "results.csv"
PAYOUTS_CSV = DATA / "payouts.csv"
ANALYSIS_CSV = DATA / "analysis_result.csv"
UNIT = 100  # 1点あたりの投資額（円）

RESULT_COLS = [
    "race_id", "date", "レース", "開催地", "馬名", "馬番",
    "着順", "人気", "確定オッズ", "source",
]
PAYOUT_COLS = [
    "race_id", "date", "レース", "開催地", "bet_type",
    "combination", "payout", "ninki", "source",
]
ANALYSIS_COLS = [
    "date", "race", "race_id", "開催地", "bet_type", "勝負ランク",
    "推奨券種", "購入対象",
    "prediction", "result", "hit", "payout", "investment", "profit", "roi",
]


def _clean(s) -> str:
    return clean_name(s) if s is not None else ""


def _norm_race_id(x) -> str:
    """CSV 読み書きで float 化した race_id を 12桁文字列へ揃える。"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    try:
        if re.fullmatch(r"\d+\.0+", s):
            s = str(int(float(s)))
    except Exception:
        pass
    return s


def _load_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


def _merge_by_race(old: pd.DataFrame, new: pd.DataFrame, key: str = "race_id") -> pd.DataFrame:
    if new.empty:
        return old
    if old.empty:
        return new.reset_index(drop=True)
    drop_ids = set(new[key].astype(str))
    kept = old[~old[key].astype(str).isin(drop_ids)]
    return pd.concat([kept, new], ignore_index=True)


def prediction_dates() -> list[str]:
    found = []
    for f in PRED_DIR.glob("predictions_*.csv"):
        m = re.fullmatch(r"predictions_(\d{4}-\d{2}-\d{2})\.csv", f.name)
        if m:
            found.append(m.group(1))
    return sorted(found, reverse=True)


def resolve_target_dates(client: NetkeibaClient, dates: list[str] | None, latest: bool) -> list[str]:
    if dates:
        return sorted({d.replace("/", "-") for d in dates})
    if latest:
        # 結果が出ている直近開催日を優先（未来日の予想だけがある場合はスキップ）
        for d in client.discover_kaisai_dates(lookback=21, lookahead=2):
            ymd = d.replace("-", "")
            try:
                ids = client.list_race_ids(ymd)
            except Exception:
                ids = []
            if not ids:
                continue
            detail = client.fetch_result_detail(ids[0])
            if detail.get("horses"):
                return [d]
        # フォールバック: 予想がある過去日
        for d in prediction_dates():
            return [d]
    return prediction_dates()[:1]


def fetch_date_results(client: NetkeibaClient, date_str: str, source: str = "jra") -> tuple[pd.DataFrame, pd.DataFrame]:
    ymd = date_str.replace("-", "")
    print(f"\n🏁 {source.upper()} {date_str} 結果取得")
    try:
        race_ids = client.list_race_ids(ymd) if source == "jra" else _list_nar_race_ids(client, ymd)
    except Exception as e:
        print(f"⚠️ レース一覧取得失敗: {e}")
        return pd.DataFrame(columns=RESULT_COLS), pd.DataFrame(columns=PAYOUT_COLS)

    if not race_ids:
        print("⚠️ レースなし")
        return pd.DataFrame(columns=RESULT_COLS), pd.DataFrame(columns=PAYOUT_COLS)

    horse_rows = []
    pay_rows = []
    for i, rid in enumerate(race_ids, 1):
        print(f"  [{i}/{len(race_ids)}] {rid}")
        try:
            detail = client.fetch_result_detail(rid, source=source)
        except Exception as e:
            print(f"    ⚠️ 取得失敗: {e}")
            continue
        if not detail.get("horses"):
            print("    ⚠️ 結果未確定")
            continue
        race_no = detail.get("race_no") or ""
        venue = detail.get("venue") or ""
        d = detail.get("date") or date_str
        rid_s = _norm_race_id(rid)
        for h in detail["horses"]:
            horse_rows.append({
                "race_id": rid_s,
                "date": d,
                "レース": race_no,
                "開催地": venue,
                "馬名": h.get("馬名", ""),
                "馬番": h.get("馬番", ""),
                "着順": h.get("着順", ""),
                "人気": h.get("人気", ""),
                "確定オッズ": h.get("確定オッズ", ""),
                "source": source,
            })
        for p in detail.get("payouts") or []:
            pay_rows.append({
                "race_id": rid_s,
                "date": d,
                "レース": race_no,
                "開催地": venue,
                "bet_type": p.get("bet_type", ""),
                "combination": p.get("combination", ""),
                "payout": p.get("payout") if p.get("payout") is not None else "",
                "ninki": p.get("ninki", ""),
                "source": source,
            })
        print(f"    ✅ {len(detail['horses'])}頭 / 払戻 {len(detail.get('payouts') or [])}件")

    return pd.DataFrame(horse_rows, columns=RESULT_COLS), pd.DataFrame(pay_rows, columns=PAYOUT_COLS)


def _list_nar_race_ids(client: NetkeibaClient, ymd: str) -> list[str]:
    """地方: nar.netkeiba の開催一覧から race_id を拾う。"""
    url = f"https://nar.netkeiba.com/top/race_list_sub.html?kaisai_date={ymd}"
    try:
        soup = client._get(url)
    except Exception:
        return []
    return sorted(set(re.findall(r"race_id=(\d{8,12})", str(soup))))


def _ban_map(results: pd.DataFrame, race_id: str) -> dict[str, str]:
    rid = _norm_race_id(race_id)
    g = results[results["race_id"].map(_norm_race_id) == rid]
    out = {}
    for _, r in g.iterrows():
        ban = str(r["馬番"]).strip()
        if not ban.isdigit():
            continue
        out[_clean(r["馬名"])] = str(int(ban))
    return out


def _finish_map(results: pd.DataFrame, race_id: str) -> dict[str, str]:
    rid = _norm_race_id(race_id)
    g = results[results["race_id"].map(_norm_race_id) == rid]
    return {_clean(r["馬名"]): str(r["着順"]).strip() for _, r in g.iterrows()}


def _payout_index(payouts: pd.DataFrame, race_id: str) -> dict[str, list[dict]]:
    rid = _norm_race_id(race_id)
    g = payouts[payouts["race_id"].map(_norm_race_id) == rid]
    idx: dict[str, list[dict]] = {}
    for _, r in g.iterrows():
        bt = str(r["bet_type"])
        idx.setdefault(bt, []).append({
            "combination": str(r["combination"]),
            "payout": int(float(r["payout"])) if str(r["payout"]).strip() not in ("", "nan") else 0,
            "ninki": str(r["ninki"]),
        })
    return idx


def _combo_key(bans: list[str], ordered: bool = False) -> str:
    nums = []
    for b in bans:
        s = str(b).strip()
        if s.isdigit():
            nums.append(str(int(s)))
    if not nums:
        return ""
    if ordered:
        return "-".join(nums)
    return "-".join(sorted(nums, key=lambda x: int(x)))


def _parse_ticket_horses(text: str) -> list[list[str]]:
    """『A － B（仮想的中…）｜C － D － E（…）』→ [[A,B],[C,D,E]]"""
    raw = str(text or "").strip()
    if not raw or raw in {"見送り", "なし", "nan", "None"}:
        return []
    tickets = []
    for part in raw.split("｜"):
        core = re.split(r"[（(]", part, maxsplit=1)[0].strip()
        if not core:
            continue
        horses = [_clean(x) for x in re.split(r"\s*[－\-−–—]\s*", core) if _clean(x)]
        if horses:
            tickets.append(horses)
    return tickets


def _marks_from_prediction(row: dict) -> list[tuple[str, str]]:
    marks = [("◎", _clean(row.get("本命", "")))]
    try:
        data = json.loads(row.get("印データ", "[]") or "[]")
    except Exception:
        data = []
    for x in data:
        marks.append((str(x.get("印", "")), _clean(x.get("馬名", ""))))
    return [(m, n) for m, n in marks if n]


def _eval_win(name: str, finishes: dict[str, str], pays: dict[str, list[dict]], ban_map: dict[str, str]) -> dict:
    finish = finishes.get(_clean(name), "")
    ban = ban_map.get(_clean(name), "")
    hit = finish == "1"
    payout = 0
    result_txt = ""
    for p in pays.get("単勝", []):
        result_txt = p["combination"]
        if hit and ban and str(p["combination"]).lstrip("0") == str(ban).lstrip("0"):
            payout = p["payout"]
            break
        if hit and str(p["combination"]) == str(ban):
            payout = p["payout"]
            break
    if hit and payout == 0:
        # 払戻表が取れない場合は確定オッズ×100で近似しない（実払戻のみ）
        payout = 0
    return {
        "prediction": name,
        "result": f"1着:{result_txt}" if result_txt else (f"{finish}着" if finish else "結果なし"),
        "hit": int(hit and payout > 0) if pays.get("単勝") else int(hit),
        "payout": payout if hit else 0,
        "investment": UNIT,
    }


def _eval_combo(
    kind: str,
    tickets: list[list[str]],
    ban_map: dict[str, str],
    pays: dict[str, list[dict]],
    ordered: bool = False,
) -> dict:
    if not tickets:
        return {
            "prediction": "見送り",
            "result": "",
            "hit": 0,
            "payout": 0,
            "investment": 0,
        }
    pay_rows = pays.get(kind, [])
    pay_map = {}
    for p in pay_rows:
        key = _combo_key(str(p["combination"]).split("-"), ordered=ordered)
        pay_map[key] = p["payout"]
    result_disp = " / ".join(p["combination"] for p in pay_rows) if pay_rows else ""
    total_pay = 0
    hit_any = False
    pred_parts = []
    for horses in tickets:
        bans = [ban_map.get(_clean(h), "") for h in horses]
        key = _combo_key(bans, ordered=ordered)
        pred_parts.append("－".join(horses))
        if key and key in pay_map:
            hit_any = True
            total_pay += pay_map[key]
        elif kind == "ワイド" and key:
            # ワイドは組合せ表記ゆれに備えて再キー化
            for pk, pv in pay_map.items():
                if set(pk.split("-")) == set(key.split("-")):
                    hit_any = True
                    total_pay += pv
                    break
    return {
        "prediction": "｜".join(pred_parts),
        "result": result_disp,
        "hit": int(hit_any),
        "payout": total_pay if hit_any else 0,
        "investment": UNIT * len(tickets),
    }


def _resolve_race_id(prow: dict, day_results: pd.DataFrame) -> str:
    """netkeiba の 12桁 ID を優先。旧JRA URL の場合は開催地+レースで解決。"""
    rid = _norm_race_id(prow.get("race_id", ""))
    if rid.isdigit() and len(rid) == 12:
        return rid
    venue = str(prow.get("開催地", "")).strip()
    race_no = prow.get("レース", "")
    try:
        race_i = int(float(race_no))
    except Exception:
        return ""
    g = day_results[
        (day_results["開催地"].astype(str) == venue)
        & (pd.to_numeric(day_results["レース"], errors="coerce") == race_i)
    ]
    if g.empty:
        return ""
    return _norm_race_id(g.iloc[0]["race_id"])


def analyze_predictions(
    results: pd.DataFrame,
    payouts: pd.DataFrame,
    dates: list[str] | None = None,
) -> pd.DataFrame:
    """predictions_by_date と結果を照合し analysis_result.csv 用 DataFrame を返す。"""
    rows = []
    targets = dates or sorted(results["date"].dropna().unique().tolist())
    for d in targets:
        path = PRED_DIR / f"predictions_{d}.csv"
        if not path.exists():
            print(f"↪️  予想なし: {path.name}")
            continue
        pred = pd.read_csv(path, encoding="utf-8-sig").fillna("")
        day_results = results[results["date"].astype(str) == str(d)]
        if day_results.empty:
            continue
        for _, prow in pred.iterrows():
            rid = _resolve_race_id(prow.to_dict(), day_results)
            if not rid:
                continue
            finishes = _finish_map(results, rid)
            if not finishes:
                continue
            bans = _ban_map(results, rid)
            pays = _payout_index(payouts, rid)
            venue = str(prow.get("開催地", ""))
            race_no = prow.get("レース", "")
            race_label = f"{venue}{int(float(race_no)):02d}R" if str(race_no).replace('.','',1).isdigit() else str(race_no)
            ai_rank = str(prow.get("勝負ランク", "") or "").upper()
            recommend = str(prow.get("推奨券種", "") or "").strip()

            # 本命（単勝）
            main = _clean(prow.get("本命", ""))
            if main:
                ev = _eval_win(main, finishes, pays, bans)
                rows.append(_analysis_row(d, race_label, rid, venue, "本命", ev, ai_rank, recommend))

            # ワイド / 馬連 / 三連複
            for kind, col, ordered in [
                ("ワイド", "ワイド買い目", False),
                ("馬連", "馬連買い目", False),
                ("三連複", "三連複買い目", False),
            ]:
                tickets = _parse_ticket_horses(prow.get(col, ""))
                # 見送り判定でも買い目があれば仮想検証する
                ev = _eval_combo(kind, tickets, bans, pays, ordered=ordered)
                if ev["investment"] > 0 or tickets:
                    rows.append(_analysis_row(d, race_label, rid, venue, kind, ev, ai_rank, recommend))

            # 三連単: ◎→○→▲ の順で1点
            marks = dict(_marks_from_prediction(prow.to_dict()))
            trio_names = [marks.get("◎", ""), marks.get("○", ""), marks.get("▲", "")]
            if all(trio_names):
                ev = _eval_combo("三連単", [trio_names], bans, pays, ordered=True)
                rows.append(_analysis_row(d, race_label, rid, venue, "三連単", ev, ai_rank, recommend))

    return pd.DataFrame(rows, columns=ANALYSIS_COLS)


def _analysis_row(
    date: str,
    race: str,
    race_id: str,
    venue: str,
    bet_type: str,
    ev: dict,
    ai_rank: str = "",
    recommend: str = "",
) -> dict:
    inv = int(ev.get("investment") or 0)
    pay = int(ev.get("payout") or 0)
    profit = pay - inv
    roi = round((pay / inv) * 100, 1) if inv > 0 else 0.0
    rec = str(recommend or "").strip()
    # 購入対象 = そのレースの推奨券種（AIが主戦として出す馬券）1件
    is_purchase = 1 if rec and bet_type == rec else 0
    return {
        "date": date,
        "race": race,
        "race_id": _norm_race_id(race_id),
        "開催地": venue,
        "bet_type": bet_type,
        "勝負ランク": str(ai_rank or "").upper(),
        "推奨券種": rec,
        "購入対象": is_purchase,
        "prediction": ev.get("prediction", ""),
        "result": ev.get("result", ""),
        "hit": int(ev.get("hit") or 0),
        "payout": pay,
        "investment": inv,
        "profit": profit,
        "roi": roi,
    }


def summarize(analysis: pd.DataFrame) -> None:
    if analysis.empty:
        print("集計対象なし")
        return
    inv = analysis["investment"].sum()
    pay = analysis["payout"].sum()
    hits = analysis["hit"].sum()
    n = len(analysis)
    print("\n====================")
    print("📊 結果検証サマリー")
    print(f"  件数: {n}")
    print(f"  的中: {hits} ({(hits/n*100):.1f}%)" if n else "  的中: 0")
    print(f"  投資: {inv:,}円")
    print(f"  払戻: {pay:,}円")
    print(f"  収支: {pay-inv:+,}円")
    print(f"  回収率: {(pay/inv*100):.1f}%" if inv else "  回収率: -")
    print("  --- 券種別 ---")
    for bt, g in analysis.groupby("bet_type"):
        i, p = g["investment"].sum(), g["payout"].sum()
        h = g["hit"].mean() * 100 if len(g) else 0
        print(f"  {bt}: 的中率{h:.1f}% / 回収率{(p/i*100) if i else 0:.1f}% / 収支{p-i:+,}円")
    if "勝負ランク" in analysis.columns:
        print("  --- AIランク別（購入対象のみ） ---")
        base = analysis
        if "購入対象" in analysis.columns:
            base = analysis[pd.to_numeric(analysis["購入対象"], errors="coerce").fillna(0).astype(int) == 1]
        for rk in ["S", "A", "B", "C"]:
            g = base[base["勝負ランク"].astype(str).str.upper() == rk]
            if g.empty:
                print(f"  {rk}: 対象0件")
                continue
            i, p = g["investment"].sum(), g["payout"].sum()
            h = g["hit"].mean() * 100 if len(g) else 0
            print(f"  {rk}: 対象{len(g)} / 的中率{h:.1f}% / 回収率{(p/i*100) if i else 0:.1f}% / 収支{p-i:+,}円")
        print("  --- ランク×券種 ---")
        label = {"本命": "単勝", "単勝": "単勝"}
        order = ["単勝", "ワイド", "馬連", "三連複", "三連単"]
        for rk in ["S", "A", "B", "C"]:
            g_rank = analysis[analysis["勝負ランク"].astype(str).str.upper() == rk]
            if g_rank.empty:
                continue
            print(f"  [{rk}]")
            buckets = {}
            for bt, g in g_rank.groupby("bet_type"):
                name = label.get(str(bt), str(bt))
                buckets[name] = g
            for name in order:
                g = buckets.get(name)
                if g is None or g.empty:
                    print(f"    {name}: 対象0件")
                    continue
                i, p = g["investment"].sum(), g["payout"].sum()
                h = g["hit"].mean() * 100 if len(g) else 0
                print(f"    {name}: 対象{len(g)} / 的中率{h:.1f}% / 回収率{(p/i*100) if i else 0:.1f}% / 収支{p-i:+,}円")


def run(dates: list[str] | None, source: str, latest: bool, skip_fetch: bool) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    client = NetkeibaClient(sleep=0.25)
    sources = ["jra", "nar"] if source == "all" else [source]

    results = _load_csv(RESULTS_CSV, RESULT_COLS)
    # 旧形式（JRA URL）は netkeiba ID と混在させない
    if not results.empty and results["race_id"].astype(str).str.startswith("http").any():
        print("↪️  旧形式 results.csv を置き換えます（netkeiba race_id へ移行）")
        results = pd.DataFrame(columns=RESULT_COLS)
    payouts = _load_csv(PAYOUTS_CSV, PAYOUT_COLS)

    target_dates: list[str] = []
    if not skip_fetch:
        for src in sources:
            if src == "jra":
                target_dates = resolve_target_dates(client, dates, latest or not dates)
            else:
                # NAR は指定日 or JRAと同日を試行
                target_dates = dates or target_dates or resolve_target_dates(client, None, True)
            for d in target_dates:
                h, p = fetch_date_results(client, d, source=src)
                results = _merge_by_race(results, h)
                payouts = _merge_by_race(payouts, p)
    else:
        target_dates = dates or sorted(results["date"].astype(str).unique().tolist(), reverse=True)[:1]

    if not results.empty and "race_id" in results.columns:
        results["race_id"] = results["race_id"].map(_norm_race_id)
    if not payouts.empty and "race_id" in payouts.columns:
        payouts["race_id"] = payouts["race_id"].map(_norm_race_id)
    # race_id を文字列のまま保存（Excel/pandas の float 化による照合ズレ防止）
    results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    payouts.to_csv(PAYOUTS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n💾 {RESULTS_CSV} ({len(results)}行)")
    print(f"💾 {PAYOUTS_CSV} ({len(payouts)}行)")

    # 予想がある開催日 × 結果がある開催日を照合
    result_dates = set(results["date"].astype(str).unique())
    analyze_dates = sorted(set(prediction_dates()) & result_dates)
    if not analyze_dates:
        analyze_dates = sorted(result_dates)
    analysis = analyze_predictions(results, payouts, dates=analyze_dates)
    if not analysis.empty and "race_id" in analysis.columns:
        analysis["race_id"] = analysis["race_id"].map(_norm_race_id)
    analysis.to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")
    print(f"💾 {ANALYSIS_CSV} ({len(analysis)}行)")
    summarize(analysis)


def main():
    ap = argparse.ArgumentParser(description="P0-4 結果検証パイプライン")
    ap.add_argument("--date", help="YYYY-MM-DD")
    ap.add_argument("--dates", nargs="*", help="複数日")
    ap.add_argument("--source", choices=["jra", "nar", "all"], default="jra")
    ap.add_argument("--latest", action="store_true", help="最新開催日（結果確定）を自動選択")
    ap.add_argument("--skip-fetch", action="store_true", help="取得をスキップして照合のみ")
    args = ap.parse_args()
    dates = []
    if args.date:
        dates.append(args.date)
    if args.dates:
        dates.extend(args.dates)
    latest = args.latest or not dates
    run(dates=dates or None, source=args.source, latest=latest, skip_fetch=args.skip_fetch)


if __name__ == "__main__":
    main()
