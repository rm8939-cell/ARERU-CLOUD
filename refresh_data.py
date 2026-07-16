"""P0-2: 日付更新・最新データ生成・Web反映パイプライン。

1) netkeiba から開催日レースを取得
2) score_test_data / all_history / results / odds を更新
3) replay_predict で predictions_by_date を再生成
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from areru_engine import clean_name, parse_date
from netkeiba_client import (
    ODDS_QUINELLA,
    ODDS_TRIO,
    ODDS_WIDE,
    ODDS_WIN,
    NetkeibaClient,
    iso_from_yyyymmdd,
    yyyymmdd,
)

DATA = Path("data")
ARCH = DATA / "predictions_by_date"
CACHE = DATA / "horse_history_cache"
META = DATA / "refresh_meta.json"


def discover_dates(around: date | None = None, back: int = 14, forward: int = 7) -> list[str]:
    """周辺日程のうち、実際にレースがある日を列挙。"""
    around = around or date.today()
    client = NetkeibaClient(sleep=0.15)
    found = []
    for i in range(-back, forward + 1):
        d = around + timedelta(days=i)
        key = d.strftime("%Y%m%d")
        try:
            ids = client.list_race_ids(key)
        except Exception as e:
            print(f"⚠️ {key} 一覧失敗: {e}")
            continue
        if ids:
            iso = d.isoformat()
            found.append(iso)
            print(f"✅ 開催あり {iso} / {len(ids)}R", flush=True)
    return found


def _load_score() -> pd.DataFrame:
    p = DATA / "score_test_data.csv"
    if p.exists():
        return pd.read_csv(p)
    cols = [
        "race_id",
        "日付",
        "レース",
        "馬名",
        "実着順",
        "着順1",
        "人気1",
        "着順2",
        "人気2",
        "着順3",
        "人気3",
        "着順4",
        "人気4",
        "着順5",
        "人気5",
        "horse_id",
        "馬番",
    ]
    return pd.DataFrame(columns=cols)


def _load_history() -> pd.DataFrame:
    p = DATA / "all_history.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(
        columns=[
            "今回レース",
            "馬名",
            "年月日",
            "場",
            "レース名",
            "距離",
            "馬場",
            "頭数",
            "人気",
            "着順",
            "騎手",
            "斤量",
            "馬体重",
            "タイム",
            "着差",
            "horse_id",
            "race_id",
            "日付",
        ]
    )


def _horse_cache_path(horse_id: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{horse_id}.json"


def get_horse_history(client: NetkeibaClient, horse_id: str, use_cache: bool = True) -> list[dict]:
    if not horse_id:
        return []
    path = _horse_cache_path(horse_id)
    if use_cache and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    rows = client.fetch_horse_history(horse_id, limit=16)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def build_runner_row(race_date: str, race_id: str, race_number: int, horse: dict, history: list[dict], finish=None):
    prior = [h for h in history if h.get("日付") and h["日付"] < race_date]
    prior = sorted(prior, key=lambda x: x["日付"], reverse=True)[:5]
    row = {
        "race_id": race_id,
        "日付": race_date,
        "レース": race_number,
        "馬名": horse["馬名"],
        "実着順": finish if finish not in (None, "", "NaN") else "",
        "horse_id": horse.get("horse_id", ""),
        "馬番": horse.get("馬番", ""),
    }
    for i in range(1, 6):
        if i <= len(prior):
            row[f"着順{i}"] = prior[i - 1].get("着順", "")
            row[f"人気{i}"] = prior[i - 1].get("人気", "")
        else:
            row[f"着順{i}"] = ""
            row[f"人気{i}"] = ""
    return row


def history_rows_for_horse(race_number: int, horse_name: str, horse_id: str, race_id: str, history: list[dict], race_date: str):
    rows = []
    for h in history:
        if h.get("日付") and h["日付"] >= race_date:
            continue  # 未来/当日リーク防止
        rows.append(
            {
                "今回レース": race_number,
                "馬名": horse_name,
                "年月日": h.get("年月日", ""),
                "場": h.get("場", ""),
                "レース名": h.get("レース名", ""),
                "距離": h.get("距離", ""),
                "馬場": h.get("馬場", ""),
                "頭数": h.get("頭数", ""),
                "人気": h.get("人気", ""),
                "着順": h.get("着順", ""),
                "騎手": h.get("騎手", ""),
                "斤量": h.get("斤量", ""),
                "馬体重": h.get("馬体重", ""),
                "タイム": h.get("タイム", ""),
                "着差": h.get("着差", ""),
                "horse_id": horse_id,
                "race_id": race_id,
                "日付": h.get("日付", ""),
            }
        )
    return rows


def refresh_date(target: str, client: NetkeibaClient | None = None, use_cache: bool = True) -> dict:
    client = client or NetkeibaClient()
    key = yyyymmdd(target)
    race_ids = client.list_race_ids(key)
    if not race_ids:
        raise ValueError(f"{target} にレースがありません")

    print(f"\n📅 {target} / {len(race_ids)}R 更新開始", flush=True)
    score_rows = []
    history_rows = []
    result_rows = []
    odds_rows = []
    ticket_odds_rows = []
    payout_rows = []

    for idx, race_id in enumerate(race_ids, start=1):
        meta = client.parse_race_id(race_id)
        race_number = meta["race_number"]
        print(f"🏇 {idx}/{len(race_ids)} {meta['venue']}{race_number}R ({race_id})", flush=True)

        result = client.fetch_result(race_id)
        race_date = result.get("日付") or target
        runners_by_name = {}
        if result.get("has_result"):
            for r in result["runners"]:
                runners_by_name[clean_name(r["馬名"])] = r
                result_rows.append(
                    {
                        "race_id": race_id,
                        "日付": race_date,
                        "レース": race_number,
                        "開催地": meta["venue"],
                        "馬名": r["馬名"],
                        "着順": r.get("着順", ""),
                        "馬番": r.get("馬番", ""),
                        "人気": r.get("人気", ""),
                        "単勝オッズ": r.get("単勝オッズ", ""),
                        "horse_id": r.get("horse_id", ""),
                    }
                )
            for kind, items in (result.get("payouts") or {}).items():
                for item in items:
                    payout_rows.append(
                        {
                            "race_id": race_id,
                            "日付": race_date,
                            "レース": race_number,
                            "券種": kind,
                            "組合せ": item.get("組合せ", ""),
                            "払戻": item.get("払戻", ""),
                        }
                    )

        shutuba = client.fetch_shutuba(race_id)
        horses = shutuba.get("horses") or []
        if not horses and result.get("runners"):
            horses = [
                {
                    "馬名": r["馬名"],
                    "horse_id": r.get("horse_id", ""),
                    "馬番": r.get("馬番", ""),
                }
                for r in result["runners"]
            ]

        # 単勝オッズ
        win_odds = client.fetch_odds(race_id, ODDS_WIN)
        umaban_to_odds = {}
        if win_odds and isinstance(win_odds.get("odds"), dict):
            block = win_odds["odds"].get("1") or next(iter(win_odds["odds"].values()), {})
            for umaban, vals in block.items():
                try:
                    umaban_to_odds[str(int(umaban))] = {
                        "単勝オッズ": float(str(vals[0]).replace(",", "")),
                        "人気": int(float(vals[2])) if len(vals) > 2 and str(vals[2]).replace(".", "", 1).isdigit() else "",
                    }
                except Exception:
                    continue

        # 券種別オッズ（取れた時だけ）
        for kind, otype in (("ワイド", ODDS_WIDE), ("馬連", ODDS_QUINELLA), ("三連複", ODDS_TRIO)):
            payload = client.fetch_odds(race_id, otype)
            if not payload:
                continue
            block = payload["odds"]
            # type key may be string of type
            inner = None
            if isinstance(block, dict):
                inner = block.get(str(otype)) or next(iter(block.values()), None)
            if not isinstance(inner, dict):
                continue
            for combo, vals in inner.items():
                try:
                    oddsv = float(str(vals[0]).replace(",", ""))
                except Exception:
                    continue
                ticket_odds_rows.append(
                    {
                        "race_id": race_id,
                        "日付": race_date,
                        "レース": race_number,
                        "券種": kind,
                        # 先頭ゼロ保持（CSVで数値化されないよう文字列のまま）
                        "組合せ": f"'{combo}" if str(combo)[:1].isdigit() else str(combo),
                        "オッズ": oddsv,
                        "人気": vals[2] if len(vals) > 2 else "",
                    }
                )

        for horse in horses:
            hid = horse.get("horse_id", "")
            hist = get_horse_history(client, hid, use_cache=use_cache)
            finish = None
            matched = runners_by_name.get(clean_name(horse["馬名"]))
            if matched:
                finish = matched.get("着順")
                if not horse.get("馬番"):
                    horse["馬番"] = matched.get("馬番", "")
            score_rows.append(
                build_runner_row(race_date, race_id, race_number, horse, hist, finish=finish)
            )
            history_rows.extend(
                history_rows_for_horse(race_number, horse["馬名"], hid, race_id, hist, race_date)
            )
            umaban = str(horse.get("馬番") or "").lstrip("0") or str(horse.get("馬番") or "")
            # normalize umaban key
            umaban_key = ""
            if str(horse.get("馬番") or "").isdigit():
                umaban_key = str(int(horse["馬番"]))
            o = umaban_to_odds.get(umaban_key) or umaban_to_odds.get(umaban)
            if o:
                odds_rows.append(
                    {
                        "race_id": race_id,
                        "日付": race_date,
                        "レース": race_number,
                        "馬名": horse["馬名"],
                        "馬番": horse.get("馬番", ""),
                        "単勝オッズ": o["単勝オッズ"],
                        "人気": o["人気"],
                        "horse_id": hid,
                    }
                )
            elif matched and matched.get("単勝オッズ"):
                try:
                    odds_rows.append(
                        {
                            "race_id": race_id,
                            "日付": race_date,
                            "レース": race_number,
                            "馬名": horse["馬名"],
                            "馬番": horse.get("馬番", ""),
                            "単勝オッズ": float(str(matched["単勝オッズ"]).replace(",", "")),
                            "人気": matched.get("人気", ""),
                            "horse_id": hid,
                        }
                    )
                except Exception:
                    pass

    # merge into master CSVs
    score = _load_score()
    score = score[~parse_date(score["日付"]).dt.strftime("%Y-%m-%d").eq(target)].copy() if len(score) else score
    score = pd.concat([score, pd.DataFrame(score_rows)], ignore_index=True)
    score.to_csv(DATA / "score_test_data.csv", index=False, encoding="utf-8-sig")

    # history: append unique by horse_id+日付+レース名 roughly; keep file usable
    hist_df = _load_history()
    new_hist = pd.DataFrame(history_rows)
    if len(new_hist):
        if "horse_id" not in hist_df.columns:
            hist_df["horse_id"] = ""
        if "日付" not in hist_df.columns:
            hist_df["日付"] = ""
        hist_df = pd.concat([hist_df, new_hist], ignore_index=True)
        hist_df = hist_df.drop_duplicates(
            subset=["馬名", "年月日", "レース名", "着順"], keep="last"
        )
        hist_df.to_csv(DATA / "all_history.csv", index=False, encoding="utf-8-sig")

    if result_rows:
        rp = DATA / "results.csv"
        old = pd.read_csv(rp) if rp.exists() else pd.DataFrame()
        if len(old):
            old = old[~old["race_id"].astype(str).isin([r["race_id"] for r in result_rows])]
        pd.concat([old, pd.DataFrame(result_rows)], ignore_index=True).to_csv(
            rp, index=False, encoding="utf-8-sig"
        )

    if odds_rows:
        op = DATA / "odds.csv"
        old = pd.read_csv(op) if op.exists() else pd.DataFrame()
        if len(old) and "race_id" in old.columns:
            old = old[~old["race_id"].astype(str).isin([r["race_id"] for r in odds_rows])]
        pd.concat([old, pd.DataFrame(odds_rows)], ignore_index=True).to_csv(
            op, index=False, encoding="utf-8-sig"
        )

    if ticket_odds_rows:
        tp = DATA / "ticket_odds.csv"
        old = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
        if len(old):
            old = old[~old["race_id"].astype(str).isin({r["race_id"] for r in ticket_odds_rows})]
        pd.concat([old, pd.DataFrame(ticket_odds_rows)], ignore_index=True).to_csv(
            tp, index=False, encoding="utf-8-sig"
        )

    if payout_rows:
        pp = DATA / "payouts.csv"
        old = pd.read_csv(pp) if pp.exists() else pd.DataFrame()
        if len(old):
            old = old[~old["race_id"].astype(str).isin({r["race_id"] for r in payout_rows})]
        pd.concat([old, pd.DataFrame(payout_rows)], ignore_index=True).to_csv(
            pp, index=False, encoding="utf-8-sig"
        )

    meta = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "date": target,
        "races": len(race_ids),
        "runners": len(score_rows),
        "results": len(result_rows),
        "odds": len(odds_rows),
        "ticket_odds": len(ticket_odds_rows),
        "payouts": len(payout_rows),
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"✅ {target} 保存: runners={len(score_rows)} results={len(result_rows)} "
        f"odds={len(odds_rows)} ticket_odds={len(ticket_odds_rows)}",
        flush=True,
    )
    return meta


def regenerate_predictions(dates: list[str] | None = None):
    from replay_predict import run_date

    runners = pd.read_csv(DATA / "score_test_data.csv")
    history = pd.read_csv(DATA / "all_history.csv")
    avail = sorted(parse_date(runners["日付"]).dropna().dt.strftime("%Y-%m-%d").unique())
    targets = dates or avail
    for d in targets:
        if d not in avail:
            print(f"⚠️ {d} は score_test_data に無いのでスキップ")
            continue
        print(f"🧠 予想生成 {d}")
        run_date(d, runners, history)


def main():
    ap = argparse.ArgumentParser(description="ARERU.CLOUD P0-2 データ更新")
    ap.add_argument("--discover", action="store_true", help="開催日を探索して表示")
    ap.add_argument("--dates", nargs="*", help="更新する日付 YYYY-MM-DD")
    ap.add_argument("--auto", action="store_true", help="周辺開催日を自動更新")
    ap.add_argument("--predict", action="store_true", help="predictions_by_date を再生成")
    ap.add_argument("--predict-only", action="store_true", help="予想生成のみ")
    ap.add_argument("--no-cache", action="store_true", help="馬履歴キャッシュを使わない")
    ap.add_argument("--back", type=int, default=14)
    ap.add_argument("--forward", type=int, default=7)
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    ARCH.mkdir(parents=True, exist_ok=True)

    if args.discover:
        dates = discover_dates(back=args.back, forward=args.forward)
        print("開催日:", ", ".join(dates) if dates else "(なし)")
        return

    if args.predict_only:
        regenerate_predictions(args.dates)
        return

    targets = args.dates or []
    if args.auto or not targets:
        targets = discover_dates(back=args.back, forward=args.forward)
    if not targets:
        print("更新対象日がありません", file=sys.stderr)
        sys.exit(1)

    client = NetkeibaClient(sleep=0.2)
    for d in targets:
        try:
            refresh_date(d, client=client, use_cache=not args.no_cache)
        except Exception as e:
            print(f"❌ {d} 失敗: {e}")

    if args.predict or args.auto or True:
        # Web反映のため必ず予想を再生成（対象日）
        regenerate_predictions(targets)
    print("🔥 P0-2 データ更新完了")


if __name__ == "__main__":
    main()
