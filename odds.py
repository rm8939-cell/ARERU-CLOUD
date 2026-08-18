import argparse
import re

import pandas as pd

from races import (
    find_cnames,
    get_meeting_races,
    get_selected_meeting,
    make_session,
    post,
)


# 人気順ページの cname（単勝・複勝オッズ 人気順）
POP_ODDS_CNAME = re.compile(
    r"pw151op1\d{3}\d{4}\d{2}\d{2}\d{2}\d{8}Z?/[0-9A-F]{2}"
)


def _parse_odds_table(soup):
    """単勝・複勝オッズ表を行ごとに読む。人気列が無い場合は 人気=None。"""
    table = soup.select_one("table.tanpuku")

    if table is None:
        return []

    headers = [
        th.get_text(strip=True)
        for th in table.find_all("th")
    ]

    if "馬名" not in headers or "単勝" not in headers:
        return []

    name_at = headers.index("馬名")
    odds_at = headers.index("単勝")
    pop_at = headers.index("人気") if "人気" in headers else None

    rows = []

    for row in table.find_all("tr"):
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all("td")
        ]

        if len(cells) <= max(name_at, odds_at):
            continue

        horse = cells[name_at]
        odds_text = cells[odds_at]

        # 取消・除外・発売前は数値が入らないので落とす（推測値は入れない）
        if not re.fullmatch(r"\d+(?:\.\d+)?", odds_text):
            continue

        popularity = None

        if pop_at is not None and pop_at < len(cells):
            if re.fullmatch(r"\d+", cells[pop_at]):
                popularity = int(cells[pop_at])

        rows.append({
            "馬名": horse,
            "単勝オッズ": float(odds_text),
            "人気": popularity
        })

    return rows


def get_win_odds(session, odds_cname):
    """1レースの単勝オッズ。人気順ページがあればその人気を使う。"""
    soup = post(
        session,
        "/JRADB/accessO.html",
        odds_cname
    )

    pop_cnames = find_cnames(soup, POP_ODDS_CNAME)

    if pop_cnames:
        pop_soup = post(
            session,
            "/JRADB/accessO.html",
            pop_cnames[0]
        )

        rows = _parse_odds_table(pop_soup)

        if rows and all(row["人気"] is not None for row in rows):
            return rows

    rows = _parse_odds_table(soup)

    # 人気列が無いページ用: 単勝オッズの昇順が単勝人気の定義
    if rows and any(row["人気"] is None for row in rows):
        order = sorted(
            {row["単勝オッズ"] for row in rows}
        )

        for row in rows:
            if row["人気"] is None:
                row["人気"] = order.index(row["単勝オッズ"]) + 1

    return rows


def get_odds(date="", venue=""):

    session = make_session()

    print()
    print("💰 単勝オッズ取得開始")
    print("====================")

    meeting = get_selected_meeting(date, venue, session)

    print("🏟 対象開催:", meeting["label"], meeting["date"])

    races = get_meeting_races(session, meeting["cname"])

    if not races:
        raise ValueError(
            f"開催内のレースを取得できませんでした: {meeting['label']}"
        )

    results = []

    for race in races:
        race_number = race["race_number"]

        print()
        print(f"🏇 {race_number}/{len(races)} レース")

        if not race["odds_cname"]:
            print("⚠️ オッズページのリンクが無いためスキップ")
            continue

        if not race["race_id"]:
            print("⚠️ race_id を特定できないためスキップ")
            continue

        rows = get_win_odds(session, race["odds_cname"])

        for row in rows:
            results.append({
                "race_id": race["race_id"],
                "レース": race_number,
                "馬名": row["馬名"],
                "単勝オッズ": row["単勝オッズ"],
                "人気": row["人気"]
            })

        print("取得:", len(rows), "頭")

    result = pd.DataFrame(
        results,
        columns=["race_id", "レース", "馬名", "単勝オッズ", "人気"]
    )

    if result.empty:
        raise ValueError("単勝オッズを1頭も取得できませんでした")

    result.to_csv(
        "data/odds.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("====================")
    print("💰 オッズ取得完了")
    print("レース数:", result["レース"].nunique())
    print("頭数:", len(result))
    print("📁 data/odds.csv")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="", help="開催日 YYYYMMDD")
    parser.add_argument("--venue", default="", help="開催場 例: 新潟")
    args = parser.parse_args()

    get_odds(args.date, args.venue)
