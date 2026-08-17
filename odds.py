import argparse
import re

import pandas as pd
import requests

from bs4 import BeautifulSoup
from races import get_races


JRA = "https://www.jra.go.jp"

# 出馬表トップ（開催一覧）。JRADB は cname を POST して画面遷移する。
SHUTUBA_INDEX_CNAME = "pw01dli00/F3"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# 開催一覧の cname: pw01drl1 + 場コード(3) + 年(4) + 回(2) + 日(2) + 開催日(8)
MEETING_CNAME = re.compile(
    r"pw01drl1(\d{3})(\d{4})(\d{2})(\d{2})(\d{8})/[0-9A-F]{2}"
)

# 単勝複勝オッズの cname: pw151ou1 + 場(3) + 年(4) + 回(2) + 日(2) + R(2) + 開催日(8)
WIN_ODDS_CNAME = re.compile(
    r"pw151ou1(\d{3})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})Z?/[0-9A-F]{2}"
)

# 人気順ページの cname
POP_ODDS_CNAME = re.compile(
    r"pw151op1\d{3}\d{4}\d{2}\d{2}\d{2}\d{8}Z?/[0-9A-F]{2}"
)


def _post(session, path, cname):
    """JRADB は cname の POST で画面が決まる。GET だと中身が返らない。"""
    response = session.post(
        JRA + path,
        data={"cname": cname},
        timeout=30
    )

    response.raise_for_status()
    response.encoding = response.apparent_encoding

    return BeautifulSoup(
        response.text,
        "lxml"
    )


def _cnames(soup, pattern):
    """doAction('/JRADB/xxx.html', 'cname') から cname を拾う。"""
    found = []

    for tag in soup.find_all(["a", "area"]):
        onclick = tag.get("onclick") or ""
        match = pattern.search(onclick)

        if match is None:
            continue

        cname = match.group(0)

        if cname not in found:
            found.append(cname)

    return found


def get_meetings(session):
    """開催一覧（場・開催日・cname）。"""
    soup = _post(
        session,
        "/JRADB/accessD.html",
        SHUTUBA_INDEX_CNAME
    )

    meetings = []

    for tag in soup.find_all("a"):
        onclick = tag.get("onclick") or ""
        match = MEETING_CNAME.search(onclick)

        if match is None:
            continue

        label = tag.get_text(" ", strip=True)
        venue = re.search(r"\d+回(.+?)\d+日", label)

        meetings.append({
            "cname": match.group(0),
            "label": label,
            "venue": venue.group(1) if venue else "",
            "date": match.group(5)
        })

    return meetings


def get_meeting_races(session, meeting_cname):
    """開催内の各レース（レース番号・出馬表URL・単勝オッズcname）。"""
    soup = _post(
        session,
        "/JRADB/accessD.html",
        meeting_cname
    )

    races = []

    for row in soup.find_all("tr"):
        odds_cnames = _cnames(row, WIN_ODDS_CNAME)

        if not odds_cnames:
            continue

        odds_cname = odds_cnames[0]
        race_number = int(
            WIN_ODDS_CNAME.search(odds_cname).group(5)
        )

        race_url = ""

        for link in row.find_all("a"):
            href = link.get("href") or ""

            if "accessD.html?CNAME=" in href:
                race_url = JRA + href
                break

        races.append({
            "race_number": race_number,
            "race_url": race_url,
            "odds_cname": odds_cname
        })

    races.sort(key=lambda race: race["race_number"])

    return races


def _parse_odds_table(soup):
    """単勝・複勝オッズ表を行ごとに読む。人気列が無い場合は None。"""
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
    """1レースの単勝オッズ。人気順ページがあればそちらの人気を使う。"""
    soup = _post(
        session,
        "/JRADB/accessO.html",
        odds_cname
    )

    pop_cnames = _cnames(soup, POP_ODDS_CNAME)

    if pop_cnames:
        pop_soup = _post(
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


def _races_from_races_py():
    """races.py の race_id を温存するための レース番号→race_id。"""
    try:
        races = get_races()
    except Exception as error:
        print("⚠️ races.py からレース一覧を取得できません:", error)
        return {}, ""

    by_number = {}
    dates = []

    for race in races:
        by_number[race["race_number"]] = race["race_id"]

        found = re.search(r"(\d{8})", race["race_id"])

        if found:
            dates.append(found.group(1))

    return by_number, (dates[0] if dates else "")


def _select_meeting(meetings, date="", venue=""):
    """対象開催を1つ選ぶ。日付・場の指定が無ければ最新開催。"""
    candidates = meetings

    if date:
        candidates = [m for m in candidates if m["date"] == date]

    if venue:
        candidates = [m for m in candidates if m["venue"] == venue]

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda m: (m["date"], m["venue"]),
        reverse=True
    )[0]


def get_odds(date="", venue=""):

    session = requests.Session()
    session.headers.update(HEADERS)

    print()
    print("💰 単勝オッズ取得開始")
    print("====================")

    meetings = get_meetings(session)

    if not meetings:
        raise ValueError("JRAの開催一覧を取得できませんでした")

    for meeting in meetings:
        print("開催:", meeting["label"], meeting["date"])

    race_ids, races_py_date = _races_from_races_py()

    target = _select_meeting(meetings, date, venue)

    # races.py と同じ開催日が生きていればその race_id を使い、既存CSVとの
    # 突き合わせを保つ。無ければ最新開催の出馬表URLを race_id にする。
    if target is None and races_py_date:
        target = _select_meeting(meetings, races_py_date, venue)

    if target is None:
        target = _select_meeting(meetings, "", venue)

    if target is None:
        raise ValueError("対象の開催が見つかりませんでした")

    use_races_py = bool(race_ids) and races_py_date == target["date"]

    print()
    print("🎯 対象開催:", target["label"], target["date"])
    print(
        "race_id:",
        "races.py を使用" if use_races_py else "JRA出馬表URLを使用"
    )

    races = get_meeting_races(session, target["cname"])

    if not races:
        raise ValueError(
            f"開催内のレースを取得できませんでした: {target['label']}"
        )

    results = []

    for race in races:
        race_number = race["race_number"]

        print()
        print(f"🏇 {race_number}/{len(races)} レース")

        rows = get_win_odds(session, race["odds_cname"])

        race_id = race["race_url"]

        if use_races_py and race_number in race_ids:
            race_id = race_ids[race_number]

        if not race_id:
            print("⚠️ race_id を特定できないためスキップ")
            continue

        for row in rows:
            results.append({
                "race_id": race_id,
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
