import re

import requests
from bs4 import BeautifulSoup


JRA = "https://www.jra.go.jp"

# 出馬表トップ（開催一覧）。JRADB は cname を POST しないと中身が返らない。
SHUTUBA_INDEX_CNAME = "pw01dli00/F3"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# 開催一覧の cname: pw01drl1 + 場(3) + 年(4) + 回(2) + 日(2) + 開催日(8)
MEETING_CNAME = re.compile(
    r"pw01drl1(\d{3})(\d{4})(\d{2})(\d{2})(\d{8})/[0-9A-F]{2}"
)

# 各レース出馬表の cname: pw01dde1 + 場(3) + 年(4) + 回(2) + 日(2) + R(2) + 開催日(8)
RACE_CNAME = re.compile(
    r"pw01dde1(\d{3})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})/[0-9A-F]{2}"
)

# 単勝複勝オッズの cname（odds.py が使う）
WIN_ODDS_CNAME = re.compile(
    r"pw151ou1(\d{3})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})Z?/[0-9A-F]{2}"
)


def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def post(session, path, cname):
    """JRADB の画面遷移は cname の POST。"""
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


def find_cnames(soup, pattern):
    """doAction('/JRADB/xxx.html', 'cname') 形式のリンクから cname を拾う。"""
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


def get_meetings(session=None):
    """開催一覧（場・開催日）。"""
    session = session or make_session()

    soup = post(
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


def select_meeting(meetings, date="", venue=""):
    """対象開催を1つ選ぶ。指定が無ければ最新開催。"""
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


def get_meeting_races(session, meeting_cname):
    """開催内の各レース（レース番号・出馬表URL・単勝オッズcname）。"""
    soup = post(
        session,
        "/JRADB/accessD.html",
        meeting_cname
    )

    races = {}

    for row in soup.find_all("tr"):
        race_url = ""
        race_number = None

        for link in row.find_all("a"):
            href = link.get("href") or ""
            match = RACE_CNAME.search(href)

            if match is None:
                continue

            race_url = JRA + href
            race_number = int(match.group(5))
            break

        if race_number is None:
            continue

        odds_cnames = find_cnames(row, WIN_ODDS_CNAME)

        races[race_number] = {
            "race_number": race_number,
            "race_id": race_url,
            "url": race_url,
            "odds_cname": odds_cnames[0] if odds_cnames else ""
        }

    return [races[number] for number in sorted(races)]


def get_selected_meeting(date="", venue="", session=None):
    """対象開催を解決して返す（開催日を他モジュールが使う）。"""
    session = session or make_session()

    meetings = get_meetings(session)

    if not meetings:
        raise ValueError("JRAの開催一覧を取得できませんでした")

    meeting = select_meeting(meetings, date, venue)

    if meeting is None:
        raise ValueError(
            f"対象の開催が見つかりません (date={date!r} venue={venue!r})"
        )

    return meeting


def get_races(date="", venue=""):
    """対象開催のレース一覧。指定が無ければ最新開催。"""
    print("🏇 レース一覧取得開始...")

    session = make_session()

    meeting = get_selected_meeting(date, venue, session)

    print("🏟 対象開催:", meeting["label"], meeting["date"])

    races = get_meeting_races(session, meeting["cname"])

    race_urls = []

    for race in races:
        if not race["race_id"]:
            continue

        race_urls.append({
            "race_id": race["race_id"],
            "race_number": race["race_number"],
            "url": race["url"]
        })

        print(
            f"{race['race_number']}レース",
            race["url"]
        )

    print()
    print("🏁 レース数:", len(race_urls))

    return race_urls


if __name__ == "__main__":
    get_races()
