import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re


def get_races():

    url = (
        "https://www.jra.go.jp/JRADB/accessD.html"
        "?CNAME=pw01dde0110202602060520260712%2F8D"
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    print("🏇 レース一覧取得開始...")

    response = requests.get(
        url,
        headers=headers
    )

    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(
        response.text,
        "lxml"
    )

    races = {
        5: url
    }

    for link in soup.find_all("a"):

        href = link.get("href", "")

        img = link.find("img")

        if img is None:
            continue

        alt = img.get("alt", "")

        match = re.fullmatch(
            r"(\d+)レース",
            alt
        )

        if match is None:
            continue

        if "accessD.html" not in href:
            continue

        race_number = int(
            match.group(1)
        )

        race_url = urljoin(
            "https://www.jra.go.jp",
            href
        )

        races[race_number] = race_url

    race_urls = []

    for race_number in sorted(races):

        race_url = races[race_number]

        race_urls.append({
            "race_id": race_url,
            "race_number": race_number,
            "url": race_url
        })

        print(
            f"{race_number}レース",
            race_url
        )

    print()
    print("🏁 レース数:", len(race_urls))

    return race_urls


if __name__ == "__main__":
    get_races()