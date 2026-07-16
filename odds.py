import re
import requests
import pandas as pd

from bs4 import BeautifulSoup
from races import get_races


def get_odds():

    races = get_races()

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    results = []

    print()
    print("💰 単勝オッズ取得開始")
    print("====================")

    for index, race in enumerate(
        races,
        start=1
    ):
        race_number = race["race_number"]
        race_url = race["url"]

        print()
        print(
            f"🏇 {race_number}/"
            f"{len(races)} レース"
        )

        response = requests.get(
            race_url,
            headers=headers
        )

        response.raise_for_status()
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        race_count = 0

        for row in soup.find_all("tr"):

            horse_link = row.find(
                "a",
                href=lambda href:
                href and "accessU.html" in href
            )

            if horse_link is None:
                continue

            horse = horse_link.get_text(
                strip=True
            )

            row_text = row.get_text(
                " ",
                strip=True
            )

            match = re.search(
                r"(\d+(?:\.\d+)?)\s*"
                r"\(\s*(\d+)\s*番人気\s*\)",
                row_text
            )

            if match is None:
                continue

            odds = float(
                match.group(1)
            )

            popularity = int(
                match.group(2)
            )

            results.append({
                "race_id": race["race_id"],
                "レース": race_number,
                "馬名": horse,
                "単勝オッズ": odds,
                "人気": popularity
            })

            race_count += 1

        print(
            "取得:",
            race_count,
            "頭"
        )

    result = pd.DataFrame(results)

    result.to_csv(
        "data/odds.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("====================")
    print("💰 オッズ取得完了")
    print("頭数:", len(result))
    print("📁 data/odds.csv")


if __name__ == "__main__":
    get_odds()