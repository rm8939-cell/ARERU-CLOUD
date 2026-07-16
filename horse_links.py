import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from races import get_races


def save_horse_links():
    races = get_races()

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    horse_data = []

    print()
    print("🔗 馬詳細リンク取得開始")
    print("====================")

    for number, race in enumerate(races, start=1):
        race_url = race["url"]
        print(f"🏇 {number}/{len(races)} {race['race_number']}R 取得中...")

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

        for link in soup.find_all("a"):
            href = link.get("href", "")
            horse_name = link.get_text(strip=True)

            if "accessU.html" in href and horse_name:
                horse_url = urljoin(
                    "https://www.jra.go.jp",
                    href
                )

                horse_data.append({
                    "race_id": race["race_id"],
                    "race_number": race["race_number"],
                    "horse": horse_name,
                    "url": horse_url
                })

    file_path = "data/horse_links.csv"

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["race_id", "race_number", "horse", "url"]
        )

        writer.writeheader()
        writer.writerows(horse_data)

    print()
    print("====================")
    print("✅ 馬リンクCSV保存完了")
    print("📁", file_path)
    print("🐴 保存件数:", len(horse_data))


if __name__ == "__main__":
    save_horse_links()