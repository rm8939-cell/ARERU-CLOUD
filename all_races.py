import requests
from bs4 import BeautifulSoup

from races import get_races


def get_all_horses():
    races = get_races()

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    all_horses = []

    print()
    print("🔥 全レース馬名取得開始")
    print("====================")

    for number, race in enumerate(races, start=1):
        race_url = race["url"]
        print()
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

        horses = []

        for link in soup.find_all("a"):
            href = link.get("href", "")
            text = link.get_text(strip=True)

            if "accessU.html" in href and text:
                if text not in horses:
                    horses.append(text)

        print("頭数:", len(horses))

        for horse in horses:
            print("・", horse)

            all_horses.append({
                "race_id": race["race_id"],
                "race_number": race["race_number"],
                "horse": horse
            })

    print()
    print("====================")
    print("🐴 合計馬数:", len(all_horses))

    return all_horses


if __name__ == "__main__":
    get_all_horses()