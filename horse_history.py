import csv
import requests
from bs4 import BeautifulSoup


def get_horse_history():
    with open(
        "data/horse_links.csv",
        "r",
        encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        horse = next(reader)

    print("🐴 テスト馬")
    print("馬名:", horse["horse"])
    print("URL:", horse["url"])

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        horse["url"],
        headers=headers
    )

    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(
        response.text,
        "lxml"
    )

    tables = soup.find_all("table")
    history_table = tables[0]

    rows = history_table.find_all("tr")

    history_data = []

    for row in rows:
        cells = row.find_all(["th", "td"])

        data = [
            cell.get_text(" ", strip=True)
            for cell in cells
        ]

        if data:
            history_data.append(data)

    file_path = "data/test_history.csv"

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerows(history_data)

    print()
    print("====================")
    print("✅ 過去走CSV保存完了")
    print("📁", file_path)
    print("📊 行数:", len(history_data))


if __name__ == "__main__":
    get_horse_history()