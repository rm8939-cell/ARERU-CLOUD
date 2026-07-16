import requests
import pandas as pd

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import StringIO

from races import get_races


def get_results():

    races = get_races()

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    results = []

    print()
    print("🏁 レース結果取得開始")
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
            f"{len(races)} 結果取得中..."
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

        result_url = None

        for link in soup.find_all("a"):

            href = link.get("href", "")
            text = link.get_text(
                " ",
                strip=True
            )

            if (
                "レース結果" in text
                and "accessS.html" in href
            ):

                result_url = urljoin(
                    "https://www.jra.go.jp",
                    href
                )

                break

        if result_url is None:

            print("⚠️ 結果リンクなし")
            continue

        result_response = requests.get(
            result_url,
            headers=headers
        )

        result_response.raise_for_status()
        result_response.encoding = (
            result_response.apparent_encoding
        )

        tables = pd.read_html(
            StringIO(result_response.text)
        )

        race_result = None

        for table in tables:

            table.columns = [
                "".join(
                    str(x)
                    for x in col
                    if str(x) != "nan"
                )
                if isinstance(col, tuple)
                else str(col)
                for col in table.columns
            ]

            finish_col = next(
                (
                    col
                    for col in table.columns
                    if "着順" in col
                ),
                None
            )

            horse_col = next(
                (
                    col
                    for col in table.columns
                    if "馬名" in col
                ),
                None
            )

            if (
                finish_col is not None
                and horse_col is not None
            ):

                race_result = table
                break

        if race_result is None:

            print("⚠️ 結果表なし")
            continue

        print(
            "✅ 取得:",
            len(race_result),
            "頭"
        )

        for _, row in race_result.iterrows():

            results.append({
                "race_id": race["race_id"],
                "レース": race_number,
                "馬名": row[horse_col],
                "着順": row[finish_col]
            })

    result = pd.DataFrame(results)

    result.to_csv(
        "data/results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("====================")
    print("✅ レース結果保存完了")
    print("頭数:", len(result))
    print("📁 data/results.csv")


if __name__ == "__main__":
    get_results()