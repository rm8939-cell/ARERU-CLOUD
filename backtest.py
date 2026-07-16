import re
import time
from io import StringIO
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


SEED_URLS = [
    "https://www.jra.go.jp/JRADB/accessS.html?CNAME=pw01sde1010202602041120260705%2F9D",
    "https://www.jra.go.jp/JRADB/accessS.html?CNAME=pw01sde0110202602060420260712%2F94",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

WEIGHTS = [
    1.0,
    0.8,
    0.6,
    0.4,
    0.2
]


session = requests.Session()

horse_cache = {}


def get_soup(url):

    response = session.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    response.encoding = (
        response.apparent_encoding
    )

    return BeautifulSoup(
        response.text,
        "lxml"
    ), response.text


def flatten_columns(table):

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

    return table


def find_col(table, words):

    for col in table.columns:

        for word in words:

            if word in str(col):

                return col

    return None


def get_race_urls():

    urls = set()

    for seed_url in SEED_URLS:

        print()
        print("🔎 開催取得")
        print(seed_url)

        soup, _ = get_soup(seed_url)

        urls.add(seed_url)

        for link in soup.find_all("a"):

            href = link.get("href", "")

            if "accessS.html" not in href:
                continue

            race_url = urljoin(
                "https://www.jra.go.jp",
                href
            )

            urls.add(race_url)

    first_urls = list(urls)

    for url in first_urls:

        try:

            soup, _ = get_soup(url)

            for link in soup.find_all("a"):

                href = link.get("href", "")

                if "accessS.html" not in href:
                    continue

                race_url = urljoin(
                    "https://www.jra.go.jp",
                    href
                )

                urls.add(race_url)

        except Exception:

            pass

        time.sleep(0.2)

    print()
    print("========================")
    print("🏇 発見レースURL:", len(urls))
    print("========================")

    return sorted(urls)


def get_race_info(soup):

    text = soup.get_text(
        " ",
        strip=True
    )

    date_match = re.search(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日",
        text
    )

    race_match = re.search(
        r"(\d+)レース",
        text
    )

    if date_match is None:

        return None, None

    race_date = pd.Timestamp(
        year=int(date_match.group(1)),
        month=int(date_match.group(2)),
        day=int(date_match.group(3))
    )

    if race_match:

        race_number = int(
            race_match.group(1)
        )

    else:

        race_number = None

    return race_date, race_number


def get_result_table(html):

    tables = pd.read_html(
        StringIO(html)
    )

    for table in tables:

        table = flatten_columns(table)

        finish_col = find_col(
            table,
            ["着順"]
        )

        horse_col = find_col(
            table,
            ["馬名"]
        )

        popularity_col = find_col(
            table,
            ["人気"]
        )

        if (
            finish_col is not None
            and horse_col is not None
            and popularity_col is not None
        ):

            return (
                table,
                finish_col,
                horse_col,
                popularity_col
            )

    return None, None, None, None


def get_horse_links(soup):

    horse_links = {}

    for link in soup.find_all("a"):

        href = link.get("href", "")

        horse_name = link.get_text(
            strip=True
        )

        if (
            "accessU.html" in href
            and horse_name
        ):

            horse_url = urljoin(
                "https://www.jra.go.jp",
                href
            )

            horse_links[
                horse_name
            ] = horse_url

    return horse_links


def get_horse_history(
    horse_name,
    horse_url
):

    if horse_url in horse_cache:

        return horse_cache[
            horse_url
        ].copy()

    try:

        response = session.get(
            horse_url,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        response.encoding = (
            response.apparent_encoding
        )

        tables = pd.read_html(
            StringIO(response.text)
        )

        history_table = None

        for table in tables:

            table = flatten_columns(
                table
            )

            date_col = find_col(
                table,
                [
                    "年月日",
                    "日付"
                ]
            )

            finish_col = find_col(
                table,
                ["着順"]
            )

            popularity_col = find_col(
                table,
                ["人気"]
            )

            if (
                date_col is not None
                and finish_col is not None
                and popularity_col is not None
            ):

                history_table = table

                break

        if history_table is None:

            horse_cache[
                horse_url
            ] = pd.DataFrame()

            return pd.DataFrame()

        date_col = find_col(
            history_table,
            [
                "年月日",
                "日付"
            ]
        )

        finish_col = find_col(
            history_table,
            ["着順"]
        )

        popularity_col = find_col(
            history_table,
            ["人気"]
        )

        history = pd.DataFrame({
            "年月日": history_table[
                date_col
            ],
            "人気": history_table[
                popularity_col
            ],
            "着順": history_table[
                finish_col
            ]
        })

        history["解析日付"] = pd.to_datetime(
            history["年月日"]
            .astype(str)
            .str.extract(
                r"(\d{4}年\d{1,2}月\d{1,2}日)",
                expand=False
            ),
            format="%Y年%m月%d日",
            errors="coerce"
        )

        history["人気"] = pd.to_numeric(
            history["人気"],
            errors="coerce"
        )

        history["着順"] = pd.to_numeric(
            history["着順"],
            errors="coerce"
        )

        history = history.sort_values(
            "解析日付",
            ascending=False
        )

        horse_cache[
            horse_url
        ] = history.copy()

        return history

    except Exception as e:

        print(
            "⚠️ 過去走エラー:",
            horse_name,
            e
        )

        return pd.DataFrame()


def calculate_areru(
    history,
    race_date
):

    if history.empty:

        return 0

    data = history[
        history["解析日付"]
        < race_date
    ].copy()

    data = data.sort_values(
        "解析日付",
        ascending=False
    )

    data = data.head(5)

    scores = []

    for _, row in data.iterrows():

        score = 0

        finish = row["着順"]
        popularity = row["人気"]

        if pd.notna(finish):

            if finish == 1:
                score += 30

            elif finish == 2:
                score += 24

            elif finish == 3:
                score += 20

            elif finish <= 5:
                score += 14

            elif finish <= 10:
                score += 7

        if (
            pd.notna(finish)
            and pd.notna(popularity)
        ):

            score += (
                popularity - finish
            ) * 2

        scores.append(score)

    if not scores:

        return 0

    weighted_score = 0
    weight_total = 0

    for score, weight in zip(
        scores,
        WEIGHTS
    ):

        weighted_score += (
            score * weight
        )

        weight_total += weight

    return round(
        weighted_score / weight_total,
        2
    )


def backtest():

    race_urls = get_race_urls()

    results = []

    total_urls = len(race_urls)

    print()
    print("========================")
    print("🔥 AREru 全馬時系列検証開始")
    print("========================")

    for index, race_url in enumerate(
        race_urls,
        start=1
    ):

        print()
        print(
            f"🏇 {index}/{total_urls}"
        )

        try:

            soup, html = get_soup(
                race_url
            )

            race_date, race_number = (
                get_race_info(soup)
            )

            if race_date is None:

                print("⚠️ 日付取得失敗")

                continue

            (
                result_table,
                finish_col,
                horse_col,
                popularity_col
            ) = get_result_table(html)

            if result_table is None:

                print("⚠️ 結果表なし")

                continue

            horse_links = get_horse_links(
                soup
            )

            print(
                race_date.date(),
                f"{race_number}R",
                "頭数",
                len(result_table)
            )

            race_results = []

            for _, row in (
                result_table.iterrows()
            ):

                horse_name = str(
                    row[horse_col]
                ).strip()

                horse_url = (
                    horse_links.get(
                        horse_name
                    )
                )

                if horse_url is None:

                    continue

                history = get_horse_history(
                    horse_name,
                    horse_url
                )

                score = calculate_areru(
                    history,
                    race_date
                )

                finish = pd.to_numeric(
                    row[finish_col],
                    errors="coerce"
                )

                popularity = pd.to_numeric(
                    row[popularity_col],
                    errors="coerce"
                )

                race_results.append({
                    "race_id": race_url,
                    "日付": race_date.date(),
                    "レース": race_number,
                    "馬名": horse_name,
                    "AREru指数": score,
                    "人気": popularity,
                    "実着順": finish,
                    "3着内": (
                        pd.notna(finish)
                        and finish <= 3
                    )
                })

                time.sleep(0.1)

            if not race_results:

                print("⚠️ 保存馬なし")

                continue

            race_df = pd.DataFrame(
                race_results
            )

            race_df["AREru順位"] = (
                race_df["AREru指数"]
                .rank(
                    ascending=False,
                    method="first"
                )
                .astype(int)
            )

            top = race_df.sort_values(
                "AREru順位"
            ).iloc[0]

            print(
                "🥇",
                top["馬名"],
                "指数",
                top["AREru指数"],
                "人気",
                top["人気"],
                "→",
                top["実着順"],
                "着",
                "⭕️"
                if top["3着内"]
                else "❌"
            )

            results.extend(
                race_df.to_dict(
                    "records"
                )
            )

        except Exception as e:

            print(
                "❌ レースエラー:",
                e
            )

        time.sleep(0.5)

    result = pd.DataFrame(results)

    result.to_csv(
        "data/backtest.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("========================")
    print("📊 AREru 全馬検証結果")
    print("========================")

    print(
        "保存馬数:",
        len(result)
    )

    if len(result) == 0:

        print("検証データ: 0")

        return

    top_result = result[
        result["AREru順位"] == 1
    ]

    total = len(top_result)

    hits = top_result[
        "3着内"
    ].sum()

    hit_rate = (
        hits / total
    ) * 100

    print(
        "検証レース:",
        total
    )

    print(
        "AREru1位3着内:",
        hits
    )

    print(
        "純AREru複勝率:",
        round(
            hit_rate,
            2
        ),
        "%"
    )

    print()
    print(
        "📁 data/backtest.csv"
    )


if __name__ == "__main__":
    backtest()