import itertools
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

    soup = BeautifulSoup(
        response.text,
        "lxml"
    )

    return soup, response.text


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

        if (
            finish_col is not None
            and horse_col is not None
        ):

            return (
                table,
                finish_col,
                horse_col
            )

    return None, None, None


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

            table = flatten_columns(table)

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


def create_test_data():

    race_urls = get_race_urls()

    test_rows = []

    total_urls = len(race_urls)

    print()
    print("========================")
    print("🔥 時系列テストデータ作成")
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

                continue

            (
                result_table,
                finish_col,
                horse_col
            ) = get_result_table(html)

            if result_table is None:

                continue

            horse_links = get_horse_links(
                soup
            )

            horse_count = 0

            for _, row in result_table.iterrows():

                horse_name = str(
                    row[horse_col]
                ).strip()

                horse_url = horse_links.get(
                    horse_name
                )

                if horse_url is None:

                    continue

                history = get_horse_history(
                    horse_name,
                    horse_url
                )

                if history.empty:

                    continue

                past = history[
                    history["解析日付"]
                    < race_date
                ].copy()

                past = past.sort_values(
                    "解析日付",
                    ascending=False
                )

                past = past.head(5)

                if len(past) == 0:

                    continue

                actual_finish = pd.to_numeric(
                    row[finish_col],
                    errors="coerce"
                )

                test_row = {
                    "race_id": race_url,
                    "日付": race_date,
                    "レース": race_number,
                    "馬名": horse_name,
                    "実着順": actual_finish
                }

                for i in range(5):

                    if i < len(past):

                        past_row = past.iloc[i]

                        test_row[
                            f"着順{i + 1}"
                        ] = past_row["着順"]

                        test_row[
                            f"人気{i + 1}"
                        ] = past_row["人気"]

                    else:

                        test_row[
                            f"着順{i + 1}"
                        ] = None

                        test_row[
                            f"人気{i + 1}"
                        ] = None

                test_rows.append(
                    test_row
                )

                horse_count += 1

                time.sleep(0.05)

            print(
                race_date.date(),
                f"{race_number}R",
                "保存馬",
                horse_count
            )

        except Exception as e:

            print(
                "❌ レースエラー:",
                e
            )

        time.sleep(0.3)

    test_data = pd.DataFrame(
        test_rows
    )

    test_data.to_csv(
        "data/runners.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("========================")
    print("✅ テストデータ作成完了")
    print("保存馬数:", len(test_data))
    print(
        "検証レース:",
        test_data[
            "race_id"
        ].nunique()
    )
    print("========================")

    return test_data


def base_score(
    finish,
    popularity,
    first_score,
    second_score,
    third_score,
    top5_score,
    top10_score,
    upset_weight
):

    score = 0

    if pd.notna(finish):

        if finish == 1:

            score += first_score

        elif finish == 2:

            score += second_score

        elif finish == 3:

            score += third_score

        elif finish <= 5:

            score += top5_score

        elif finish <= 10:

            score += top10_score

    if (
        pd.notna(finish)
        and pd.notna(popularity)
    ):

        score += (
            popularity - finish
        ) * upset_weight

    return score


def optimize(test_data):

    first_scores = [
        20,
        25,
        30,
        35,
        40
    ]

    second_scores = [
        15,
        20,
        24,
        28,
        32
    ]

    third_scores = [
        10,
        15,
        20,
        24
    ]

    top5_scores = [
        5,
        10,
        14,
        18
    ]

    top10_scores = [
        0,
        3,
        7,
        10
    ]

    upset_weights = [
        0,
        1,
        2,
        3,
        4
    ]

    weights = [
        1.0,
        0.8,
        0.6,
        0.4,
        0.2
    ]

    combinations = list(
        itertools.product(
            first_scores,
            second_scores,
            third_scores,
            top5_scores,
            top10_scores,
            upset_weights
        )
    )

    total_patterns = len(
        combinations
    )

    print()
    print("========================")
    print("🔥 ガチ時系列スコア総当たり")
    print("========================")
    print(
        "検証レース:",
        test_data[
            "race_id"
        ].nunique()
    )
    print(
        "検証パターン:",
        total_patterns
    )

    results = []

    for index, params in enumerate(
        combinations,
        start=1
    ):

        (
            first_score,
            second_score,
            third_score,
            top5_score,
            top10_score,
            upset_weight
        ) = params

        calculated_scores = []

        for _, row in test_data.iterrows():

            weighted_score = 0
            weight_total = 0

            for i, weight in enumerate(
                weights,
                start=1
            ):

                finish = row[
                    f"着順{i}"
                ]

                popularity = row[
                    f"人気{i}"
                ]

                if pd.isna(finish):

                    continue

                score = base_score(
                    finish,
                    popularity,
                    first_score,
                    second_score,
                    third_score,
                    top5_score,
                    top10_score,
                    upset_weight
                )

                weighted_score += (
                    score * weight
                )

                weight_total += weight

            if weight_total > 0:

                final_score = (
                    weighted_score
                    / weight_total
                )

            else:

                final_score = 0

            calculated_scores.append(
                final_score
            )

        work = test_data[
            [
                "race_id",
                "実着順"
            ]
        ].copy()

        work["テスト指数"] = (
            calculated_scores
        )

        top = (
            work
            .sort_values(
                "テスト指数",
                ascending=False
            )
            .groupby(
                "race_id",
                as_index=False
            )
            .head(1)
        )

        total_races = len(top)

        hits = (
            pd.to_numeric(
                top["実着順"],
                errors="coerce"
            )
            <= 3
        ).sum()

        if total_races > 0:

            hit_rate = (
                hits / total_races
            ) * 100

        else:

            hit_rate = 0

        results.append({
            "1着点": first_score,
            "2着点": second_score,
            "3着点": third_score,
            "5着内点": top5_score,
            "10着内点": top10_score,
            "穴補正": upset_weight,
            "レース数": total_races,
            "3着内": hits,
            "複勝率": round(
                hit_rate,
                2
            )
        })

        if (
            index % 500 == 0
            or index == total_patterns
        ):

            print(
                "進捗",
                f"{index}/{total_patterns}"
            )

    result = pd.DataFrame(
        results
    )

    result = result.sort_values(
        [
            "複勝率",
            "3着内"
        ],
        ascending=False
    )

    print()
    print("========================")
    print("🏆 ガチスコア設定 TOP20")
    print("========================")

    print(
        result.head(20).to_string(
            index=False
        )
    )

    result.to_csv(
        "data/score_optimizer.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("========================")
    print("✅ ガチ総当たり完了")
    print("========================")
    print(
        "📁 data/score_optimizer.csv"
    )


def main():

    test_data = create_test_data()

    if len(test_data) == 0:

        print("❌ テストデータ0件")

        return

    optimize(
        test_data
    )


if __name__ == "__main__":

    main()