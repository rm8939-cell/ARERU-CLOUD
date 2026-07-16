import csv
import os
import time
import requests
import pandas as pd
import re
from datetime import date
from io import StringIO


def _default_exclude_date() -> str:
    # Linuxでは %-m が使えない環境があるため手動でゼロ埋めなしにする
    t = date.today()
    return f"{t.year}年{t.month}月{t.day}日"


# 当日結果のリーク防止。環境変数 ARERU_EXCLUDE_DATE（例: 2026年7月18日）で上書き可。
TARGET_DATE = os.environ.get("ARERU_EXCLUDE_DATE", _default_exclude_date())


def normalize_date(value):

    text = str(value).strip()

    text = text.replace(" ", "")

    match = re.search(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日",
        text
    )

    if match:

        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))

        return f"{year}年{month}月{day}日"

    return text


def save_all_history():

    with open(
        "data/horse_links.csv",
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)
        horses = list(reader)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    session = requests.Session()

    file_path = "data/all_history.csv"

    columns = [
        "今回レース",
        "馬名",
        "年月日",
        "場",
        "レース名",
        "距離",
        "馬場",
        "頭数",
        "人気",
        "着順",
        "騎手",
        "斤量",
        "馬体重",
        "タイム",
        "着差"
    ]

    all_history = []

    total = len(horses)

    excluded_count = 0

    print()
    print("🔥 全馬過去走取得開始")
    print("====================")
    print("対象馬:", total)
    print("🚫 除外日:", TARGET_DATE)

    for i, horse in enumerate(
        horses,
        start=1
    ):

        horse_name = horse.get(
            "horse",
            ""
        )

        race_number = horse.get(
            "race_number",
            horse.get("race", "")
        )

        horse_url = horse.get(
            "url",
            ""
        )

        print()
        print(
            f"🐴 {i}/{total} "
            f"{horse_name} 取得中..."
        )

        try:

            response = session.get(
                horse_url,
                headers=headers,
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

                date_col = next(
                    (
                        col
                        for col in table.columns
                        if "年月日" in col
                        or "日付" in col
                    ),
                    None
                )

                finish_col = next(
                    (
                        col
                        for col in table.columns
                        if "着順" in col
                    ),
                    None
                )

                if (
                    date_col is not None
                    and finish_col is not None
                ):

                    history_table = table

                    break

            if history_table is None:

                print("⚠️ 過去走表なし")

                continue

            def find_col(words):

                for col in history_table.columns:

                    for word in words:

                        if word in str(col):

                            return col

                return None

            date_col = find_col([
                "年月日",
                "日付"
            ])

            place_col = find_col([
                "場"
            ])

            race_col = find_col([
                "レース名",
                "競走名"
            ])

            distance_col = find_col([
                "距離"
            ])

            track_col = find_col([
                "馬場"
            ])

            count_col = find_col([
                "頭数"
            ])

            popularity_col = find_col([
                "人気"
            ])

            finish_col = find_col([
                "着順"
            ])

            jockey_col = find_col([
                "騎手"
            ])

            weight_col = find_col([
                "斤量"
            ])

            body_col = find_col([
                "馬体重"
            ])

            time_col = find_col([
                "タイム"
            ])

            margin_col = find_col([
                "着差"
            ])

            def value(row, col):

                if col is None:

                    return ""

                data = row.get(
                    col,
                    ""
                )

                if pd.isna(data):

                    return ""

                return str(data).strip()

            count = 0

            horse_excluded = 0

            for _, row in history_table.iterrows():

                raw_date = value(
                    row,
                    date_col
                )

                date = normalize_date(
                    raw_date
                )

                if not date:

                    continue

                # 今回レース結果を除外
                if date == TARGET_DATE:

                    excluded_count += 1

                    horse_excluded += 1

                    continue

                finish = value(
                    row,
                    finish_col
                )

                history = [
                    race_number,
                    horse_name,
                    date,
                    value(row, place_col),
                    value(row, race_col),
                    value(row, distance_col),
                    value(row, track_col),
                    value(row, count_col),
                    value(row, popularity_col),
                    finish,
                    value(row, jockey_col),
                    value(row, weight_col),
                    value(row, body_col),
                    value(row, time_col),
                    value(row, margin_col)
                ]

                all_history.append(
                    history
                )

                count += 1

            print(
                f"✅ 取得完了 {count}走"
            )

            if horse_excluded > 0:

                print(
                    f"🚫 今回結果除外 "
                    f"{horse_excluded}件"
                )

        except Exception as e:

            print(
                "❌ エラー:",
                e
            )

        time.sleep(1)

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(columns)

        writer.writerows(
            all_history
        )

    print()
    print("====================")
    print("🔥 全馬過去走CSV保存完了")
    print("📁", file_path)
    print(
        "📊 データ数:",
        len(all_history)
    )
    print(
        "🚫 今回結果除外:",
        excluded_count,
        "件"
    )


if __name__ == "__main__":
    save_all_history()