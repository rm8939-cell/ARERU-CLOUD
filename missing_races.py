import pandas as pd


def main():

    backtest = pd.read_csv(
        "data/backtest.csv",
        encoding="utf-8-sig"
    )

    print("======================")
    print("AREru レース確認")
    print("======================")

    race_count = (
        backtest["race_id"]
        .nunique()
    )

    print("保存レース:", race_count)

    race_size = (
        backtest
        .groupby(
            ["日付", "レース"]
        )
        .size()
        .reset_index(name="頭数")
        .sort_values(
            ["日付", "レース"]
        )
    )

    print()
    print(race_size.to_string(index=False))

    race_size.to_csv(
        "data/race_check.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("保存しました")
    print("data/race_check.csv")


if __name__ == "__main__":
    main()