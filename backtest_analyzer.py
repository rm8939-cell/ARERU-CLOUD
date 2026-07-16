import pandas as pd


def find_col(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name

    for col in df.columns:
        text = str(col)

        for name in candidates:
            if name in text:
                return col

    return None


def analyze():
    file_path = "data/backtest.csv"

    df = pd.read_csv(file_path)

    score_col = find_col(
        df,
        [
            "AREru指数",
            "AREru値",
            "テスト指数",
            "指数",
        ],
    )

    finish_col = find_col(
        df,
        [
            "実着順",
            "着順",
        ],
    )

    race_col = find_col(
        df,
        [
            "race_id",
            "今回レース",
        ],
    )

    horse_col = find_col(
        df,
        [
            "馬名",
        ],
    )

    print()
    print("========================")
    print("📊 AREru指数 詳細分析")
    print("========================")

    print("列:", list(df.columns))

    if score_col is None:
        raise KeyError(
            "指数列が見つかりません"
        )

    if finish_col is None:
        raise KeyError(
            "着順列が見つかりません"
        )

    print("指数列:", score_col)
    print("着順列:", finish_col)

    df[score_col] = pd.to_numeric(
        df[score_col],
        errors="coerce",
    )

    df[finish_col] = pd.to_numeric(
        df[finish_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            score_col,
            finish_col,
        ]
    ).copy()

    df["3着内"] = (
        df[finish_col] <= 3
    ).astype(int)

    if race_col is not None:
        race_count = df[
            race_col
        ].nunique()
    else:
        race_count = len(df)

    hit_count = int(
        df["3着内"].sum()
    )

    hit_rate = (
        hit_count / len(df) * 100
        if len(df) > 0
        else 0
    )

    print()
    print("検証データ:", len(df))
    print("検証レース:", race_count)
    print("3着内:", hit_count)
    print(
        "全体3着内率:",
        round(hit_rate, 2),
        "%",
    )

    bins = [
        -float("inf"),
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        50,
        float("inf"),
    ]

    labels = [
        "10以下",
        "10〜15",
        "15〜20",
        "20〜25",
        "25〜30",
        "30〜35",
        "35〜40",
        "40〜50",
        "50超",
    ]

    df["指数帯"] = pd.cut(
        df[score_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    band = (
        df.groupby(
            "指数帯",
            observed=False,
        )
        .agg(
            レース数=(
                score_col,
                "size",
            ),
            的中数=(
                "3着内",
                "sum",
            ),
            平均指数=(
                score_col,
                "mean",
            ),
        )
        .reset_index()
    )

    band["複勝率"] = (
        band["的中数"]
        / band["レース数"]
        * 100
    ).fillna(0)

    print()
    print("========================")
    print("🔥 指数帯別成績")
    print("========================")
    print(
        band.to_string(
            index=False
        )
    )

    top20 = df.sort_values(
        score_col,
        ascending=False,
    ).head(20)

    show_cols = []

    if race_col is not None:
        show_cols.append(race_col)

    if horse_col is not None:
        show_cols.append(horse_col)

    show_cols.extend(
        [
            score_col,
            finish_col,
        ]
    )

    print()
    print("========================")
    print("🏆 高指数順 TOP20")
    print("========================")
    print(
        top20[
            show_cols
        ].to_string(
            index=False
        )
    )

    band.to_csv(
        "data/areru_band_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top20.to_csv(
        "data/areru_top20.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("========================")
    print("✅ 分析保存完了")
    print("========================")
    print(
        "📁 data/areru_band_analysis.csv"
    )
    print(
        "📁 data/areru_top20.csv"
    )


if __name__ == "__main__":
    analyze()
