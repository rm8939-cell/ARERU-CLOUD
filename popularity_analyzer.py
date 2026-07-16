import pandas as pd


def analyze():

    file_path = "data/backtest.csv"

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    print("📊 データ数:", len(df))
    print("📋 列:", list(df.columns))

    # 数値化
    df["AREru指数"] = pd.to_numeric(
        df["AREru指数"],
        errors="coerce"
    )

    df["実着順"] = pd.to_numeric(
        df["実着順"],
        errors="coerce"
    )

    # 人気列確認
    if "人気" not in df.columns:

        print()
        print("❌ backtest.csv に人気列がありません")
        print("📋 現在の列:")
        print(list(df.columns))

        return

    df["人気"] = pd.to_numeric(
        df["人気"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "AREru指数",
            "実着順",
            "人気"
        ]
    )

    # 指数帯
    df["指数帯"] = pd.cut(
        df["AREru指数"],
        bins=[
            20,
            25,
            30,
            35,
            40,
            50
        ],
        labels=[
            "20〜25",
            "25〜30",
            "30〜35",
            "35〜40",
            "40〜50"
        ],
        right=False
    )

    # 人気帯
    df["人気帯"] = pd.cut(
        df["人気"],
        bins=[
            0,
            3,
            6,
            10,
            100
        ],
        labels=[
            "1〜2人気",
            "3〜5人気",
            "6〜9人気",
            "10人気以下"
        ],
        right=False
    )

    results = []

    grouped = df.groupby(
        [
            "指数帯",
            "人気帯"
        ],
        observed=True
    )

    for (
        index_band,
        popularity_band
    ), data in grouped:

        total = len(data)

        hits = (
            data["実着順"] <= 3
        ).sum()

        hit_rate = (
            hits / total * 100
            if total > 0
            else 0
        )

        results.append({
            "指数帯": index_band,
            "人気帯": popularity_band,
            "頭数": total,
            "3着内": hits,
            "複勝率": round(
                hit_rate,
                2
            )
        })

    result = pd.DataFrame(results)

    result = result.sort_values(
        [
            "指数帯",
            "人気帯"
        ]
    )

    print()
    print("====================")
    print("🔥 AREru指数 × 人気分析")
    print("====================")

    print(
        result.to_string(
            index=False
        )
    )

    print()
    print("====================")
    print("💣 穴馬ゾーン")
    print("====================")

    danger = result[
        result["人気帯"].isin([
            "6〜9人気",
            "10人気以下"
        ])
    ]

    print(
        danger.sort_values(
            "複勝率",
            ascending=False
        ).to_string(
            index=False
        )
    )

    result.to_csv(
        "data/areru_popularity_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("====================")
    print("✅ 人気帯分析保存完了")
    print("====================")
    print(
        "📁 data/areru_popularity_analysis.csv"
    )


if __name__ == "__main__":
    analyze()