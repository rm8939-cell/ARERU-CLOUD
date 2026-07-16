import pandas as pd


def analyze():

    file_path = "data/backtest.csv"

    data = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    print("📊 データ数:", len(data))
    print("📋 列:", list(data.columns))

    required = [
        "AREru指数",
        "人気",
        "実着順"
    ]

    for col in required:

        if col not in data.columns:

            print(
                "❌ 必要な列がありません:",
                col
            )

            return

    data["AREru指数"] = pd.to_numeric(
        data["AREru指数"],
        errors="coerce"
    )

    data["人気"] = pd.to_numeric(
        data["人気"],
        errors="coerce"
    )

    data["実着順"] = pd.to_numeric(
        data["実着順"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "AREru指数",
            "人気"
        ]
    ).copy()

    # 人気補正
    data["人気補正"] = (
        data["人気"] - 1
    ) * 2.5

    # 人気の割に指数が高い馬を評価
    data["補正AREru指数"] = (
        data["AREru指数"]
        + data["人気補正"]
    )

    data["3着内"] = (
        data["実着順"] <= 3
    )

    data["補正順位"] = (
        data.groupby("race_id")
        ["補正AREru指数"]
        .rank(
            ascending=False,
            method="first"
        )
    )

    top = data[
        data["補正順位"] == 1
    ].copy()

    total = len(top)

    hits = top[
        "3着内"
    ].sum()

    if total > 0:

        hit_rate = (
            hits / total
        ) * 100

    else:

        hit_rate = 0

    print()
    print("========================")
    print("🔥 人気補正AREru検証")
    print("========================")

    print(
        "検証レース:",
        total
    )

    print(
        "3着内:",
        hits
    )

    print(
        "複勝率:",
        round(
            hit_rate,
            2
        ),
        "%"
    )

    print()
    print("🎯 補正AREru上位")
    print(
        top[
            [
                "日付",
                "レース",
                "馬名",
                "AREru指数",
                "人気",
                "補正AREru指数",
                "実着順"
            ]
        ]
        .sort_values(
            "補正AREru指数",
            ascending=False
        )
        .head(30)
        .to_string(
            index=False
        )
    )

    top.to_csv(
        "data/adjusted_areru.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("========================")
    print("✅ 人気補正検証完了")
    print("========================")
    print(
        "📁 data/adjusted_areru.csv"
    )


if __name__ == "__main__":
    analyze()