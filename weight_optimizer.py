import pandas as pd


def optimize():

    data = pd.read_csv(
        "data/backtest.csv",
        encoding="utf-8-sig"
    )

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

    results = []

    # 0.0 ～ 2.5を0.1刻み
    weights = [
        x / 10
        for x in range(26)
    ]

    print()
    print("========================")
    print("🔥 人気補正係数 総当たり")
    print("========================")

    for weight in weights:

        test = data.copy()

        test["補正AREru指数"] = (
            test["AREru指数"]
            + (
                test["人気"] - 1
            ) * weight
        )

        test["補正順位"] = (
            test.groupby("race_id")
            ["補正AREru指数"]
            .rank(
                ascending=False,
                method="first"
            )
        )

        top = test[
            test["補正順位"] == 1
        ]

        total = len(top)

        hits = (
            top["実着順"] <= 3
        ).sum()

        hit_rate = (
            hits / total * 100
            if total > 0
            else 0
        )

        avg_popularity = (
            top["人気"].mean()
        )

        results.append({
            "補正係数": weight,
            "検証レース": total,
            "3着内": hits,
            "複勝率": round(
                hit_rate,
                2
            ),
            "平均人気": round(
                avg_popularity,
                2
            )
        })

        print(
            f"係数 {weight:.1f} "
            f"→ 複勝率 {hit_rate:.2f}% "
            f"3着内 {hits}/{total} "
            f"平均人気 {avg_popularity:.2f}"
        )

    result = pd.DataFrame(results)

    result = result.sort_values(
        [
            "複勝率",
            "平均人気"
        ],
        ascending=[
            False,
            False
        ]
    )

    print()
    print("========================")
    print("🏆 補正係数 TOP10")
    print("========================")

    print(
        result.head(10).to_string(
            index=False
        )
    )

    result.to_csv(
        "data/weight_optimizer.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("✅ 総当たり完了")
    print("📁 data/weight_optimizer.csv")


if __name__ == "__main__":
    optimize()