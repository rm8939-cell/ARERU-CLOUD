import pandas as pd


def verify():

    predictions = pd.read_csv(
        "data/predictions.csv",
        encoding="utf-8-sig"
    )

    results = pd.read_csv(
        "data/results.csv",
        encoding="utf-8-sig"
    )

    predictions["レース"] = pd.to_numeric(
        predictions["レース"],
        errors="coerce"
    )

    results["レース"] = pd.to_numeric(
        results["レース"],
        errors="coerce"
    )

    results["着順"] = pd.to_numeric(
        results["着順"],
        errors="coerce"
    )

    verification = []

    for _, pred in predictions.iterrows():

        race = pred["レース"]
        main_horse = str(pred["本命"]).strip()

        race_results = results[
            results["レース"] == race
        ]

        main_result = race_results[
            race_results["馬名"]
            .astype(str)
            .str.strip()
            == main_horse
        ]

        matched = len(main_result) > 0

        if matched:
            main_finish = main_result.iloc[0]["着順"]
            top3 = (
                pd.notna(main_finish)
                and main_finish <= 3
            )
        else:
            main_finish = None
            top3 = False

        verification.append({
            "レース": race,
            "本命": main_horse,
            "照合成功": matched,
            "本命着順": main_finish,
            "本命TOP3": top3
        })

    result = pd.DataFrame(verification)

    print()
    print("========================")
    print("🔬 AREru 厳格検証")
    print("========================")

    for _, row in result.iterrows():

        race = int(row["レース"])

        if row["照合成功"]:

            mark = (
                "⭕️"
                if row["本命TOP3"]
                else "❌"
            )

            print(
                f"{race}R "
                f"{row['本命']} "
                f"→ {row['本命着順']}着 "
                f"{mark}"
            )

        else:

            print(
                f"{race}R "
                f"{row['本命']} "
                "→ ⚠️ 結果照合失敗"
            )

    total = len(result)

    matched_count = result[
        "照合成功"
    ].sum()

    hits = result[
        "本命TOP3"
    ].sum()

    missing = total - matched_count

    print()
    print("========================")
    print("📊 厳格成績")
    print("========================")

    print("予想数:", total)
    print("照合成功:", matched_count)
    print("照合失敗:", missing)
    print("3着内:", hits)

    if matched_count > 0:

        hit_rate = (
            hits / matched_count
        ) * 100

        print(
            "照合成功ベース複勝率:",
            round(hit_rate, 2),
            "%"
        )

    strict_rate = (
        hits / total
    ) * 100

    print(
        "全予想ベース厳格複勝率:",
        round(strict_rate, 2),
        "%"
    )

    result.to_csv(
        "data/verification.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("✅ 厳格検証保存完了")


if __name__ == "__main__":
    verify()