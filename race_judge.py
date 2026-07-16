import pandas as pd


def judge_races():

    df = pd.read_csv(
        "data/areru_ranking.csv",
        encoding="utf-8-sig"
    )

    results = []

    for race_id, race_data in df.groupby("race_id"):

        race = race_data.iloc[0]["レース"]

        race_data = race_data.sort_values(
            "AREru指数",
            ascending=False
        ).copy()

        scores = race_data["AREru指数"]

        horse_count = len(race_data)

        average_score = scores.mean()

        high_score_horses = (
            scores >= 15
        ).sum()

        close_horses = (
            scores >= scores.max() * 0.7
        ).sum()

        positive_scores = (scores > 0).sum()

        if positive_scores == 0:
            race_score = 0
            judge = "⚪ 過去走データ不足"
        else:
            race_score = 0
            race_score += average_score * 2
            race_score += high_score_horses * 3
            race_score += close_horses * 2
            race_score += horse_count * 0.5

            if race_score >= 55:
                judge = "🔥 大波乱警戒"
            elif race_score >= 40:
                judge = "⚠️ 波乱注意"
            else:
                judge = "🟢 比較的平穏"

        top3 = race_data.head(3)

        hole_horses = " / ".join(
            top3["馬名"].tolist()
        )

        results.append({
            "race_id": race_id,
            "レース": race,
            "荒れ度": round(race_score, 2),
            "判定": judge,
            "穴候補TOP3": hole_horses
        })

    result = pd.DataFrame(results)

    result = result.sort_values(
        "荒れ度",
        ascending=False
    )

    print()
    print("========================")
    print("🔥 AREru 波乱レース判定")
    print("========================")

    for rank, (_, row) in enumerate(
        result.iterrows(),
        start=1
    ):

        print()
        print(
            f"{rank}位｜"
            f"レース {row['レース']}"
        )

        print(
            f"荒れ度: {row['荒れ度']}"
        )

        print(
            f"判定: {row['判定']}"
        )

        print(
            f"🐴 候補: {row['穴候補TOP3']}"
        )

    result.to_csv(
        "data/race_judgement.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("========================")
    print("✅ 波乱レース判定保存完了")
    print("📁 data/race_judgement.csv")


if __name__ == "__main__":
    judge_races()