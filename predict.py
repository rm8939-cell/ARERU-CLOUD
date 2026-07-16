import pandas as pd


def predict():
    areru = pd.read_csv("data/areru_ranking.csv", encoding="utf-8-sig")
    judgement = pd.read_csv("data/race_judgement.csv", encoding="utf-8-sig")
    odds = pd.read_csv("data/odds.csv", encoding="utf-8-sig")

    areru["レース"] = pd.to_numeric(areru["レース"], errors="coerce")
    odds["レース"] = pd.to_numeric(odds["レース"], errors="coerce")
    odds["単勝オッズ"] = pd.to_numeric(odds["単勝オッズ"], errors="coerce")
    odds["人気"] = pd.to_numeric(odds["人気"], errors="coerce")

    merged = pd.merge(
        areru,
        odds[["race_id", "馬名", "単勝オッズ", "人気"]],
        on=["race_id", "馬名"],
        how="inner",
    )

    if merged.empty:
        raise ValueError("AREruランキングとオッズが1頭も一致しません")

    predictions = []

    for race_id, race_data in merged.groupby("race_id"):
        race = race_data.iloc[0]["レース"]
        race_data = race_data.sort_values(["AREru指数", "人気"], ascending=[False, True]).copy()

        judge_data = judgement[judgement["race_id"] == race_id]
        if len(judge_data) > 0:
            judge = judge_data.iloc[0]
            areru_level = judge["荒れ度"]
            judge_text = judge["判定"]
        else:
            areru_level = 0
            judge_text = "判定なし"

        main_candidates = race_data[race_data["人気"] <= 5]
        hole_candidates = race_data[
            (race_data["人気"] >= 6) & (race_data["単勝オッズ"] >= 10) & (race_data["AREru指数"] > 0)
        ]

        main = main_candidates.iloc[0] if len(main_candidates) > 0 else race_data.iloc[0]
        holes = hole_candidates.head(3)

        print()
        print("==============================")
        print(f"🏇 レース {int(race)}")
        print("==============================")
        print(f"🔥 荒れ度: {areru_level}")
        print(f"判定: {judge_text}")
        print()
        print(
            f"◎ 本命 {main['馬名']} "
            f"AREru {main['AREru指数']} "
            f"/ {main['単勝オッズ']}倍 "
            f"/ {int(main['人気'])}人気"
        )
        print()
        print("💣 穴候補")

        hole_names = []
        for _, horse in holes.iterrows():
            print(
                f"・{horse['馬名']} "
                f"AREru {horse['AREru指数']} "
                f"/ {horse['単勝オッズ']}倍 "
                f"/ {int(horse['人気'])}人気"
            )
            hole_names.append(horse["馬名"])

        if len(holes) == 0:
            print("・該当なし")

        predictions.append({
            "race_id": race_id,
            "レース": int(race),
            "荒れ度": areru_level,
            "判定": judge_text,
            "本命": main["馬名"],
            "本命AREru指数": main["AREru指数"],
            "本命オッズ": main["単勝オッズ"],
            "穴候補": " / ".join(hole_names) if hole_names else "該当なし",
        })

    result = pd.DataFrame(predictions).sort_values("レース")

    if result["本命オッズ"].isna().any():
        bad = result.loc[result["本命オッズ"].isna(), ["レース", "本命"]]
        raise ValueError("本命オッズ欠損が残っています:\n" + bad.to_string(index=False))

    result.to_csv("data/predictions.csv", index=False, encoding="utf-8-sig")

    print()
    print("==============================")
    print("✅ 最終予想作成完了")
    print("レース数:", len(result))
    print("本命オッズ欠損:", int(result["本命オッズ"].isna().sum()))
    print("📁 data/predictions.csv")


if __name__ == "__main__":
    predict()
