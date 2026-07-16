from pathlib import Path

def patch(name, replacements):
    p = Path(name)
    s = p.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in s:
            raise RuntimeError(f"{name}: 修正箇所が見つかりません")
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print("✅", name)

patch("races.py", [(
'''        race_urls.append(race_url)

        print(
            f"{race_number}レース",
            race_url
        )
''',
'''        race_urls.append({
            "race_id": race_url,
            "race_number": race_number,
            "url": race_url
        })

        print(
            f"{race_number}レース",
            race_url
        )
'''
)])

patch("all_races.py", [
("    race_urls = get_races()", "    races = get_races()"),
("    for number, race_url in enumerate(race_urls, start=1):",
 "    for number, race in enumerate(races, start=1):\n        race_url = race[\"url\"]"),
('f"🏇 {number}/{len(race_urls)} レース取得中..."',
 'f"🏇 {number}/{len(races)} {race[\'race_number\']}R 取得中..."'),
('''            all_horses.append({
                "race": number,
                "horse": horse
            })
''',
'''            all_horses.append({
                "race_id": race["race_id"],
                "race_number": race["race_number"],
                "horse": horse
            })
''')
])

patch("save_horses.py", [
('fieldnames=["race", "horse"]',
 'fieldnames=["race_id", "race_number", "horse"]')
])

patch("horse_links.py", [
("    race_urls = get_races()", "    races = get_races()"),
("    for number, race_url in enumerate(race_urls, start=1):",
 "    for number, race in enumerate(races, start=1):\n        race_url = race[\"url\"]"),
('f"🏇 {number}/{len(race_urls)} レース取得中..."',
 'f"🏇 {number}/{len(races)} {race[\'race_number\']}R 取得中..."'),
('''                horse_data.append({
                    "race": number,
                    "horse": horse_name,
                    "url": horse_url
                })
''',
'''                horse_data.append({
                    "race_id": race["race_id"],
                    "race_number": race["race_number"],
                    "horse": horse_name,
                    "url": horse_url
                })
'''),
('fieldnames=["race", "horse", "url"]',
 'fieldnames=["race_id", "race_number", "horse", "url"]')
])

patch("all_history.py", [
('''    columns = [
        "今回レース",
        "馬名",
''',
'''    columns = [
        "race_id",
        "今回レース",
        "馬名",
'''),
('''                history = [
                    horse["race"],
                    horse["horse"],
''',
'''                history = [
                    horse["race_id"],
                    horse["race_number"],
                    horse["horse"],
''')
])

patch("analyzer.py", [
('''    grouped = history.groupby(
        ["今回レース", "馬名"]
    )

    for (race, horse), data in grouped:
''',
'''    grouped = history.groupby(
        ["race_id", "今回レース", "馬名"]
    )

    for (race_id, race, horse), data in grouped:
'''),
('''        results.append({
            "レース": race,
''',
'''        results.append({
            "race_id": race_id,
            "レース": race,
'''),
('result.groupby("レース")', 'result.groupby("race_id")')
])

patch("race_judge.py", [
('for race, race_data in df.groupby("レース"):',
 'for race_id, race_data in df.groupby("race_id"):\n\n        race = race_data.iloc[0]["レース"]'),
('''        results.append({
            "レース": race,
''',
'''        results.append({
            "race_id": race_id,
            "レース": race,
''')
])

patch("odds.py", [
("    race_urls = get_races()", "    races = get_races()"),
('''    for race_number, race_url in enumerate(
        race_urls,
        start=1
    ):
''',
'''    for index, race in enumerate(
        races,
        start=1
    ):
        race_number = race["race_number"]
        race_url = race["url"]
'''),
('f"{len(race_urls)} レース"', 'f"{len(races)} レース"'),
('''            results.append({
                "レース": race_number,
''',
'''            results.append({
                "race_id": race["race_id"],
                "レース": race_number,
''')
])

patch("results.py", [
("    race_urls = get_races()", "    races = get_races()"),
('''    for race_number, race_url in enumerate(
        race_urls,
        start=1
    ):
''',
'''    for index, race in enumerate(
        races,
        start=1
    ):
        race_number = race["race_number"]
        race_url = race["url"]
'''),
('f"{len(race_urls)} 結果取得中..."', 'f"{len(races)} 結果取得中..."'),
('''            results.append({
                "レース": race_number,
''',
'''            results.append({
                "race_id": race["race_id"],
                "レース": race_number,
''')
])

patch("predict.py", [
('''    merged = pd.merge(
        areru,
        odds,
        on=["レース", "馬名"],
        how="left"
    )
''',
'''    merged = pd.merge(
        areru,
        odds[["race_id", "馬名", "単勝オッズ", "人気"]],
        on=["race_id", "馬名"],
        how="left"
    )
'''),
('for race, race_data in merged.groupby("レース"):',
 'for race_id, race_data in merged.groupby("race_id"):\n\n        race = race_data.iloc[0]["レース"]'),
('''        judge_data = judgement[
            judgement["レース"] == race
        ]
''',
'''        judge_data = judgement[
            judgement["race_id"] == race_id
        ]
'''),
('''        predictions.append({
            "レース": race,
''',
'''        predictions.append({
            "race_id": race_id,
            "レース": race,
''')
])

patch("verify.py", [
('''        race = pred["レース"]
        main_horse = str(pred["本命"]).strip()

        race_results = results[
            results["レース"] == race
        ]
''',
'''        race = pred["レース"]
        race_id = pred["race_id"]
        main_horse = str(pred["本命"]).strip()

        race_results = results[
            results["race_id"] == race_id
        ]
'''),
('''        verification.append({
            "レース": race,
''',
'''        verification.append({
            "race_id": race_id,
            "レース": race,
''')
])

print()
print("🔥 race_id統一修正 完了")
print("次は全CSVを作り直してください")
