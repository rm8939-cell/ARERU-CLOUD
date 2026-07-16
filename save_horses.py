import csv

from all_races import get_all_horses


def save_horses():
    horses = get_all_horses()

    file_path = "data/all_horses.csv"

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["race_id", "race_number", "horse"]
        )

        writer.writeheader()
        writer.writerows(horses)

    print()
    print("====================")
    print("✅ CSV保存完了")
    print("📁", file_path)
    print("🐴 保存件数:", len(horses))


if __name__ == "__main__":
    save_horses()