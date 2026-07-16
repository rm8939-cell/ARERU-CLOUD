import csv
import os


def save_csv(filename, data):
    os.makedirs("data", exist_ok=True)

    filepath = f"data/{filename}"

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for row in data:
            writer.writerow(row)

    print(f"✅ {filepath} を保存しました")