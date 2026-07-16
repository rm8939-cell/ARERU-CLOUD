import pandas as pd
from pathlib import Path

SCORE_FILE = Path("data/score_test_data.csv")
OUT_FILE = Path("data/arerU_ticket_analysis.csv")

print()
print("=" * 60)
print("💰 ARERU.EXE 馬券戦略探索")
print("=" * 60)

if not SCORE_FILE.exists():
    raise FileNotFoundError(f"見つかりません: {SCORE_FILE}")

df = pd.read_csv(SCORE_FILE)

print("読込行数:", len(df))
print("列:", list(df.columns))

required = [
    "race_id",
    "着順1", "人気1",
    "着順2", "人気2",
    "着順3", "人気3",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"必要列なし: {missing}")

for c in ["着順1", "人気1", "着順2", "人気2", "着順3", "人気3"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

score_cols = ["着順1", "着順2", "着順3", "着順4", "着順5"]
score_points = [30, 25, 10, 5, 5]

df["ARERU_SIM"] = 0.0

for col, point in zip(score_cols, score_points):
    if col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce")
        df["ARERU_SIM"] += (x == 1).astype(float) * point

pop_cols = ["人気1", "人気2", "人気3", "人気4", "人気5"]

for i, col in enumerate(pop_cols):
    if col not in df.columns:
        continue

    pop = pd.to_numeric(df[col], errors="coerce")
    finish_col = f"着順{i+1}"

    if finish_col not in df.columns:
        continue

    finish = pd.to_numeric(df[finish_col], errors="coerce")

    df["ARERU_SIM"] += (
        ((pop >= 6) & (finish <= 3)).astype(float)
        * (pop - 5)
        * 2
    )

df["TOP3_POP_SUM"] = (
    df["人気1"].fillna(99)
    + df["人気2"].fillna(99)
    + df["人気3"].fillna(99)
)

df["HOLE_IN_TOP3"] = (
    (df["人気1"] >= 6)
    | (df["人気2"] >= 6)
    | (df["人気3"] >= 6)
).astype(int)

bands = [
    (0, 15, "静穏"),
    (15, 20, "やや注意"),
    (20, 25, "波乱注意"),
    (25, 30, "高波乱"),
    (30, 9999, "大波乱"),
]

rows = []

for low, high, label in bands:
    part = df[
        (df["ARERU_SIM"] >= low)
        & (df["ARERU_SIM"] < high)
    ]

    if len(part) == 0:
        continue

    rows.append({
        "指数帯": f"{low}〜{high}",
        "判定": label,
        "レース数": len(part),
        "穴3着内数": int(part["HOLE_IN_TOP3"].sum()),
        "穴3着内率": round(part["HOLE_IN_TOP3"].mean() * 100, 2),
        "平均上位人気合計": round(part["TOP3_POP_SUM"].mean(), 2),
    })

result = pd.DataFrame(rows)

print()
print("🔥 指数帯別 馬券傾向")
print(result.to_string(index=False))

def ticket_type(row):
    score = row["ARERU_SIM"]

    if score < 15:
        return "単勝・馬連候補"
    elif score < 20:
        return "ワイド候補"
    elif score < 25:
        return "ワイド・3連複候補"
    elif score < 30:
        return "3連複候補"
    else:
        return "穴ワイド・3連複候補"

df["推奨券種"] = df.apply(ticket_type, axis=1)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(
    OUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

df[
    [
        "race_id",
        "ARERU_SIM",
        "TOP3_POP_SUM",
        "HOLE_IN_TOP3",
        "推奨券種",
    ]
].to_csv(
    "data/arerU_ticket_races.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 60)
print("🏆 ARERU.EXE 馬券戦略 仮判定")
print("=" * 60)

best = result[
    result["レース数"] >= 30
].sort_values(
    "穴3着内率",
    ascending=False
)

if len(best):
    print(best.head(5).to_string(index=False))
else:
    print("⚠️ 母数30以上の指数帯なし")

print()
print("✅ 保存完了")
print("📄 data/arerU_ticket_analysis.csv")
print("📄 data/arerU_ticket_races.csv")
print("=" * 60)
