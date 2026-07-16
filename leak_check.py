import pandas as pd


history = pd.read_csv(
    "data/all_history.csv",
    encoding="utf-8-sig"
)

TARGET_DATE = pd.Timestamp("2026-07-12")

date_text = history["年月日"].astype(str)

history["解析日付"] = pd.to_datetime(
    date_text.str.extract(
        r"(\d{4}年\d{1,2}月\d{1,2}日)",
        expand=False
    ),
    format="%Y年%m月%d日",
    errors="coerce"
)

leak = history[
    history["解析日付"] >= TARGET_DATE
]

print()
print("========================")
print("🚨 AREru ガチカンニング検査")
print("========================")

print("全過去走データ:", len(history))
print("解析成功:", history["解析日付"].notna().sum())
print("解析失敗:", history["解析日付"].isna().sum())

print()
print("最古日付:", history["解析日付"].min())
print("最新日付:", history["解析日付"].max())
print("2026年7月12日以降:", len(leak))

if len(leak) > 0:

    print()
    print("❌ データ漏洩あり！！")

    print(
        leak[
            [
                "今回レース",
                "馬名",
                "年月日",
                "着順"
            ]
        ].to_string(index=False)
    )

else:

    print()
    print("✅ データ漏洩なし")
    print("当日結果は予想に入っていません")