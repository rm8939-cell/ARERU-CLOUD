from save_horses import save_horses
from horse_links import save_horse_links
from all_history import save_all_history
from analyzer import analyze_races
from race_judge import judge_races
from odds import get_odds
from predict import predict


def main():
    print("==============================")
    print("🏇 AREru v1.0 FINAL")
    print("==============================")
    print()

    print("① 出走馬取得")
    save_horses()

    print()
    print("② 馬リンク取得")
    save_horse_links()

    print()
    print("③ オッズ・現出走馬確認")
    get_odds()

    print()
    print("④ 過去走取得")
    save_all_history()

    print()
    print("⑤ AREru指数計算")
    analyze_races()

    print()
    print("⑥ 波乱判定")
    judge_races()

    print()
    print("⑦ 最終予想")
    predict()

    print()
    print("==============================")
    print("✅ 全処理完了")
    print("==============================")
    print("📁 data/predictions.csv")


if __name__ == "__main__":
    main()
