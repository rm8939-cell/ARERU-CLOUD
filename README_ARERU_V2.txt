ARERU.EXE v2

【最短起動】
1. ターミナルで ARERU.EXE_FINAL に移動
2. python3 replay_predict.py --all
3. python3 web_app.py
4. iPhoneで http://MacのIP:5001 を開く

Macでは run_areru.command をダブルクリックでも起動できます。

【今回の改修】
・score_test_data.csv に存在する全開催日を自動認識
・--all で全開催日を一括再現
・Webの日付候補を自動表示
・未生成日はWeb表示時に自動生成
・対象日より前の履歴だけを参照（未来データ混入防止）
・近走パフォーマンス / 人気着順乖離 / 安定感 / 上向き度 / 過小評価傾向 / 条件適性を複合指数化
・人気馬危険度 / 穴馬上昇評価 / 荒れ度 / 理由 / 推奨券種を表示
・optimize_v2.py で時系列ホールドアウト付き重み最適化

【P0-2】
・python3 refresh_data.py --latest-only で最新開催日を netkeiba から自動取得
・canonical 出走データは data/runners.csv（score_test_data.csv 依存を廃止）
・predictions_by_date を自動生成
・Web日付プルダウンは runners.csv ∪ 生成済み predictions を表示

【重要なデータ上の制約】
Webに出せる開催日は data/runners.csv（なければ旧 score_test_data.csv）と predictions_by_date の日付が基準です。
また過去日出馬データに当日の競馬場・距離・馬場・斤量・騎手が保存されていないため、v2の「条件適性」は対象日前の履歴から推定しています。
本当の当日条件ベース（距離替わり、馬場替わり、斤量変化、騎手替わり）にするには、今後スクレイパー側で各開催日の出馬表条件を保存する必要があります。

【最適化】
python3 optimize_v2.py
python3 replay_predict.py --all

実着順は最適化の評価ラベルだけに使用し、予想指数の入力には使用していません。
