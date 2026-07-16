ARERU.EXE v2

【最短起動】
1. ターミナルでリポジトリへ移動
2. python3 refresh_data.py --latest-only
3. python3 web_app.py
4. iPhoneで http://MacのIP:5001 を開く

Macでは run_areru.command をダブルクリックでも起動できます。

【P0-2 改修】
・最新開催日を netkeiba から自動検出（refresh_data.py）
・出走データの正本を data/runners.csv に変更（score_test_data.csv 依存を廃止）
・predictions_by_date を日付指定 / --all / Web表示時に自動生成
・Web日付プルダウンは runners.csv + 生成済み predictions を合流し最新日まで表示
・/refresh で最新開催週末の再取得が可能

【従来からの機能】
・対象日より前の履歴だけを参照（未来データ混入防止）
・近走パフォーマンス / 人気着順乖離 / 安定感 / 上向き度 / 過小評価傾向 / 条件適性を複合指数化
・人気馬危険度 / 穴馬上昇評価 / 荒れ度 / 理由 / 推奨券種を表示
・optimize_v2.py で時系列ホールドアウト付き重み最適化

【データ更新】
python3 refresh_data.py --list
python3 refresh_data.py --latest-only
python3 refresh_data.py --dates 2026-07-19
python3 replay_predict.py --all

【最適化】
python3 optimize_v2.py
python3 replay_predict.py --all

実着順は最適化の評価ラベルだけに使用し、予想指数の入力には使用していません。
