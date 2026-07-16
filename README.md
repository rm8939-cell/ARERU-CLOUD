# ARERU.CLOUD β

競馬AI予想クラウド（JRA中央 / NAR地方）。

## 最短起動

```bash
pip install -r requirements.txt
python3 refresh_data.py --dates 2026-07-18 2026-07-19 --predict   # JRA最新
python3 refresh_nar.py --dates 2026-07-16                          # NAR地方
gunicorn web_app:app -b 0.0.0.0:${PORT:-5001}
```

ブラウザで `http://localhost:5001` を開く。

## 主要機能（β）

- JRA: 20,000回仮想レース / S-A-B-C / ワイド・馬連・三連複候補圧縮
- 日付自動認識と `refresh_data.py` による最新データ生成→Web反映
- 券種別実オッズ接続時のみ合成オッズ・期待回収率を表示
- 払戻ベース実回収率（分析タブ、取得済み日のみ）
- NAR地方専用エンジン（ダート適性・開催場実績軸、JRAロジック非流用）
- 結果検証タブ（◎○▲△☆ × 実着順）

## データ更新

| コマンド | 内容 |
|---|---|
| `python3 refresh_data.py --auto --predict` | JRA開催日探索→取込→予想 |
| `python3 refresh_nar.py --auto` | NAR開催日探索→取込→予想 |
| `python3 replay_predict.py --all` | 保存済みJRA日付の予想再生成 |
| `python3 roi_analyzer.py --all` | 払戻ベース回収率集計 |

## デプロイ

- Render: `render.yaml`（`gunicorn web_app:app`）
- Railway: `Procfile` / `railway.toml`
- Health: `GET /health`

## 注意

- 未取得のオッズ・払戻・回収率は数字を作りません
- 買い期待度は回収率ではありません
- 実着順は予想入力に未使用（検証・最適化ラベル専用）
