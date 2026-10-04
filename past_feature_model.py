"""過去走特徴量を確率に載せるための、学習済み係数とその適用。

係数は `scripts/fit_past_feature_model.py` が train 期間だけで推定し
`data/past_feature_model.json` に書く。ここでは読んで当てはめるだけで、
手で重みを決めることはしない。

適用のしかたは確率v2の合成と同じ。レース内 softmax の線形項に足す。

    eta = log(市場勝率) + Σ βᵢ · zᵢ

zᵢ はレース内 z スコアで、欠損馬は 0（＝レース平均）になる。
データが無い馬を減点しないため。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path("data/past_feature_model.json")

_model: dict | None = None
_model_loaded = False
_panel: pd.DataFrame | None = None
_panel_loaded = False


def enabled() -> bool:
    """過去走特徴量を最終確率に載せるか。既定は off。"""
    return str(os.environ.get("ARERU_PAST_FEATURES") or "").strip().lower() in (
        "1", "true", "yes",
    )


def load_model(path: str | Path | None = None) -> dict | None:
    global _model, _model_loaded
    if _model_loaded and path is None:
        return _model
    p = Path(path or os.environ.get("ARERU_PAST_MODEL") or MODEL_PATH)
    m = None
    if p.exists():
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            m = None
    if path is None:
        _model, _model_loaded = m, True
    return m


def load_panel(path: str | Path | None = None) -> pd.DataFrame | None:
    """(race_id, 馬名) で引ける特徴量パネル。

    バックテストでは事前に作ったCSVを環境変数 ARERU_PAST_PANEL で渡す。
    本番でこれを使う場合はキャッシュから同じ特徴量をその場で計算する実装に
    差し替える想定で、どちらも同じ列名を返す。
    """
    global _panel, _panel_loaded
    if _panel_loaded and path is None:
        return _panel
    p = os.environ.get("ARERU_PAST_PANEL") if path is None else str(path)
    df = None
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, low_memory=False)
            df["race_id"] = df["race_id"].astype(str)
            df["馬名"] = df["馬名"].astype(str).str.strip()
            df = df.drop_duplicates(["race_id", "馬名"]).set_index(["race_id", "馬名"])
        except Exception:
            df = None
    if path is None:
        _panel, _panel_loaded = df, True
    return df


def reset_cache() -> None:
    global _model, _model_loaded, _panel, _panel_loaded
    _model = _panel = None
    _model_loaded = _panel_loaded = False


def race_z(values: np.ndarray) -> np.ndarray:
    """レース内 z スコア。欠損は 0（レース平均と同じ扱い）。"""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.zeros_like(v)
    mu = v[ok].mean()
    sd = v[ok].std()
    out = np.zeros_like(v)
    if sd > 0:
        out[ok] = (v[ok] - mu) / sd
    return out


def linear_term(race_id: str, names) -> np.ndarray | None:
    """この出走表に対する Σ βᵢ·zᵢ。使えないときは None。"""
    model = load_model()
    if not model or not model.get("係数"):
        return None
    panel = load_panel()
    if panel is None:
        return None
    names = [str(n).strip() for n in names]
    keys = [(str(race_id), n) for n in names]
    try:
        sub = panel.reindex(keys)
    except Exception:
        return None
    if sub is None or sub.empty:
        return None
    total = np.zeros(len(names), dtype=float)
    used = 0
    for col, beta in model["係数"].items():
        if col not in sub.columns:
            continue
        total += float(beta) * race_z(pd.to_numeric(sub[col], errors="coerce").to_numpy())
        used += 1
    if used == 0:
        return None
    return total
