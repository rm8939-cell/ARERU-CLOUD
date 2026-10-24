"""レースを「買うべき」「少額なら買える」「見送る」に分ける。

期待値だけで買い目を決めると、当たりにくい人気薄ばかりが残る。
12,468レースの検証では、期待値上位で買う方針は的中率10.2%・最大35連敗・
最大ドローダウン144単位で、資金が少ない人には使えなかった。
いっぽうレース単位で「読みやすさ」を見て買うレースを絞ると、本命の勝率が
42.3%→60.6%、最大ドローダウンが600単位→145単位まで下がった。

ここで使うのは、レース内の勝率の形から出る7つの指標だけ。着順や配当は
使わないので、発走前に計算できる。

  1位2位差      1位と2位の勝率差
  上位3集中度    上位3頭の勝率合計
  確率分散      勝率のばらつき
  AI1位信頼度    1位の勝率そのもの
  人気乖離      AI評価と人気の順位相関のずれ
  穴馬候補      オッズ10倍以上でAI評価が高い馬の最大勝率
  SIM安定性     揺らしても1位が入れ替わらない割合

係数は data/race_grade_model.json に入れてある。学習は 2026-01-01〜2026-07-26
の9,351レースで、検証は 2026-07-26〜2026-09-30 の別3,117レース。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "data" / "race_grade_model.json"
LONGSHOT_ODDS = 10.0
SIM_NOISE = 0.35
SIM_DRAWS = 400
FEATURES = ["1位2位差", "上位3集中度", "確率分散", "AI1位信頼度",
            "人気乖離", "穴馬候補", "SIM安定性"]

_CACHE: dict | None = None
_LOADED = False


def enabled() -> bool:
    """ARERU_BUY_V2 が立っているときだけ使う。既定は従来どおり。"""
    return str(os.getenv("ARERU_BUY_V2", "")).strip().lower() in ("1", "true", "yes", "on")


def reset_cache() -> None:
    global _CACHE, _LOADED
    _CACHE, _LOADED = None, False


def load_model() -> dict | None:
    global _CACHE, _LOADED
    if _LOADED:
        return _CACHE
    _LOADED = True
    path = Path(os.getenv("ARERU_GRADE_MODEL", "") or MODEL_PATH)
    try:
        _CACHE = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _CACHE = None
    return _CACHE


def sim_stability(win_prob: np.ndarray, seed: int = 0) -> float:
    """log勝率を揺らして並べ替えたとき、1位が1位のままでいる割合。

    係数も過去走もノイズを含むので、少し揺らしても順位が動かないレースは
    「確率の出方が安定している」と見なせる。実機のモンテカルロと同じ考え方。
    """
    p = np.asarray(win_prob, dtype=float)
    if len(p) < 2:
        return 1.0
    lo = np.log(np.clip(p, 1e-9, None))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, SIM_NOISE, size=(SIM_DRAWS, len(p)))
    top = int(np.argmax(p))
    return float((np.argmax(lo[None, :] + noise, axis=1) == top).mean())


def signals(win_prob, odds=None, seed: int = 0) -> dict | None:
    """レース単位の7指標。4頭未満や勝率が全部0のレースは None。"""
    p = np.asarray(win_prob, dtype=float)
    p = np.where(np.isfinite(p), p, 0.0)
    if len(p) < 4 or p.sum() <= 0:
        return None
    p = p / p.sum()
    srt = np.sort(p)[::-1]

    dev = 0.0
    hole = 0.0
    if odds is not None:
        o = np.asarray(odds, dtype=float)
        if len(o) == len(p) and np.isfinite(o).all() and (o > 0).all():
            ra = np.argsort(np.argsort(-p)).astype(float)
            rm = np.argsort(np.argsort(o)).astype(float)
            if np.std(ra) > 1e-9 and np.std(rm) > 1e-9:
                dev = 1.0 - float(np.corrcoef(ra, rm)[0, 1])
            ls = p[o >= LONGSHOT_ODDS]
            hole = float(ls.max()) if ls.size else 0.0

    return {
        "1位2位差": float(srt[0] - srt[1]),
        "上位3集中度": float(srt[:3].sum()),
        "確率分散": float(np.std(p) * np.sqrt(len(p))),
        "AI1位信頼度": float(srt[0]),
        "人気乖離": dev,
        "穴馬候補": hole,
        "SIM安定性": sim_stability(p, seed=seed),
    }


def _logit(z: np.ndarray, beta: list[float]) -> float:
    v = float(beta[0] + float(np.dot(z, beta[1:])))
    return 1.0 / (1.0 + np.exp(-max(min(v, 30.0), -30.0)))


def grade(win_prob, odds=None, seed: int = 0) -> dict | None:
    """レース区分と、その根拠になった確率を返す。

    判定は2軸。予想しやすさが上位で荒れ寄りでなければ「買うべき」、
    予想しやすさが下位（＝混戦）か荒れ寄りなら「見送る」。
    期待値は一切見ない。期待値はこの判定を通ったあとの馬選びにだけ使う。
    """
    m = load_model()
    sig = signals(win_prob, odds, seed=seed)
    if m is None or sig is None:
        return None
    cols = m.get("特徴量") or FEATURES
    mu, sd = m["平均"], m["標準偏差"]
    z = np.array([(sig[c] - mu[c]) / (sd[c] if sd[c] > 1e-9 else 1.0)
                  for c in cols], dtype=float)
    ph = _logit(z, m["予想しやすさ係数"])
    pu = _logit(z, m["荒れやすさ係数"])

    if ph < m["見送り下限"] or pu >= m["荒れ上限"]:
        cls = "見送る"
    elif ph >= m["買うべき下限"]:
        cls = "買うべき"
    else:
        cls = "少額なら買える"
    return {"レース区分": cls,
            "予想しやすさ": round(float(ph) * 100, 1),
            "荒れやすさ": round(float(pu) * 100, 1),
            "指標": {k: round(float(v), 4) for k, v in sig.items()}}
