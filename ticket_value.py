"""券種別実オッズから合成オッズ・期待回収率を算出する。

未取得のオッズは数字を作らない（空文字を返す）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA = Path("data")


def _norm_umaban(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        return f"{int(float(s)):02d}"
    except Exception:
        return s if s.isdigit() else ""


def combo_key(umabans: list[str], kind: str) -> str:
    nums = sorted({_norm_umaban(u) for u in umabans if _norm_umaban(u)})
    if kind in ("ワイド", "馬連"):
        if len(nums) != 2:
            return ""
        return "".join(nums)
    if kind == "三連複":
        if len(nums) != 3:
            return ""
        return "".join(nums)
    return "".join(nums)


def _pad_combo(kind: str, combo: str) -> str:
    digits = re.sub(r"\D", "", str(combo))
    if not digits:
        return ""
    # CSV経由で先頭ゼロが落ちるため券種別に桁を復元
    width = {"ワイド": 4, "馬連": 4, "三連複": 6}.get(kind, 0)
    if width and len(digits) < width:
        digits = digits.zfill(width)
    return digits


def load_ticket_odds(path: Path | None = None) -> pd.DataFrame:
    p = path or DATA / "ticket_odds.csv"
    if not p.exists():
        return pd.DataFrame(columns=["race_id", "券種", "組合せ", "オッズ"])
    df = pd.read_csv(p, dtype={"組合せ": str, "race_id": str})
    if df.empty:
        return df
    df["race_id"] = df["race_id"].astype(str)
    df["券種"] = df["券種"].astype(str)
    df["組合せ"] = [
        _pad_combo(k, str(c).lstrip("'"))
        for k, c in zip(df["券種"].tolist(), df["組合せ"].tolist())
    ]
    df["オッズ"] = pd.to_numeric(df["オッズ"], errors="coerce")
    return df


def parse_bet_line(line: str) -> tuple[list[str], float | None]:
    """'馬A － 馬B（仮想的中 12.3%）' -> names, hit%"""
    text = str(line).strip()
    if not text or text == "見送り":
        return [], None
    hit = None
    m = re.search(r"仮想的中\s*([0-9.]+)\s*%", text)
    if m:
        hit = float(m.group(1))
    names_part = re.sub(r"（.*?）", "", text).strip()
    names = [n.strip() for n in re.split(r"\s*[－\-]\s*", names_part) if n.strip()]
    return names, hit


def name_to_umaban_map(scores: pd.DataFrame) -> dict[str, str]:
    m = {}
    if scores is None or scores.empty:
        return m
    for _, row in scores.iterrows():
        name = str(row.get("馬名", "")).strip()
        uma = _norm_umaban(row.get("馬番", ""))
        if name and uma:
            m[name] = uma
            m[re.sub(r"\s+", "", name)] = uma
    return m


def lookup_odds(ticket_odds: pd.DataFrame, race_id: str, kind: str, key: str) -> float | None:
    if ticket_odds is None or ticket_odds.empty or not key:
        return None
    sub = ticket_odds[
        (ticket_odds["race_id"].astype(str) == str(race_id))
        & (ticket_odds["券種"] == kind)
        & (ticket_odds["組合せ"] == key)
    ]
    if sub.empty:
        # try unsorted variants already normalized
        return None
    val = sub.iloc[0]["オッズ"]
    return float(val) if pd.notna(val) else None


def enrich_race_row(row: dict, scores_race: pd.DataFrame, ticket_odds: pd.DataFrame) -> dict:
    """1レース分の予測行に合成オッズ/期待回収率を付与。"""
    name_map = name_to_umaban_map(scores_race)
    race_id = str(row.get("race_id", ""))
    best_kind = str(row.get("推奨券種", ""))
    synth_parts = []
    ev_parts = []

    for kind in ("ワイド", "馬連", "三連複"):
        lines = str(row.get(f"{kind}買い目", "見送り")).split("｜")
        odds_list = []
        weighted = []
        for line in lines:
            names, hit = parse_bet_line(line)
            if not names:
                continue
            umas = [name_map.get(n) or name_map.get(re.sub(r"\s+", "", n), "") for n in names]
            key = combo_key(umas, kind)
            oddsv = lookup_odds(ticket_odds, race_id, kind, key)
            if oddsv is None:
                continue
            odds_list.append(oddsv)
            if hit is not None:
                # 仮想的中率(%) × オッズ → 期待回収率(%)
                weighted.append(hit * oddsv)
        if odds_list:
            # 複数点は単純平均（未取得は混ぜない）
            avg_odds = sum(odds_list) / len(odds_list)
            row[f"{kind}合成オッズ"] = round(avg_odds, 1)
            if weighted:
                row[f"{kind}期待回収率"] = round(sum(weighted) / len(weighted), 1)
            else:
                row[f"{kind}期待回収率"] = ""
            if kind == best_kind:
                synth_parts.append(f"{kind} {avg_odds:.1f}倍")
                if weighted:
                    ev_parts.append(f"{sum(weighted)/len(weighted):.1f}%")
        else:
            row[f"{kind}合成オッズ"] = ""
            row[f"{kind}期待回収率"] = ""

    if synth_parts:
        row["合成オッズ"] = " / ".join(synth_parts)
        row["期待回収率"] = " / ".join(ev_parts) if ev_parts else ""
    else:
        row["合成オッズ"] = "券種別オッズ待ち"
        row["期待回収率"] = "オッズ接続後に算出"
    return row


def enrich_predictions(result: pd.DataFrame, scores: pd.DataFrame, ticket_odds: pd.DataFrame | None = None) -> pd.DataFrame:
    ticket_odds = load_ticket_odds() if ticket_odds is None else ticket_odds
    rows = []
    for _, row in result.iterrows():
        d = row.to_dict()
        race_id = str(d.get("race_id", ""))
        sr = scores[scores["race_id"].astype(str) == race_id] if scores is not None and len(scores) else pd.DataFrame()
        rows.append(enrich_race_row(d, sr, ticket_odds))
    return pd.DataFrame(rows)
