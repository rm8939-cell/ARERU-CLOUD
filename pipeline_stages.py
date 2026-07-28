"""パイプライン7工程の統一ログ（地方・JRA共通）。

各工程は OK / ERROR で Render Logs に出力する。
"""
from __future__ import annotations

STAGE_NAMES: dict[int, str] = {
    1: "開催取得",
    2: "レース一覧取得",
    3: "出馬表取得",
    4: "AI予想生成",
    5: "predictions保存",
    6: "キャッシュ更新",
    7: "画面反映",
}


def _src_label(source: str) -> str:
    s = str(source or "all").strip().lower()
    if s == "nar":
        return "NAR"
    if s == "jra":
        return "JRA"
    return "ALL"


def log_pipeline_stage(
    source: str,
    step: int,
    ok: bool,
    date_str: str = "",
    **kv,
) -> None:
    """[pipeline][NAR] 1.開催取得 OK date=2026-07-28 races=12"""
    label = _src_label(source)
    status = "OK" if ok else "ERROR"
    name = STAGE_NAMES.get(int(step), f"step{step}")
    bits = [f"[pipeline][{label}]", f"{step}.{name}", status]
    if date_str:
        bits.append(f"date={date_str}")
    for k, v in kv.items():
        if v is None or v == "" or v == []:
            continue
        bits.append(f"{k}={v}")
    print(" ".join(bits), flush=True)


def log_orchestrator(actor: str, action: str, source: str = "", **kv) -> None:
    """[orchestrator] actor=cron action=START source=nar mode=morning"""
    bits = [f"[orchestrator]", f"actor={actor}", f"action={action}"]
    if source:
        bits.append(f"source={source}")
    for k, v in kv.items():
        if v is None or v == "":
            continue
        bits.append(f"{k}={v}")
    print(" ".join(bits), flush=True)


def predictions_label(source: str, date_str: str) -> str:
    """ログ用ファイル名（実体は predictions_{date}.csv、ソース別件数で検証）。"""
    src = str(source or "all").strip().lower()
    if src in ("nar", "jra"):
        return f"predictions_{src}_{date_str}.csv"
    return f"predictions_{date_str}.csv"
