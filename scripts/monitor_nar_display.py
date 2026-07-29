#!/usr/bin/env python3
"""地方表示の健全性チェック（開催場・レース一覧・予想）。

exit 0: すべて OK
exit 1: いずれか失敗
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))


def fetch(url: str, timeout: int = 90) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "areru-nar-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.status, res.read().decode("utf-8", errors="replace")


def main() -> int:
    base = (os.environ.get("APP_URL") or "https://areru-cloud.onrender.com").rstrip("/")
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S%z")
    report = {"ok": False, "at": now, "base": base, "checks": {}}

    # 1) healthz
    try:
        code, body = fetch(f"{base}/healthz", timeout=60)
        health = json.loads(body)
        report["checks"]["healthz"] = {"ok": code == 200 and bool(health.get("ok")), "code": code}
    except Exception as e:
        report["checks"]["healthz"] = {"ok": False, "error": str(e)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    # 2) job-status
    try:
        code, body = fetch(f"{base}/api/job-status?source=nar", timeout=60)
        st = json.loads(body)
        report["checks"]["job_status"] = {
            "ok": code == 200 and bool(st.get("ok")),
            "ready": bool(st.get("ready")),
            "heavy": st.get("heavy") or "",
            "today": st.get("today"),
        }
    except Exception as e:
        report["checks"]["job_status"] = {"ok": False, "error": str(e)}

    # 3) venue picker (cache-first)
    try:
        code, html = fetch(f"{base}/?source=nar&mode=predict", timeout=90)
        generating = ("<h2>データ取得中</h2>" in html) or ("初回データ" in html)
        venues = re.findall(r'data-venue="([^"]+)"', html)
        selected = re.findall(r'<option value="(\d{4}-\d{2}-\d{2})"[^>]*selected', html)
        date = selected[0] if selected else ""
        report["checks"]["venues"] = {
            "ok": code == 200 and (not generating) and len(venues) > 0,
            "code": code,
            "generating": generating,
            "count": len(venues),
            "venues": venues,
            "selected": date,
        }
    except Exception as e:
        report["checks"]["venues"] = {"ok": False, "error": str(e)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    # 4) first venue races + predictions
    # history なしでも開けること（キャッシュ日の開くリンク退行防止）
    venues = report["checks"]["venues"].get("venues") or []
    date = report["checks"]["venues"].get("selected") or ""
    if venues and date:
        v = venues[0]
        q = urllib.parse.urlencode(
            {
                "source": "nar",
                "mode": "predict",
                "date": date,
                "venue": v,
            }
        )
        try:
            code, html = fetch(f"{base}/?{q}", timeout=120)
            generating = ("<h2>データ取得中</h2>" in html) or ("初回データ" in html)
            err = "通信エラー" in html
            stuck_picker = bool(re.findall(r'data-venue="([^"]+)"', html)) and ("レース一覧" not in html)
            has_races = ("レース一覧" in html) or ("本命" in html)
            has_pred = ("本命" in html) or ("推奨" in html)
            report["checks"]["races"] = {
                "ok": code == 200 and (not generating) and (not err) and has_races and (not stuck_picker),
                "code": code,
                "venue": v,
                "date": date,
                "generating": generating,
                "error": err,
                "stuck_on_picker": stuck_picker,
                "has_races": has_races,
            }
            report["checks"]["predictions"] = {
                "ok": code == 200 and (not generating) and (not err) and has_pred and (not stuck_picker),
                "has_predictions": has_pred,
                "honmei_count": html.count("本命"),
            }
        except Exception as e:
            report["checks"]["races"] = {"ok": False, "error": str(e)}
            report["checks"]["predictions"] = {"ok": False, "error": str(e)}
    else:
        report["checks"]["races"] = {"ok": False, "error": "no venues"}
        report["checks"]["predictions"] = {"ok": False, "error": "no venues"}

    ok = all(bool((report["checks"].get(k) or {}).get("ok")) for k in ("healthz", "venues", "races", "predictions"))
    report["ok"] = ok
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
