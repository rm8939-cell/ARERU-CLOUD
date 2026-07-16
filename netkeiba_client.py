"""netkeiba から JRA 開催日・出馬表・結果・馬履歴を取得するクライアント。"""
from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://race.netkeiba.com"
NAR_BASE = "https://nar.netkeiba.com"
DB = "https://db.netkeiba.com"
CACHE = Path("data/cache/horse_results")
CACHE.mkdir(parents=True, exist_ok=True)

VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


class NetkeibaClient:
    def __init__(self, sleep: float = 0.25):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; ARERU-CLOUD/1.0)",
            "Accept-Language": "ja,en;q=0.8",
        })
        self.sleep = sleep

    def _get(self, url: str, encoding: Optional[str] = None) -> BeautifulSoup:
        r = self.session.get(url, timeout=40)
        r.raise_for_status()
        if encoding:
            r.encoding = encoding
        else:
            # db.netkeiba は euc-jp、race.netkeiba は utf-8 が多い
            if "db.netkeiba.com" in url:
                r.encoding = r.apparent_encoding or "euc-jp"
            else:
                r.encoding = r.apparent_encoding or "utf-8"
        time.sleep(self.sleep)
        return BeautifulSoup(r.text, "lxml")

    def list_race_ids(self, yyyymmdd: str) -> list[str]:
        url = f"{BASE}/top/race_list_sub.html?kaisai_date={yyyymmdd}"
        soup = self._get(url)
        ids = sorted(set(re.findall(r"race_id=(\d{12})", str(soup))))
        return ids

    def discover_kaisai_dates(
        self,
        center: Optional[date] = None,
        lookback: int = 28,
        lookahead: int = 14,
    ) -> list[str]:
        """周辺カレンダーを走査し、レースがある開催日を新しい順で返す。"""
        center = center or date.today()
        found = []
        for offset in range(-lookback, lookahead + 1):
            d = center + timedelta(days=offset)
            key = d.strftime("%Y%m%d")
            try:
                ids = self.list_race_ids(key)
            except Exception:
                ids = []
            if ids:
                found.append(d.strftime("%Y-%m-%d"))
        return sorted(found, reverse=True)

    def latest_kaisai_date(self, center: Optional[date] = None) -> Optional[str]:
        dates = self.discover_kaisai_dates(center=center)
        return dates[0] if dates else None

    def parse_race_id(self, race_id: str) -> dict:
        m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})", str(race_id))
        if not m:
            return {"race_id": race_id, "venue": "開催地不明", "race_no": None, "date": None}
        year, venue, _kai, _day, race_no = m.groups()
        return {
            "race_id": race_id,
            "year": year,
            "venue_code": venue,
            "venue": VENUE_CODES.get(venue, "開催地不明"),
            "race_no": int(race_no),
        }

    def fetch_entries(self, race_id: str) -> list[dict]:
        url = f"{BASE}/race/shutuba.html?race_id={race_id}&rf=race_list"
        soup = self._get(url)
        meta = self.parse_race_id(race_id)
        # タイトルから日付を拾う
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        dm = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
        race_date = f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}" if dm else None

        rows = []
        for tr in soup.select("tr.HorseList"):
            a = tr.select_one("a[href*='/horse/']")
            if not a:
                continue
            name = a.get_text(strip=True)
            hm = re.search(r"/horse/(\d+)", a.get("href", ""))
            horse_id = hm.group(1) if hm else ""
            umaban = tr.select_one("td.Umaban1, td[class*=Umaban]")
            ban = umaban.get_text(strip=True) if umaban else ""
            # 出馬表に出ている暫定オッズ/人気（未発表は ---.-**）
            odds_td = tr.select_one("td.Txt_R.Popular, td.Popular")
            ninki_td = tr.select_one("td.Popular_Ninki")
            shutuba_odds = _parse_odds_text(odds_td.get_text(strip=True) if odds_td else "")
            shutuba_ninki = _num_or_blank(ninki_td.get_text(strip=True) if ninki_td else "")
            rows.append({
                "race_id": race_id,
                "日付": race_date,
                "レース": meta["race_no"],
                "馬名": name,
                "horse_id": horse_id,
                "馬番": ban,
                "開催地": meta["venue"],
                "単勝オッズ": shutuba_odds,
                "人気": shutuba_ninki,
            })
        return rows

    def fetch_odds_api(self, race_id: str, odds_type: int = 1) -> dict:
        """netkeiba JRA オッズ API。

        type: 1単勝 2複勝 3枠連 4馬連 5ワイド 6馬単 7三連複 8三連単
        戻り値: {status, updated_at, odds: {key: {...}}}
        """
        url = f"{BASE}/api/api_get_jra_odds.html?type={odds_type}&race_id={race_id}"
        r = self.session.get(url, timeout=40)
        r.raise_for_status()
        time.sleep(self.sleep)
        try:
            payload = r.json()
        except Exception:
            return {"status": "error", "updated_at": "", "odds": {}}
        status = str(payload.get("status") or "")
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return {"status": status or "empty", "updated_at": "", "odds": {}}
        updated = str(data.get("official_datetime") or "")
        raw = (data.get("odds") or {}).get(str(odds_type), {}) or {}
        parsed = {}
        for key, vals in raw.items():
            if not isinstance(vals, (list, tuple)) or not vals:
                continue
            odds_val = _parse_odds_text(vals[0])
            odds_hi = _parse_odds_text(vals[1]) if len(vals) > 1 else ""
            ninki = _num_or_blank(vals[2]) if len(vals) > 2 else ""
            entry = {"単勝オッズ" if odds_type == 1 else "オッズ": odds_val, "人気": ninki}
            if odds_type in (2, 5) and odds_hi:
                entry["オッズ下限"] = odds_val
                entry["オッズ上限"] = odds_hi
                # ワイド/複勝は中央値を代表オッズに
                try:
                    lo = float(odds_val); hi = float(odds_hi)
                    entry["オッズ"] = f"{(lo + hi) / 2:.1f}"
                except Exception:
                    entry["オッズ"] = odds_val
            parsed[str(key)] = entry
        return {"status": status, "updated_at": updated, "odds": parsed}

    def fetch_win_odds(self, race_id: str) -> dict[str, dict]:
        """馬番(str) -> {単勝オッズ, 人気, オッズ更新日時, オッズ状態}"""
        api = self.fetch_odds_api(race_id, 1)
        out = {}
        for ban, info in api["odds"].items():
            ban_key = str(int(ban)) if str(ban).isdigit() else str(ban).lstrip("0") or "0"
            out[ban_key] = {
                "単勝オッズ": info.get("単勝オッズ", ""),
                "人気": info.get("人気", ""),
                "オッズ更新日時": api.get("updated_at", ""),
                "オッズ状態": api.get("status", ""),
            }
            # zero-padded key も残す
            out[str(ban).zfill(2)] = out[ban_key]
        return out

    def fetch_ticket_odds_maps(self, race_id: str) -> dict:
        """券種別オッズマップ。キーは馬番ゼロ埋め連結（昇順）。"""
        mapping = {"馬連": 4, "ワイド": 5, "三連複": 7}
        out = {"updated_at": "", "status": ""}
        for kind, t in mapping.items():
            api = self.fetch_odds_api(race_id, t)
            if api.get("updated_at") and not out["updated_at"]:
                out["updated_at"] = api["updated_at"]
            if api.get("status"):
                out["status"] = api["status"]
            table = {}
            for key, info in api["odds"].items():
                odds = info.get("オッズ") or info.get("単勝オッズ") or ""
                table[str(key)] = odds
            out[kind] = table
        return out

    def fetch_results(self, race_id: str) -> dict[str, str]:
        """馬名 -> 着順"""
        detail = self.fetch_result_detail(race_id)
        return {r["馬名"]: r["着順"] for r in detail.get("horses", []) if r.get("着順")}

    def fetch_result_detail(self, race_id: str, source: str = "jra") -> dict:
        """着順・人気・確定オッズ・払戻を一括取得。

        戻り値:
          {
            race_id, date, venue, race_no, source,
            horses: [{馬名, 馬番, 着順, 人気, 確定オッズ}, ...],
            payouts: [{bet_type, combination, payout, ninki}, ...],
          }
        """
        base = NAR_BASE if source == "nar" else BASE
        url = f"{base}/race/result.html?race_id={race_id}"
        soup = self._get(url)
        meta = self.parse_race_id(race_id)
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        dm = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
        race_date = (
            f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}" if dm else None
        )

        horses = []
        # 走行データ等の別表にも HorseList があるため、結果表だけを対象にする
        rows = soup.select("table.RaceTable01 tr.HorseList") or soup.select("tr.HorseList")
        for tr in rows:
            table = tr.find_parent("table")
            table_cls = " ".join((table.get("class") if table else None) or [])
            if table_cls and "RaceTable01" not in table_cls:
                continue
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            finish = tds[0].get_text(strip=True)
            a = tr.select_one("a[href*='/horse/']")
            if not a:
                continue
            name = a.get_text(strip=True)
            # 標準列: 着順, 枠, 馬番, 馬名, ...
            ban = tds[2].get_text(strip=True) if len(tds) > 2 else ""
            if not str(ban).isdigit():
                for td in tds:
                    cls = " ".join(td.get("class") or [])
                    if "Txt_C" in cls and "Num" in cls:
                        cand = td.get_text(strip=True)
                        if cand.isdigit():
                            ban = cand
                            break
            # 人気・単勝オッズは Odds クラスの並び（人気 → オッズ）
            odds_cells = [td.get_text(strip=True) for td in tds if "Odds" in " ".join(td.get("class") or [])]
            ninki = _num_or_blank(odds_cells[0]) if odds_cells else ""
            kakutei = _parse_odds_text(odds_cells[1]) if len(odds_cells) > 1 else ""
            if not finish.isdigit() or not str(ban).isdigit():
                continue
            horses.append({
                "馬名": name,
                "馬番": str(int(ban)),
                "着順": finish,
                "人気": ninki,
                "確定オッズ": kakutei,
            })

        payouts = _parse_payout_tables(soup)
        return {
            "race_id": str(race_id),
            "date": race_date,
            "venue": meta.get("venue") or "開催地不明",
            "race_no": meta.get("race_no"),
            "source": source,
            "horses": horses,
            "payouts": payouts,
        }

    def fetch_horse_history(self, horse_id: str, use_cache: bool = True) -> list[dict]:
        cache_path = CACHE / f"{horse_id}.json"
        if use_cache and cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        url = f"{DB}/horse/result/{horse_id}/"
        soup = self._get(url, encoding="euc-jp")
        table = soup.select_one("table.db_h_race_results") or soup.select_one("table")
        hist = []
        if table:
            headers = [th.get_text(strip=True) for th in table.select("tr")[0].find_all(["th", "td"])] if table.select("tr") else []
            for tr in table.select("tr")[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(cells) < 12:
                    continue
                # 標準レイアウト: 日付,開催,天気,R,レース名,...,頭数,枠番,馬番,オッズ,人気,着順,...
                row = {
                    "年月日": cells[0].replace("/", "-") if cells[0] else "",
                    "場": cells[1],
                    "レース": cells[3] if len(cells) > 3 else "",
                    "レース名": cells[4] if len(cells) > 4 else "",
                    "頭数": cells[6] if len(cells) > 6 else "",
                    "人気": cells[10] if len(cells) > 10 else "",
                    "着順": cells[11] if len(cells) > 11 else "",
                    "騎手": cells[12] if len(cells) > 12 else "",
                    "斤量": cells[13] if len(cells) > 13 else "",
                    "距離": cells[14] if len(cells) > 14 else "",
                    "馬場": cells[2] if len(cells) > 2 else "",
                }
                # 距離列がずれる場合のフォールバック
                if not re.search(r"(芝|ダ|障)", str(row["距離"])):
                    for c in cells:
                        if re.match(r"^(芝|ダ|障)\d+", c):
                            row["距離"] = c
                            break
                hist.append(row)
        cache_path.write_text(json.dumps(hist, ensure_ascii=False), encoding="utf-8")
        return hist

    def past_five_for_score(self, history: list[dict], before_date: str) -> dict:
        """対象日より前の直近5走の着順/人気を score 用列に変換。"""
        target = before_date.replace("/", "-")
        past = []
        for h in history:
            d = str(h.get("年月日", "")).replace("/", "-")
            if not d or d >= target:
                continue
            past.append(h)
            if len(past) >= 5:
                break
        out = {"実着順": ""}
        for i in range(1, 6):
            if i <= len(past):
                out[f"着順{i}"] = _num_or_blank(past[i - 1].get("着順"))
                out[f"人気{i}"] = _num_or_blank(past[i - 1].get("人気"))
            else:
                out[f"着順{i}"] = ""
                out[f"人気{i}"] = ""
        return out


def _num_or_blank(v) -> str:
    s = str(v).strip()
    if not s or s in {"--", "**", "除外", "中止", "取消", "失格"}:
        return ""
    m = re.search(r"\d+", s)
    return m.group(0) if m else ""


def _parse_odds_text(v) -> str:
    s = str(v).strip().replace(",", "")
    if not s or s in {"---.-", "****", "**", "-", "--", "---"}:
        return ""
    m = re.search(r"\d+(?:\.\d+)?", s)
    return m.group(0) if m else ""


def _parse_yen(text: str) -> Optional[int]:
    s = str(text).replace(",", "").replace("円", "").strip()
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None


def _normalize_bet_type(label: str) -> str:
    t = str(label).strip().replace("３", "3").replace("　", "")
    mapping = {
        "単勝": "単勝",
        "複勝": "複勝",
        "枠連": "枠連",
        "馬連": "馬連",
        "ワイド": "ワイド",
        "馬単": "馬単",
        "3連複": "三連複",
        "三連複": "三連複",
        "3連単": "三連単",
        "三連単": "三連単",
    }
    return mapping.get(t, t)


def _parse_payout_tables(soup: BeautifulSoup) -> list[dict]:
    """Payout_Detail_Table から券種別払戻を抽出。"""
    out: list[dict] = []
    for table in soup.select("table.Payout_Detail_Table"):
        for tr in table.select("tr"):
            th = tr.select_one("th")
            if not th:
                continue
            bet_type = _normalize_bet_type(th.get_text(strip=True))
            result_td = tr.select_one("td.Result")
            payout_td = tr.select_one("td.Payout")
            ninki_td = tr.select_one("td.Ninki")
            if result_td is None or payout_td is None:
                continue

            combos: list[str] = []
            for ul in result_td.select("ul"):
                nums = re.findall(r"\d+", ul.get_text(" ", strip=True))
                if nums:
                    combos.append("-".join(nums))
            if not combos:
                # 単勝/複勝は div 並び
                nums = [
                    d.get_text(strip=True)
                    for d in result_td.select("div")
                    if d.get_text(strip=True).isdigit()
                ]
                if bet_type == "複勝":
                    combos = nums[:]  # 各馬番を個別扱い
                elif nums:
                    combos = ["-".join(nums)]

            pays = [_parse_yen(x) for x in re.findall(r"[\d,]+円", payout_td.get_text(" ", strip=True))]
            pays = [p for p in pays if p is not None]
            ninkis = re.findall(r"\d+", ninki_td.get_text(" ", strip=True) if ninki_td else "")

            if bet_type == "複勝":
                for i, combo in enumerate(combos):
                    out.append({
                        "bet_type": bet_type,
                        "combination": str(combo),
                        "payout": pays[i] if i < len(pays) else (pays[0] if pays else None),
                        "ninki": ninkis[i] if i < len(ninkis) else "",
                    })
                continue

            if bet_type == "ワイド" and len(combos) == len(pays) and combos:
                for i, combo in enumerate(combos):
                    out.append({
                        "bet_type": bet_type,
                        "combination": combo,
                        "payout": pays[i],
                        "ninki": ninkis[i] if i < len(ninkis) else "",
                    })
                continue

            # 単一組合せ券種
            if combos:
                out.append({
                    "bet_type": bet_type,
                    "combination": combos[0],
                    "payout": pays[0] if pays else None,
                    "ninki": ninkis[0] if ninkis else "",
                })
    return out


def venue_from_netkeiba_id(race_id: str) -> str:
    m = re.fullmatch(r"(\d{4})(\d{2})\d{6}", str(race_id))
    if m:
        return VENUE_CODES.get(m.group(2), "開催地不明")
    return "開催地不明"
