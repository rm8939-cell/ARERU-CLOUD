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
            rows.append({
                "race_id": race_id,
                "日付": race_date,
                "レース": meta["race_no"],
                "馬名": name,
                "horse_id": horse_id,
                "馬番": ban,
                "開催地": meta["venue"],
            })
        return rows

    def fetch_results(self, race_id: str) -> dict[str, str]:
        """馬名 -> 着順"""
        url = f"{BASE}/race/result.html?race_id={race_id}"
        soup = self._get(url)
        out = {}
        for tr in soup.select("tr.HorseList"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            finish = tds[0].get_text(strip=True)
            a = tr.select_one("a[href*='/horse/']")
            if not a:
                continue
            name = a.get_text(strip=True)
            if finish.isdigit():
                out[name] = finish
        return out

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


def venue_from_netkeiba_id(race_id: str) -> str:
    m = re.fullmatch(r"(\d{4})(\d{2})\d{6}", str(race_id))
    if m:
        return VENUE_CODES.get(m.group(2), "開催地不明")
    return "開催地不明"
