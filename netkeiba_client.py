"""netkeiba から JRA 開催日・出馬表・結果・馬履歴を取得するクライアント。"""
from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://race.netkeiba.com"
DB = "https://db.netkeiba.com"
CACHE = Path("data/cache/horse_results")
CACHE.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}

VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


class NetkeibaClient:
    def __init__(self, sleep: float = 0.2, session: requests.Session | None = None):
        self.sleep = sleep
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)

    def _get(self, url: str, encoding: Optional[str] = None) -> requests.Response:
        time.sleep(self.sleep)
        resp = self.session.get(url, timeout=40)
        resp.raise_for_status()
        if encoding:
            resp.encoding = encoding
        elif "db.netkeiba.com" in url:
            resp.encoding = resp.apparent_encoding or "euc-jp"
        else:
            if not resp.encoding or resp.encoding.lower() in ("iso-8859-1", "ascii"):
                resp.encoding = resp.apparent_encoding or "utf-8"
        return resp

    def _soup(self, url: str, encoding: Optional[str] = None) -> BeautifulSoup:
        return BeautifulSoup(self._get(url, encoding=encoding).text, "lxml")

    def list_race_ids(self, yyyymmdd: str) -> list[str]:
        url = f"{BASE}/top/race_list_sub.html?kaisai_date={yyyymmdd}"
        html = self._get(url).text
        return sorted(set(re.findall(r"race_id=(\d{12})", html)))

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

    def parse_race_id(self, race_id: str) -> dict[str, Any]:
        rid = str(race_id)
        if len(rid) != 12 or not rid.isdigit():
            return {"race_id": rid, "venue": "開催地不明", "race_no": None}
        venue_code = rid[4:6]
        return {
            "race_id": rid,
            "year": int(rid[0:4]),
            "venue_code": venue_code,
            "venue": VENUE_CODES.get(venue_code, "開催地不明"),
            "kai": int(rid[6:8]),
            "day": int(rid[8:10]),
            "race_no": int(rid[10:12]),
            "race_number": int(rid[10:12]),
        }

    def fetch_entries(self, race_id: str) -> list[dict]:
        """出馬表から出走馬一覧を返す（score用）。"""
        shutuba = self.fetch_shutuba(race_id)
        rows = []
        for h in shutuba.get("horses", []):
            rows.append({
                "race_id": race_id,
                "日付": shutuba.get("日付"),
                "レース": shutuba.get("race_no") or shutuba.get("race_number"),
                "馬名": h["馬名"],
                "horse_id": h.get("horse_id", ""),
                "馬番": h.get("馬番", ""),
                "開催地": shutuba.get("venue", "開催地不明"),
            })
        return rows

    def fetch_shutuba(self, race_id: str) -> dict[str, Any]:
        url = f"{BASE}/race/shutuba.html?race_id={race_id}"
        html = self._get(url).text
        soup = BeautifulSoup(html, "lxml")
        meta = self.parse_race_id(race_id)
        title = soup.title.get_text(strip=True) if soup.title else ""
        date_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title) or re.search(
            r"(\d{4})年(\d{1,2})月(\d{1,2})日", html
        )
        race_date = None
        if date_m:
            race_date = (
                f"{int(date_m.group(1)):04d}-"
                f"{int(date_m.group(2)):02d}-"
                f"{int(date_m.group(3)):02d}"
            )

        race_name = ""
        h1 = soup.select_one("div.RaceName, h1")
        if h1:
            race_name = h1.get_text(" ", strip=True)

        horses = []
        for tr in soup.select("table.Shutuba_Table tr.HorseList"):
            a = tr.select_one("span.HorseName a") or tr.select_one("a[href*='/horse/']")
            if not a:
                continue
            name = a.get_text(strip=True)
            href = a.get("href") or ""
            hid_m = re.search(r"/horse/(\w+)", href)
            horse_id = hid_m.group(1) if hid_m else ""
            umaban = ""
            waku = ""
            umaban_td = tr.select_one("td.Umaban")
            if umaban_td:
                utxt = umaban_td.get_text(strip=True)
                if utxt.isdigit():
                    umaban = utxt
            waku_td = tr.select_one("td.Waku")
            if waku_td:
                wtxt = re.sub(r"\D", "", waku_td.get_text())
                if wtxt:
                    waku = wtxt
            # 枠順未定でも tr_N / select name=N に馬番が入る
            if not umaban:
                tr_id = tr.get("id") or ""
                m = re.search(r"tr_(\d+)", tr_id)
                if m:
                    umaban = m.group(1)
            if not umaban:
                sel = tr.select_one("select[name]")
                if sel and str(sel.get("name", "")).isdigit():
                    umaban = str(sel.get("name"))
            horses.append({
                "race_id": race_id,
                "馬名": name,
                "horse_id": horse_id,
                "馬番": umaban,
                "枠番": waku,
                "horse_url": urljoin(f"{DB}/", href),
            })
        return {
            **meta,
            "日付": race_date,
            "レース名": race_name,
            "horses": horses,
            "url": url,
        }

    def fetch_results(self, race_id: str) -> dict[str, str]:
        """馬名 -> 着順（未確定なら空辞書）。"""
        result = self.fetch_result(race_id)
        out: dict[str, str] = {}
        for r in result.get("runners", []):
            finish = str(r.get("着順", "")).strip()
            name = r.get("馬名", "")
            if name and finish.isdigit():
                out[name] = finish
        return out

    def fetch_result(self, race_id: str) -> dict[str, Any]:
        url = f"{BASE}/race/result.html?race_id={race_id}"
        html = self._get(url).text
        soup = BeautifulSoup(html, "lxml")
        meta = self.parse_race_id(race_id)
        title = soup.title.get_text(strip=True) if soup.title else ""
        date_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title) or re.search(
            r"(\d{4})年(\d{1,2})月(\d{1,2})日", html
        )
        race_date = None
        if date_m:
            race_date = (
                f"{int(date_m.group(1)):04d}-"
                f"{int(date_m.group(2)):02d}-"
                f"{int(date_m.group(3)):02d}"
            )

        table = soup.select_one("table.RaceTable01")
        runners = []
        if table:
            for tr in table.find_all("tr"):
                a = tr.select_one("a[href*='/horse/']")
                if not a:
                    continue
                tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(tds) < 5:
                    continue
                finish = tds[0]
                waku = tds[1] if len(tds) > 1 else ""
                umaban = tds[2] if len(tds) > 2 else ""
                pop = ""
                odds = ""
                for td in tr.find_all("td"):
                    cls = " ".join(td.get("class") or [])
                    txt = td.get_text(strip=True)
                    if "Popular" in cls or "Ninki" in cls:
                        pop = txt
                    if "Odds" in cls:
                        odds = txt
                if not pop:
                    for x in reversed(tds):
                        if re.fullmatch(r"\d{1,2}", x):
                            pop = x
                            break
                hid_m = re.search(r"/horse/(\w+)", a.get("href") or "")
                runners.append({
                    "race_id": race_id,
                    "馬名": a.get_text(strip=True),
                    "horse_id": hid_m.group(1) if hid_m else "",
                    "着順": finish,
                    "枠番": waku,
                    "馬番": umaban,
                    "人気": pop,
                    "単勝オッズ": odds,
                })
        return {
            **meta,
            "日付": race_date,
            "runners": runners,
            "url": url,
            "has_result": bool(runners),
        }

    def fetch_horse_history(
        self,
        horse_id: str,
        limit: int = 20,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        if not horse_id:
            return []
        cache_path = CACHE / f"{horse_id}.json"
        if use_cache and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return data[:limit] if limit else data
            except Exception:
                pass

        url = f"{DB}/horse/result/{horse_id}/"
        soup = self._soup(url, encoding="euc-jp")
        table = soup.select_one("table.db_h_race_results")
        if not table:
            cache_path.write_text("[]", encoding="utf-8")
            return []

        rows = []
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 12:
                continue
            cells = [td.get_text(strip=True) for td in tds]
            date_raw = cells[0]
            date_m = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})", date_raw)
            if not date_m:
                continue
            y, mo, d = map(int, date_m.groups())
            iso = f"{y:04d}-{mo:02d}-{d:02d}"
            jp = f"{y}年{mo}月{d}日"
            race_a = tr.select_one("a[href*='/race/']")
            race_id = ""
            if race_a:
                rm = re.search(r"/race/(\d{12})", race_a.get("href") or "")
                if rm:
                    race_id = rm.group(1)
            place_raw = cells[1]
            venue = re.sub(r"^\d+", "", place_raw)
            venue = re.sub(r"\d+$", "", venue)
            dist = cells[14] if len(cells) > 14 else ""
            if not re.search(r"(芝|ダ|障)", str(dist)):
                for c in cells:
                    if re.match(r"^(芝|ダ|障)\d+", c):
                        dist = c
                        break
            rows.append({
                "horse_id": horse_id,
                "年月日": jp,
                "日付": iso,
                "場": venue,
                "レース": cells[3],
                "レース名": cells[4],
                "頭数": cells[6],
                "枠番": cells[7],
                "馬番": cells[8],
                "単勝オッズ": cells[9],
                "人気": cells[10],
                "着順": cells[11],
                "騎手": cells[12] if len(cells) > 12 else "",
                "斤量": cells[13] if len(cells) > 13 else "",
                "距離": dist,
                "馬場": cells[16] if len(cells) > 16 else "",
                "タイム": cells[18] if len(cells) > 18 else "",
                "着差": cells[19] if len(cells) > 19 else "",
                "馬体重": cells[28] if len(cells) > 28 else "",
                "history_race_id": race_id,
            })
            if limit and len(rows) >= limit:
                break

        cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return rows

    def past_five_for_score(self, history: list[dict], before_date: str) -> dict:
        """対象日より前の直近5走の着順/人気を score 用列に変換。"""
        target = before_date.replace("/", "-")
        past = []
        for h in history:
            d = str(h.get("日付") or h.get("年月日") or "")
            d = (
                d.replace("年", "-")
                .replace("月", "-")
                .replace("日", "")
                .replace("/", "-")
            )
            parts = d.split("-")
            if len(parts) == 3:
                try:
                    d = f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                except Exception:
                    continue
            else:
                continue
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


def yyyymmdd(date_str: str) -> str:
    return date_str.replace("-", "")


def iso_from_yyyymmdd(value: str) -> str:
    v = re.sub(r"\D", "", value)
    return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
