"""netkeiba 取得クライアント（JRA中央）。

P0-2以降のデータ更新・オッズ・結果照合の共通基盤。
未取得の数字は返さない（空/None）。
"""
from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}

VENUE_CODES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

# netkeiba odds API type
ODDS_WIN = 1
ODDS_PLACE = 2
ODDS_QUINELLA = 4  # 馬連
ODDS_WIDE = 5
ODDS_EXACTA = 6  # 馬単
ODDS_TRIO = 7  # 三連複
ODDS_TRIFECTA = 8  # 三連単


class NetkeibaClient:
    def __init__(self, sleep: float = 0.25, session: requests.Session | None = None):
        self.sleep = sleep
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)

    def _get(self, url: str, **kwargs) -> requests.Response:
        time.sleep(self.sleep)
        resp = self.session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        if not resp.encoding or resp.encoding.lower() in ("iso-8859-1", "ascii"):
            resp.encoding = resp.apparent_encoding or "utf-8"
        return resp

    def list_race_ids(self, yyyymmdd: str) -> list[str]:
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={yyyymmdd}"
        html = self._get(url).text
        ids = sorted(set(re.findall(r"race_id=(\d{12})", html)))
        return ids

    def parse_race_id(self, race_id: str) -> dict[str, Any]:
        rid = str(race_id)
        if len(rid) != 12 or not rid.isdigit():
            raise ValueError(f"invalid race_id: {race_id}")
        venue_code = rid[4:6]
        return {
            "race_id": rid,
            "year": int(rid[0:4]),
            "venue_code": venue_code,
            "venue": VENUE_CODES.get(venue_code, "開催地不明"),
            "kai": int(rid[6:8]),
            "day": int(rid[8:10]),
            "race_number": int(rid[10:12]),
        }

    def fetch_shutuba(self, race_id: str) -> dict[str, Any]:
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        html = self._get(url).text
        soup = BeautifulSoup(html, "lxml")
        meta = self.parse_race_id(race_id)
        title = soup.title.get_text(strip=True) if soup.title else ""
        date_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title) or re.search(
            r"(\d{4})年(\d{1,2})月(\d{1,2})日", html
        )
        race_date = None
        if date_m:
            race_date = f"{int(date_m.group(1)):04d}-{int(date_m.group(2)):02d}-{int(date_m.group(3)):02d}"

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
            horses.append(
                {
                    "race_id": race_id,
                    "馬名": name,
                    "horse_id": horse_id,
                    "馬番": umaban,
                    "枠番": waku,
                    "horse_url": urljoin("https://db.netkeiba.com/", href),
                }
            )
        return {
            **meta,
            "日付": race_date,
            "レース名": race_name,
            "horses": horses,
            "url": url,
        }

    def fetch_result(self, race_id: str) -> dict[str, Any]:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        html = self._get(url).text
        soup = BeautifulSoup(html, "lxml")
        meta = self.parse_race_id(race_id)
        title = soup.title.get_text(strip=True) if soup.title else ""
        date_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title) or re.search(
            r"(\d{4})年(\d{1,2})月(\d{1,2})日", html
        )
        race_date = None
        if date_m:
            race_date = f"{int(date_m.group(1)):04d}-{int(date_m.group(2)):02d}-{int(date_m.group(3)):02d}"

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
                # popularity often near end-ish; find numeric popularity cell
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
                    # fallback: last small integer-like
                    for x in reversed(tds):
                        if re.fullmatch(r"\d{1,2}", x):
                            pop = x
                            break
                hid_m = re.search(r"/horse/(\w+)", a.get("href") or "")
                runners.append(
                    {
                        "race_id": race_id,
                        "馬名": a.get_text(strip=True),
                        "horse_id": hid_m.group(1) if hid_m else "",
                        "着順": finish,
                        "枠番": waku,
                        "馬番": umaban,
                        "人気": pop,
                        "単勝オッズ": odds,
                    }
                )

        payouts = self._parse_payouts(soup)
        return {
            **meta,
            "日付": race_date,
            "runners": runners,
            "payouts": payouts,
            "url": url,
            "has_result": bool(runners),
        }

    def _parse_payouts(self, soup: BeautifulSoup) -> dict[str, list[dict[str, Any]]]:
        """結果ページの払戻を券種別に抽出。取れない券種はキー自体を持たない。"""
        out: dict[str, list[dict[str, Any]]] = {}
        pay_root = soup.select_one("div.Result_Pay_Back, div.FullResult_Pay_Back")
        if not pay_root:
            return out
        blob = pay_root.get_text(" ", strip=True)
        if not blob:
            return out

        def add(kind: str, combo: str, yen: str):
            try:
                amount = int(str(yen).replace(",", ""))
            except Exception:
                return
            combo_n = re.sub(r"\s+", "-", combo.strip())
            out.setdefault(kind, []).append({"組合せ": combo_n, "払戻": amount})

        # 単勝 5 780円
        for m in re.finditer(r"単勝\s+(\d+)\s+([\d,]+)\s*円", blob):
            add("単勝", m.group(1), m.group(2))
        # 馬連 4 5 7,600円
        for m in re.finditer(r"馬連\s+(\d+)\s+(\d+)\s+([\d,]+)\s*円", blob):
            add("馬連", f"{m.group(1)}-{m.group(2)}", m.group(3))
        # ワイドは「4 5 2 5 2 4 1,620円 550円 1,230円」形式が多い
        wide = re.search(r"ワイド(.+?)(?:馬単|3連複|三連複|三連単|$)", blob)
        if wide:
            section = wide.group(1)
            yens = [y.replace(",", "") for y in re.findall(r"([\d,]+)\s*円", section)]
            nums = re.findall(r"\b(\d{1,2})\b", re.split(r"円", section)[0] if "円" in section else section)
            # 人気数字が混ざる前の馬番ペアを優先。円より前の偶数個数字をペア化
            pairs = []
            for i in range(0, len(nums) - 1, 2):
                pairs.append((nums[i], nums[i + 1]))
            for i, (a, b) in enumerate(pairs):
                if i >= len(yens):
                    break
                add("ワイド", f"{a}-{b}", yens[i])
        # 3連複 / 三連複 2 4 5 8,790円
        for m in re.finditer(r"(?:3連複|三連複)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d,]+)\s*円", blob):
            add("三連複", f"{m.group(1)}-{m.group(2)}-{m.group(3)}", m.group(4))
        return out

    def fetch_horse_history(self, horse_id: str, limit: int = 12) -> list[dict[str, Any]]:
        url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
        html = self._get(url).text
        soup = BeautifulSoup(html, "lxml")
        table = soup.select_one("table.db_h_race_results")
        if not table:
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
            place_raw = cells[1]  # e.g. 1函館10
            venue = re.sub(r"^\d+", "", place_raw)
            venue = re.sub(r"\d+$", "", venue)
            rows.append(
                {
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
                    "距離": cells[14] if len(cells) > 14 else "",
                    "馬場": cells[16] if len(cells) > 16 else "",
                    "タイム": cells[18] if len(cells) > 18 else "",
                    "着差": cells[19] if len(cells) > 19 else "",
                    "馬体重": cells[28] if len(cells) > 28 else "",
                    "history_race_id": race_id,
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def fetch_odds(self, race_id: str, odds_type: int) -> dict[str, Any] | None:
        url = (
            "https://race.netkeiba.com/api/api_get_jra_odds.html"
            f"?race_id={race_id}&type={odds_type}"
        )
        resp = self._get(url)
        try:
            payload = resp.json()
        except Exception:
            return None
        if payload.get("status") not in ("result", "middle"):
            return None
        data = payload.get("data")
        if not data or data == "":
            return None
        odds = data.get("odds") if isinstance(data, dict) else None
        if not odds:
            return None
        return {"status": payload.get("status"), "odds": odds, "raw": data}


def yyyymmdd(date_str: str) -> str:
    return date_str.replace("-", "")


def iso_from_yyyymmdd(value: str) -> str:
    v = re.sub(r"\D", "", value)
    return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
