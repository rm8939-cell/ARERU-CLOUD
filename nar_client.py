"""NAR（地方競馬）netkeiba クライアント。

JRA用 NetkeibaClient を継承し、ホストと開催場コードだけ分離する。
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from netkeiba_client import NetkeibaClient

NAR_VENUE_CODES = {
    "30": "門別",
    "35": "盛岡",
    "36": "水沢",
    "42": "浦和",
    "43": "船橋",
    "44": "大井",
    "45": "川崎",
    "46": "金沢",
    "47": "笠松",
    "48": "名古屋",
    "50": "園田",
    "51": "姫路",
    "54": "高知",
    "55": "佐賀",
    "65": "帯広",
}


class NarClient(NetkeibaClient):
    BASE = "https://nar.netkeiba.com"

    def list_race_ids(self, yyyymmdd: str) -> list[str]:
        url = f"{self.BASE}/top/race_list_sub.html?kaisai_date={yyyymmdd}"
        html = self._get(url).text
        return sorted(set(re.findall(r"race_id=(\d{12})", html)))

    def parse_race_id(self, race_id: str) -> dict[str, Any]:
        rid = str(race_id)
        if len(rid) != 12 or not rid.isdigit():
            raise ValueError(f"invalid nar race_id: {race_id}")
        venue_code = rid[4:6]
        return {
            "race_id": rid,
            "year": int(rid[0:4]),
            "venue_code": venue_code,
            "venue": NAR_VENUE_CODES.get(venue_code, f"地方{venue_code}"),
            "mmdd": rid[6:10],
            "race_number": int(rid[10:12]),
            "source": "nar",
        }

    def fetch_shutuba(self, race_id: str) -> dict[str, Any]:
        url = f"{self.BASE}/race/shutuba.html?race_id={race_id}"
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
        horses = []
        seen = set()
        rows = soup.select("tr.HorseList") or soup.select("table.Shutuba_Table tr")
        for tr in rows:
            a = tr.select_one("span.HorseName a") or tr.select_one("a[href*='/horse/']")
            if not a:
                continue
            name = a.get_text(strip=True)
            href = a.get("href") or ""
            hid_m = re.search(r"/horse/(\w+)", href)
            horse_id = hid_m.group(1) if hid_m else ""
            key = horse_id or name
            if not name or key in seen:
                continue
            seen.add(key)
            umaban = ""
            umaban_td = tr.select_one("td.Umaban")
            if umaban_td and umaban_td.get_text(strip=True).isdigit():
                umaban = umaban_td.get_text(strip=True)
            if not umaban:
                m = re.search(r"tr_(\d+)", tr.get("id") or "")
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
                    "horse_url": urljoin("https://db.netkeiba.com/", href),
                }
            )
        # 出馬表が薄い場合は結果表の出走馬で補完
        if len(horses) <= 1:
            result = self.fetch_result(race_id)
            if result.get("runners"):
                horses = [
                    {
                        "race_id": race_id,
                        "馬名": r["馬名"],
                        "horse_id": r.get("horse_id", ""),
                        "馬番": r.get("馬番", ""),
                        "horse_url": f"https://db.netkeiba.com/horse/{r.get('horse_id','')}/",
                    }
                    for r in result["runners"]
                ]
                if not race_date:
                    race_date = result.get("日付")
        return {**meta, "日付": race_date, "horses": horses, "url": url}

    def fetch_result(self, race_id: str) -> dict[str, Any]:
        url = f"{self.BASE}/race/result.html?race_id={race_id}"
        html = self._get(url).text
        soup = BeautifulSoup(html, "lxml")
        meta = self.parse_race_id(race_id)
        title = soup.title.get_text(strip=True) if soup.title else ""
        date_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
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
                hid_m = re.search(r"/horse/(\w+)", a.get("href") or "")
                pop = ""
                odds = ""
                for td in tr.find_all("td"):
                    cls = " ".join(td.get("class") or [])
                    txt = td.get_text(strip=True)
                    if "Popular" in cls or "Ninki" in cls:
                        pop = txt
                    if "Odds" in cls:
                        odds = txt
                runners.append(
                    {
                        "race_id": race_id,
                        "馬名": a.get_text(strip=True),
                        "horse_id": hid_m.group(1) if hid_m else "",
                        "着順": tds[0],
                        "枠番": tds[1] if len(tds) > 1 else "",
                        "馬番": tds[2] if len(tds) > 2 else "",
                        "人気": pop,
                        "単勝オッズ": odds,
                    }
                )
        return {
            **meta,
            "日付": race_date,
            "runners": runners,
            "payouts": self._parse_payouts(soup),
            "url": url,
            "has_result": bool(runners),
        }
