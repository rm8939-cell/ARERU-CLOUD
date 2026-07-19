"""netkeiba から JRA/NAR 開催日・出馬表・結果・馬履歴を取得するクライアント。"""
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

# JRA: 01-10 / NAR: 30台〜
JRA_VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
NAR_VENUE_CODES = {
    "30": "門別", "35": "盛岡", "36": "水沢",
    "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎",
    "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路",
    "54": "高知", "55": "佐賀",
    "65": "帯広",
}
VENUE_CODES = {**JRA_VENUE_CODES, **NAR_VENUE_CODES}

# NAR 券種 HTML type コード
_NAR_ODDS_TYPE = {
    1: "b1",  # 単勝・複勝
    4: "b4",  # 馬連
    5: "b5",  # ワイド
    7: "b7",  # 三連複
    8: "b8",  # 三連単
}


def infer_source(race_id: str) -> str:
    """race_id の会場コードから jra / nar を推定。"""
    s = str(race_id).strip()
    if re.fullmatch(r"\d{12}", s):
        code = s[4:6]
        if code in NAR_VENUE_CODES:
            return "nar"
        if code in JRA_VENUE_CODES:
            return "jra"
        # 11以降は地方寄り（帯広・季節開催など）
        try:
            if int(code) >= 30:
                return "nar"
        except Exception:
            pass
    return "jra"


def base_url(source: str = "jra") -> str:
    return NAR_BASE if source == "nar" else BASE


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

    def list_race_ids(self, yyyymmdd: str, source: str = "jra") -> list[str]:
        base = base_url(source)
        url = f"{base}/top/race_list_sub.html?kaisai_date={yyyymmdd}"
        soup = self._get(url)
        # NAR も基本12桁。幅を持たせて拾う
        pat = r"race_id=(\d{12})" if source == "jra" else r"race_id=(\d{8,12})"
        ids = sorted(set(re.findall(pat, str(soup))))
        if source == "nar":
            # 12桁以外は除外（万一のゴミ対策）
            ids = [x for x in ids if len(x) == 12 and infer_source(x) == "nar"]
        return ids

    def discover_kaisai_dates(
        self,
        center: Optional[date] = None,
        lookback: int = 28,
        lookahead: int = 14,
        source: str = "jra",
    ) -> list[str]:
        """周辺カレンダーを走査し、レースがある開催日を新しい順で返す。"""
        center = center or date.today()
        found = []
        for offset in range(-lookback, lookahead + 1):
            d = center + timedelta(days=offset)
            key = d.strftime("%Y%m%d")
            try:
                ids = self.list_race_ids(key, source=source)
            except Exception:
                ids = []
            if ids:
                found.append(d.strftime("%Y-%m-%d"))
        return sorted(found, reverse=True)

    def latest_kaisai_date(
        self, center: Optional[date] = None, source: str = "jra"
    ) -> Optional[str]:
        dates = self.discover_kaisai_dates(center=center, source=source)
        return dates[0] if dates else None

    def parse_race_id(self, race_id: str) -> dict:
        m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})", str(race_id))
        if not m:
            return {
                "race_id": race_id,
                "venue": "開催地不明",
                "race_no": None,
                "date": None,
                "source": infer_source(race_id),
            }
        year, venue, a, b, race_no = m.groups()
        src = infer_source(race_id)
        # NAR: YYYY + venue + MMDD + RR / JRA: YYYY + venue + kai + day + RR
        race_date = None
        if src == "nar":
            try:
                race_date = f"{year}-{a}-{b}"
            except Exception:
                race_date = None
        return {
            "race_id": race_id,
            "year": year,
            "venue_code": venue,
            "venue": VENUE_CODES.get(venue, "開催地不明"),
            "race_no": int(race_no),
            "date": race_date,
            "source": src,
        }

    def fetch_entries(self, race_id: str, source: Optional[str] = None) -> list[dict]:
        src = source or infer_source(race_id)
        base = base_url(src)
        url = f"{base}/race/shutuba.html?race_id={race_id}&rf=race_list"
        soup = self._get(url)
        meta = self.parse_race_id(race_id)
        # タイトルから日付を拾う
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        dm = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
        race_date = (
            f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}"
            if dm
            else meta.get("date")
        )
        # タイトルから開催地名を補正
        vm = re.search(r"\d{4}年\d{1,2}月\d{1,2}日\s+(\S+?)\d+R", title)
        venue = vm.group(1) if vm else meta.get("venue") or "開催地不明"

        rows = []
        for tr in soup.select("tr.HorseList"):
            a = tr.select_one("a[href*='/horse/']")
            if not a:
                # HorseInfo 内テキストだけの場合
                hi = tr.select_one("td.HorseInfo")
                if not hi:
                    continue
                name = hi.get_text(strip=True)
                horse_id = ""
            else:
                name = a.get_text(strip=True)
                hm = re.search(r"/horse/(\d+)", a.get("href", ""))
                horse_id = hm.group(1) if hm else ""

            umaban = tr.select_one("td.Umaban1, td[class*=Umaban]")
            ban = umaban.get_text(strip=True) if umaban else ""
            waku_td = tr.select_one("td[class*=Waku]")
            waku = waku_td.get_text(strip=True) if waku_td else ""
            # 斤量: Txt_C の数値（性齢の次）
            kinryo = ""
            for td in tr.find_all("td"):
                cls = " ".join(td.get("class") or [])
                txt = td.get_text(strip=True)
                if "Jockey_Info" in cls or (cls == "Txt_C" and re.fullmatch(r"\d+(?:\.\d+)?", txt)):
                    if re.fullmatch(r"\d+(?:\.\d+)?", txt):
                        kinryo = txt
                        break
            jockey_td = tr.select_one("td.Jockey")
            jockey = ""
            if jockey_td:
                ja = jockey_td.select_one("a")
                jockey = (ja.get_text(strip=True) if ja else jockey_td.get_text(strip=True))

            odds_td = tr.select_one("td.Txt_R.Popular, td.Popular.Txt_R, td.Popular")
            ninki_td = tr.select_one("td.Popular_Ninki, td.Popular.Txt_C")
            # 単勝オッズは Popular Txt_R、人気は Popular Txt_C（NAR出馬表）
            shutuba_odds = ""
            shutuba_ninki = ""
            pop_cells = tr.select("td.Popular")
            if len(pop_cells) >= 2:
                shutuba_odds = _parse_odds_text(pop_cells[0].get_text(strip=True))
                shutuba_ninki = _num_or_blank(pop_cells[1].get_text(strip=True))
            else:
                shutuba_odds = _parse_odds_text(odds_td.get_text(strip=True) if odds_td else "")
                shutuba_ninki = _num_or_blank(ninki_td.get_text(strip=True) if ninki_td else "")

            name = (name or "").strip()
            ban = (ban or "").strip()
            # 取消・空行・馬番なし重複行はスキップ（馬番欠落のまま残すと後段で潰れる）
            if not name:
                continue
            if not ban or not re.fullmatch(r"\d+", ban):
                continue
            rows.append({
                "race_id": race_id,
                "日付": race_date,
                "レース": meta["race_no"],
                "馬名": name,
                "horse_id": horse_id,
                "馬番": ban,
                "枠": waku,
                "騎手": jockey,
                "斤量": kinryo,
                "開催地": venue,
                "単勝オッズ": shutuba_odds,
                "人気": shutuba_ninki,
                "source": src,
            })
        # 同一馬名が複数ある場合は馬番付きを優先
        uniq = {}
        for r in rows:
            uniq[r["馬名"]] = r
        return list(uniq.values())

    def fetch_odds_api(self, race_id: str, odds_type: int = 1, source: Optional[str] = None) -> dict:
        """オッズ取得。JRAは公式API、NARはHTMLスクレイピング。

        type: 1単勝 2複勝 3枠連 4馬連 5ワイド 6馬単 7三連複 8三連単
        戻り値: {status, updated_at, odds: {key: {...}}}
        """
        src = source or infer_source(race_id)
        if src == "nar":
            return self._fetch_nar_odds_html(race_id, odds_type)

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

    def _fetch_nar_odds_html(self, race_id: str, odds_type: int) -> dict:
        """NAR オッズHTMLから券種別オッズを抽出。"""
        type_code = _NAR_ODDS_TYPE.get(odds_type)
        if not type_code:
            return {"status": "unsupported", "updated_at": "", "odds": {}}

        if odds_type == 1:
            url = f"{NAR_BASE}/odds/index.html?type=b1&race_id={race_id}"
        else:
            url = f"{NAR_BASE}/odds/index.html?type={type_code}&race_id={race_id}&housiki=c99"

        try:
            soup = self._get(url)
        except Exception:
            return {"status": "error", "updated_at": "", "odds": {}}

        if odds_type == 1:
            return self._parse_nar_win_odds(soup)
        return self._parse_nar_combo_odds(soup, odds_type)

    def _parse_nar_win_odds(self, soup: BeautifulSoup) -> dict:
        tables = soup.select("table.RaceOdds_HorseList_Table")
        if not tables:
            return {"status": "empty", "updated_at": "", "odds": {}}
        # 1つ目=単勝、2つ目=複勝
        # 現行HTML: [枠][馬番(class無し)][印][馬名][Odds]
        parsed = {}
        rows_odds = []
        for tr in tables[0].select("tr"):
            odds_td = tr.select_one("td.Odds")
            if not odds_td:
                continue
            ban = ""
            ban_td = tr.select_one("td.W31")
            if ban_td:
                ban = ban_td.get_text(strip=True)
            if not ban.isdigit():
                for td in tr.find_all("td"):
                    cls = " ".join(td.get("class") or [])
                    txt = td.get_text(strip=True)
                    if cls.startswith("Waku"):
                        continue
                    if (not cls or cls == "W31") and txt.isdigit():
                        ban = txt
                        break
            odds_val = _parse_odds_text(odds_td.get_text(strip=True))
            if not ban.isdigit() or not odds_val:
                continue
            rows_odds.append((int(ban), float(odds_val), odds_val))
        # 人気はオッズ昇順で付与
        rows_odds.sort(key=lambda x: x[1])
        for ninki, (ban, _f, odds_val) in enumerate(rows_odds, 1):
            key = f"{ban:02d}"
            parsed[key] = {"単勝オッズ": odds_val, "人気": str(ninki)}
            parsed[str(ban)] = parsed[key]
        return {
            "status": "ok" if parsed else "empty",
            "updated_at": "",
            "odds": parsed,
        }

    def _parse_nar_combo_odds(self, soup: BeautifulSoup, odds_type: int) -> dict:
        """NAR 馬連/ワイド/三連複/三連単。現行は td.Combi + td.Txt_R 形式。"""
        table = soup.select_one("table.RaceOdds_HorseList_Table")
        if not table:
            return {"status": "empty", "updated_at": "", "odds": {}}
        parsed = {}
        need = 3 if odds_type in (7, 8) else 2
        for tr in table.select("tr"):
            ninki_td = tr.select_one("td.Ninki")
            if not ninki_td:
                continue
            ninki = _num_or_blank(ninki_td.get_text(strip=True))
            if not ninki:
                continue

            # 組合せ: "2 4" / "2 4 10" / "4 2 10"
            combi_txt = ""
            combi_td = tr.select_one("td.Combi")
            if combi_td and "Txt_R" not in " ".join(combi_td.get("class") or []):
                combi_txt = combi_td.get_text(" ", strip=True)
            if not combi_txt:
                for td in tr.find_all("td"):
                    cls = " ".join(td.get("class") or [])
                    txt = td.get_text(" ", strip=True)
                    if cls:
                        continue
                    nums = re.findall(r"\d+", txt)
                    if len(nums) >= need and len(txt) <= 20:
                        combi_txt = txt
                        break
            bans = [int(x) for x in re.findall(r"\d+", combi_txt)]
            if len(bans) < need:
                continue
            if odds_type == 8:
                use = bans[:3]  # 三連単は順序保持
            elif odds_type == 7:
                use = sorted(bans[:3])
            else:
                use = sorted(bans[:2])

            # オッズ: ワイドは "1.8 2.0"、他は単一値
            odds_td = (
                tr.select_one("td.Combi.Txt_R")
                or tr.select_one("td.Txt_R")
                or tr.select_one("td.Name_Odds")
                or tr.select_one("td[class*=Odds]")
            )
            if not odds_td:
                continue
            spans = [s.get_text(strip=True) for s in odds_td.select("span.Odds")]
            raw = " ".join(spans) if spans else odds_td.get_text(" ", strip=True)
            nums = re.findall(r"\d+(?:\.\d+)?", raw.replace(",", ""))
            if odds_type == 5 and len(nums) >= 2:
                lo, hi = nums[0], nums[1]
                try:
                    odds_mid = f"{(float(lo) + float(hi)) / 2:.1f}"
                except Exception:
                    odds_mid = lo
                entry = {
                    "オッズ": odds_mid,
                    "オッズ下限": lo,
                    "オッズ上限": hi,
                    "人気": ninki,
                }
            else:
                odds_val = _parse_odds_text(nums[0] if nums else raw)
                if not odds_val:
                    continue
                entry = {"オッズ": odds_val, "人気": ninki}
            key = "".join(f"{b:02d}" for b in use)
            parsed[key] = entry
        return {
            "status": "ok" if parsed else "empty",
            "updated_at": "",
            "odds": parsed,
        }

    def fetch_win_odds(self, race_id: str, source: Optional[str] = None) -> dict[str, dict]:
        """馬番(str) -> {単勝オッズ, 人気, オッズ更新日時, オッズ状態}"""
        api = self.fetch_odds_api(race_id, 1, source=source)
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
            out[str(ban).zfill(2) if str(ban).isdigit() else str(ban)] = out[ban_key]
        return out

    def fetch_ticket_odds_maps(self, race_id: str, source: Optional[str] = None) -> dict:
        """券種別オッズマップ。キーは馬番ゼロ埋め連結（昇順）。"""
        src = source or infer_source(race_id)
        mapping = {"馬連": 4, "ワイド": 5, "三連複": 7}
        out = {"updated_at": "", "status": "", "source": src}
        for kind, t in mapping.items():
            api = self.fetch_odds_api(race_id, t, source=src)
            if api.get("updated_at") and not out["updated_at"]:
                out["updated_at"] = api["updated_at"]
            if api.get("status"):
                out["status"] = api["status"]
            table = {}
            for key, info in api["odds"].items():
                odds = info.get("オッズ") or info.get("単勝オッズ") or ""
                table[str(key)] = odds
            out[kind] = table
        # 三連単も取得（検証用）
        try:
            api8 = self.fetch_odds_api(race_id, 8, source=src)
            table = {}
            for key, info in api8["odds"].items():
                odds = info.get("オッズ") or ""
                table[str(key)] = odds
            out["三連単"] = table
        except Exception:
            out["三連単"] = {}
        return out

    def fetch_results(self, race_id: str, source: Optional[str] = None) -> dict[str, str]:
        """馬名 -> 着順"""
        detail = self.fetch_result_detail(race_id, source=source or infer_source(race_id))
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
        src = source or infer_source(race_id)
        base = base_url(src)
        url = f"{base}/race/result.html?race_id={race_id}"
        soup = self._get(url)
        meta = self.parse_race_id(race_id)
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        dm = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
        race_date = (
            f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}" if dm else meta.get("date")
        )
        vm = re.search(r"\d{4}年\d{1,2}月\d{1,2}日\s+(\S+?)\d+R", title)
        venue = vm.group(1) if vm else (meta.get("venue") or "開催地不明")

        horses = []
        # NAR: Result_Num 行が本体。JRA: HorseList。混在時は Result_Num を優先。
        rows = [
            tr for tr in soup.select("table.RaceTable01 tr")
            if tr.select_one("td.Result_Num")
        ]
        if not rows:
            rows = soup.select("table.RaceTable01 tr.HorseList") or soup.select("tr.HorseList")
        for tr in rows:
            table = tr.find_parent("table")
            table_cls = " ".join((table.get("class") if table else None) or [])
            if table_cls and "RaceTable01" not in table_cls:
                continue
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue

            finish_td = tr.select_one("td.Result_Num")
            finish = finish_td.get_text(strip=True) if finish_td else tds[0].get_text(strip=True)

            a = tr.select_one("a[href*='/horse/']")
            if a:
                name = a.get_text(strip=True)
            else:
                hi = tr.select_one("td.Horse_Info")
                name = hi.get_text(strip=True) if hi else ""
            if not name:
                continue

            # 馬番: Num.Waku（枠クラスを持たない方）または 3列目
            ban = ""
            ban_td = None
            for td in tr.select("td.Num"):
                cls = " ".join(td.get("class") or [])
                # Waku7 のような枠セルは除外し、素の Waku / Num を取る
                if re.search(r"Waku\d", cls):
                    continue
                ban_td = td
                break
            if ban_td is None and len(tds) > 2:
                ban_td = tds[2]
            ban = ban_td.get_text(strip=True) if ban_td else ""
            if not str(ban).isdigit():
                for td in tds:
                    cls = " ".join(td.get("class") or [])
                    if "Txt_C" in cls and "Num" in cls:
                        cand = td.get_text(strip=True)
                        if cand.isdigit():
                            ban = cand
                            break

            # 人気・単勝オッズは Odds クラスの並び（人気 → オッズ）
            odds_cells = [
                td.get_text(strip=True)
                for td in tds
                if "Odds" in " ".join(td.get("class") or [])
            ]
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
            "venue": venue,
            "race_no": meta.get("race_no"),
            "source": src,
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
