import requests
from bs4 import BeautifulSoup


def get_jra():
    url = "https://www.jra.go.jp/"

    print("🌐 JRAへ接続中...")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "lxml")

    print("✅ 接続成功")

    race_links = []

    for link in soup.find_all("a"):
        text = link.get_text(strip=True)
        href = link.get("href")

        if href and ("出馬表" in text or "レース" in text):
            race_links.append([text, href])

    print()
    print("🏇 レース関連リンク")
    print("======================")

    for text, href in race_links:
        print(text, "→", href)

    return race_links