import requests
from bs4 import BeautifulSoup


def get_horses():
    url = (
        "https://www.jra.go.jp/JRADB/accessD.html"
        "?CNAME=pw01dde0110202602060520260712%2F8D"
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    print("🏇 JRA出馬表取得開始...")

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "lxml")

    horses = []

    for link in soup.find_all("a"):
        href = link.get("href", "")
        text = link.get_text(strip=True)

        if "accessU.html" in href and text:
            if text not in horses:
                horses.append(text)

    print()
    print("🐴 取得した馬")
    print("================")

    for horse in horses:
        print(horse)

    print()
    print("頭数:", len(horses))

    return horses