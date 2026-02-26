import requests
from bs4 import BeautifulSoup
import time
import json

BASE_URL = "https://raamde-bf.net/page/"
OUTPUT_FILE = "raamde_corpus.json"


def get_article_links(start_page, end_page):
    links = set()
    print(f"--- Fetching links from pages {start_page} to {end_page} ---")

    for p in range(start_page, end_page + 1):
        url = f"{BASE_URL}{p}/"
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            for img_tag in soup.find_all("img", class_="attachment-bam-featured"):
                parent_a = img_tag.find_parent("a")
                if parent_a:
                    href = parent_a.get("href")
                    if href:
                        links.add(href)

            print(f"Page {p}: Found {len(links)} unique links so far...")
            time.sleep(1.5)
        except Exception as e:
            print(f"Error on page {p}: {e}")

    return list(links)


def scrape_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No Title"
        content_div = soup.find("div", class_="entry-content")

        if content_div:
            for extra in content_div.find_all(["figure", "script", "style"]):
                extra.decompose()
            paragraphs = [p.get_text(strip=True) for p in content_div.find_all("p") if p.get_text(strip=True)]

            return {"url": url, "title": title, "text_units": paragraphs}
    except Exception as e:
        print(f"Failed: {url} | Error: {e}")
    return None


if __name__ == "__main__":
    all_links = get_article_links(1, 43)
    corpus = []
    for i, link in enumerate(all_links):
        print(f"Scraping ({i + 1}/{len(all_links)}): {link}")
        data = scrape_article_content(link)
        if data:
            corpus.append(data)
        time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
