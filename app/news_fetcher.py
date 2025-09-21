import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
BASE = "https://newsapi.org/v2/everything"

def fetch_news(query, from_days=1, page_size=5):
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY not found in .env file")
    from_date = (datetime.utcnow() - timedelta(days=from_days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "relevancy",
        "apiKey": NEWSAPI_KEY
    }
    r = requests.get(BASE, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("articles", [])

if __name__ == "__main__":
    articles = fetch_news("electric vehicles", from_days=2, page_size=3)
    for i,a in enumerate(articles, start=1):
        print(f"\n--- ARTICLE {i} ---")
        print("Title:", a.get("title"))
        print("Source:", a.get("source", {}).get("name"))
        print("URL:", a.get("url"))
        print("Published:", a.get("publishedAt"))
