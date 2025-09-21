# app/ingestion.py
import os, hashlib, json
from dotenv import load_dotenv
from newspaper import Article
from app.news_fetcher import fetch_news

load_dotenv()

DATA_DIR = "data"  # folder to store ingested JSON files
os.makedirs(DATA_DIR, exist_ok=True)

def doc_id_from_url(url: str) -> str:
    """Create a unique id from article URL."""
    return hashlib.sha1(url.encode()).hexdigest()

def extract_full_text(url: str):
    """Try to scrape article full text from URL using newspaper3k."""
    try:
        a = Article(url)
        a.download()
        a.parse()
        return {
            "title": a.title,
            "text": a.text,
            "publish_date": str(a.publish_date) if a.publish_date else None,
            "authors": a.authors,
        }
    except Exception as e:
        return {"title": None, "text": None, "publish_date": None, "authors": []}

def chunk_text(text, chunk_size=350, overlap=50):
    """Split long text into overlapping word chunks for retrieval."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest_query(query: str, from_days=1, max_articles=5):
    """
    Fetch articles for a query, extract full text, split into chunks,
    and save to data/<doc_id>.json files.
    """
    articles = fetch_news(query, from_days=from_days, page_size=max_articles)
    ingested_docs = []

    for art in articles:
        url = art.get("url")
        if not url:
            continue
        doc_id = doc_id_from_url(url)

        # Try scraping for full text
        details = extract_full_text(url)
        full_text = details.get("text") or art.get("content") or art.get("description")

        if not full_text:
            continue  # skip empty

        chunks = chunk_text(full_text)

        doc_record = {
            "doc_id": doc_id,
            "title": details.get("title") or art.get("title"),
            "url": url,
            "source": art.get("source", {}).get("name"),
            "publishedAt": art.get("publishedAt"),
            "chunks": chunks,
        }

        # Save JSON for persistence
        path = os.path.join(DATA_DIR, f"{doc_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc_record, f, ensure_ascii=False, indent=2)

        ingested_docs.append(doc_record)

    return ingested_docs

if __name__ == "__main__":
    # Test ingestion
    docs = ingest_query("artificial intelligence", from_days=1, max_articles=3)
    for d in docs:
        print(f"\n=== {d['title']} ({d['source']}) ===")
        print("URL:", d['url'])
        print("Chunks:", len(d['chunks']))
        print("First chunk sample:\n", d['chunks'][0][:200], "...")
