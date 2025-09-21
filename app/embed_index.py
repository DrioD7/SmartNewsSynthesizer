# app/embed_index.py
import os
import glob
import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Try to import faiss; give helpful error if it fails
try:
    import faiss
except Exception as e:
    raise RuntimeError(
        "faiss import failed. On Windows try: 'pip install faiss-cpu' or "
        "'pip install faiss-cpu==1.7.3'. If that still fails, install chromadb "
        "or run on Linux. Original error: " + str(e)
    )

# CONFIG
DATA_DIR = Path("data")
INDEX_DIR = Path("index_data")
INDEX_DIR.mkdir(exist_ok=True)
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
METADATA_FILE = INDEX_DIR / "metadatas.pkl"
FAISS_INDEX_FILE = INDEX_DIR / "index.faiss"

# Load model (will download first time)
print("Loading embedding model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

def collect_docs(data_dir=DATA_DIR):
    docs = []        # texts to embed
    metadatas = []   # parallel metadata dicts
    paths = sorted(glob.glob(str(data_dir / "*.json")))
    if not paths:
        raise RuntimeError(f"No json files found in {data_dir}. Run ingestion first.")
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            rec = json.load(f)
        doc_id = rec.get("doc_id")
        title = rec.get("title")
        url = rec.get("url")
        source = rec.get("source")
        publishedAt = rec.get("publishedAt")
        chunks = rec.get("chunks", [])
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metadatas.append({
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "source": source,
                "publishedAt": publishedAt,
                "chunk_index": i,
            })
    print(f"Collected {len(docs)} chunks from {len(paths)} files.")
    return docs, metadatas

def build_faiss_index(docs, metadatas, index_path=FAISS_INDEX_FILE):
    # create embeddings
    print("Computing embeddings for docs...")
    embs = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    # ensure float32
    if embs.dtype != np.float32:
        embs = embs.astype("float32")

    dim = embs.shape[1]
    print(f"Embedding dim = {dim}; building FAISS IndexFlatL2 ...")
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    # persist
    print("Saving index and metadata to", INDEX_DIR)
    faiss.write_index(index, str(index_path))
    np.save(str(EMBEDDINGS_FILE), embs)
    with open(str(METADATA_FILE), "wb") as f:
        pickle.dump(metadatas, f)
    print("Saved:", FAISS_INDEX_FILE, EMBEDDINGS_FILE, METADATA_FILE)

def load_index(index_path=FAISS_INDEX_FILE):
    if not index_path.exists():
        raise RuntimeError("Index file not found. Run build first.")
    index = faiss.read_index(str(index_path))
    with open(str(METADATA_FILE), "rb") as f:
        metadatas = pickle.load(f)
    embs = np.load(str(EMBEDDINGS_FILE))
    return index, metadatas, embs

def query_index(query_text, top_k=5):
    # embed query
    q_emb = model.encode([query_text], convert_to_numpy=True).astype("float32")
    index, metadatas, _ = load_index()
    D, I = index.search(q_emb, top_k)
    results = []
    for rank, idx in enumerate(I[0]):
        meta = metadatas[idx]
        score = float(D[0][rank])
        results.append({"rank": rank+1, "score": score, "metadata": meta})
    return results

if __name__ == "__main__":
    # Build index (run once after ingestion or when data changes)
    docs, metadatas = collect_docs()
    build_faiss_index(docs, metadatas)

    # Quick test query
    q = "artificial intelligence regulation"
    print("\nTesting query:", q)
    res = query_index(q, top_k=5)
    for r in res:
        md = r["metadata"]
        print(f"R{r['rank']} score={r['score']:.4f} - {md['title']} ({md['source']})")
        print("  url:", md["url"])
        print("  chunk_index:", md["chunk_index"])
        print()
