# app/rag.py
import os
import textwrap
import json
from dotenv import load_dotenv

load_dotenv()

# Import helper functions from other modules in the app package
# Run this as module (python -m app.rag) from project root to have imports work.
from app.embed_index import query_index  # returns list of {metadata, score}
from app.ollama_client import generate_with_ollama

# Config: how many chunks to retrieve and max tokens to request from Ollama
TOP_K = int(os.getenv("RAG_TOP_K", 5))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", 512))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

# Prompt template pieces
SYSTEM_INSTRUCTION = (
   "You are a professional news summarizer. Use ONLY the numbered evidence passages provided below "
    "to write a concise, factual news-style summary that directly answers the user's query. "
    "Do NOT invent facts or use external knowledge. If the evidence does not mention the query, "
    "explicitly say 'No relevant evidence found for the query: <query>'."
)

PROMPT_INSTRUCTIONS = (
    "TASK:\n"
    "1) Using only the Evidence passages below, write a NEWS SUMMARY (120-250 words) that answers the user's query exactly.\n"
    "2) Do NOT provide unrelated background. If the evidence doesn't support the query, say so clearly.\n"
    "3) Use inline numeric citations [n] only where evidence supports a fact. If you cannot support a fact with provided evidence, do not state it.\n\n"
    "OUTPUT FORMAT:\n"
    "Only output plain human-readable text. No JSON, no lists, no metadata. Write 1-3 short paragraphs.\n"
)

def build_evidence_block(retrieved):
    """
    Build the Evidence section text from retrieved results.
    retrieved: list of dicts returned by query_index:
      each item: {"rank":..., "score":..., "metadata": {...}}
    We will number them starting at 1 and include a short header with source/title/date.
    """
    lines = []
    mapping = []  # map citation number -> metadata
    for i, item in enumerate(retrieved, start=1):
        md = item["metadata"]
        title = md.get("title") or "Untitled"
        source = md.get("source") or ""
        url = md.get("url") or ""
        pub = md.get("publishedAt") or ""
        chunk_index = md.get("chunk_index")
        # Short passage text must be retrieved from the original data store; we only have metadata here.
        # embed_index stored the chunks as the "docs" list — but we didn't persist the chunk text by id here.
        # For simplicity we will embed the chunk text into metadata earlier (if desired). For now we assume:
        # metadata contains a 'chunk_text' key. If not present, we'll put a placeholder.
        chunk_text = md.get("chunk_text") or md.get("snippet") or "(passage text unavailable)"
        header = f"[{i}] ({source}) {title} — {pub} — chunk:{chunk_index}"
        lines.append(header)
        # wrap the passage to reasonable width
        wrapped = textwrap.fill(chunk_text, width=500)
        lines.append(wrapped)
        lines.append("")  # blank line
        mapping.append({"idx": i, "title": title, "url": url, "source": source})
    evidence_text = "\n".join(lines)
    return evidence_text, mapping

def build_prompt(query, evidence_text):
    """
    Build the final prompt given a user query and evidence block.
    """
    prompt = []
    prompt.append(SYSTEM_INSTRUCTION)
    prompt.append(f"\nUser Query: {query}\n")
    prompt.append("Evidence:\n")
    prompt.append(evidence_text)
    prompt.append("\n" + PROMPT_INSTRUCTIONS)
    return "\n\n".join(prompt)

def prepare_retrieved_with_texts(raw_results, index_data_dir="index_data"):
    """
    The embed_index stored metadatas but chunk text may not be in metadata.
    If 'chunk_text' is missing, attempt to read chunk content from `data/` JSON files.
    This function will try to enrich metadata entries with the actual chunk text.
    """
    # Try to load data files into a dict by doc_id
    import glob, json
    data_map = {}
    for p in glob.glob("data/*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
            data_map[rec.get("doc_id")] = rec
        except Exception:
            continue

    enriched = []
    for item in raw_results:
        md = dict(item["metadata"])  # copy
        doc_id = md.get("doc_id")
        chunk_i = md.get("chunk_index")
        chunk_text = None
        if doc_id and doc_id in data_map:
            chunks = data_map[doc_id].get("chunks", [])
            if isinstance(chunk_i, int) and 0 <= chunk_i < len(chunks):
                chunk_text = chunks[chunk_i]
        if chunk_text:
            md["chunk_text"] = chunk_text
        else:
            md["chunk_text"] = md.get("chunk_text") or "(passage text unavailable)"
        enriched.append({"rank": item["rank"], "score": item["score"], "metadata": md})
    return enriched

def run_rag(query, top_k=TOP_K, model_name=OLLAMA_MODEL, max_tokens=OLLAMA_MAX_TOKENS, verbose=True):
    """
    End-to-end RAG call:
    1) Retrieve top_k passages
    2) Build evidence & prompt
    3) Call Ollama
    4) Return a dict with summary, raw_ollama, sources mapping
    """
    # 1) retrieve
    raw = query_index(query, top_k=top_k)
    if verbose:
        print(f"Retrieved {len(raw)} passages (top_k={top_k}).")
    # Enrich with actual chunk text from data/*.json
    retrieved = prepare_retrieved_with_texts(raw)

    # 2) build evidence and map
    evidence_text, mapping = build_evidence_block(retrieved)
    prompt = build_prompt(query, evidence_text)

    if verbose:
        print("\n--- Prompt preview (first 800 chars) ---\n")
        print(prompt[:800])
        print("\n--- End prompt preview ---\n")

    # 3) call Ollama
    # Use chat or generate endpoint depending on your ollama_client implementation
    try:
        ollama_out = generate_with_ollama(prompt, model=model_name, max_tokens=max_tokens, use_chat=False)
    except TypeError:
        # older generate_with_ollama signature (no use_chat param)
        ollama_out = generate_with_ollama(prompt, model=model_name, max_tokens=max_tokens)

    # The ollama_client returns a text string (extracted). If it returns structured JSON, keep raw as well.
    raw_response = ollama_out
    if isinstance(ollama_out, dict):
        # try to find extracted text
        text = ollama_out.get("text") or ollama_out.get("output") or str(ollama_out)
    else:
        text = str(ollama_out)

    # 4) return structured result
    result = {
        "query": query,
        "summary_text": text,
        "raw_ollama": raw_response,
        "sources": mapping,
        "retrieved": retrieved
    }
    return result

if __name__ == "__main__":
    # Quick manual test - run this module from project root:
    # python -m app.rag
    test_q = "artificial intelligence regulation"
    out = run_rag(test_q, top_k=5, verbose=True)
    print("\n=== GENERATED SUMMARY ===\n")
    print(out["summary_text"][:2000])
    print("\n=== SOURCES ===\n")
    for s in out["sources"]:
        print(f"[{s['idx']}] {s['title']} — {s['url']}")
