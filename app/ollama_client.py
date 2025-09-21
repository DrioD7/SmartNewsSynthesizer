# app/ollama_client.py  (replace generate_with_ollama or add this helper)
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

GENERATE_ENDPOINT = f"{OLLAMA_URL}/api/generate"
CHAT_ENDPOINT = f"{OLLAMA_URL}/api/chat"

def _extract_json_objects_from_text(s: str):
    """
    Extract JSON objects from a concatenated/streaming text blob.
    Returns a list of parsed dicts. Tolerant to NDJSON / concatenated JSON.
    """
    objs = []
    decoder = json.JSONDecoder()
    pos = 0
    L = len(s)
    while True:
        # find next '{'
        brace = s.find('{', pos)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(s, idx=brace)
            objs.append(obj)
            pos = end
        except ValueError:
            # skip this brace and continue
            pos = brace + 1
            continue
    return objs

def _assemble_responses_from_objs(objs):
    """
    Given a list of parsed JSON objects (from Ollama streaming),
    join their 'response' or 'text' fields in order to form final text.
    """
    parts = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        # common keys: "response", "text", "output", "message"
        if "response" in o and isinstance(o["response"], str):
            parts.append(o["response"])
        elif "text" in o and isinstance(o["text"], str):
            parts.append(o["text"])
        elif "output" in o:
            # could be list or string
            out = o["output"]
            if isinstance(out, list):
                for it in out:
                    if isinstance(it, str):
                        parts.append(it)
                    elif isinstance(it, dict) and "response" in it:
                        parts.append(it["response"])
            elif isinstance(out, str):
                parts.append(out)
        elif "message" in o:
            # message may contain content
            msg = o["message"]
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("text")
                if isinstance(content, str):
                    parts.append(content)
    # join without extra separators (Ollama streaming already splits words sensibly)
    return "".join(parts).strip()

def generate_with_ollama(prompt, model=None, max_tokens=512, use_chat=False, timeout=120):
    """
    Calls Ollama /api/generate (or /api/chat) and returns the assembled text
    (concatenation of streaming 'response' pieces). This avoids JSON parse errors.
    """
    model = model or OLLAMA_MODEL
    url = CHAT_ENDPOINT if use_chat else GENERATE_ENDPOINT

    if use_chat:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
        }
    else:
        payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}

    headers = {"Content-Type": "application/json"}
    # Use stream=True to get raw bytes (some Ollama versions stream NDJSON)
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()

    # Read entire response bytes (streaming or not)
    try:
        # decode safely
        raw_text = resp.content.decode("utf-8", errors="replace")
    except Exception:
        raw_text = resp.text

    # Try parse as single JSON object first
    try:
        parsed = json.loads(raw_text)
        # If parsed JSON contains a simple text field, return it
        if isinstance(parsed, dict):
            # Common keys
            for k in ("text", "response", "output"):
                if k in parsed:
                    val = parsed[k]
                    if isinstance(val, str):
                        return val.strip()
                    if isinstance(val, list):
                        # join list into text
                        return " ".join(map(str, val)).strip()
            # As fallback, return pretty json string
            return json.dumps(parsed)
    except json.JSONDecodeError:
        pass

    # If not single JSON, try to extract multiple JSON objects (NDJSON / concatenated)
    objs = _extract_json_objects_from_text(raw_text)
    if objs:
        assembled = _assemble_responses_from_objs(objs)
        if assembled:
            return assembled

    # Last-resort: return raw_text as fallback
    return raw_text
