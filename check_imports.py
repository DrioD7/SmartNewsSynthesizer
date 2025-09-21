# check_imports.py
modules = [
    "streamlit", "fastapi", "uvicorn", "requests", "dotenv",
    "newspaper", "bs4", "lxml", "lxml_html_clean",
    "sentence_transformers", "transformers", "tokenizers", "accelerate", "datasets", "sentencepiece",
    "faiss", "faiss.cpu", "pptx", "reportlab", "PIL", "codecarbon", "tqdm", "regex"
]

for m in modules:
    try:
        __import__(m.split('.')[0])
        print(f"OK: {m}")
    except Exception as e:
        print(f"FAILED: {m} -> {e.__class__.__name__}: {e}")

