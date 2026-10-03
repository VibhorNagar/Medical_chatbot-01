"""Optional: build the search index ahead of time.   python build_index.py [path/to/book.pdf]"""
import os
import sys

from chatbot import CosineEngine, EngineError

BASE = os.path.dirname(os.path.abspath(__file__))
pdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "data", "KML_Medical_book.pdf")
try:
    e = CosineEngine(pdf, os.path.join(BASE, "data"), os.getenv("MEDBOT_METHOD", "tfidf"))
    print(f"Index ready: {len(e.chunks)} passages")
except EngineError as err:
    sys.exit(f"[MedBot] {err}")
