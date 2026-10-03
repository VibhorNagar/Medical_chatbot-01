"""Cosine-similarity search engine over a medical PDF.

    cosine similarity = (A . B) / (|A| * |B|)

The PDF is split into chunks, each chunk and each question becomes a vector (TF-IDF by default, or
sentence embeddings with method="embed") and the chunks with the highest cosine score are returned.
"""
import os
import pickle
import re
from typing import Dict, List

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CACHE_VERSION = 3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EngineError(Exception):
    """A problem the user can fix (missing PDF, missing package...)."""


def _chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                      # words broken across lines
    text = re.sub(r"GEM\s*-\s*\d+\s*to\s*\d+\s*-\s*[A-Z]\s*\d+/\d+/\d+\s*\d+:\d+\s*[AP]M\s*Page\s*\d+", " ", text)
    text = re.sub(r"GALE\s+ENCYCLOPEDIA\s+OF\s+MEDICINE(?:\s+2)?\s*\d*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                                            # cut at a sentence/space, not mid-word
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) > 40:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class CosineEngine:
    def __init__(self, pdf_path: str, cache_dir: str, method: str = "tfidf"):
        if method not in ("tfidf", "embed"):
            raise EngineError("method must be 'tfidf' or 'embed'")
        self.pdf_path, self.method = pdf_path, method
        self.cache_path = os.path.join(cache_dir, f"index_{method}.pkl")
        self._model = None
        idx = self._load_cache() or self._build()
        self.chunks, self.matrix, self.vectorizer = idx["chunks"], idx["matrix"], idx.get("vectorizer")

    # ---- index: load from cache or build from the PDF ----
    def _load_cache(self):
        try:
            with open(self.cache_path, "rb") as f:
                idx = pickle.load(f)
            if idx.get("version") == CACHE_VERSION and idx.get("method") == self.method:
                return idx
        except Exception:               # missing, corrupt or from another library version -> rebuild
            pass
        return None

    def _build(self):
        if not os.path.isfile(self.pdf_path):
            raise EngineError(
                f"The index is missing and the PDF was not found at:\n  {self.pdf_path}\n"
                "Put the book in the data folder (data/KML_Medical_book.pdf) or set MEDBOT_PDF=/path/to/book.pdf.")
        if self.method == "embed":
            self._embedder()                    # fail early if sentence-transformers is not installed
        print("[MedBot] Building the index from the PDF (first run only, about a minute)...", flush=True)
        chunks: List[Dict] = []
        for page_no, page in enumerate(PdfReader(self.pdf_path).pages, 1):
            for piece in _chunk_text(page.extract_text() or ""):
                chunks.append({"text": piece, "page": page_no})
        if not chunks:
            raise EngineError("No text could be read from the PDF (is it a scanned image?).")
        texts = [c["text"] for c in chunks]
        idx = {"version": CACHE_VERSION, "method": self.method, "chunks": chunks}
        if self.method == "embed":
            idx["matrix"] = np.asarray(self._embedder().encode(texts, batch_size=64, normalize_embeddings=True))
        else:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            idx["matrix"], idx["vectorizer"] = vec.fit_transform(texts), vec
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(idx, f)
        return idx

    def _embedder(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise EngineError("method 'embed' needs:  pip install sentence-transformers")
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    # ---- search ----
    def search(self, question: str, top_k: int = 3, min_score: float = 0.15) -> List[Dict]:
        if self.method == "embed":
            q = self._embedder().encode([question], normalize_embeddings=True)
        else:
            q = self.vectorizer.transform([question])
        scores = cosine_similarity(q, self.matrix).ravel()             # one cosine score per chunk
        best = np.argsort(-scores)[:top_k]
        return [{"score": round(float(scores[i]), 3), "page": self.chunks[i]["page"], "text": self.chunks[i]["text"]}
                for i in best if scores[i] >= min_score]
