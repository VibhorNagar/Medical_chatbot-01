"""Flask backend for the cosine-similarity medical chatbot.

Run:  python app.py      then open http://localhost:5000
"""
import os
import re
import sys

from flask import Flask, jsonify, render_template, request

from chatbot import CosineEngine, EngineError

BASE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.environ.get("MEDBOT_PDF", os.path.join(BASE, "data", "KML_Medical_book.pdf"))
METHOD = os.environ.get("MEDBOT_METHOD", "tfidf")          # "tfidf" (default) or "embed"
MIN_SCORE = float(os.environ.get("MEDBOT_MIN_SCORE", "0.15"))
DISCLAIMER = "Educational use only. This is text from a medical reference book, not medical advice."
EMERGENCY = re.compile(r"chest pain|can'?t breathe|cannot breathe|difficulty breathing|stroke|suicid|"
                       r"kill myself|overdose|unconscious|severe bleeding|poison", re.I)

app = Flask(__name__)

try:
    engine = CosineEngine(PDF_PATH, os.path.join(BASE, "data"), METHOD)
except EngineError as e:
    sys.exit(f"[MedBot] {e}")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", passages=len(engine.chunks), method=METHOD)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("message") or "").strip()[:500]
    if not question:
        return jsonify(error="Please type a question."), 400
    results = engine.search(question, top_k=3, min_score=MIN_SCORE)
    return jsonify(matched=bool(results), results=results, disclaimer=DISCLAIMER,
                   emergency=bool(EMERGENCY.search(question)))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
