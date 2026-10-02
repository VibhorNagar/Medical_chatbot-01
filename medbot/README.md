# MedBot – medical chatbot built from KML_Medical_book.pdf

Offline retrieval chatbot (Flask backend + single-page frontend). Every answer is quoted from the book
with its PDF page number; nothing is invented. No API keys or internet needed.

## Run
    pip install -r requirements.txt
    python app.py              # open http://localhost:5000

## Rebuild the index (e.g. with a different book)
    # needs poppler's `pdftotext` on PATH
    python build_index.py path/to/book.pdf

## Files
- `build_index.py` – parses the PDF into entries/sections and builds a TF-IDF index (`data/index.pkl`)
- `app.py` – `/api/chat` (POST {message,last_topic}), `/api/topics`, serves the UI
- `static/index.html` – chat UI (follow-up questions, suggestion chips, emergency banner, dark mode)

Note: the PDF is the Gale Encyclopedia of Medicine, 2nd ed., Vol. 1 (A–B), so only topics A–B are covered.
