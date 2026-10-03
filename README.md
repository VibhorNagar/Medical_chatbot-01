# 🩺 Cosine Medical ChatBot

A medical question-answering chatbot with a **Flask backend** and a **plain HTML/CSS/JavaScript frontend**.
It answers from a medical PDF (*Gale Encyclopedia of Medicine, 2nd ed., Vol. 1, A–B*) using **cosine similarity**.
No API keys, no database, no internet needed once the packages are installed.

> Educational use only. Not a substitute for professional medical advice.

## How it works
```
PDF -> split into chunks -> vectors (TF-IDF) -> saved index
Question -> vector -> cosine similarity with every chunk -> top 3 passages (with PDF page numbers)
```
`cosine similarity = (A · B) / (|A| × |B|)` — the closer to 1, the more similar the question and the passage.
The bot shows the best matching passages from the book; it does not write new sentences.

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py                     # open http://localhost:5000
```
On the first start the index is built from `data/KML_Medical_book.pdf` (about a minute).
If the index file is missing or corrupt it is rebuilt automatically; if the PDF is missing too, the app stops
with a message telling you where to put it (or set `MEDBOT_PDF=/path/to/book.pdf`).

**Note:** the PDF is copyrighted and is listed in `.gitignore`, so it is not uploaded to GitHub. Put your own copy in `data/`.

## Options (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `MEDBOT_PDF` | `data/KML_Medical_book.pdf` | path to the book |
| `MEDBOT_METHOD` | `tfidf` | `embed` = sentence embeddings (better meaning-matching; `pip install sentence-transformers`, ~90 MB model download) |
| `MEDBOT_MIN_SCORE` | `0.15` | minimum cosine score to accept a match |
| `PORT` | `5000` | web port |

## Project structure
```
app.py                Flask backend: GET /, GET /health, POST /api/chat
chatbot/engine.py     PDF chunking, vectors, cosine search, index cache
build_index.py        optional: build the index ahead of time
templates/index.html  chat page
static/css/style.css, static/js/script.js
tests/test_app.py     smoke tests:  python -m unittest discover tests
data/                 put the PDF here (index_*.pkl is generated)
```

## API
`POST /api/chat` with JSON `{"message": "What is asthma?"}` returns
`{"matched": true, "results": [{"score": 0.2, "page": 393, "text": "..."}], "emergency": false, "disclaimer": "..."}`

## Limitations
- Covers only the topics in the supplied book (A–B), so other questions get "no good match".
- TF-IDF matches shared words, not meaning, so the top passage is sometimes related but not the best one.
- Each question is answered on its own (no chat memory).
- `index_*.pkl` is a pickle file: only load index files you built yourself.

## Put it on GitHub
```bash
git init
git add .
git commit -m "Cosine medical chatbot"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
GitHub only stores the code. **GitHub Pages cannot run the Flask backend**, so to show the chatbot online you need a
Python host (for example Render or PythonAnywhere); otherwise run it locally for your demo.
