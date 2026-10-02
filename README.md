# 🩺 Medical ChatBot (RAG)

A Retrieval-Augmented Generation chatbot that answers questions from `data/KML_Medical_book.pdf`
(Gale Encyclopedia of Medicine, 2nd ed., Vol. 1, A–B).

**Stack:** Flask · LangChain · Hugging Face embeddings (`all-MiniLM-L6-v2`, 384-d) · Pinecone · Groq LLM (`openai/gpt-oss-20b`)

> Educational use only. Not a substitute for professional medical advice.

## How it works
PDF → PyPDFLoader → text splitter → embeddings → Pinecone.
Question → embedding → top-3 similar chunks → prompt + context → Groq LLM → answer (with PDF page numbers).

## Setup
1. Python 3.10+ (3.12 recommended). Create an environment and install:

       python -m venv .venv
       source .venv/bin/activate        # Windows: .venv\Scripts\activate
       pip install -r requirements.txt

2. Get free API keys from Pinecone (https://www.pinecone.io) and Groq (https://console.groq.com).
3. Create your env file and paste the keys in:

       cp .env.example .env             # Windows: copy .env.example .env

4. Build the vector index (one time, a few minutes; it downloads the ~90 MB embedding model first):

       python store_index.py

5. Run the app and open http://localhost:8080:

       python app.py

## Project structure
    app.py            Flask app: GET / (UI), POST /get (msg=...), GET /health
    store_index.py    one-time PDF -> embeddings -> Pinecone
    src/helper.py     PDF loading, chunking, embedding model
    src/prompt.py     system prompt
    templates/chat.html, static/css/style.css, static/js/script.js   frontend
    data/             put PDFs here
    .env.example      required keys

## Troubleshooting
- **"Missing PINECONE_API_KEY…" in the chat:** create `.env` from `.env.example` and restart.
- **"Pinecone index 'medicalbot' does not exist":** run `python store_index.py`.
- **Import errors from LangChain:** package layouts change between versions; run `pip install -U -r requirements.txt`,
  or pin the versions of a working environment with `pip freeze`.
- Re-running `store_index.py` on an existing index adds the chunks again (duplicates). Delete the index in the
  Pinecone dashboard first if you want a clean rebuild.
- Never commit `.env`.

## API
`POST /get` with form field `msg` → `{"answer": "...", "pages": [393, 394]}` or `{"error": "..."}`.
