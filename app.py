"""Flask web app for the RAG medical chatbot (Hugging Face embeddings + Pinecone + Groq)."""
import os
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalbot")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TOP_K = int(os.getenv("TOP_K", "3"))

app = Flask(__name__)

_rag = None
_lock = threading.Lock()


class SetupError(Exception):
    """A problem the user can fix (missing key, missing index...). Shown in the chat window."""


def build_rag():
    """Create the retriever + LLM chain once. Raises SetupError with a clear message if setup is incomplete."""
    missing = [k for k in ("PINECONE_API_KEY", "GROQ_API_KEY") if not os.getenv(k)]
    if missing:
        raise SetupError("Missing " + " and ".join(missing) + ". Copy .env.example to .env, fill it in, then restart.")

    from pinecone import Pinecone
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    from langchain_pinecone import PineconeVectorStore
    from src.helper import download_hugging_face_embeddings
    from src.prompt import system_prompt

    if not Pinecone(api_key=os.environ["PINECONE_API_KEY"]).has_index(INDEX_NAME):
        raise SetupError(f"Pinecone index '{INDEX_NAME}' does not exist yet. Run:  python store_index.py")

    docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=download_hugging_face_embeddings())
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt | llm | StrOutputParser()

    def answer(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        text = chain.invoke({"context": context, "input": question})
        sources = sorted({d.metadata.get("page") for d in docs if d.metadata.get("page")})
        return text, sources

    return answer


def get_rag():
    global _rag
    with _lock:
        if _rag is None:
            _rag = build_rag()      # not cached on failure, so fixing .env + retrying works after restart
        return _rag


@app.get("/")
def index():
    return render_template("chat.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/get")
def chat():
    question = (request.form.get("msg") or (request.get_json(silent=True) or {}).get("msg") or "").strip()
    if not question:
        return jsonify(error="Please type a question."), 400
    try:
        text, pages = get_rag()(question)
    except SetupError as e:
        return jsonify(error=str(e)), 503
    except Exception as e:      # network / API errors
        return jsonify(error=f"Something went wrong while answering ({type(e).__name__}). Check your API keys and internet connection."), 500
    return jsonify(answer=text, pages=pages)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
