"""One-time step: read the PDF(s) in data/, embed them and upload them to Pinecone.

Usage:  python store_index.py
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalbot")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        sys.exit("PINECONE_API_KEY is missing. Copy .env.example to .env and fill it in.")

    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    from src.helper import (EMBEDDING_DIMENSION, download_hugging_face_embeddings,
                            filter_to_minimal_docs, load_pdf_files, text_split)

    print("Loading PDF(s)...")
    docs = filter_to_minimal_docs(load_pdf_files(DATA_DIR))
    chunks = text_split(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")

    pc = Pinecone(api_key=key)
    if not pc.has_index(INDEX_NAME):
        print(f"Creating Pinecone index '{INDEX_NAME}' (dimension {EMBEDDING_DIMENSION}, cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                region=os.getenv("PINECONE_REGION", "us-east-1")),
        )
    else:
        print(f"Index '{INDEX_NAME}' already exists; adding the chunks to it.")

    print("Embedding and uploading (this can take several minutes the first time)...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=download_hugging_face_embeddings(),
        index_name=INDEX_NAME,
        batch_size=100,
    )
    print("Done. Now run:  python app.py")


if __name__ == "__main__":
    main()
