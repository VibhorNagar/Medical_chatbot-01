"""Helpers for loading the PDF, chunking it and creating the embedding model."""
import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dimensional vectors
EMBEDDING_DIMENSION = 384


def load_pdf_files(data_dir: str) -> List[Document]:
    """Load every PDF inside `data_dir` (one Document per page)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    if not docs:
        raise FileNotFoundError(f"No PDF files found in {data_dir}. Put the medical book there first.")
    return docs


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only the text, the source file name and the page number (small metadata = small index)."""
    minimal = []
    for d in docs:
        page = d.metadata.get("page")
        minimal.append(Document(
            page_content=d.page_content,
            metadata={
                "source": os.path.basename(str(d.metadata.get("source", ""))),
                "page": (int(page) + 1) if isinstance(page, int) else None,   # PyPDF pages start at 0
            },
        ))
    return minimal


def text_split(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if len(c.page_content.strip()) > 40]    # drop empty / header-only chunks


def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """Downloads the model on first use (about 90 MB), then loads it from the local cache."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
