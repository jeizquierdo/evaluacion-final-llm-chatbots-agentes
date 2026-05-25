"""
utils/rag_loader.py

Handles loading, chunking, and indexing of local documents from the data/ folder.
Supports PDF, TXT, and DOCX formats.

The index is built once on first call and cached in memory for the rest of
the session. If the data/ folder is empty or missing, all functions degrade
gracefully so the rest of the graph is not affected.
"""

import os
from pathlib import Path
from typing import Optional

# LangChain document loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings and vector store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR        = Path("data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # lightweight, fast, local
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K_RESULTS   = 4  # number of chunks to retrieve per query

# Supported file extensions and their corresponding loaders
SUPPORTED_EXTENSIONS = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
}

# ---------------------------------------------------------------------------
# MODULE-LEVEL CACHE
# In-memory cache so the index is only built once per session.
# ---------------------------------------------------------------------------

_vector_store: Optional[FAISS] = None
_embeddings:   Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the HuggingFace embeddings instance, creating it once and
    caching it for the rest of the session.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def data_folder_has_documents() -> bool:
    """
    Checks whether the data/ folder exists and contains at least one
    supported document. Used by the RAG tool to decide whether to proceed
    or return a graceful fallback message.

    Returns:
        True  — data/ exists and has at least one PDF, TXT, or DOCX file.
        False — data/ is missing, empty, or has no supported files.
    """
    if not DATA_DIR.exists():
        return False

    for ext in SUPPORTED_EXTENSIONS:
        if any(DATA_DIR.glob(f"*{ext}")):
            return True

    return False


def _load_documents() -> list:
    """
    Loads all supported documents from the data/ folder.

    Returns a flat list of LangChain Document objects, one per page/chunk
    depending on the loader. Returns an empty list if no documents are found.
    """
    documents = []

    for ext, LoaderClass in SUPPORTED_EXTENSIONS.items():
        for file_path in DATA_DIR.glob(f"*{ext}"):
            try:
                loader = LoaderClass(str(file_path))
                docs   = loader.load()
                documents.extend(docs)
                print(f"[rag_loader] Loaded: {file_path.name} ({len(docs)} chunks)")
            except Exception as e:
                print(f"[rag_loader] Failed to load {file_path.name}: {e}")

    return documents


def _build_vector_store() -> Optional[FAISS]:
    """
    Loads documents, splits them into chunks, and builds the FAISS index.

    Returns None if no documents are available, so callers can detect
    an empty index without raising exceptions.
    """
    documents = _load_documents()

    if not documents:
        print("[rag_loader] No documents found in data/. Index not built.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"[rag_loader] Built index with {len(chunks)} chunks from {len(documents)} documents.")

    embeddings   = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def get_vector_store() -> Optional[FAISS]:
    """
    Returns the cached FAISS vector store, building it on first call.

    Subsequent calls return the cached instance without reloading documents,
    making retrieval fast after the first invocation.

    Returns:
        FAISS vector store if documents are available, None otherwise.
    """
    global _vector_store

    if _vector_store is None:
        _vector_store = _build_vector_store()

    return _vector_store


def retrieve(query: str) -> tuple[str, list[str]]:
    """
    Retrieves the most relevant document chunks for the given query.

    Args:
        query: The search query string.

    Returns:
        A tuple of:
          - context (str): concatenated text of the top-K chunks.
          - sources (list[str]): deduplicated list of source filenames.

    If the index is empty or unavailable, returns empty strings/lists
    so the researcher can fall back to web search gracefully.
    """
    store = get_vector_store()

    if store is None:
        return "", []

    results = store.similarity_search(query, k=TOP_K_RESULTS)

    if not results:
        return "", []

    # Concatenate chunk texts with separators for readability
    context = "\n\n---\n\n".join(doc.page_content for doc in results)

    # Extract unique source filenames from document metadata
    sources = list(dict.fromkeys(
        Path(doc.metadata.get("source", "unknown")).name
        for doc in results
    ))

    return context, sources