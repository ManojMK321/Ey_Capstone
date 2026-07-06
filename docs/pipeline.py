"""
Shared singleton for all ingestion + retrieval components.
Both upload_api and chat_api import from here so they share
the same Pinecone index — uploads are instantly visible to chat.
"""
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

logger = logging.getLogger(__name__)


class _UnavailableParser:
    def parse(self, pdf_path):
        raise RuntimeError("PDF parsing dependencies are unavailable.")


class _UnavailableChunker:
    def chunk(self, pages, doc_id, filename):
        raise RuntimeError("Text chunking dependencies are unavailable.")


class _UnavailableEmbedder:
    embedding_model = None


class _UnavailableVectorStore:
    vector_store = None

    def add_documents(self, documents):
        raise RuntimeError(
            "Vector store is not configured. "
            "Add PINECONE_API_KEY and PINECONE_INDEX_NAME to your .env file."
        )

    def delete_document(self, doc_id):
        return None

    def similarity_search(self, query: str, k: int = 5, doc_id=None):
        return []

    def similarity_search_with_score(self, query: str, k: int = 5, doc_id=None):
        return []

    def document_count(self):
        return 0

    def get_all_doc_ids(self):
        return []

    def get_all_documents(self, doc_id=None):
        return []


_parser = None
_chunker = None
_embedder = None
_vector_store = None


def get_pipeline():
    global _parser, _chunker, _embedder, _vector_store
    if _embedder is None:
        logger.info("Initialising shared ingestion pipeline...")
        try:
            import os
            from docs.parser import PDFParser
            from docs.chunking import DocumentChunker
            from docs.embedding import OpenAIEmbedder

            _parser   = PDFParser()
            _chunker  = DocumentChunker()
            _embedder = OpenAIEmbedder()

            from docs.vector_store_pinecone import PineconeVectorStoreManager
            _vector_store = PineconeVectorStoreManager(
                embedding_model=_embedder.embedding_model,
            )
            logger.info("Pipeline ready — vector store: Pinecone.")
        except Exception as exc:
            logger.warning("Pipeline init failed — using stub components: %s", exc)
            _parser       = _UnavailableParser()
            _chunker      = _UnavailableChunker()
            _embedder     = _UnavailableEmbedder()
            _vector_store = _UnavailableVectorStore()
    return _parser, _chunker, _vector_store


def get_vector_store():
    _, _, vs = get_pipeline()
    return vs


def get_embedder():
    get_pipeline()
    return _embedder
