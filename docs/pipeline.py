"""
Shared singleton for all ingestion + retrieval components.
Both upload_api and chat_api import from here so they share
the same in-memory FAISS index — uploads are instantly visible to chat.
"""
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

from docs.parser import PDFParser
from docs.chunking import DocumentChunker
from docs.embedding import OpenAIEmbedder
from docs.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

_parser:       PDFParser | None        = None
_chunker:      DocumentChunker | None  = None
_embedder:     OpenAIEmbedder | None   = None
_vector_store: FAISSVectorStore | None = None


def get_pipeline() -> tuple[PDFParser, DocumentChunker, FAISSVectorStore]:
    global _parser, _chunker, _embedder, _vector_store
    if _embedder is None:
        logger.info("Initialising shared ingestion pipeline...")
        _parser       = PDFParser()
        _chunker      = DocumentChunker()
        _embedder     = OpenAIEmbedder()
        _vector_store = FAISSVectorStore(
            index_dir="faiss_index",
            embedding_model=_embedder.embedding_model,
        )
        logger.info("Pipeline ready.")
    return _parser, _chunker, _vector_store


def get_vector_store() -> FAISSVectorStore:
    _, _, vs = get_pipeline()
    return vs


def get_embedder() -> OpenAIEmbedder:
    get_pipeline()
    return _embedder
