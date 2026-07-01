"""
src/agents/retriever.py
Hybrid BM25 + dense retriever and cross-encoder reranker.

Dependencies (add to requirements.txt):
    rank_bm25>=0.2.2
    sentence-transformers>=2.7.0
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Lazy-loaded singleton — model loads on first call (~200 MB, once per process)
_cross_encoder: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model (first call only)...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder ready")
    return _cross_encoder


def build_hybrid_retriever(
    vector_store,           # FAISSVectorStore instance from docs/vector_store.py
    k: int = 20,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> EnsembleRetriever:
    """
    Build a BM25 + dense EnsembleRetriever from the current FAISS index state.

    BM25 handles exact-match contract terms ("Effective Date", "Force Majeure").
    Dense handles semantic similarity.
    Weights are used in Reciprocal Rank Fusion.

    NOTE: call this after each batch of uploads; BM25 is built from the snapshot
    of documents in the index at call time.
    """
    all_docs: list[Document] = list(vector_store.vector_store.docstore._dict.values())

    if not all_docs:
        raise RuntimeError(
            "FAISS index is empty — upload at least one PDF before building the retriever."
        )

    logger.info("Building BM25 from %d chunks", len(all_docs))
    bm25 = BM25Retriever.from_documents(all_docs, k=k)

    dense = vector_store.vector_store.as_retriever(search_kwargs={"k": k})

    logger.info(
        "EnsembleRetriever ready (BM25=%.1f, dense=%.1f, k=%d)",
        bm25_weight, dense_weight, k,
    )
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_weight, dense_weight],
    )


def rerank(
    query: str,
    docs: list[Document],
    top_n: int = 5,
) -> list[Document]:
    """
    Re-rank candidate documents with a cross-encoder and return the top_n.

    The cross-encoder scores (query, passage) pairs jointly, catching relevance
    signals that cosine similarity on embeddings misses (e.g. negation, specificity).
    """
    if not docs:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top = [doc for doc, _ in ranked[:top_n]]

    logger.info(
        "Reranked %d -> %d  (top score: %.4f, bottom kept: %.4f)",
        len(docs), len(top),
        ranked[0][1] if ranked else 0.0,
        ranked[top_n - 1][1] if len(ranked) >= top_n else 0.0,
    )
    return top
