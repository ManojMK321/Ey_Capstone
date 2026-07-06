"""
knowledge_rag.py — Production-ready single-hop RAG pipeline.

Flow: Question → Hybrid Retrieve (BM25 + Dense) → Cross-encoder Rerank → LLM Answer

Graceful degradation:
  - rank_bm25 / sentence-transformers missing → dense-only FAISS fallback
  - Reranker fails at runtime                 → returns un-reranked candidates
  - LLM returns empty                         → safe fallback message
"""

import logging
import time
from typing import Optional

from langchain_core.documents import Document
from openai import OpenAI

from src.observability.langsmith import traceable_operation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional hybrid retrieval — graceful degradation if deps absent
# ---------------------------------------------------------------------------
try:
    from src.agents.retriever import build_hybrid_retriever, rerank as _rerank
    _HYBRID_AVAILABLE = True
    logger.info("KnowledgeRAG: hybrid retrieval (BM25 + dense + rerank) enabled.")
except ImportError:
    _HYBRID_AVAILABLE = False
    logger.warning(
        "KnowledgeRAG: rank_bm25 / sentence-transformers not installed. "
        "Falling back to dense-only Pinecone retrieval. "
        "Install with: pip install rank_bm25 sentence-transformers"
    )


_SYSTEM_PROMPT = (
    "You are a precise contract knowledge assistant. "
    "Answer the user's question using ONLY the provided context. "
    "For EVERY factual claim you make, include an inline citation in the exact "
    "format [doc_id, page]. "
    "If the context does not contain the answer, say so explicitly — do not guess."
)


def _extract_response_text(response) -> str:
    if getattr(response, "output_text", None):
        return response.output_text
    if getattr(response, "output", None):
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for block in getattr(item, "content", []) or []:
                    if getattr(block, "type", None) == "output_text":
                        return getattr(block, "text", "")
    return ""


def _extract_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens":  getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }


def _build_context(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents were retrieved."
    parts = []
    for doc in docs:
        if isinstance(doc, dict):
            meta = doc.get("metadata") or {}
            content = str(doc.get("content") or "").strip()
            filename = meta.get("filename") or meta.get("source") or "document"
            page = meta.get("page")
            header = f"[doc_id={meta.get('doc_id', 'unknown')}, page={page if page is not None else '?'}, file={filename}]"
            parts.append(f"{header}\n{content}")
            continue

        meta = getattr(doc, "metadata", None) or {}
        header = (
            f"[doc_id={meta.get('doc_id', 'unknown')}, "
            f"page={meta.get('page', '?')}, "
            f"file={meta.get('filename') or meta.get('source', 'document')}]"
        )
        parts.append(f"{header}\n{getattr(doc, 'page_content', '').strip()}")
    return "\n\n---\n\n".join(parts)


def _docs_to_sources(docs: list[Document]) -> list[dict]:
    sources = []
    for doc in docs:
        if isinstance(doc, dict):
            meta = doc.get("metadata") or {}
            filename = meta.get("filename") or meta.get("source") or "document"
            page = meta.get("page")
            sources.append({
                "doc_id": meta.get("doc_id"),
                "page": page,
                "filename": filename,
                "display": f"{filename} (page {page})" if page is not None else filename,
            })
            continue

        meta = getattr(doc, "metadata", None) or {}
        filename = meta.get("filename") or meta.get("source") or "document"
        page = meta.get("page")
        sources.append({
            "doc_id": meta.get("doc_id"),
            "page": page,
            "filename": filename,
            "display": f"{filename} (page {page})" if page is not None else filename,
        })
    return sources


class KnowledgeRAG:
    """
    Production-ready single-hop RAG pipeline.

    Retrieval (best-available strategy):
        Hybrid  → EnsembleRetriever (BM25 + dense, configurable weights)
                  then cross-encoder rerank: k_initial candidates → k_final
        Fallback → FAISS similarity_search when hybrid deps are absent
    """

    def __init__(
        self,
        client: OpenAI,
        vector_store,
        model: str = "gpt-4o",
        k_initial: int = 20,
        k_final: int | None = None,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        max_output_tokens: int = 600,
        temperature: float = 0.0,
        top_k: int | None = None,
    ):
        self.client = client
        self.vector_store = vector_store
        self.model = model
        self.k_initial = k_initial
        self.k_final = k_final if k_final is not None else (top_k if top_k is not None else 5)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

    @traceable_operation(
        name="KnowledgeRAG retrieval",
        tags=["knowledge_rag", "retrieval"],
        metadata={"component": "knowledge_rag"},
    )
    def retrieve(self, query: str) -> tuple[list[Document], str]:
        if _HYBRID_AVAILABLE:
            try:
                retriever = build_hybrid_retriever(
                    self.vector_store,
                    k=self.k_initial,
                    bm25_weight=self.bm25_weight,
                    dense_weight=self.dense_weight,
                )
                candidates = retriever.invoke(query)
                docs = _rerank(query, candidates, top_n=self.k_final)
                logger.debug(
                    "KnowledgeRAG: hybrid retrieval returned %d docs for query: %.80s",
                    len(docs), query,
                )
                return docs, "hybrid"
            except Exception as exc:
                logger.warning(
                    "KnowledgeRAG: hybrid retrieval failed (%s); falling back to dense.", exc
                )

        docs = self.vector_store.similarity_search(query=query, k=self.k_final)
        if not docs:
            logger.info("KnowledgeRAG: no documents retrieved for query: %.80s", query)
        else:
            logger.debug("KnowledgeRAG: dense retrieval returned %d docs.", len(docs))
        return docs, "dense"

    @traceable_operation(
        name="KnowledgeRAG answer",
        tags=["knowledge_rag", "llm"],
        metadata={"component": "knowledge_rag"},
    )
    def answer(
        self,
        query: str,
        docs: list[Document],
        history: Optional[str] = None,
    ) -> str:
        context = _build_context(docs)
        history_section = f"Previous conversation:\n{history}\n\n" if history else ""

        user_prompt = (
            f"{history_section}"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer (cite every claim as [doc_id, page]):"
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            logger.error("KnowledgeRAG: LLM call failed for query: %.80s — %s", query, exc)
            raise

        self.last_usage = _extract_usage(response)
        answer_text = _extract_response_text(response).strip()
        if not answer_text:
            logger.warning("KnowledgeRAG: empty response from model for query: %.80s", query)
            return "I couldn't generate an answer from the retrieved documents."

        return answer_text

    def run(self, query: str, history: Optional[str] = None) -> dict:
        t0 = time.perf_counter()

        docs, retrieval_mode = self.retrieve(query)
        answer_text = self.answer(query, docs, history)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "KnowledgeRAG.run | query=%.80s | mode=%s | docs=%d | latency=%.1fms",
            query, retrieval_mode, len(docs), latency_ms,
        )

        return {
            "answer":  answer_text,
            "sources": _docs_to_sources(docs),
            "docs":    docs,
            "meta": {
                "retrieval_mode": retrieval_mode,
                "docs_retrieved": len(docs),
                "latency_ms":     round(latency_ms, 2),
            },
        }
