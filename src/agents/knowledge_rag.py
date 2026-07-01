import logging

from langchain_core.documents import Document
from openai import OpenAI

from docs.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

# Optional enhanced retrieval — gracefully degrades if deps not installed.
# Catches Exception (not just ImportError) because a broken PyTorch install
# raises OSError / WinError 1114 when the DLL fails to load, which ImportError
# does not cover.
try:
    from src.agents.retriever import build_hybrid_retriever, rerank as _rerank
    _HYBRID_AVAILABLE = True
except Exception:
    _HYBRID_AVAILABLE = False
    logger.info(
        "rank_bm25 / sentence-transformers not available or failed to load; "
        "KnowledgeRAG falling back to dense-only retrieval."
    )


def _extract_response_text(response) -> str:
    if getattr(response, "output_text", None):
        return response.output_text
    if getattr(response, "output", None):
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for content_item in getattr(item, "content", []) or []:
                    if getattr(content_item, "type", None) == "output_text":
                        return getattr(content_item, "text", "")
    return ""


_SYSTEM_PROMPT = (
    "You are a precise contract knowledge assistant. "
    "Answer the user's question using ONLY the provided context. "
    "For EVERY factual claim you make, include an inline citation in the exact format [doc_id, page]. "
    "If the context does not contain the answer, say so explicitly — do not guess."
)


class KnowledgeRAG:
    """
    Single-hop RAG branch: Question -> Retrieve -> Answer.

    Retrieval strategy (when enhanced deps are installed):
      - Hybrid BM25 + dense (EnsembleRetriever, 0.4 / 0.6)
      - Cross-encoder rerank: top-20 candidates -> top-5
    Falls back to dense-only if rank_bm25 / sentence-transformers are absent.

    Interface expected by chat.py: retrieve(query) -> docs,
    answer(query, source_payload, history) -> str,
    run(query, history) -> dict.
    """

    def __init__(
        self,
        client: OpenAI,
        vector_store: FAISSVectorStore,
        top_k: int = 5,
        model: str = "gpt-4o",
        k_initial: int = 20,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        self.client = client
        self.vector_store = vector_store
        self.top_k = top_k
        self.model = model
        self.k_initial = k_initial
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> list:
        if _HYBRID_AVAILABLE:
            try:
                retriever = build_hybrid_retriever(
                    self.vector_store,
                    k=self.k_initial,
                    bm25_weight=self.bm25_weight,
                    dense_weight=self.dense_weight,
                )
                candidates = retriever.invoke(query)
                return _rerank(query, candidates, top_n=self.top_k)
            except Exception as exc:
                logger.warning("Hybrid retrieval failed (%s); falling back to dense.", exc)

        docs = self.vector_store.similarity_search(query=query, k=self.top_k)
        if not docs:
            logger.info("KnowledgeRAG: no documents retrieved for query: %s", query)
        return docs

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------
    @staticmethod
    def _build_context(source_payload: list[dict]) -> str:
        if not source_payload:
            return "No relevant documents were retrieved."
        return "\n\n---\n\n".join(
            f"[doc_id={item.get('doc_id', 'unknown')}, page={item.get('page', '?')}, "
            f"file={item.get('source', 'document')}]\n{item['content']}"
            for item in source_payload
        )

    def answer(self, query: str, source_payload: list[dict], history: str | None = None) -> str:
        context = self._build_context(source_payload)
        history_section = f"Previous conversation:\n{history}\n\n" if history else ""

        user_prompt = (
            f"{history_section}"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer (cite every claim as [doc_id, page]):"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_output_tokens=600,
        )
        answer_text = _extract_response_text(response).strip()
        if not answer_text:
            logger.warning("KnowledgeRAG: empty response from model for query: %s", query)
            return "I couldn't generate an answer from the retrieved documents."
        return answer_text

    # ------------------------------------------------------------------
    # run — full branch in one call
    # ------------------------------------------------------------------
    def run(self, query: str, history: str | None = None) -> dict:
        docs = self.retrieve(query)
        source_payload = []
        for doc in docs:
            metadata = doc.metadata or {}
            source_name = metadata.get("filename") or metadata.get("source") or "document"
            page_number = metadata.get("page")
            if page_number is not None:
                source_name = f"{source_name} (page {page_number})"
            source_payload.append({
                "source":  source_name,
                "doc_id":  metadata.get("doc_id"),
                "page":    page_number,
                "content": doc.page_content.strip(),
            })

        answer_text = self.answer(query, source_payload, history)
        return {"answer": answer_text, "docs": docs, "sources": source_payload}


# ---------------------------------------------------------------------------
# EnhancedKnowledgeRAG — dev/notebook class (always uses hybrid + reranking)
# ---------------------------------------------------------------------------

class EnhancedKnowledgeRAG:
    """
    Notebook / evaluation variant. Always uses hybrid BM25+dense retrieval
    and cross-encoder reranking. Returns structured sources with doc_id + page.

    Use KnowledgeRAG in production (graceful fallback); use this in notebooks
    when you want explicit control over every knob.
    """

    def __init__(
        self,
        client: OpenAI,
        vector_store: FAISSVectorStore,
        model: str = "gpt-4o",
        k_initial: int = 20,
        k_final: int = 5,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        self.client = client
        self.vector_store = vector_store
        self.model = model
        self.k_initial = k_initial
        self.k_final = k_final
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def retrieve(self, query: str) -> list[Document]:
        from src.agents.retriever import build_hybrid_retriever, rerank
        retriever = build_hybrid_retriever(
            self.vector_store,
            k=self.k_initial,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )
        candidates = retriever.invoke(query)
        return rerank(query, candidates, top_n=self.k_final)

    @staticmethod
    def _build_context(docs: list[Document]) -> str:
        if not docs:
            return "No relevant documents were retrieved."
        parts = []
        for doc in docs:
            meta = doc.metadata or {}
            parts.append(
                f"[doc_id={meta.get('doc_id', 'unknown')}, page={meta.get('page', '?')}, "
                f"file={meta.get('filename', 'document')}]\n{doc.page_content.strip()}"
            )
        return "\n\n---\n\n".join(parts)

    def answer(self, query: str, docs: list[Document], history: str | None = None) -> dict:
        context = self._build_context(docs)
        history_section = f"Previous conversation:\n{history}\n\n" if history else ""
        user_prompt = (
            f"{history_section}"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer (cite every claim as [doc_id, page]):"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_output_tokens=600,
        )
        answer_text = _extract_response_text(response).strip()
        if not answer_text:
            logger.warning("EnhancedKnowledgeRAG: empty model response for: %s", query)
            answer_text = "I couldn't generate an answer from the retrieved documents."

        sources = [
            {
                "doc_id":   (doc.metadata or {}).get("doc_id"),
                "page":     (doc.metadata or {}).get("page"),
                "filename": (doc.metadata or {}).get("filename"),
            }
            for doc in docs
        ]
        return {"answer": answer_text, "sources": sources}

    def run(self, query: str, history: str | None = None) -> dict:
        docs = self.retrieve(query)
        result = self.answer(query, docs, history)
        result["docs"] = docs
        return result
