import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, status
from openai import OpenAI

from src.agents.agentic_rag import AgenticRAG
from src.observability.langsmith import traceable_operation
from src.observability import metrics
from src.agents.intent_detection import IntentDetector, Workflow
from src.agents.knowledge_rag import KnowledgeRAG
from src.orchestrator.session_store_postgres import SessionStore
from src.retrieval.doc_registry import list_documents
from guardrails import input_guardrail, retrieval_guardrail, output_guardrail, describe_block
from guardrails.audit_logger import audit_logger
from src.schema.validation import (
    ChatHistoryItem,
    ChatRequest,
    ChatResponse,
    DocumentListResponse,
    DocumentStatus,
    SessionHistoryResponse,
)
from docs.pipeline import get_vector_store

load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

logger = logging.getLogger(__name__)
router = APIRouter()

session_man = SessionStore()
MAX_HISTORY_ITEMS = 20

# ---------------------------------------------------------------------------
# Lazy-initialised singletons (created on first request, not at import time)
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None
_intent_detector: IntentDetector | None = None
_knowledge_rag: KnowledgeRAG | None = None
_agentic_rag: AgenticRAG | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_agents() -> tuple[IntentDetector, KnowledgeRAG, AgenticRAG]:
    global _intent_detector, _knowledge_rag, _agentic_rag
    client = _get_client()
    vector_store = get_vector_store()
    if _intent_detector is None:
        _intent_detector = IntentDetector()
    if _knowledge_rag is None:
        _knowledge_rag = KnowledgeRAG(client=client, vector_store=vector_store, top_k=5)
    if _agentic_rag is None:
        _agentic_rag = AgenticRAG(client=client, vector_store=vector_store, top_k=5)
    return _intent_detector, _knowledge_rag, _agentic_rag

# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------

def _build_sources(docs: list) -> tuple[list[str], list[dict]]:
    sources: list[str] = []
    source_payload: list[dict] = []
    seen_content: set[str] = set()
    for item in docs:
        if isinstance(item, dict):
            content = item.get("content") or ""
            source_name = item.get("source") or "document"
        else:
            meta = getattr(item, "metadata", None) or {}
            source_name = meta.get("filename") or meta.get("source") or "document"
            page = meta.get("page")
            if page is not None:
                source_name = f"{source_name} (page {int(page)})"
            content = getattr(item, "page_content", "").strip()

        # Retrieval can surface the same passage more than once (re-uploads,
        # overlapping chunks). Identical content adds no new information to
        # the LLM's context, so it's dropped here rather than paid for twice.
        content_key = content.strip()
        if content_key and content_key in seen_content:
            continue
        seen_content.add(content_key)

        sources.append(source_name)
        source_payload.append({"source": source_name, "content": content})
    return sources, source_payload


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@traceable_operation(
    name="Chat request",
    tags=["api", "chat"],
    metadata={"endpoint": "/chat/"},
)
@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a chat query — auto-routed to KnowledgeRAG or AgenticRAG.",
)
async def chat_documents(request: ChatRequest) -> ChatResponse:
    raw_query = request.query.strip()
    if not raw_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query text cannot be empty.",
        )

    overall_start = time.perf_counter()
    workflow_type = "unknown"
    model_used = "unknown"
    usage: dict = {}

    try:
        client = _get_client()

        # ── Input Guardrail ────────────────────────────────────────────
        input_guard = input_guardrail.check_input(raw_query, client)
        audit_logger.log("input_guardrail", input_guard, session_id=request.session_id)
        if not input_guard.allowed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=describe_block(input_guard),
            )
        query = input_guard.data

        session_id = session_man.make_session(request.session_id, request.reset_session)
        history_context = session_man.history_context(session_id)
        session_man.append_history(session_id, "user", query)

        vector_store = get_vector_store()
        if getattr(vector_store, "vector_store", None) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No indexed documents are available. Upload a PDF first via POST /upload/.",
            )

        intent_detector, knowledge_rag, agentic_rag = _get_agents()

        start = time.perf_counter()
        intent = intent_detector.detect(query)
        workflow_type = intent.workflow.value
        logger.info(
            "Intent detected: %s (%.2f) — %s",
            intent.workflow.value, intent.confidence, intent.reason,
        )

        agent_blocked_info = None
        if intent.workflow == Workflow.KNOWLEDGE_RAG:
            model_used = knowledge_rag.model
            raw_retrieval = knowledge_rag.retrieve(query)
            if isinstance(raw_retrieval, tuple):
                docs, _ = raw_retrieval
            else:
                docs = raw_retrieval

            # ── Retrieval Guardrail ─────────────────────────────────────
            retrieval_guard = retrieval_guardrail.check_retrieval(query, docs)
            audit_logger.log("retrieval_guardrail", retrieval_guard, session_id=session_id)
            if not retrieval_guard.allowed:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=describe_block(retrieval_guard),
                )
            docs = retrieval_guard.data

            sources, source_payload = _build_sources(docs)
            llm_start = time.perf_counter()
            answer = knowledge_rag.answer(query, source_payload, history_context)
            llm_latency_ms = (time.perf_counter() - llm_start) * 1000
            usage = knowledge_rag.last_usage
        else:
            model_used = agentic_rag.model
            llm_start = time.perf_counter()
            agent_result = agentic_rag.run(query, history_context, session_id=session_id)
            llm_latency_ms = (time.perf_counter() - llm_start) * 1000
            answer = agent_result.get("answer", "")
            usage = {
                "input_tokens":  agent_result.get("input_tokens", 0),
                "output_tokens": agent_result.get("output_tokens", 0),
            }
            docs = []
            for item in agent_result.get("retrieved", []):
                docs.extend(item.get("docs", []))
            sources, source_payload = _build_sources(docs)
            agent_blocked_info = agent_result.get("blocked_info")

        # ── Output Guardrail ────────────────────────────────────────────
        context_texts = [c.get("content", "") for c in source_payload if c.get("content")]
        output_guard = output_guardrail.check_output(answer, context_texts, client)
        audit_logger.log("output_guardrail", output_guard, session_id=session_id)
        answer = output_guard.data if output_guard.allowed else output_guard.reason

        # An AgenticRAG-internal guardrail (retrieval/tool) is the root cause
        # if one fired; otherwise fall back to whatever the output guardrail decided.
        blocked_info = agent_blocked_info or (describe_block(output_guard) if not output_guard.allowed else None)

        elapsed = time.perf_counter() - start
        logger.info(
            "Query answered in %.2fs. intent=%s sources=%d",
            elapsed, intent.workflow.value, len(sources),
        )

        audit_logger.log_final_response(
            session_id=session_id,
            query=query,
            workflow=intent.workflow.value,
            allowed=blocked_info is None,
            stage_results={"input": input_guard, "output": output_guard},
        )

        session_man.append_history(session_id, "assistant", answer)
        metrics.CHAT_HISTORY_SIZE.observe(len(session_man.get_history(session_id)))
        response = ChatResponse(
            session_id=session_id,
            query=query,
            answer=answer,
            sources=sources,
            chunks=source_payload,
            intent=intent.workflow.value,
            intent_reason=intent.reason,
            intent_confidence=intent.confidence,
            llm_latency_ms=llm_latency_ms,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            blocked=blocked_info is not None,
            threat_type=blocked_info.get("threat_type") if blocked_info else None,
            threat_detail=blocked_info.get("threat_detail") if blocked_info else None,
            risk_level=blocked_info.get("risk_level") if blocked_info else None,
            risk_score=blocked_info.get("risk_score") if blocked_info else None,
        )
    except HTTPException as http_exc:
        metrics.CHAT_ERRORS.labels(workflow_type=workflow_type, error_type=f"http_{http_exc.status_code}").inc()
        raise
    except Exception as exc:
        metrics.CHAT_ERRORS.labels(workflow_type=workflow_type, error_type=type(exc).__name__).inc()
        raise

    metrics.record_chat_request(
        workflow_type=workflow_type,
        status="success",
        duration=time.perf_counter() - overall_start,
        tokens_used=usage.get("input_tokens", 0),
        tokens_generated=usage.get("output_tokens", 0),
        model=model_used,
    )
    return response


@traceable_operation(
    name="Chat session history request",
    tags=["api", "chat", "history"],
    metadata={"endpoint": "/chat/history/{session_id}"},
)
@router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Retrieve chat history for a session.",
)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    session = session_man.get_session_info(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    history = session_man.get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        history=[ChatHistoryItem(**item) for item in history],
    )


@traceable_operation(
    name="Document list request",
    tags=["api", "chat", "documents"],
    metadata={"endpoint": "/chat/documents"},
)
@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents.",
)
async def get_document_list() -> DocumentListResponse:
    docs = list_documents()
    return DocumentListResponse(
        count=len(docs),
        documents=[DocumentStatus(**doc) for doc in docs],
    )
