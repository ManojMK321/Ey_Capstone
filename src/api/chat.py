import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, status
from openai import OpenAI

from src.agents.agentic_rag import AgenticRAG
from src.agents.intent_detection import IntentDetector, Workflow
from src.agents.knowledge_rag import KnowledgeRAG
from src.retrieval.doc_registry import list_documents
# from guardrails import GuardResult, check_input, check_output
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

SESSION_STORE: dict[str, dict[str, Any]] = {}
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
# Session helpers
# ---------------------------------------------------------------------------

def _make_session(session_id: str | None = None, reset: bool = False) -> str:
    if reset or not session_id or session_id not in SESSION_STORE:
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "last_active": datetime.utcnow().isoformat() + "Z",
            "history": [],
            "turn_count": 0,
        }
    return session_id


def _append_history(session_id: str, role: str, text: str) -> None:
    session = SESSION_STORE[session_id]
    session["history"].append({
        "role": role,
        "text": text,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })
    session["history"] = session["history"][-MAX_HISTORY_ITEMS:]
    session["last_active"] = datetime.utcnow().isoformat() + "Z"
    session["turn_count"] = len(session["history"]) // 2


def _history_context(session_id: str) -> str | None:
    history = SESSION_STORE.get(session_id, {}).get("history", [])
    if not history:
        return None
    return "\n".join(
        f"{item['role'].capitalize()}: {item['text']}" for item in history
    )


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------

def _build_sources(docs: list) -> tuple[list[str], list[dict]]:
    sources: list[str] = []
    source_payload: list[dict] = []
    for doc in docs:
        meta = doc.metadata or {}
        name = meta.get("filename") or meta.get("source") or "document"
        page = meta.get("page")
        if page is not None:
            name = f"{name} (page {page})"
        sources.append(name)
        source_payload.append({"source": name, "content": doc.page_content.strip()})
    return sources, source_payload


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a chat query — auto-routed to KnowledgeRAG or AgenticRAG.",
)
async def chat_documents(request: ChatRequest) -> ChatResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query text cannot be empty.",
        )

    # # --- Input guardrail -------------------------------------------------------
    # input_guard = check_input(query, _get_client())
    # if not input_guard.allowed:
    #     logger.warning(
    #         "Input guard blocked query. reason=%s detail=%s",
    #         input_guard.reason, input_guard.detail,
    #     )
    #     raise HTTPException(
    #         status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    #         detail=input_guard.detail,
    #     )
    # # --------------------------------------------------------------------------

    session_id = _make_session(request.session_id, request.reset_session)
    history_context = _history_context(session_id)
    _append_history(session_id, "user", query)

    vector_store = get_vector_store()
    if vector_store.vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No indexed documents are available. Upload a PDF first via POST /upload/.",
        )

    intent_detector, knowledge_rag, agentic_rag = _get_agents()

    start = time.perf_counter()
    intent = intent_detector.detect(query)
    logger.info(
        "Intent detected: %s (%.2f) — %s",
        intent.workflow.value, intent.confidence, intent.reason,
    )

    if intent.workflow == Workflow.KNOWLEDGE_RAG:
        docs = knowledge_rag.retrieve(query)
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant document chunks found for the query.",
            )
        sources, source_payload = _build_sources(docs)
        answer = knowledge_rag.answer(query, source_payload, history_context)
        context_chunks = [item["content"] for item in source_payload]
    else:
        agent_result = agentic_rag.run(query, history_context)
        answer = agent_result.get("answer", "")
        docs = []
        for item in agent_result.get("retrieved", []):
            docs.extend(item.get("docs", []))
        sources, _ = _build_sources(docs)
        context_chunks = [doc.page_content for doc in docs]

    # --- Output guardrail ------------------------------------------------------
    # output_guard = check_output(answer, context_chunks, _get_client())
    # if not output_guard.allowed:
    #     logger.warning(
    #         "Output guard blocked answer. reason=%s detail=%s",
    #         output_guard.reason, output_guard.detail,
    #     )
    #     answer = output_guard.detail
    # --------------------------------------------------------------------------

    elapsed = time.perf_counter() - start
    logger.info(
        "Query answered in %.2fs. intent=%s sources=%d",
        elapsed, intent.workflow.value, len(sources),
    )

    _append_history(session_id, "assistant", answer)
    return ChatResponse(
        session_id=session_id,
        query=query,
        answer=answer,
        sources=sources,
        intent=intent.workflow.value,
        intent_reason=intent.reason,
    )


@router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Retrieve chat history for a session.",
)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    session = SESSION_STORE.get(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return SessionHistoryResponse(
        session_id=session_id,
        history=[ChatHistoryItem(**item) for item in session["history"]],
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
