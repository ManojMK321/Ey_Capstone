from typing import List, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class UploadedFile(BaseModel):
    file_id: str
    original_name: str
    size_bytes: int
    page_count: int = 0
    chunk_count: int = 0


class UploadResponse(BaseModel):
    session_id: str
    message: str
    uploaded_count: int
    failed_count: int
    files: List[UploadedFile]
    errors: List[dict]


# ---------------------------------------------------------------------------
# Chat — request / response
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    reset_session: bool = False


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources: List[str]
    chunks: List[dict] = []   # [{"source": str, "content": str}] — used by RAGAS eval
    intent: str = ""
    intent_reason: str = ""
    intent_confidence: float = 0.0
    llm_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Chat — session history
# ---------------------------------------------------------------------------

class ChatHistoryItem(BaseModel):
    role: str
    text: str
    timestamp: str


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    last_active: str
    turn_count: int


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatHistoryItem]


# ---------------------------------------------------------------------------
# Documents list
# ---------------------------------------------------------------------------

class DocumentStatus(BaseModel):
    file_id: str
    original_name: str
    status: str = "indexed"
    page_count: int = 0
    chunk_count: int = 0


class DocumentListResponse(BaseModel):
    count: int
    documents: List[DocumentStatus]
