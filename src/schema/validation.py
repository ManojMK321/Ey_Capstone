from typing import List, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class UploadedFile(BaseModel):
    file_id: str
    original_name: str
    blob_url: str
    size_bytes: int
    page_count: int = 0
    chunk_count: int = 0


class UploadResponse(BaseModel):
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
    query: str
    answer: str
    sources: List[str]
    session_id: str = ""
    intent: str = ""
    intent_reason: str = ""


# ---------------------------------------------------------------------------
# Chat — session history
# ---------------------------------------------------------------------------

class ChatHistoryItem(BaseModel):
    role: str
    text: str
    timestamp: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatHistoryItem]


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    last_active: str
    turn_count: int


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
