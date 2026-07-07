import asyncio
import json
import tempfile
import time
import uuid
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, HTTPException, status, Request
from fastapi.responses import StreamingResponse

from src.schema.validation import UploadedFile, UploadResponse
from src.retrieval.blob_storage import upload_blob, list_blobs, delete_blob, blob_exists
from src.retrieval.doc_registry import register_document
from src.orchestrator.session_store_postgres import session_store
from src.observability.langsmith import traceable_operation
from src.observability import metrics
from docs.pipeline import get_pipeline, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Outside the project tree on purpose: uvicorn's --reload file watcher covers
# the whole project directory, and every create/delete here was slipping past
# reload_excludes and restarting the worker mid-upload — killing whatever
# request was in flight during the slowest step (embedding + indexing).
TEMP_DIR              = Path(tempfile.gettempdir()) / "contract_intelligence_uploads"
ALLOWED_EXTENSIONS    = {".pdf"}
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream", "binary/octet-stream"}
MAX_FILE_SIZE_MB      = 50
MAX_FILE_SIZE         = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_FILES             = 10

TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_file(file: UploadFile) -> str | None:
    if not file.filename:
        return "Filename is missing."
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return f"'{file.filename}' is not a PDF."
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        return f"'{file.filename}' has unsupported content type: {file.content_type}."
    return None


async def read_upload(file: UploadFile) -> tuple[bytes, int]:
    data = b""
    while chunk := await file.read(1024 * 1024):
        data += chunk
        if len(data) > MAX_FILE_SIZE:
            raise ValueError(f"'{file.filename}' exceeds the {MAX_FILE_SIZE_MB} MB limit.")
    return data, len(data)


def normalize_session_id(session_id: str) -> str | None:
    """Strip whitespace and validate the session_id is a well-formed UUID."""
    if not session_id:
        return None
    cleaned = session_id.strip()
    try:
        uuid.UUID(cleaned)
        return cleaned
    except ValueError:
        return None


def _resolve_session(raw_id: str | None) -> str:
    """
    Resolve or create a session from a raw session ID string.
    Calls make_session so callers always get a valid session_id back.
    """
    normalized = normalize_session_id(raw_id or "")
    return session_store.make_session(normalized, reset=False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@traceable_operation(
    name="Upload documents",
    tags=["api", "upload"],
    metadata={"endpoint": "/upload/"},
)
@router.post(
    "/",
    response_model=UploadResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "One or more PDF files",
                            }
                        },
                    }
                }
            },
            "required": True,
        }
    },
)
async def upload_documents(request: Request) -> UploadResponse:
    form       = await request.form()
    files      = form.getlist("files")
    # Accept session_id from form data or X-Session-ID header
    raw_session = str(form.get("session_id") or "") or request.headers.get("X-Session-ID", "")
    session_id  = _resolve_session(raw_session)

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Max {MAX_FILES} files per request.")

    uploaded: List[UploadedFile] = []
    errors:   List[dict]         = []

    parser, chunker, vector_store = get_pipeline()

    for file in files:
        temp_path: Path | None = None
        file_type = (Path(file.filename).suffix.lstrip(".").lower() or "unknown") if file.filename else "unknown"
        upload_start = time.perf_counter()
        try:
            # ── 1. Validate ───────────────────────────────────────────
            err = validate_file(file)
            if err:
                errors.append({"filename": file.filename, "reason": err})
                metrics.UPLOAD_ERRORS.labels(file_type=file_type, error_type="validation").inc()
                continue

            # ── 2. Read into memory & save to local temp ──────────────
            data, size = await read_upload(file)
            file_id   = str(uuid.uuid4())
            stem      = Path(file.filename).stem
            temp_path = TEMP_DIR / f"{stem}_{file_id}.pdf"
            temp_path.write_bytes(data)
            logger.info("Saved locally: %s (%d bytes)", temp_path, size)

            # ── 3. Parse PDF → pages ──────────────────────────────────
            stage_start = time.perf_counter()
            pages = parser.parse(pdf_path=temp_path)
            metrics.DOCUMENT_PROCESSING_DURATION_SECONDS.labels(file_type=file_type, stage="parse").observe(time.perf_counter() - stage_start)
            page_count = len(pages)
            logger.info("Parsed %d page(s) from '%s'.", page_count, file.filename)

            # ── 4. Chunk pages ────────────────────────────────────────
            stage_start = time.perf_counter()
            chunks = chunker.chunk(pages=pages, doc_id=file_id, filename=file.filename)
            metrics.DOCUMENT_PROCESSING_DURATION_SECONDS.labels(file_type=file_type, stage="chunk").observe(time.perf_counter() - stage_start)
            chunk_count = len(chunks)
            logger.info("Generated %d chunk(s) from '%s'.", chunk_count, file.filename)

            # ── 5. Embed & store in Pinecone ──────────────────────────
            stage_start = time.perf_counter()
            vector_store.add_documents(chunks)
            metrics.DOCUMENT_PROCESSING_DURATION_SECONDS.labels(file_type=file_type, stage="embed_store").observe(time.perf_counter() - stage_start)
            logger.info("Stored %d chunks in Pinecone for doc_id '%s'.", chunk_count, file_id)

            # ── 6. Upload to Azure Blob Storage (optional) ────────────
            # blob_name = f"{stem}_{file_id}.pdf"
            # blob_url  = upload_blob(blob_name, data)
            # logger.info("Uploaded to blob: %s", blob_url)

            # ── 7. Delete local temp file ─────────────────────────────
            temp_path.unlink(missing_ok=True)
            logger.info("Deleted local temp file: %s", temp_path)
            temp_path = None

            # ── 8. Register in session ────────────────────────────────
            session_store.add_document(
                session_id=session_id,
                file_id=file_id,
                original_name=file.filename,
                size_bytes=size,
            )

            uploaded.append(UploadedFile(
                file_id       = file_id,
                original_name = file.filename,
                size_bytes    = size,
                page_count    = page_count,
                chunk_count   = chunk_count,
            ))
            register_document(
                file_id=file_id,
                original_name=file.filename,
                page_count=page_count,
                chunk_count=chunk_count,
            )
            metrics.record_document_upload(
                file_type=file_type,
                total_duration=time.perf_counter() - upload_start,
                num_chunks=chunk_count,
            )

        except ValueError as ve:
            logger.warning(str(ve))
            errors.append({"filename": file.filename, "reason": str(ve)})
            metrics.UPLOAD_ERRORS.labels(file_type=file_type, error_type="value_error").inc()

        except Exception as ex:
            logger.error("Unexpected error for '%s': %s", file.filename, ex, exc_info=True)
            errors.append({"filename": file.filename, "reason": "An unexpected error occurred."})
            metrics.UPLOAD_ERRORS.labels(file_type=file_type, error_type=type(ex).__name__).inc()

        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    logger.info("Upload complete — %d succeeded, %d failed.", len(uploaded), len(errors))

    return UploadResponse(
        session_id     = session_id,
        message        = f"{len(uploaded)} file(s) uploaded successfully.",
        uploaded_count = len(uploaded),
        failed_count   = len(errors),
        files          = uploaded,
        errors         = errors,
    )


@traceable_operation(
    name="Upload stream",
    tags=["api", "upload", "stream"],
    metadata={"endpoint": "/upload/stream"},
)
@router.post("/stream", summary="Upload one PDF and stream pipeline progress via SSE")
async def upload_stream(request: Request):
    """
    Streams Server-Sent Events (text/event-stream) so the UI can animate
    each pipeline step as it completes. Handles exactly one file per request.
    """
    form       = await request.form()
    files      = form.getlist("files")
    raw_session = str(form.get("session_id") or "") or request.headers.get("X-Session-ID", "")
    session_id  = _resolve_session(raw_session)

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")

    file = files[0]

    async def generate():
        temp_path: Path | None = None
        file_type = (Path(file.filename).suffix.lstrip(".").lower() or "unknown") if file.filename else "unknown"
        upload_start = time.perf_counter()
        try:
            # ── validate ──────────────────────────────────────────────
            err = validate_file(file)
            if err:
                metrics.UPLOAD_ERRORS.labels(file_type=file_type, error_type="validation").inc()
                yield f"data: {json.dumps({'error': err})}\n\n"
                return

            # ── read & save ───────────────────────────────────────────
            data, size = await read_upload(file)
            file_id    = str(uuid.uuid4())
            stem       = Path(file.filename).stem
            temp_path  = TEMP_DIR / f"{stem}_{file_id}.pdf"
            temp_path.write_bytes(data)

            parser, chunker, vector_store = get_pipeline()

            # ── step 0 – parse ────────────────────────────────────────
            yield f"data: {json.dumps({'step': 0, 'status': 'running'})}\n\n"
            await asyncio.sleep(0)
            stage_start = time.perf_counter()
            pages      = parser.parse(pdf_path=temp_path)
            metrics.DOCUMENT_PROCESSING_DURATION_SECONDS.labels(file_type=file_type, stage="parse").observe(time.perf_counter() - stage_start)
            page_count = len(pages)
            logger.info("Parsed %d page(s) from '%s'.", page_count, file.filename)
            yield f"data: {json.dumps({'step': 0, 'status': 'done', 'pages': page_count})}\n\n"
            await asyncio.sleep(0)

            # ── step 1 – clean (runs inside chunker) ──────────────────
            yield f"data: {json.dumps({'step': 1, 'status': 'running'})}\n\n"
            await asyncio.sleep(0)
            stage_start = time.perf_counter()
            chunks      = chunker.chunk(pages=pages, doc_id=file_id, filename=file.filename)
            metrics.DOCUMENT_PROCESSING_DURATION_SECONDS.labels(file_type=file_type, stage="chunk").observe(time.perf_counter() - stage_start)
            chunk_count = len(chunks)
            logger.info("Generated %d chunk(s) from '%s'.", chunk_count, file.filename)
            yield f"data: {json.dumps({'step': 1, 'status': 'done'})}\n\n"
            await asyncio.sleep(0)

            # ── step 2 – chunk (done in same call) ───────────────────
            yield f"data: {json.dumps({'step': 2, 'status': 'done', 'chunks': chunk_count})}\n\n"
            await asyncio.sleep(0)

            # ── step 3 – embed (calls OpenAI — slow) ─────────────────
            yield f"data: {json.dumps({'step': 3, 'status': 'running'})}\n\n"
            await asyncio.sleep(0)
            stage_start = time.perf_counter()
            vector_store.add_documents(chunks)
            metrics.DOCUMENT_PROCESSING_DURATION_SECONDS.labels(file_type=file_type, stage="embed_store").observe(time.perf_counter() - stage_start)
            logger.info("Stored %d chunks in Pinecone vector database for doc_id '%s'.", chunk_count, file_id)
            yield f"data: {json.dumps({'step': 3, 'status': 'done'})}\n\n"
            await asyncio.sleep(0)

            # ── steps 4 & 5 – index (done inside add_documents) ──────
            yield f"data: {json.dumps({'step': 4, 'status': 'done'})}\n\n"
            await asyncio.sleep(0)
            yield f"data: {json.dumps({'step': 5, 'status': 'done'})}\n\n"
            await asyncio.sleep(0)

            session_store.add_document(
                session_id=session_id,
                file_id=file_id,
                original_name=file.filename,
                size_bytes=size,
            )
            register_document(
                file_id=file_id,
                original_name=file.filename,
                page_count=page_count,
                chunk_count=chunk_count,
            )

            metrics.record_document_upload(
                file_type=file_type,
                total_duration=time.perf_counter() - upload_start,
                num_chunks=chunk_count,
            )

            result = {
                "file_id":       file_id,
                "original_name": file.filename,
                "size_bytes":    size,
                "page_count":    page_count,
                "chunk_count":   chunk_count,
            }
            logger.info("Stream upload complete: %s (%d chunks)", file.filename, chunk_count)
            yield f"data: {json.dumps({'done': True, 'file': result})}\n\n"

        except ValueError as ve:
            logger.warning(str(ve))
            metrics.UPLOAD_ERRORS.labels(file_type=file_type, error_type="value_error").inc()
            yield f"data: {json.dumps({'error': str(ve)})}\n\n"

        except Exception as ex:
            logger.error("Stream upload error: %s", ex, exc_info=True)
            metrics.UPLOAD_ERRORS.labels(file_type=file_type, error_type=type(ex).__name__).inc()
            yield f"data: {json.dumps({'error': 'An unexpected error occurred.'})}\n\n"

        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/list/{session_id}", summary="List documents uploaded to a session")
async def list_session_documents(session_id: str):
    normalized = normalize_session_id(session_id)
    if not normalized or not session_store.session_exists(normalized):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    documents = session_store.list_documents(normalized)
    return {"session_id": normalized, "count": len(documents), "files": documents}


@router.delete("/{session_id}/{file_id}", summary="Delete a document from a session")
async def delete_session_document(session_id: str, file_id: str):
    normalized = normalize_session_id(session_id)
    if not normalized or not session_store.session_exists(normalized):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    deleted = session_store.delete_document(normalized, file_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    try:
        vs = get_vector_store()
        vs.delete_document(file_id)
    except Exception:
        logger.warning(
            "Unable to remove doc '%s' from Pinecone for session '%s'.",
            file_id, normalized,
        )

    return {"message": f"'{file_id}' removed from session '{normalized}'."}


@router.get("/list", summary="List all PDFs in Azure Blob Storage")
async def list_blobs_in_container():
    blobs = list_blobs()
    return {"count": len(blobs), "files": blobs}


@router.delete("/{blob_name}", summary="Delete a PDF from Azure Blob Storage")
async def delete_blob_file(blob_name: str):
    if not blob_exists(blob_name):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Blob '{blob_name}' not found.")
    delete_blob(blob_name)
    return {"message": f"'{blob_name}' deleted from blob storage."}
