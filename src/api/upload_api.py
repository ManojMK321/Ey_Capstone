import asyncio
import json
import uuid
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, HTTPException, status, Request
from fastapi.responses import StreamingResponse

from src.schema.validation import UploadedFile, UploadResponse
from src.retrieval.blob_storage import upload_blob, list_blobs, delete_blob, blob_exists
from src.retrieval.doc_registry import register_document
from docs.parser import PDFParser
from docs.chunking import DocumentChunker
from docs.vector_store import FAISSVectorStore
from docs.pipeline import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEMP_DIR              = Path("temp_uploads")
ALLOWED_EXTENSIONS    = {".pdf"}
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream", "binary/octet-stream"}
MAX_FILE_SIZE_MB      = 50
MAX_FILE_SIZE         = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_FILES             = 10

TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Shared pipeline singleton lives in docs/pipeline.py

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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
    form  = await request.form()
    files = form.getlist("files")

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Max {MAX_FILES} files per request.")

    uploaded: List[UploadedFile] = []
    errors:   List[dict]         = []

    parser, chunker, vector_store = get_pipeline()

    for file in files:
        temp_path: Path | None = None
        try:
            # ── 1. Validate ───────────────────────────────────────────
            err = validate_file(file)
            if err:
                errors.append({"filename": file.filename, "reason": err})
                continue

            # ── 2. Read into memory & save to local temp ──────────────
            data, size = await read_upload(file)
            file_id   = str(uuid.uuid4())
            stem      = Path(file.filename).stem
            temp_path = TEMP_DIR / f"{stem}_{file_id}.pdf"
            temp_path.write_bytes(data)
            logger.info("Saved locally: %s (%d bytes)", temp_path, size)

            # ── 3. Parse PDF → pages ──────────────────────────────────
            pages = parser.parse(pdf_path=temp_path)
            page_count = len(pages)
            logger.info("Parsed %d page(s) from '%s'.", page_count, file.filename)

            # ── 4. Chunk pages ────────────────────────────────────────
            chunks = chunker.chunk(pages=pages, doc_id=file_id, filename=file.filename)
            chunk_count = len(chunks)
            logger.info("Generated %d chunk(s) from '%s'.", chunk_count, file.filename)

            # ── 5. Embed & store in FAISS ─────────────────────────────
            vector_store.add_documents(chunks)
            logger.info("Stored %d chunks in FAISS for doc_id '%s'.", len(chunks), file_id)

            # ── 6. Upload to Azure Blob Storage ───────────────────────
            # blob_name = f"{stem}_{file_id}.pdf"
            # blob_url  = upload_blob(blob_name, data)
            # logger.info("Uploaded to blob: %s", blob_url)
            blob_url = ""

            # ── 7. Delete local temp file ─────────────────────────────
            # temp_path.unlink()
            # logger.info("Deleted local temp file: %s", temp_path)
            # temp_path = None

            uploaded.append(UploadedFile(
                file_id       = file_id,
                original_name = file.filename,
                blob_url      = blob_url,
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

        except ValueError as ve:
            logger.warning(str(ve))
            errors.append({"filename": file.filename, "reason": str(ve)})

        except Exception as ex:
            logger.error("Unexpected error for '%s': %s", file.filename, ex, exc_info=True)
            errors.append({"filename": file.filename, "reason": "An unexpected error occurred."})

        # finally:
        #     if temp_path and temp_path.exists():
        #         temp_path.unlink(missing_ok=True)

    logger.info("Upload complete — %d succeeded, %d failed.", len(uploaded), len(errors))

    return UploadResponse(
        message        = f"{len(uploaded)} file(s) uploaded successfully.",
        uploaded_count = len(uploaded),
        failed_count   = len(errors),
        files          = uploaded,
        errors         = errors,
    )


@router.post("/stream", summary="Upload one PDF and stream pipeline progress via SSE")
async def upload_stream(request: Request):
    """
    Streams Server-Sent Events (text/event-stream) so the UI can animate
    each pipeline step as it completes.  Handles exactly one file per request.
    """
    form  = await request.form()
    files = form.getlist("files")

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")

    file = files[0]

    async def generate():
        temp_path: Path | None = None
        try:
            # ── validate ──────────────────────────────────────────────
            err = validate_file(file)
            if err:
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
            pages      = parser.parse(pdf_path=temp_path)
            page_count = len(pages)
            logger.info("Parsed %d page(s) from '%s'.", page_count, file.filename)
            yield f"data: {json.dumps({'step': 0, 'status': 'done', 'pages': page_count})}\n\n"
            await asyncio.sleep(0)

            # ── step 1 – clean (runs inside chunker) ──────────────────
            yield f"data: {json.dumps({'step': 1, 'status': 'running'})}\n\n"
            await asyncio.sleep(0)
            chunks      = chunker.chunk(pages=pages, doc_id=file_id, filename=file.filename)
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
            vector_store.add_documents(chunks)
            logger.info("Stored %d chunks in FAISS for doc_id '%s'.", chunk_count, file_id)
            yield f"data: {json.dumps({'step': 3, 'status': 'done'})}\n\n"
            await asyncio.sleep(0)

            # ── steps 4 & 5 – index (done inside add_documents) ──────
            yield f"data: {json.dumps({'step': 4, 'status': 'done'})}\n\n"
            await asyncio.sleep(0)
            yield f"data: {json.dumps({'step': 5, 'status': 'done'})}\n\n"
            await asyncio.sleep(0)

            result = {
                "file_id":       file_id,
                "original_name": file.filename,
                "blob_url":      "",
                "size_bytes":    size,
                "page_count":    page_count,
                "chunk_count":   chunk_count,
            }
            register_document(
                file_id=file_id,
                original_name=file.filename,
                page_count=page_count,
                chunk_count=chunk_count,
            )
            logger.info("Stream upload complete: %s (%d chunks)", file.filename, chunk_count)
            yield f"data: {json.dumps({'done': True, 'file': result})}\n\n"

        except ValueError as ve:
            logger.warning(str(ve))
            yield f"data: {json.dumps({'error': str(ve)})}\n\n"

        except Exception as ex:
            logger.error("Stream upload error: %s", ex, exc_info=True)
            yield f"data: {json.dumps({'error': 'An unexpected error occurred.'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
