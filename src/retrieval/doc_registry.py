"""
In-memory document registry.

upload_api calls register_document() after each successful ingest.
The chat /documents endpoint calls list_documents() to read the registry.
"""
from __future__ import annotations
import threading
from typing import Any

_lock: threading.Lock = threading.Lock()
_registry: list[dict[str, Any]] = []


def register_document(
    file_id: str,
    original_name: str,
    page_count: int = 0,
    chunk_count: int = 0,
    status: str = "indexed",
) -> None:
    entry = {
        "file_id": file_id,
        "original_name": original_name,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "status": status,
    }
    with _lock:
        # avoid duplicates by file_id
        _registry[:] = [d for d in _registry if d["file_id"] != file_id]
        _registry.append(entry)


def list_documents() -> list[dict[str, Any]]:
    with _lock:
        return list(_registry)


def clear_registry() -> None:
    with _lock:
        _registry.clear()
