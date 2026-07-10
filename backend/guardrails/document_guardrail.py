"""
document_guardrail.py

Runs at ingestion time — BEFORE an uploaded file is written to disk,
parsed, chunked, or embedded. Independent of the query-side guardrails; a
document passes through this one once, at upload.

Checks (cheapest-first):
    1. Extension / declared content-type
    2. Empty-file / size-limit
    3. Magic-byte signature — confirms the bytes are actually a PDF, not a
       renamed file of another type
    4. Active-content scan — flags PDFs carrying embedded JavaScript or
       auto-launch actions, a common malicious-PDF vector that a
       filename/content-type check alone would miss
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .guardrail_config import GuardrailResult, RiskLevel, config

logger = logging.getLogger(__name__)

STAGE = "document_guardrail"

_PDF_MAGIC = b"%PDF-"

_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream", "binary/octet-stream"}

# Classic malicious-PDF indicators: embedded scripts / auto-actions that
# execute on open. Legitimate contract PDFs never need these.
_ACTIVE_CONTENT_MARKERS = (b"/JavaScript", b"/JS", b"/OpenAction", b"/Launch", b"/AA")


def _check_extension(filename: Optional[str]) -> Optional[GuardrailResult]:
    if not filename:
        return GuardrailResult.blocked(STAGE, "Filename is missing.", RiskLevel.LOW, {"check": "filename"})
    if Path(filename).suffix.lower() not in config.allowed_extensions:
        return GuardrailResult.blocked(
            STAGE, f"'{filename}' is not a supported file type.", RiskLevel.LOW,
            {"check": "extension", "filename": filename},
        )
    return None


def _check_content_type(filename: str, content_type: Optional[str]) -> Optional[GuardrailResult]:
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        return GuardrailResult.blocked(
            STAGE, f"'{filename}' has unsupported content type: {content_type}.", RiskLevel.LOW,
            {"check": "content_type", "content_type": content_type},
        )
    return None


def _check_size(filename: str, data: bytes) -> Optional[GuardrailResult]:
    if not data:
        return GuardrailResult.blocked(STAGE, f"'{filename}' is empty.", RiskLevel.LOW,
                                        {"check": "size", "size_bytes": 0})
    max_bytes = config.max_file_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        return GuardrailResult.blocked(
            STAGE, f"'{filename}' exceeds the {config.max_file_size_mb} MB limit.", RiskLevel.LOW,
            {"check": "size", "size_bytes": len(data)},
        )
    return None


def _check_magic_bytes(filename: str, data: bytes) -> Optional[GuardrailResult]:
    if _PDF_MAGIC not in data[:1024]:
        return GuardrailResult.blocked(
            STAGE, f"'{filename}' does not look like a valid PDF file.", RiskLevel.MEDIUM,
            {"check": "magic_bytes"},
        )
    return None


def _check_active_content(filename: str, data: bytes) -> Optional[GuardrailResult]:
    hits = [marker.decode() for marker in _ACTIVE_CONTENT_MARKERS if marker in data]
    if hits:
        logger.warning("Document guardrail — active-content markers found in '%s': %s", filename, hits)
        return GuardrailResult.blocked(
            STAGE, f"'{filename}' contains embedded active content ({', '.join(hits)}) and was rejected.",
            RiskLevel.HIGH, {"check": "active_content", "markers": hits},
        )
    return None


def check_document(filename: Optional[str], content_type: Optional[str], data: bytes) -> GuardrailResult:
    """
    Validate an uploaded document before it enters the ingestion pipeline.
    On ALLOWED, `.data` is the (unmodified) file bytes.
    """
    if not config.enable_document_guardrail:
        return GuardrailResult.ok(STAGE, data=data, reason="document guardrail disabled")

    for check in (
        lambda: _check_extension(filename),
        lambda: _check_content_type(filename, content_type),
        lambda: _check_size(filename, data),
        lambda: _check_magic_bytes(filename, data),
        lambda: _check_active_content(filename, data),
    ):
        result = check()
        if result is not None:
            return result

    return GuardrailResult.ok(STAGE, data=data, metadata={"filename": filename, "size_bytes": len(data)})
