"""
retrieval_guardrail.py

Runs AFTER Document Retrieval and BEFORE the retrieved passages are
handed to an LLM (KnowledgeRAG's answer call, or AgenticRAG's specialist
agents). Works on both LangChain Document objects and the plain
{"source": ..., "content": ...} dict shape used across this codebase.

Checks:
    1. Emptiness  — no context retrieved at all
    2. Volume cap — truncate to `max_retrieved_chunks` so a runaway
       retrieval doesn't blow the LLM's context window / cost
    3. Sensitive-data scan — flags PII patterns surfacing in retrieved
       content. Never blocks on its own (the contract's own data
       legitimately contains this), but raises the risk level so it's
       visible in the audit trail.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .guardrail_config import GuardrailResult, RiskLevel, config

logger = logging.getLogger(__name__)

STAGE = "retrieval_guardrail"

_PII_PATTERNS: dict[str, re.Pattern] = {
    "SSN":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "IBAN":        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,}([A-Z0-9]?\d?){0,16}\b"),
}


def _content_of(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content") or "")
    return str(getattr(item, "page_content", "") or "")


def _scan_pii(items: list) -> list[str]:
    found: set[str] = set()
    for item in items:
        text = _content_of(item)
        for label, pattern in _PII_PATTERNS.items():
            if pattern.search(text):
                found.add(label)
    return sorted(found)


def check_retrieval(query: str, docs: list) -> GuardrailResult:
    """
    Validate retrieved chunks before they enter the LLM context.
    `.data` is the (possibly truncated) list the caller should use downstream.
    """
    if not config.enable_retrieval_guardrail:
        return GuardrailResult.ok(STAGE, data=docs, reason="retrieval guardrail disabled")

    if not docs:
        return GuardrailResult.blocked(
            STAGE, "No relevant document chunks were retrieved for this query.",
            RiskLevel.LOW, {"check": "empty", "query_len": len(query or "")},
        )

    truncated = False
    if len(docs) > config.max_retrieved_chunks:
        docs = docs[: config.max_retrieved_chunks]
        truncated = True

    pii_hits = _scan_pii(docs)

    if truncated or pii_hits:
        reason_parts = []
        if truncated:
            reason_parts.append(f"truncated to {config.max_retrieved_chunks} chunks")
        if pii_hits:
            reason_parts.append(f"sensitive data detected in source content: {', '.join(pii_hits)}")
        return GuardrailResult.modified(
            STAGE, data=docs, reason="; ".join(reason_parts),
            risk_level=RiskLevel.MEDIUM if pii_hits else RiskLevel.LOW,
            metadata={"check": "retrieval", "truncated": truncated, "pii_labels": pii_hits, "doc_count": len(docs)},
        )

    return GuardrailResult.ok(STAGE, data=docs, metadata={"doc_count": len(docs)})
