"""
guardrails/input_guard.py

Input guardrail — runs BEFORE the query enters the RAG pipeline.

Three checks executed in cheapest-first order:
  1. Length gate    — O(1), rejects empty or oversized queries immediately
  2. Injection scan — O(n) regex, blocks prompt-injection / jailbreak attempts
  3. Scope filter   — LLM call (gpt-4o-mini), confirms question is contract-related
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result type — used by BOTH input_guard and output_guard
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    """
    Returned by every guardrail check.

    allowed : bool   — True means the content passed; False means it was blocked.
    reason  : str    — machine-readable code for metrics/logging.
                       One of: "ok" | "length" | "injection" | "off_scope"
                                "pii" | "ungrounded"
    detail  : str    — human-readable sentence shown in the API response.
    """
    allowed: bool
    reason: str
    detail: str = ""

    @classmethod
    def ok(cls) -> "GuardResult":
        return cls(allowed=True, reason="ok", detail="")


# ---------------------------------------------------------------------------
# Check 1 — Length gate
# ---------------------------------------------------------------------------

MIN_QUERY_LEN: int = 3        # chars — catches empty / single-char noise
MAX_QUERY_LEN: int = 2_000    # chars — prevents token-stuffing / resource abuse


def _check_length(query: str) -> Optional[GuardResult]:
    if len(query) < MIN_QUERY_LEN:
        return GuardResult(
            allowed=False,
            reason="length",
            detail="Query is too short. Please ask a complete question.",
        )
    if len(query) > MAX_QUERY_LEN:
        return GuardResult(
            allowed=False,
            reason="length",
            detail=f"Query exceeds the {MAX_QUERY_LEN}-character limit. Please shorten your question.",
        )
    return None


# ---------------------------------------------------------------------------
# Check 2 — Injection scan
# ---------------------------------------------------------------------------

# Compiled once at import time for performance.
# Patterns are intentionally narrow to avoid false positives on contract language
# (e.g. "act as the agent", "ignore the payment instructions in section 3").
_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Classic instruction override.
    # Requires "your" possessive so we match "ignore your previous instructions"
    # (targeting the AI) but not "ignore the prior instructions in Exhibit A"
    # (referencing a contract section).
    (re.compile(r"ignore\s+(?:all\s+)?your\s+(?:previous|above|prior)\s+instructions?", re.I),
     "instruction override"),

    # Identity hijack — "you are now a/an/the <role>".
    # The article requirement avoids matching "you are now required to pay...".
    (re.compile(r"\byou\s+are\s+now\s+(?:a|an|the)\b", re.I),
     "identity hijack"),

    # "act as" only when paired with a bypass adjective.
    # Avoids matching contract language like "shall act as the sole distributor".
    (re.compile(
        r"\bact\s+as\s+(?:a\s+|an\s+)?(?:unrestricted|uncensored|jailbroken|different|alternative|evil)\b",
        re.I,
    ), "restricted role bypass"),

    # "pretend to be" — specific enough not to appear in contract language.
    (re.compile(r"\bpretend\s+(?:you\s+are|to\s+be)\b", re.I),
     "identity pretend"),

    # System-prompt leakage — only when paired with an extraction verb.
    # "system prompt" alone can appear in legitimate tech/SaaS contract questions.
    (re.compile(
        r"\b(?:reveal|show|print|output|repeat|leak|expose|tell\s+me)\s+(?:your\s+|the\s+)?(?:system\s+prompt|system\s+instructions)\b",
        re.I,
    ), "prompt leakage"),

    # XML/HTML tag injection targeting the system role
    (re.compile(r"<\s*/?\s*system\s*>", re.I),
     "system tag injection"),

    # Template injection (Jinja / Handlebars style): {{ ... }}
    (re.compile(r"\{\{.*?\}\}", re.S),
     "template injection"),

    # Special model tokens: <|endoftext|>, <|im_start|>, etc.
    (re.compile(r"<\|.*?\|>"),
     "special token injection"),

    # Named jailbreak modes circulated publicly
    (re.compile(r"\b(?:jailbreak|DAN\s+mode|do\s+anything\s+now)\b", re.I),
     "jailbreak keyword"),

    # Direct safety bypass vocabulary
    (re.compile(r"\b(?:override|bypass|disable)\s+(?:safety|filter|guardrail|restriction)\b", re.I),
     "safety bypass"),
]


def _check_injection(query: str) -> Optional[GuardResult]:
    for pattern, label in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning("Input guard — injection pattern matched: '%s'", label)
            return GuardResult(
                allowed=False,
                reason="injection",
                detail="The query contains patterns that are not permitted in this system.",
            )
    return None


# ---------------------------------------------------------------------------
# Check 3 — Scope filter  (gpt-4o-mini — fast, ~$0.0001 / call)
# ---------------------------------------------------------------------------

_SCOPE_SYSTEM_PROMPT = (
    "You are a strict scope classifier for a Contract Intelligence System. "
    "Decide if the user's question is about contracts, legal documents, business agreements, "
    "their clauses, terms, obligations, parties, dates, or related legal concepts. "
    "Respond ONLY with valid JSON: "
    '{"in_scope": true_or_false, "reason": "<one short sentence>"}'
)


def _check_scope(query: str, client: OpenAI) -> Optional[GuardResult]:
    """
    LLM-based scope check. Failure (exception) is non-fatal — the query is
    allowed through rather than blocking users due to an API hiccup.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=80,
            messages=[
                {"role": "system", "content": _SCOPE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data: dict = json.loads(raw)
        if not data.get("in_scope", True):
            reason = data.get("reason", "Question is outside the contract domain.")
            logger.info("Input guard — scope filter blocked query. reason=%s", reason)
            return GuardResult(
                allowed=False,
                reason="off_scope",
                detail=(
                    "This system only answers questions about contracts and legal documents. "
                    f"{reason}"
                ),
            )
    except Exception as exc:
        logger.warning("Input guard — scope check failed (%s); query allowed through.", exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_input(query: str, client: OpenAI) -> GuardResult:
    """
    Run all input guardrail checks in cheapest-first order.

    Returns the first GuardResult with allowed=False, or GuardResult.ok()
    if every check passes.

    Args:
        query   : raw query string from the user request.
        client  : initialised OpenAI client (reused from the caller — no extra init cost).

    Returns:
        GuardResult  — caller should raise HTTP 422 if .allowed is False.
    """
    # Cheap checks first — avoid unnecessary LLM calls
    length_result = _check_length(query)
    if length_result is not None:
        return length_result

    injection_result = _check_injection(query)
    if injection_result is not None:
        return injection_result

    # Expensive check last
    scope_result = _check_scope(query, client)
    if scope_result is not None:
        return scope_result

    logger.debug("Input guard — all checks passed for query (len=%d)", len(query))
    return GuardResult.ok()
