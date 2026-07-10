"""
input_guardrail.py

Runs BEFORE a user query enters the Query Router / RAG pipeline. Checks
execute cheapest-first so an obviously-bad query never pays for an LLM
call:

    1. Length gate    — O(1)
    2. Injection scan — O(n) regex
    3. Scope filter   — LLM call (gpt-4o-mini), confirms the question is
                         contract-related
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from .guardrail_config import GuardrailResult, RiskLevel, config

logger = logging.getLogger(__name__)

STAGE = "input_guardrail"

# Patterns are intentionally narrow to avoid false positives on contract
# language (e.g. "act as the agent", "ignore the payment instructions in
# section 3").
_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(?:all\s+)?your\s+(?:previous|above|prior)\s+instructions?", re.I),
     "instruction override"),
    (re.compile(r"\byou\s+are\s+now\s+(?:a|an|the)\b", re.I),
     "identity hijack"),
    (re.compile(
        r"\bact\s+as\s+(?:a\s+|an\s+)?(?:unrestricted|uncensored|jailbroken|different|alternative|evil)\b",
        re.I,
    ), "restricted role bypass"),
    (re.compile(r"\bpretend\s+(?:you\s+are|to\s+be)\b", re.I),
     "identity pretend"),
    (re.compile(
        r"\b(?:reveal|show|print|output|repeat|leak|expose|tell\s+me)\s+(?:your\s+|the\s+)?(?:system\s+prompt|system\s+instructions)\b",
        re.I,
    ), "prompt leakage"),
    (re.compile(r"<\s*/?\s*system\s*>", re.I),
     "system tag injection"),
    (re.compile(r"\{\{.*?\}\}", re.S),
     "template injection"),
    (re.compile(r"<\|.*?\|>"),
     "special token injection"),
    (re.compile(r"\b(?:jailbreak|DAN\s+mode|do\s+anything\s+now)\b", re.I),
     "jailbreak keyword"),
    (re.compile(r"\b(?:override|bypass|disable)\s+(?:safety|filter|restriction)\b", re.I),
     "safety bypass"),
]

_SCOPE_SYSTEM_PROMPT = (
    "You are a strict scope classifier for a Contract Intelligence System. "
    "Decide if the user's question is about contracts, legal documents, business agreements, "
    "their clauses, terms, obligations, parties, dates, or related legal concepts. "
    "Respond ONLY with valid JSON: "
    '{"in_scope": true_or_false, "reason": "<one short sentence>"}'
)


def _check_length(query: str) -> Optional[GuardrailResult]:
    if len(query) < config.min_query_len:
        return GuardrailResult.blocked(
            STAGE, "Query is too short. Please ask a complete question.",
            RiskLevel.LOW, {"check": "length", "len": len(query)},
        )
    if len(query) > config.max_query_len:
        return GuardrailResult.blocked(
            STAGE, f"Query exceeds the {config.max_query_len}-character limit. Please shorten your question.",
            RiskLevel.LOW, {"check": "length", "len": len(query)},
        )
    return None


def _check_injection(query: str) -> Optional[GuardrailResult]:
    for pattern, label in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning("Input guardrail — injection pattern matched: '%s'", label)
            return GuardrailResult.blocked(
                STAGE, "The query contains patterns that are not permitted in this system.",
                RiskLevel.HIGH, {"check": "injection", "pattern": label},
            )
    return None


def _check_scope(query: str, client: OpenAI) -> Optional[GuardrailResult]:
    """
    LLM-based scope check. Failure (exception) is non-fatal — the query is
    allowed through rather than blocking users due to an API hiccup.
    """
    try:
        response = client.chat.completions.create(
            model=config.scope_model,
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
            logger.info("Input guardrail — scope filter blocked query. reason=%s", reason)
            return GuardrailResult.blocked(
                STAGE,
                f"This system only answers questions about contracts and legal documents. {reason}",
                RiskLevel.MEDIUM, {"check": "scope", "model_reason": reason},
            )
    except Exception as exc:
        logger.warning("Input guardrail — scope check failed (%s); query allowed through.", exc)
    return None


def check_input(query: str, client: OpenAI) -> GuardrailResult:
    """
    Run every input-guardrail check in cheapest-first order.

    On ALLOWED, `.data` is the cleaned (whitespace-stripped) query the
    caller should use downstream instead of the raw input.
    """
    if not config.enable_input_guardrail:
        return GuardrailResult.ok(STAGE, data=query.strip(), reason="input guardrail disabled")

    cleaned = query.strip()

    length_result = _check_length(cleaned)
    if length_result is not None:
        return length_result

    injection_result = _check_injection(cleaned)
    if injection_result is not None:
        return injection_result

    scope_result = _check_scope(cleaned, client)
    if scope_result is not None:
        return scope_result

    logger.debug("Input guardrail — all checks passed for query (len=%d)", len(cleaned))
    return GuardrailResult.ok(STAGE, data=cleaned)
