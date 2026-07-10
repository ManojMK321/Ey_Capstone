"""
output_guardrail.py

Runs AFTER LLM Response Generation, on the final answer text before it is
returned to the user — for both KnowledgeRAG and AgenticRAG.

Checks (cheapest-first):
    1. PII scan      — O(n) regex. Blocks by default; set
       GUARDRAIL_PII_MODE=redact to mask matches and let the answer through
       instead.
    2. Groundedness  — LLM call (gpt-4o-mini), verifies every claim is
       backed by the retrieved context
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from .guardrail_config import GuardrailResult, GuardrailStatus, RiskLevel, config

logger = logging.getLogger(__name__)

STAGE = "output_guardrail"

_PII_PATTERNS: dict[str, re.Pattern] = {
    "SSN":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "passport":    re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
    "IBAN":        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,}([A-Z0-9]?\d?){0,16}\b"),
}

_GROUNDEDNESS_SYSTEM_PROMPT = (
    "You are a strict groundedness auditor for a Contract Intelligence System. "
    "You receive: (1) retrieved context chunks from contract documents, and "
    "(2) an AI-generated answer. "
    "Your task: verify that EVERY factual claim in the answer is explicitly supported "
    "by the provided context. "
    "If the answer introduces facts, names, dates, amounts, or clauses NOT present in "
    "the context, mark it as NOT grounded. "
    "Respond ONLY with valid JSON: "
    '{"grounded": true_or_false, "reason": "<one sentence naming any unsupported claim>"}'
)

_MAX_CHUNKS_FOR_AUDIT = 5


def _redact(text: str, label: str, pattern: re.Pattern) -> str:
    return pattern.sub(f"[REDACTED:{label}]", text)


def _check_pii(answer: str) -> Optional[GuardrailResult]:
    hits: list[str] = []
    cleaned = answer
    for label, pattern in _PII_PATTERNS.items():
        if pattern.search(cleaned):
            hits.append(label)
            if config.pii_mode == "redact":
                cleaned = _redact(cleaned, label, pattern)

    if not hits:
        return None

    logger.warning("Output guardrail — PII pattern(s) detected in model output: %s", hits)

    if config.pii_mode == "redact":
        return GuardrailResult.modified(
            STAGE, data=cleaned,
            reason=f"Sensitive data ({', '.join(hits)}) was redacted from the answer.",
            risk_level=RiskLevel.HIGH, metadata={"check": "pii", "labels": hits},
        )

    return GuardrailResult.blocked(
        STAGE, f"The generated answer contained sensitive data ({', '.join(hits)}) and was blocked to protect privacy.",
        RiskLevel.HIGH, {"check": "pii", "labels": hits},
    )


def _check_groundedness(answer: str, context_chunks: list[str], client: OpenAI) -> Optional[GuardrailResult]:
    """
    LLM-based groundedness check. Failure (exception / API error) is
    non-fatal — the answer is allowed through rather than blocking users
    due to a transient fault.
    """
    if not context_chunks:
        return None

    context_text = "\n\n---\n\n".join(context_chunks[:_MAX_CHUNKS_FOR_AUDIT])
    user_content = f"Context chunks:\n{context_text}\n\nAnswer to audit:\n{answer}"

    try:
        response = client.chat.completions.create(
            model=config.groundedness_model,
            temperature=0.0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _GROUNDEDNESS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data: dict = json.loads(raw)

        if not data.get("grounded", True):
            reason = data.get("reason", "Answer contains claims not found in the retrieved context.")
            logger.warning("Output guardrail — groundedness check failed: %s", reason)
            return GuardrailResult.blocked(
                STAGE, f"The answer could not be fully verified against the source documents. {reason}",
                RiskLevel.MEDIUM, {"check": "groundedness", "model_reason": reason},
            )
    except Exception as exc:
        logger.warning("Output guardrail — groundedness check failed (%s); answer allowed through.", exc)

    return None


def check_output(answer: str, context_chunks: list[str], client: OpenAI) -> GuardrailResult:
    """
    Validate a generated answer before it is returned to the user.
    `.data` is the answer to actually send back — unchanged on ALLOWED,
    redacted on MODIFIED, or None on BLOCKED (use `.reason` as the
    user-facing replacement message in that case).
    """
    if not config.enable_output_guardrail:
        return GuardrailResult.ok(STAGE, data=answer, reason="output guardrail disabled")

    pii_result = _check_pii(answer)
    if pii_result is not None and pii_result.status == GuardrailStatus.BLOCKED:
        return pii_result

    working_answer = pii_result.data if (pii_result and pii_result.status == GuardrailStatus.MODIFIED) else answer

    groundedness_result = _check_groundedness(working_answer, context_chunks, client)
    if groundedness_result is not None:
        return groundedness_result

    if pii_result is not None:  # MODIFIED case, survived the groundedness check
        return pii_result

    logger.debug("Output guardrail — all checks passed.")
    return GuardrailResult.ok(STAGE, data=working_answer)
