"""
guardrails/output_guard.py

Output guardrail — runs AFTER the RAG pipeline generates an answer.

Two checks executed in cheapest-first order:
  1. PII scan       — O(n) regex, blocks answers that expose personal / financial data
  2. Groundedness   — LLM call (gpt-4o-mini), verifies every claim is backed by retrieved context
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from guardrails.input_guard import GuardResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check 1 — PII scan
# ---------------------------------------------------------------------------

# Map of label -> compiled regex.
# Patterns target high-risk personal / financial identifiers that should
# never appear in contract analysis answers.
_PII_PATTERNS: dict[str, re.Pattern] = {
    # US Social Security Number:  123-45-6789
    "SSN": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    # Payment card numbers (Visa / MC / Amex / Discover — with or without separators)
    "credit_card": re.compile(
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    ),
    # Passport numbers (1-2 uppercase letters followed by 6-9 digits)
    "passport": re.compile(
        r"\b[A-Z]{1,2}\d{6,9}\b"
    ),
    # IBAN (International Bank Account Number): GB29NWBK60161331926819
    "IBAN": re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,}([A-Z0-9]?\d?){0,16}\b"
    ),
}


def _check_pii(answer: str) -> Optional[GuardResult]:
    for label, pattern in _PII_PATTERNS.items():
        if pattern.search(answer):
            logger.warning("Output guard — PII pattern '%s' detected in model output.", label)
            return GuardResult(
                allowed=False,
                reason="pii",
                detail=(
                    f"The generated answer contained sensitive data ({label}) and was blocked "
                    "to protect privacy."
                ),
            )
    return None


# ---------------------------------------------------------------------------
# Check 2 — Groundedness check  (gpt-4o-mini)
# ---------------------------------------------------------------------------

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

# Cap how many context chunks we send to the auditor — keeps token cost low
_MAX_CHUNKS_FOR_AUDIT: int = 5


def _check_groundedness(
    answer: str,
    context_chunks: list[str],
    client: OpenAI,
) -> Optional[GuardResult]:
    """
    LLM-based groundedness check. Failure (exception / API error) is non-fatal —
    the answer is allowed through rather than blocking users due to a transient fault.
    """
    if not context_chunks:
        # Nothing to compare against — skip rather than false-blocking
        return None

    context_text = "\n\n---\n\n".join(context_chunks[:_MAX_CHUNKS_FOR_AUDIT])
    user_content = (
        f"Context chunks:\n{context_text}\n\n"
        f"Answer to audit:\n{answer}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
            reason = data.get(
                "reason",
                "Answer contains claims not found in the retrieved context.",
            )
            logger.warning("Output guard — groundedness check failed: %s", reason)
            return GuardResult(
                allowed=False,
                reason="ungrounded",
                detail=(
                    "The answer could not be fully verified against the source documents. "
                    f"{reason}"
                ),
            )

    except Exception as exc:
        logger.warning("Output guard — groundedness check failed (%s); answer allowed through.", exc)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_output(
    answer: str,
    context_chunks: list[str],
    client: OpenAI,
) -> GuardResult:
    """
    Run all output guardrail checks in cheapest-first order.

    Returns the first GuardResult with allowed=False, or GuardResult.ok()
    if every check passes.

    Args:
        answer         : text generated by the RAG pipeline.
        context_chunks : list of page_content strings from retrieved documents
                         (used for groundedness verification).
        client         : initialised OpenAI client.

    Returns:
        GuardResult — caller should replace the answer with .detail if .allowed is False.
    """
    pii_result = _check_pii(answer)
    if pii_result is not None:
        return pii_result

    groundedness_result = _check_groundedness(answer, context_chunks, client)
    if groundedness_result is not None:
        return groundedness_result

    logger.debug("Output guard — all checks passed.")
    return GuardResult.ok()
