"""
agent_guardrail.py

"Agent Plan Validation" — validates AgenticRAG's plan before it is
executed: the sub-questions produced by query decomposition, and the
task-routing decision that selects a specialist agent. Deterministic,
structural checks only — no LLM call — so this adds negligible latency.

Call it once per piece of the plan you have in hand:

    plan = check_plan(question, subquestions=subquestions)
    ...
    plan = check_plan(question, task=route.task)
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .guardrail_config import GuardrailResult, RiskLevel, config, escalate

logger = logging.getLogger(__name__)

STAGE = "agent_guardrail"

_KNOWN_TASKS = {"comparison", "compliance", "general"}

# A sub-question is model-generated (from analyze_query), not raw user
# text, so the risk surface is narrower than the input guardrail's full
# injection scan — this just catches a decomposition step echoing back an
# instruction-override attempt.
_PLAN_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(?:all\s+)?(?:your\s+)?(?:previous|above|prior)\s+instructions?", re.I),
     "instruction override"),
    (re.compile(r"\byou\s+are\s+now\s+(?:a|an|the)\b", re.I),
     "identity hijack"),
    (re.compile(r"<\s*/?\s*system\s*>", re.I),
     "system tag injection"),
    (re.compile(r"<\|.*?\|>"),
     "special token injection"),
]


def _clean_subquestions(subquestions: list[str]) -> tuple[list[str], list[str]]:
    kept, dropped = [], []
    for sub in subquestions:
        text = (sub or "").strip()
        if not text:
            continue
        if any(pattern.search(text) for pattern, _ in _PLAN_INJECTION_PATTERNS):
            dropped.append(text)
            continue
        kept.append(text)
    return kept, dropped


def check_plan(
    question: str,
    subquestions: Optional[list[str]] = None,
    task: Optional[str] = None,
) -> GuardrailResult:
    """
    Validate one or both parts of an AgenticRAG plan.

    `.data` is always `{"subquestions": [...], "task": ...}` — untouched
    fields (the one not passed in this call) come back as given.
    """
    if not config.enable_agent_guardrail:
        return GuardrailResult.ok(
            STAGE, data={"subquestions": subquestions, "task": task},
            reason="agent guardrail disabled",
        )

    metadata: dict = {}
    risk = RiskLevel.NONE
    notes: list[str] = []

    cleaned_subquestions = subquestions
    if subquestions is not None:
        kept, dropped = _clean_subquestions(subquestions)
        if dropped:
            risk = escalate(risk, RiskLevel.MEDIUM)
            notes.append(f"dropped {len(dropped)} subquestion(s) matching injection patterns")
            metadata["dropped_subquestions"] = dropped
        if len(kept) > config.max_subquestions:
            kept = kept[: config.max_subquestions]
            risk = escalate(risk, RiskLevel.LOW)
            notes.append(f"capped subquestions to {config.max_subquestions}")
        cleaned_subquestions = kept or [question]

    validated_task = task
    if task is not None and task not in _KNOWN_TASKS:
        logger.warning("Agent guardrail — unknown task '%s' from router, defaulting to 'general'.", task)
        risk = escalate(risk, RiskLevel.LOW)
        notes.append(f"unknown task '{task}' replaced with 'general'")
        metadata["original_task"] = str(task)
        validated_task = "general"

    data = {"subquestions": cleaned_subquestions, "task": validated_task}

    if notes:
        return GuardrailResult.modified(STAGE, data=data, reason="; ".join(notes), risk_level=risk, metadata=metadata)
    return GuardrailResult.ok(STAGE, data=data)
