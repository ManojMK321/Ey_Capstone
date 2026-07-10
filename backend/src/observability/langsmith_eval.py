"""
langsmith_eval.py
-----------------
Pushes the live RAGAS scores computed by `eval/ragas_judge.py` onto the
matching LangSmith run as feedback, so grounding/relevancy show up next to
the trace instead of only in the Streamlit "Evaluate RAGAS" panel.

Safe no-op if LangSmith isn't configured — same "missing API key => disabled"
behaviour as the rest of this package (see langsmith.py).
"""

from __future__ import annotations

import logging
from typing import Any

from . import langsmith as ls_module

logger = logging.getLogger(__name__)

_METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def log_ragas_feedback(run_id: str, score: Any) -> None:
    """
    Attach a RAGAS score to a LangSmith run as feedback.

    `score` is a `ragas_judge.TurnRagasScore` (or any object/dict exposing
    the same four fields); `None`-valued metrics are skipped. `run_id` is
    the LangSmith run id of the traced chat turn being scored. No-ops
    quietly if LangSmith isn't configured.
    """
    if not ls_module.is_enabled():
        return

    client = ls_module._get_client()
    if client is None:
        return

    for key in _METRIC_KEYS:
        value = score.get(key) if isinstance(score, dict) else getattr(score, key, None)
        if value is None:
            continue
        try:
            client.create_feedback(run_id, key=key, score=float(value))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to log LangSmith feedback '%s' for run %s: %s", key, run_id, exc)


def evaluate_and_log(question: str, answer: str, contexts: list[str], run_id: str | None = None) -> Any:
    """Score a chat turn with RAGAS and, if `run_id` is given, log it to LangSmith."""
    from eval.ragas_judge import score_turn

    score = score_turn(question, answer, contexts)
    if run_id:
        log_ragas_feedback(run_id, score)
    return score
