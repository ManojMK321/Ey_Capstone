"""
evaluation/comparison_agent.py
-------------------------------
Evaluator for the Comparison sub-agent inside AgenticRAG.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from evaluation.base import (
    CitationChecker,
    EvalSample,
    MetricsResult,
    RagasRunner,
    SemanticSimilarity,
)

logger = logging.getLogger(__name__)

_PRIMARY_RAGAS = ["faithfulness", "answer_correctness"]
_SECONDARY_RAGAS = ["context_precision", "context_recall"]


def _normalise_tool_call(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _tool_call_metrics(
    predicted: list[str],
    expected: list[str],
    sim_threshold: float = 0.70,
) -> dict[str, float]:
    """
    Compare predicted tool calls (subquestions) against expected ones.
    A predicted call matches an expected call if:
      - exact normalised match, OR
      - semantic similarity >= sim_threshold
    Each predicted call matches at most one expected call.
    """
    if not expected and not predicted:
        return {
            "tool_call_precision": 1.0,
            "tool_call_recall": 1.0,
            "tool_call_f1": 1.0,
            "tool_call_accuracy": 1.0,
        }
    if not expected:
        return {
            "tool_call_precision": 0.0,
            "tool_call_recall": 0.0,
            "tool_call_f1": 0.0,
            "tool_call_accuracy": 0.0,
        }
    if not predicted:
        return {
            "tool_call_precision": 0.0,
            "tool_call_recall": 0.0,
            "tool_call_f1": 0.0,
            "tool_call_accuracy": 0.0,
        }

    norm_pred = [_normalise_tool_call(p) for p in predicted]
    norm_exp = [_normalise_tool_call(e) for e in expected]

    matched_expected: set[int] = set()
    true_positives = 0

    for pi, np_ in enumerate(norm_pred):
        for ei, ne in enumerate(norm_exp):
            if ei in matched_expected:
                continue
            if np_ == ne:
                matched_expected.add(ei)
                true_positives += 1
                break
            try:
                sim = SemanticSimilarity.score(predicted[pi], expected[ei])
            except Exception:
                sim = 0.0
            if sim >= sim_threshold:
                matched_expected.add(ei)
                true_positives += 1
                break

    precision = true_positives / len(predicted)
    recall = true_positives / len(expected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # accuracy: exact set match (1.0 if sizes equal and all match, 0.0 otherwise)
    accuracy = 1.0 if (len(predicted) == len(expected) and true_positives == len(expected)) else 0.0

    return {
        "tool_call_precision": precision,
        "tool_call_recall": recall,
        "tool_call_f1": f1,
        "tool_call_accuracy": accuracy,
    }


class ComparisonAgentEvaluator:

    def __init__(
        self,
        use_ragas: bool = True,
        include_secondary: bool = True,
        client=None,
    ):
        self.use_ragas = use_ragas
        self.include_secondary = include_secondary
        self.client = client

    def evaluate(self, samples: list[EvalSample]) -> MetricsResult:
        t0 = time.perf_counter()
        errors: list[str] = []
        metrics: dict[str, float] = {}
        details: dict[str, Any] = {}

        if not samples:
            return MetricsResult(
                component="ComparisonAgent",
                metrics=metrics,
                details=details,
                errors=["No samples provided"],
                elapsed_s=0.0,
            )

        # ---- RAGAS primary ----
        ragas_names = list(_PRIMARY_RAGAS)
        if self.include_secondary:
            ragas_names += list(_SECONDARY_RAGAS)

        if self.use_ragas:
            try:
                ragas_scores = RagasRunner.run(
                    samples=samples,
                    metric_names=ragas_names,
                    client=self.client,
                )
                metrics.update(ragas_scores)
            except Exception as exc:
                logger.warning("RAGAS evaluation failed: %s", exc)
                errors.append(f"RAGAS: {exc}")
                for m in ragas_names:
                    metrics.setdefault(m, 0.0)
        else:
            for m in ragas_names:
                metrics[m] = 0.0

        # ---- Semantic similarity ----
        sim_scores = []
        for s in samples:
            if s.answer and s.ground_truth:
                try:
                    sim_scores.append(SemanticSimilarity.score(s.answer, s.ground_truth))
                except Exception as exc:
                    errors.append(f"SemanticSimilarity: {exc}")
        metrics["semantic_similarity"] = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0

        # ---- Tool call metrics ----
        tc_precision_list, tc_recall_list, tc_f1_list, tc_acc_list = [], [], [], []
        for s in samples:
            tc = _tool_call_metrics(s.tool_calls, s.expected_tool_calls)
            tc_precision_list.append(tc["tool_call_precision"])
            tc_recall_list.append(tc["tool_call_recall"])
            tc_f1_list.append(tc["tool_call_f1"])
            tc_acc_list.append(tc["tool_call_accuracy"])

        metrics["tool_call_precision"] = sum(tc_precision_list) / len(tc_precision_list)
        metrics["tool_call_recall"] = sum(tc_recall_list) / len(tc_recall_list)
        metrics["tool_call_f1"] = sum(tc_f1_list) / len(tc_f1_list)
        metrics["tool_call_accuracy"] = sum(tc_acc_list) / len(tc_acc_list)

        # ---- Multi-doc citation rate ----
        multi_doc_count = 0
        for s in samples:
            cited = CitationChecker.parse_citations(s.answer)
            doc_ids = {c["doc_id"] for c in cited}
            if len(doc_ids) >= 2:
                multi_doc_count += 1
        metrics["multi_doc_citation_rate"] = multi_doc_count / len(samples)

        elapsed = time.perf_counter() - t0
        return MetricsResult(
            component="ComparisonAgent",
            metrics=metrics,
            details=details,
            errors=errors,
            elapsed_s=elapsed,
        )
