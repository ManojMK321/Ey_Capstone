"""
evaluation/compliance_agent.py
-------------------------------
Evaluator for the Compliance sub-agent inside AgenticRAG.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from evaluation.base import (
    EvalSample,
    MetricsResult,
    RagasRunner,
    SemanticSimilarity,
)

logger = logging.getLogger(__name__)


def _jaccard_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 0.0


def _match_issues(
    predicted: list[str],
    expected: list[str],
    overlap_thresh: float,
    sim_thresh: float,
) -> tuple[int, set[int]]:
    """
    Returns (n_true_positives, set of matched expected indices).
    Each predicted issue matches at most one expected issue.
    """
    matched_expected: set[int] = set()
    tp = 0
    for pred in predicted:
        for ei, exp in enumerate(expected):
            if ei in matched_expected:
                continue
            jac = _jaccard_overlap(pred, exp)
            if jac >= overlap_thresh:
                matched_expected.add(ei)
                tp += 1
                break
            try:
                sim = SemanticSimilarity.score(pred, exp)
            except Exception:
                sim = 0.0
            if sim >= sim_thresh:
                matched_expected.add(ei)
                tp += 1
                break
    return tp, matched_expected


class ComplianceAgentEvaluator:

    def __init__(
        self,
        use_ragas: bool = True,
        issue_overlap_thresh: float = 0.35,
        issue_sim_thresh: float = 0.65,
        client=None,
    ):
        self.use_ragas = use_ragas
        self.issue_overlap_thresh = issue_overlap_thresh
        self.issue_sim_thresh = issue_sim_thresh
        self.client = client

    def evaluate(self, samples: list[EvalSample]) -> MetricsResult:
        t0 = time.perf_counter()
        errors: list[str] = []
        metrics: dict[str, float] = {}
        details: dict[str, Any] = {}

        if not samples:
            return MetricsResult(
                component="ComplianceAgent",
                metrics=metrics,
                details=details,
                errors=["No samples provided"],
                elapsed_s=0.0,
            )

        # ---- RAGAS: faithfulness ----
        if self.use_ragas:
            try:
                ragas_scores = RagasRunner.run(
                    samples=samples,
                    metric_names=["faithfulness"],
                    client=self.client,
                )
                metrics.update(ragas_scores)
            except Exception as exc:
                logger.warning("RAGAS evaluation failed: %s", exc)
                errors.append(f"RAGAS: {exc}")
                metrics.setdefault("faithfulness", 0.0)
        else:
            metrics["faithfulness"] = 0.0

        # ---- Issue precision / recall / f1 ----
        precision_list, recall_list, f1_list = [], [], []
        fp_count = 0
        total_predicted = 0

        for s in samples:
            predicted = s.predicted_issues
            expected = s.expected_issues
            n_pred = len(predicted)
            n_exp = len(expected)
            total_predicted += n_pred

            if n_pred == 0 and n_exp == 0:
                precision_list.append(1.0)
                recall_list.append(1.0)
                f1_list.append(1.0)
                continue

            if n_pred == 0:
                precision_list.append(0.0)
                recall_list.append(0.0)
                f1_list.append(0.0)
                continue

            tp, matched = _match_issues(
                predicted, expected,
                self.issue_overlap_thresh, self.issue_sim_thresh,
            )
            fp = n_pred - tp
            fp_count += fp

            prec = tp / n_pred if n_pred > 0 else 0.0
            rec = tp / n_exp if n_exp > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)

        metrics["issue_precision"] = sum(precision_list) / len(precision_list) if precision_list else 0.0
        metrics["issue_recall"] = sum(recall_list) / len(recall_list) if recall_list else 0.0
        metrics["issue_f1"] = sum(f1_list) / len(f1_list) if f1_list else 0.0
        metrics["issue_false_positive_rate"] = (
            fp_count / total_predicted if total_predicted > 0 else 0.0
        )

        # ---- Severity accuracy ----
        sev_samples = [
            s for s in samples
            if s.expected_severity is not None and s.predicted_severity is not None
        ]
        if sev_samples:
            correct = sum(
                s.expected_severity.lower() == s.predicted_severity.lower()
                for s in sev_samples
            )
            metrics["severity_accuracy"] = correct / len(sev_samples)

            for level in ["high", "medium", "low"]:
                level_samples = [s for s in sev_samples if s.expected_severity.lower() == level]
                if level_samples:
                    correct_level = sum(
                        s.predicted_severity.lower() == level for s in level_samples
                    )
                    metrics[f"severity_accuracy_{level}"] = correct_level / len(level_samples)
                else:
                    metrics[f"severity_accuracy_{level}"] = 0.0
        else:
            metrics["severity_accuracy"] = 0.0
            for level in ["high", "medium", "low"]:
                metrics[f"severity_accuracy_{level}"] = 0.0

        # ---- Semantic similarity ----
        sim_scores = []
        for s in samples:
            if s.answer and s.ground_truth:
                try:
                    sim_scores.append(SemanticSimilarity.score(s.answer, s.ground_truth))
                except Exception as exc:
                    errors.append(f"SemanticSimilarity: {exc}")
        metrics["semantic_similarity"] = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0

        elapsed = time.perf_counter() - t0
        return MetricsResult(
            component="ComplianceAgent",
            metrics=metrics,
            details=details,
            errors=errors,
            elapsed_s=elapsed,
        )
