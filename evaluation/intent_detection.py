"""
evaluation/intent_detection.py
-------------------------------
Evaluator for the IntentDetector component.

Computes workflow-level and task-level classification metrics.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from evaluation.base import EvalSample, MetricsResult

logger = logging.getLogger(__name__)

WORKFLOW_LABELS = ["KnowledgeRAG", "AgenticRAG"]
TASK_LABELS = [
    "lookup", "summary", "comparison", "compliance",
    "reasoning", "risk_analysis", "multi_step",
]


def _safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if denom > 0 else default


def _precision_recall_f1(
    y_true: list[str], y_pred: list[str], label: str
) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s = []
    for label in labels:
        _, _, f1 = _precision_recall_f1(y_true, y_pred, label)
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _weighted_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    total = len(y_true)
    if total == 0:
        return 0.0
    f1_sum = 0.0
    for label in labels:
        support = sum(1 for t in y_true if t == label)
        _, _, f1 = _precision_recall_f1(y_true, y_pred, label)
        f1_sum += f1 * support
    return f1_sum / total


def _confusion_matrix(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1
    return matrix


def _ece(
    y_true: list[str], y_pred: list[str], confidences: list[float], n_bins: int = 10
) -> float:
    """Expected Calibration Error with equal-width bins."""
    if len(y_true) < n_bins:
        return 0.0
    bin_size = 1.0 / n_bins
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo = b * bin_size
        hi = lo + bin_size
        indices = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not indices:
            continue
        acc = sum(1 for i in indices if y_true[i] == y_pred[i]) / len(indices)
        avg_conf = sum(confidences[i] for i in indices) / len(indices)
        ece += (len(indices) / n) * abs(acc - avg_conf)
    return ece


class IntentDetectionEvaluator:

    def __init__(self, use_sklearn: bool = True):
        self.use_sklearn = use_sklearn
        self._sklearn_available: bool | None = None

    def _check_sklearn(self) -> bool:
        if self._sklearn_available is not None:
            return self._sklearn_available
        try:
            import sklearn  # noqa: F401
            self._sklearn_available = True
        except ImportError:
            self._sklearn_available = False
        return self._sklearn_available

    def evaluate(self, samples: list[EvalSample]) -> MetricsResult:
        t0 = time.perf_counter()
        errors: list[str] = []

        # Filter samples that have both expected and predicted intent
        wf_samples = [
            s for s in samples
            if s.expected_intent is not None and s.predicted_intent is not None
        ]
        task_samples = [
            s for s in samples
            if s.expected_task is not None and s.predicted_task is not None
        ]

        metrics: dict[str, float] = {}
        details: dict[str, Any] = {}

        # ---- Workflow metrics ----
        if wf_samples:
            y_true_wf = [s.expected_intent for s in wf_samples]
            y_pred_wf = [s.predicted_intent for s in wf_samples]

            observed_wf = sorted(set(y_true_wf + y_pred_wf))

            metrics["workflow_accuracy"] = _accuracy(y_true_wf, y_pred_wf)
            metrics["workflow_f1_macro"] = _macro_f1(y_true_wf, y_pred_wf, observed_wf)
            metrics["workflow_f1_weighted"] = _weighted_f1(y_true_wf, y_pred_wf, observed_wf)

            for label in WORKFLOW_LABELS:
                p, r, f = _precision_recall_f1(y_true_wf, y_pred_wf, label)
                metrics[f"precision_{label}"] = p
                metrics[f"recall_{label}"] = r
                metrics[f"f1_{label}"] = f

            cm_wf = _confusion_matrix(y_true_wf, y_pred_wf, WORKFLOW_LABELS)
            details["workflow_confusion_matrix"] = {
                "labels": WORKFLOW_LABELS,
                "matrix": cm_wf,
            }

            if self.use_sklearn and self._check_sklearn():
                try:
                    from sklearn.metrics import classification_report
                    details["workflow_classification_report"] = classification_report(
                        y_true_wf, y_pred_wf, labels=observed_wf, zero_division=0
                    )
                except Exception as exc:
                    errors.append(f"sklearn workflow report: {exc}")
        else:
            for key in ["workflow_accuracy", "workflow_f1_macro", "workflow_f1_weighted"]:
                metrics[key] = 0.0
            for label in WORKFLOW_LABELS:
                metrics[f"precision_{label}"] = 0.0
                metrics[f"recall_{label}"] = 0.0
                metrics[f"f1_{label}"] = 0.0

        # ---- Task metrics ----
        if task_samples:
            y_true_task = [s.expected_task for s in task_samples]
            y_pred_task = [s.predicted_task for s in task_samples]

            observed_tasks = sorted(set(y_true_task + y_pred_task))

            metrics["task_accuracy"] = _accuracy(y_true_task, y_pred_task)
            metrics["task_f1_macro"] = _macro_f1(y_true_task, y_pred_task, observed_tasks)
            metrics["task_f1_weighted"] = _weighted_f1(y_true_task, y_pred_task, observed_tasks)

            for label in observed_tasks:
                p, r, f = _precision_recall_f1(y_true_task, y_pred_task, label)
                metrics[f"task_precision_{label}"] = p
                metrics[f"task_recall_{label}"] = r
                metrics[f"task_f1_{label}"] = f

            cm_task = _confusion_matrix(y_true_task, y_pred_task, observed_tasks)
            details["task_confusion_matrix"] = {
                "labels": observed_tasks,
                "matrix": cm_task,
            }

            if self.use_sklearn and self._check_sklearn():
                try:
                    from sklearn.metrics import classification_report
                    details["task_classification_report"] = classification_report(
                        y_true_task, y_pred_task, labels=observed_tasks, zero_division=0
                    )
                except Exception as exc:
                    errors.append(f"sklearn task report: {exc}")
        else:
            metrics["task_accuracy"] = 0.0
            metrics["task_f1_macro"] = 0.0
            metrics["task_f1_weighted"] = 0.0

        # ---- Confidence / calibration ----
        confidences = [
            s.metadata.get("confidence")
            for s in wf_samples
            if isinstance(s.metadata.get("confidence"), (int, float))
        ]
        if confidences:
            metrics["avg_confidence"] = sum(confidences) / len(confidences)
            if len(wf_samples) >= 10:
                y_true_wf_ece = [s.expected_intent for s in wf_samples
                                  if isinstance(s.metadata.get("confidence"), (int, float))]
                y_pred_wf_ece = [s.predicted_intent for s in wf_samples
                                  if isinstance(s.metadata.get("confidence"), (int, float))]
                metrics["expected_calibration_error"] = _ece(
                    y_true_wf_ece, y_pred_wf_ece, confidences
                )

        elapsed = time.perf_counter() - t0
        return MetricsResult(
            component="IntentDetection",
            metrics=metrics,
            details=details,
            errors=errors,
            elapsed_s=elapsed,
        )
