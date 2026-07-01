"""
evaluation/system.py
---------------------
System-level evaluator: end-to-end metrics across all pipeline stages.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from evaluation.base import CitationChecker, EvalSample, MetricsResult, RagasRunner

logger = logging.getLogger(__name__)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100) * (len(sorted_v) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


class SystemEvaluator:

    def __init__(
        self,
        use_ragas: bool = True,
        latency_sla_ms: float = 2000.0,
        client=None,
    ):
        self.use_ragas = use_ragas
        self.latency_sla_ms = latency_sla_ms
        self.client = client

    def evaluate(self, samples: list[EvalSample]) -> MetricsResult:
        t0 = time.perf_counter()
        errors: list[str] = []
        metrics: dict[str, float] = {}
        details: dict[str, Any] = {}

        if not samples:
            return MetricsResult(
                component="System",
                metrics=metrics,
                details=details,
                errors=["No samples provided"],
                elapsed_s=0.0,
            )

        # ---- Intent routing accuracy ----
        intent_samples = [
            s for s in samples
            if s.expected_intent is not None and s.predicted_intent is not None
        ]
        if intent_samples:
            correct = sum(
                s.expected_intent == s.predicted_intent for s in intent_samples
            )
            metrics["intent_routing_accuracy"] = correct / len(intent_samples)
        else:
            metrics["intent_routing_accuracy"] = 0.0

        # ---- Citation accuracy by workflow ----
        knowledge_cf1 = []
        agentic_cf1 = []
        all_cf1 = []

        for s in samples:
            try:
                cit = CitationChecker.score(s.answer, s.sources)
                cf1 = cit["citation_f1"]
            except Exception as exc:
                errors.append(f"Citation: {exc}")
                cf1 = 0.0
            all_cf1.append(cf1)
            if s.predicted_intent == "KnowledgeRAG":
                knowledge_cf1.append(cf1)
            elif s.predicted_intent == "AgenticRAG":
                agentic_cf1.append(cf1)

        metrics["citation_accuracy_knowledge"] = (
            sum(knowledge_cf1) / len(knowledge_cf1) if knowledge_cf1 else 0.0
        )
        metrics["citation_accuracy_agentic"] = (
            sum(agentic_cf1) / len(agentic_cf1) if agentic_cf1 else 0.0
        )
        metrics["citation_accuracy_overall"] = (
            sum(all_cf1) / len(all_cf1) if all_cf1 else 0.0
        )

        # ---- Latency metrics ----
        latency_values = [
            s.response_time_ms
            for s in samples
            if s.response_time_ms is not None
        ]
        if latency_values:
            metrics["response_time_p50_ms"] = _percentile(latency_values, 50)
            metrics["response_time_p95_ms"] = _percentile(latency_values, 95)
            metrics["response_time_p99_ms"] = _percentile(latency_values, 99)
            metrics["avg_response_time_ms"] = sum(latency_values) / len(latency_values)
            sla_ok = sum(1 for v in latency_values if v <= self.latency_sla_ms)
            metrics["sla_compliance_rate"] = sla_ok / len(latency_values)
        else:
            metrics["response_time_p50_ms"] = 0.0
            metrics["response_time_p95_ms"] = 0.0
            metrics["response_time_p99_ms"] = 0.0
            metrics["avg_response_time_ms"] = 0.0
            metrics["sla_compliance_rate"] = 1.0

        # ---- Hallucination rate ----
        # Basic: non-empty answer with no contexts
        flagged_basic: set[int] = set()
        for i, s in enumerate(samples):
            if s.answer and len(s.answer.strip()) >= 10 and not s.contexts:
                flagged_basic.add(i)

        # RAGAS faithfulness supplement
        if self.use_ragas:
            try:
                ragas_scores = RagasRunner.run(
                    samples=samples,
                    metric_names=["faithfulness"],
                    client=self.client,
                )
                batch_faithfulness = ragas_scores.get("faithfulness", None)
                # If batch faithfulness < 0.5, flag all non-empty samples
                if batch_faithfulness is not None and batch_faithfulness < 0.5:
                    for i, s in enumerate(samples):
                        if s.answer and len(s.answer.strip()) >= 10:
                            flagged_basic.add(i)
                # Store for transparency
                metrics["system_faithfulness"] = float(batch_faithfulness or 0.0)
            except Exception as exc:
                errors.append(f"RAGAS faithfulness: {exc}")

        metrics["hallucination_rate"] = len(flagged_basic) / len(samples)

        # ---- Share of each workflow ----
        knowledge_count = sum(
            1 for s in samples if s.predicted_intent == "KnowledgeRAG"
        )
        agentic_count = sum(
            1 for s in samples if s.predicted_intent == "AgenticRAG"
        )
        metrics["knowledge_rag_share"] = knowledge_count / len(samples)
        metrics["agentic_rag_share"] = agentic_count / len(samples)

        # ---- Error rate (empty or < 10 char answers) ----
        error_count = sum(
            1 for s in samples
            if not s.answer or len(s.answer.strip()) < 10
        )
        metrics["error_rate"] = error_count / len(samples)

        # ---- Average answer length ----
        answer_lengths = [len(s.answer) for s in samples if s.answer]
        metrics["avg_answer_length_chars"] = (
            sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0
        )

        # ---- User satisfaction ----
        satisfactions = [
            s.metadata.get("user_satisfaction")
            for s in samples
            if isinstance(s.metadata.get("user_satisfaction"), (int, float))
        ]
        if satisfactions:
            # Normalise 1-5 → 0-1
            metrics["user_satisfaction"] = (
                sum((v - 1) / 4 for v in satisfactions) / len(satisfactions)
            )

        elapsed = time.perf_counter() - t0
        return MetricsResult(
            component="System",
            metrics=metrics,
            details=details,
            errors=errors,
            elapsed_s=elapsed,
        )
