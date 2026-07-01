"""
evaluation/knowledge_rag.py
---------------------------
Evaluator for the KnowledgeRAG component.
"""
from __future__ import annotations

import logging
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

_DEFAULT_RAGAS_METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "answer_correctness",
]


class KnowledgeRAGEvaluator:

    def __init__(
        self,
        use_ragas: bool = True,
        ragas_metrics: list[str] | None = None,
        client=None,
    ):
        self.use_ragas = use_ragas
        self.ragas_metrics = ragas_metrics or _DEFAULT_RAGAS_METRICS
        self.client = client

    def evaluate(self, samples: list[EvalSample]) -> MetricsResult:
        t0 = time.perf_counter()
        errors: list[str] = []
        metrics: dict[str, float] = {}
        details: dict[str, Any] = {}

        if not samples:
            return MetricsResult(
                component="KnowledgeRAG",
                metrics=metrics,
                details=details,
                errors=["No samples provided"],
                elapsed_s=0.0,
            )

        # ---- RAGAS metrics ----
        if self.use_ragas:
            try:
                ragas_scores = RagasRunner.run(
                    samples=samples,
                    metric_names=self.ragas_metrics,
                    client=self.client,
                )
                metrics.update(ragas_scores)
            except Exception as exc:
                logger.warning("RAGAS evaluation failed: %s", exc)
                errors.append(f"RAGAS: {exc}")
                for m in self.ragas_metrics:
                    metrics.setdefault(m, 0.0)
        else:
            for m in self.ragas_metrics:
                metrics[m] = 0.0

        # ---- Citation metrics ----
        citation_per_sample = []
        citation_precision_list = []
        citation_recall_list = []
        citation_f1_list = []

        for s in samples:
            try:
                cit = CitationChecker.score(s.answer, s.sources)
                citation_per_sample.append({
                    "query": s.query,
                    "citation_precision": cit["citation_precision"],
                    "citation_recall": cit["citation_recall"],
                    "citation_f1": cit["citation_f1"],
                })
                citation_precision_list.append(cit["citation_precision"])
                citation_recall_list.append(cit["citation_recall"])
                citation_f1_list.append(cit["citation_f1"])
            except Exception as exc:
                logger.warning("CitationChecker failed for sample: %s", exc)
                errors.append(f"Citation: {exc}")
                citation_precision_list.append(0.0)
                citation_recall_list.append(0.0)
                citation_f1_list.append(0.0)

        metrics["citation_precision"] = (
            sum(citation_precision_list) / len(citation_precision_list)
            if citation_precision_list else 0.0
        )
        metrics["citation_recall"] = (
            sum(citation_recall_list) / len(citation_recall_list)
            if citation_recall_list else 0.0
        )
        metrics["citation_f1"] = (
            sum(citation_f1_list) / len(citation_f1_list)
            if citation_f1_list else 0.0
        )
        details["citation_per_sample"] = citation_per_sample

        # ---- Semantic similarity ----
        sim_scores = []
        sim_per_sample = []
        for s in samples:
            if s.answer and s.ground_truth:
                try:
                    sc = SemanticSimilarity.score(s.answer, s.ground_truth)
                    sim_scores.append(sc)
                    sim_per_sample.append({"query": s.query, "score": sc})
                except Exception as exc:
                    logger.warning("SemanticSimilarity failed: %s", exc)
                    errors.append(f"SemanticSimilarity: {exc}")
            else:
                sim_per_sample.append({"query": s.query, "score": 0.0})

        metrics["semantic_similarity"] = (
            sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
        )
        details["semantic_similarity_per_sample"] = sim_per_sample

        # ---- Context coverage ----
        samples_with_context = sum(1 for s in samples if s.contexts)
        metrics["context_coverage_rate"] = samples_with_context / len(samples)

        doc_counts = [len(s.contexts) for s in samples]
        metrics["avg_retrieved_docs"] = (
            sum(doc_counts) / len(doc_counts) if doc_counts else 0.0
        )

        elapsed = time.perf_counter() - t0
        return MetricsResult(
            component="KnowledgeRAG",
            metrics=metrics,
            details=details,
            errors=errors,
            elapsed_s=elapsed,
        )
