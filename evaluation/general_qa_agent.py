"""
evaluation/general_qa_agent.py
-------------------------------
Evaluator for the General QA sub-agent inside AgenticRAG.
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

_PRIMARY_RAGAS = ["faithfulness", "answer_relevancy"]
_SECONDARY_RAGAS = ["answer_correctness"]


class GeneralQAAgentEvaluator:

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
                component="GeneralQAAgent",
                metrics=metrics,
                details=details,
                errors=["No samples provided"],
                elapsed_s=0.0,
            )

        # ---- RAGAS metrics ----
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

        # ---- Multi-hop coverage ----
        # Fraction of subquestions (tool_calls) that have at least one context chunk
        coverage_scores = []
        for s in samples:
            if not s.tool_calls:
                continue
            # Approximate: if contexts is non-empty, assume subquestions are covered
            covered = min(len(s.tool_calls), len(s.contexts))
            coverage_scores.append(covered / len(s.tool_calls))
        metrics["multi_hop_coverage"] = (
            sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        )

        # ---- Citation metrics ----
        cp_list, cr_list, cf_list = [], [], []
        for s in samples:
            try:
                cit = CitationChecker.score(s.answer, s.sources)
                cp_list.append(cit["citation_precision"])
                cr_list.append(cit["citation_recall"])
                cf_list.append(cit["citation_f1"])
            except Exception as exc:
                errors.append(f"Citation: {exc}")
                cp_list.append(0.0)
                cr_list.append(0.0)
                cf_list.append(0.0)
        metrics["citation_precision"] = sum(cp_list) / len(cp_list) if cp_list else 0.0
        metrics["citation_recall"] = sum(cr_list) / len(cr_list) if cr_list else 0.0
        metrics["citation_f1"] = sum(cf_list) / len(cf_list) if cf_list else 0.0

        # ---- Subquestion stats ----
        sub_counts = [len(s.tool_calls) for s in samples]
        metrics["avg_subquestions_per_query"] = (
            sum(sub_counts) / len(sub_counts) if sub_counts else 0.0
        )
        metrics["max_subquestions"] = float(max(sub_counts)) if sub_counts else 0.0

        # ---- No-context answer rate (hallucination proxy) ----
        no_ctx = sum(1 for s in samples if s.answer and not s.contexts)
        metrics["no_context_answer_rate"] = no_ctx / len(samples)

        elapsed = time.perf_counter() - t0
        return MetricsResult(
            component="GeneralQAAgent",
            metrics=metrics,
            details=details,
            errors=errors,
            elapsed_s=elapsed,
        )
