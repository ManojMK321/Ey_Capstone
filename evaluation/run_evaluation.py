"""
evaluation/run_evaluation.py
-----------------------------
Orchestrates all component evaluators into a single pipeline run.

CLI usage:
    python -m evaluation.run_evaluation \\
        --dataset eval.json \\
        --output  report.json \\
        [--skip-ragas] [--sla-ms 2000]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from evaluation.base import EvalSample, MetricsResult
from evaluation.intent_detection import IntentDetectionEvaluator
from evaluation.knowledge_rag import KnowledgeRAGEvaluator
from evaluation.comparison_agent import ComparisonAgentEvaluator
from evaluation.compliance_agent import ComplianceAgentEvaluator
from evaluation.general_qa_agent import GeneralQAAgentEvaluator
from evaluation.ingestion_pipeline import IngestionPipelineEvaluator
from evaluation.system import SystemEvaluator

logger = logging.getLogger(__name__)

_GENERAL_TASKS = {"reasoning", "multi_step", "risk_analysis"}


def load_dataset(path: str) -> list[EvalSample]:
    """Load a JSON file and convert each record to an EvalSample."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = []
    if isinstance(data, dict):
        # Support {"samples": [...]} wrapper
        data = data.get("samples", [data])
    for record in data:
        if not isinstance(record, dict):
            continue
        samples.append(
            EvalSample(
                query=record.get("query", ""),
                answer=record.get("answer", ""),
                ground_truth=record.get("ground_truth", ""),
                contexts=record.get("contexts", []),
                sources=record.get("sources", []),
                expected_intent=record.get("expected_intent"),
                predicted_intent=record.get("predicted_intent"),
                expected_task=record.get("expected_task"),
                predicted_task=record.get("predicted_task"),
                tool_calls=record.get("tool_calls", []),
                expected_tool_calls=record.get("expected_tool_calls", []),
                expected_issues=record.get("expected_issues", []),
                predicted_issues=record.get("predicted_issues", []),
                expected_severity=record.get("expected_severity"),
                predicted_severity=record.get("predicted_severity"),
                response_time_ms=record.get("response_time_ms"),
                metadata=record.get("metadata", {}),
            )
        )
    return samples


class EvaluationRunner:

    def __init__(
        self,
        use_ragas: bool = True,
        latency_sla_ms: float = 2000.0,
        verbose: bool = True,
    ):
        self.use_ragas = use_ragas
        self.latency_sla_ms = latency_sla_ms
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)

    def run(
        self,
        samples: list[EvalSample],
        chunks: list | None = None,
        expected_doc_ids: list[str] | None = None,
        vector_store=None,
    ) -> dict:
        t0 = time.perf_counter()
        results: dict[str, dict] = {}
        flat_scores: dict[str, float] = {}

        def _store(result: MetricsResult) -> None:
            results[result.component] = result.to_dict()
            for k, v in result.metrics.items():
                flat_scores[f"{result.component}.{k}"] = v

        # ---- Route samples ----
        knowledge_rag_samples = [
            s for s in samples if s.predicted_intent == "KnowledgeRAG"
        ]
        comparison_samples = [
            s for s in samples if s.predicted_task == "comparison"
        ]
        compliance_samples = [
            s for s in samples if s.predicted_task == "compliance"
        ]
        general_qa_samples = [
            s for s in samples if s.predicted_task in _GENERAL_TASKS
        ]

        # 1. Intent Detection (all samples)
        self._log("Running IntentDetection evaluator...")
        try:
            result = IntentDetectionEvaluator().evaluate(samples)
            _store(result)
        except Exception as exc:
            logger.error("IntentDetection evaluator failed: %s", exc)
            _store(MetricsResult(
                component="IntentDetection",
                metrics={},
                errors=[str(exc)],
            ))

        # 2. KnowledgeRAG (routed samples or fallback to all)
        self._log("Running KnowledgeRAG evaluator...")
        try:
            rag_samples = knowledge_rag_samples if knowledge_rag_samples else samples
            result = KnowledgeRAGEvaluator(use_ragas=self.use_ragas).evaluate(rag_samples)
            _store(result)
        except Exception as exc:
            logger.error("KnowledgeRAG evaluator failed: %s", exc)
            _store(MetricsResult(
                component="KnowledgeRAG",
                metrics={},
                errors=[str(exc)],
            ))

        # 3. ComparisonAgent
        self._log("Running ComparisonAgent evaluator...")
        try:
            if comparison_samples:
                result = ComparisonAgentEvaluator(use_ragas=self.use_ragas).evaluate(
                    comparison_samples
                )
            else:
                result = MetricsResult(
                    component="ComparisonAgent",
                    metrics={},
                    errors=["No comparison samples in this run"],
                )
            _store(result)
        except Exception as exc:
            logger.error("ComparisonAgent evaluator failed: %s", exc)
            _store(MetricsResult(
                component="ComparisonAgent",
                metrics={},
                errors=[str(exc)],
            ))

        # 4. ComplianceAgent
        self._log("Running ComplianceAgent evaluator...")
        try:
            if compliance_samples:
                result = ComplianceAgentEvaluator(use_ragas=self.use_ragas).evaluate(
                    compliance_samples
                )
            else:
                result = MetricsResult(
                    component="ComplianceAgent",
                    metrics={},
                    errors=["No compliance samples in this run"],
                )
            _store(result)
        except Exception as exc:
            logger.error("ComplianceAgent evaluator failed: %s", exc)
            _store(MetricsResult(
                component="ComplianceAgent",
                metrics={},
                errors=[str(exc)],
            ))

        # 5. GeneralQAAgent
        self._log("Running GeneralQAAgent evaluator...")
        try:
            if general_qa_samples:
                result = GeneralQAAgentEvaluator(use_ragas=self.use_ragas).evaluate(
                    general_qa_samples
                )
            else:
                result = MetricsResult(
                    component="GeneralQAAgent",
                    metrics={},
                    errors=["No general QA samples in this run"],
                )
            _store(result)
        except Exception as exc:
            logger.error("GeneralQAAgent evaluator failed: %s", exc)
            _store(MetricsResult(
                component="GeneralQAAgent",
                metrics={},
                errors=[str(exc)],
            ))

        # 6. IngestionPipeline (if chunks provided)
        if chunks is not None:
            self._log("Running IngestionPipeline evaluator...")
            try:
                result = IngestionPipelineEvaluator(vector_store=vector_store).evaluate(
                    chunks=chunks,
                    expected_doc_ids=expected_doc_ids,
                )
                _store(result)
            except Exception as exc:
                logger.error("IngestionPipeline evaluator failed: %s", exc)
                _store(MetricsResult(
                    component="IngestionPipeline",
                    metrics={},
                    errors=[str(exc)],
                ))

        # 7. System (all samples)
        self._log("Running System evaluator...")
        try:
            result = SystemEvaluator(
                use_ragas=self.use_ragas,
                latency_sla_ms=self.latency_sla_ms,
            ).evaluate(samples)
            _store(result)
        except Exception as exc:
            logger.error("System evaluator failed: %s", exc)
            _store(MetricsResult(
                component="System",
                metrics={},
                errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t0

        # Build human-readable summary
        summary_lines = [
            f"Evaluation complete in {elapsed:.1f}s | {len(samples)} samples",
        ]
        for comp, comp_result in results.items():
            top_metrics = list(comp_result.get("metrics", {}).items())[:4]
            metric_str = "  ".join(f"{k}={v:.3f}" for k, v in top_metrics)
            summary_lines.append(f"  [{comp}] {metric_str}")
        summary = "\n".join(summary_lines)

        if self.verbose:
            logger.info(summary)

        return {
            "results": results,
            "summary": summary,
            "scores": flat_scores,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Contract Intelligence evaluation suite."
    )
    p.add_argument("--dataset", required=True, help="Path to eval JSON file")
    p.add_argument("--output", default="report.json", help="Output report path")
    p.add_argument("--skip-ragas", action="store_true", help="Disable RAGAS metrics")
    p.add_argument("--sla-ms", type=float, default=2000.0, help="Latency SLA in ms")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    samples = load_dataset(args.dataset)
    logger.info("Loaded %d samples from %s", len(samples), args.dataset)

    runner = EvaluationRunner(
        use_ragas=not args.skip_ragas,
        latency_sla_ms=args.sla_ms,
        verbose=True,
    )
    report = runner.run(samples)

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(report["results"], indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Report written to %s", output_path)
    print(report["summary"])


if __name__ == "__main__":
    main()
