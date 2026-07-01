"""
evaluation/ingestion_pipeline.py
---------------------------------
Evaluator for the document ingestion pipeline (chunking quality, embedding coverage).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from evaluation.base import EvalSample, MetricsResult, SemanticSimilarity

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100) * (len(sorted_v) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


class IngestionPipelineEvaluator:

    def __init__(
        self,
        vector_store=None,
        min_chunk: int = 100,
        max_chunk: int = 1200,
    ):
        self.vector_store = vector_store
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk

    def evaluate(
        self,
        chunks: list,
        expected_doc_ids: list[str] | None = None,
    ) -> MetricsResult:
        t0 = time.perf_counter()
        errors: list[str] = []
        metrics: dict[str, float] = {}
        details: dict[str, Any] = {}

        if not chunks:
            return MetricsResult(
                component="IngestionPipeline",
                metrics={"total_chunks": 0.0, "total_docs": 0.0},
                details=details,
                errors=["No chunks provided"],
                elapsed_s=0.0,
            )

        # ---- Basic stats ----
        total_chunks = len(chunks)
        metrics["total_chunks"] = float(total_chunks)

        chunk_sizes = []
        doc_ids_seen: set[str] = set()
        pages_seen: set[str] = set()
        chunks_by_doc: dict[str, list] = {}
        metadata_accurate_count = 0

        for chunk in chunks:
            text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            meta = chunk.metadata if hasattr(chunk, "metadata") else {}
            size = len(text)
            chunk_sizes.append(size)

            doc_id = str(meta.get("doc_id", ""))
            page = str(meta.get("page", ""))
            doc_ids_seen.add(doc_id)

            if doc_id and page:
                pages_seen.add(f"{doc_id}::{page}")

            if doc_id:
                chunks_by_doc.setdefault(doc_id, []).append(chunk)

            # metadata accuracy: chunk_size field matches actual length ± 5
            reported_size = meta.get("chunk_size")
            if reported_size is not None:
                if abs(int(reported_size) - size) <= 5:
                    metadata_accurate_count += 1

        metrics["total_docs"] = float(len(chunks_by_doc))
        metrics["avg_chunk_size"] = sum(chunk_sizes) / len(chunk_sizes)
        metrics["avg_chunks_per_doc"] = (
            total_chunks / len(chunks_by_doc) if chunks_by_doc else 0.0
        )
        metrics["undersized_chunk_rate"] = sum(
            1 for s in chunk_sizes if s < self.min_chunk
        ) / total_chunks
        metrics["oversized_chunk_rate"] = sum(
            1 for s in chunk_sizes if s > self.max_chunk
        ) / total_chunks
        metrics["unique_pages_indexed"] = float(len(pages_seen))

        # metadata accuracy
        chunks_with_size_meta = sum(
            1 for c in chunks
            if (c.metadata if hasattr(c, "metadata") else {}).get("chunk_size") is not None
        )
        metrics["chunk_size_metadata_accuracy"] = (
            metadata_accurate_count / chunks_with_size_meta
            if chunks_with_size_meta > 0 else 1.0
        )

        # ---- Embedding coverage ----
        if expected_doc_ids:
            expected_set = set(expected_doc_ids)
            covered = expected_set & doc_ids_seen
            metrics["embedding_coverage"] = len(covered) / len(expected_set)
            orphan_chunks = sum(
                1 for c in chunks
                if str((c.metadata if hasattr(c, "metadata") else {}).get("doc_id", ""))
                not in expected_set
            )
            metrics["orphan_chunk_rate"] = orphan_chunks / total_chunks
        else:
            metrics["embedding_coverage"] = 1.0
            metrics["orphan_chunk_rate"] = 0.0

        # ---- Chunk Boundary Quality (CBQ) ----
        cbq_per_doc: dict[str, dict] = {}
        intra_scores_global: list[float] = []
        inter_scores_global: list[float] = []

        for doc_id, doc_chunks in chunks_by_doc.items():
            # Sort by chunk_index
            try:
                doc_chunks_sorted = sorted(
                    doc_chunks,
                    key=lambda c: (c.metadata if hasattr(c, "metadata") else {}).get(
                        "chunk_index", 0
                    ),
                )
            except Exception:
                doc_chunks_sorted = doc_chunks

            if len(doc_chunks_sorted) < 2:
                cbq_per_doc[doc_id] = {
                    "intra": 0.0, "inter": 0.0, "cbq": 0.0,
                    "n_chunks": len(doc_chunks_sorted),
                }
                continue

            # Compute intra-chunk coherence
            intra_scores: list[float] = []
            for chunk in doc_chunks_sorted:
                text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                sents = _split_sentences(text)
                if len(sents) >= 2:
                    pairs = [(sents[i], sents[i + 1]) for i in range(len(sents) - 1)]
                    try:
                        scores = SemanticSimilarity.batch_score(pairs)
                        intra_scores.extend(scores)
                    except Exception as exc:
                        logger.warning("Intra-chunk similarity failed: %s", exc)

            avg_intra = sum(intra_scores) / len(intra_scores) if intra_scores else 0.0
            intra_scores_global.extend(intra_scores)

            # Compute inter-chunk separation
            inter_scores: list[float] = []
            for i in range(len(doc_chunks_sorted) - 1):
                text_a = (
                    doc_chunks_sorted[i].page_content
                    if hasattr(doc_chunks_sorted[i], "page_content")
                    else str(doc_chunks_sorted[i])
                )
                text_b = (
                    doc_chunks_sorted[i + 1].page_content
                    if hasattr(doc_chunks_sorted[i + 1], "page_content")
                    else str(doc_chunks_sorted[i + 1])
                )
                sents_a = _split_sentences(text_a)
                sents_b = _split_sentences(text_b)
                if sents_a and sents_b:
                    last_sent = sents_a[-1]
                    first_sent = sents_b[0]
                    try:
                        sim = SemanticSimilarity.score(last_sent, first_sent)
                        inter_scores.append(sim)
                    except Exception as exc:
                        logger.warning("Inter-chunk similarity failed: %s", exc)

            avg_inter = sum(inter_scores) / len(inter_scores) if inter_scores else 0.0
            inter_scores_global.extend(inter_scores)

            cbq = max(0.0, min(1.0, avg_intra - avg_inter))
            cbq_per_doc[doc_id] = {
                "intra": avg_intra,
                "inter": avg_inter,
                "cbq": cbq,
                "n_chunks": len(doc_chunks_sorted),
            }

        # Global CBQ
        cbq_values = [v["cbq"] for v in cbq_per_doc.values()]
        metrics["chunk_boundary_quality"] = (
            sum(cbq_values) / len(cbq_values) if cbq_values else 0.0
        )
        metrics["intra_chunk_coherence"] = (
            sum(intra_scores_global) / len(intra_scores_global)
            if intra_scores_global else 0.0
        )
        metrics["inter_chunk_separation"] = (
            sum(inter_scores_global) / len(inter_scores_global)
            if inter_scores_global else 0.0
        )

        details["cbq_per_doc"] = cbq_per_doc
        details["size_distribution"] = {
            "min": float(min(chunk_sizes)),
            "max": float(max(chunk_sizes)),
            "p25": _percentile(chunk_sizes, 25),
            "p50": _percentile(chunk_sizes, 50),
            "p75": _percentile(chunk_sizes, 75),
        }

        elapsed = time.perf_counter() - t0
        return MetricsResult(
            component="IngestionPipeline",
            metrics=metrics,
            details=details,
            errors=errors,
            elapsed_s=elapsed,
        )
