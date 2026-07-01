"""
evaluation/base.py
------------------
Core data structures and shared utility classes for the evaluation suite.
"""
from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    query: str
    answer: str = ""
    ground_truth: str = ""
    contexts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    expected_intent: str | None = None
    predicted_intent: str | None = None
    expected_task: str | None = None
    predicted_task: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    expected_tool_calls: list[str] = field(default_factory=list)
    expected_issues: list[str] = field(default_factory=list)
    predicted_issues: list[str] = field(default_factory=list)
    expected_severity: str | None = None
    predicted_severity: str | None = None
    response_time_ms: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MetricsResult:
    component: str
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def summary(self) -> str:
        lines = [f"[{self.component}] elapsed={self.elapsed_s:.2f}s"]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        if self.errors:
            lines.append(f"  errors: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "metrics": self.metrics,
            "details": self.details,
            "errors": self.errors,
            "elapsed_s": self.elapsed_s,
        }


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def Timer():
    """Yields a dict with key 'elapsed' populated after the block exits."""
    state = {"elapsed": 0.0}
    t0 = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# SemanticSimilarity
# ---------------------------------------------------------------------------

class SemanticSimilarity:
    """
    Cosine similarity between text pairs using sentence-transformers.
    Falls back to Jaccard token overlap if the library is unavailable.
    """

    _model = None
    _model_name = "all-MiniLM-L6-v2"
    _available: bool | None = None

    @classmethod
    def _ensure_model(cls) -> bool:
        if cls._available is not None:
            return cls._available
        try:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(cls._model_name)
            cls._available = True
        except Exception as exc:
            logger.warning("sentence-transformers unavailable, using Jaccard: %s", exc)
            cls._available = False
        return cls._available

    @classmethod
    def score(cls, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if cls._ensure_model():
            try:
                import numpy as np
                vecs = cls._model.encode([a, b], convert_to_numpy=True)
                num = float(np.dot(vecs[0], vecs[1]))
                denom = float(np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
                return float(num / denom) if denom > 0 else 0.0
            except Exception as exc:
                logger.warning("SemanticSimilarity encode failed: %s", exc)
        return cls._jaccard(a, b)

    @classmethod
    def batch_score(cls, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        if cls._ensure_model():
            try:
                import numpy as np
                texts_a = [p[0] for p in pairs]
                texts_b = [p[1] for p in pairs]
                all_texts = texts_a + texts_b
                vecs = cls._model.encode(all_texts, convert_to_numpy=True)
                n = len(pairs)
                scores = []
                for i in range(n):
                    va, vb = vecs[i], vecs[n + i]
                    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
                    scores.append(float(np.dot(va, vb) / denom) if denom > 0 else 0.0)
                return scores
            except Exception as exc:
                logger.warning("SemanticSimilarity batch_score failed: %s", exc)
        return [cls._jaccard(a, b) for a, b in pairs]

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a and not tokens_b:
            return 1.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# CitationChecker
# ---------------------------------------------------------------------------

# Matches: [doc_id=X, page=N]  or  [doc_id=X,page=N]  (spaces optional)
_CITATION_RE = re.compile(
    r"\[doc_id\s*=\s*([^,\]]+?)\s*,\s*page\s*=\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)


class CitationChecker:
    """
    Parse inline citations in the format [doc_id=X, page=N] from answer text
    and score them against the actual retrieved sources.
    """

    @staticmethod
    def parse_citations(text: str) -> list[dict]:
        """Return list of {'doc_id': ..., 'page': ...} dicts found in text."""
        matches = _CITATION_RE.findall(text)
        return [{"doc_id": m[0].strip(), "page": m[1].strip()} for m in matches]

    @staticmethod
    def score(answer: str, sources: list[dict]) -> dict:
        """
        Returns:
            citation_precision  — fraction of cited refs that match a source
            citation_recall     — fraction of sources that were cited
            citation_f1         — harmonic mean
            n_cited             — number of citations found in answer
            n_valid             — number of valid citations
            n_sources           — number of sources
        """
        cited = CitationChecker.parse_citations(answer)
        n_cited = len(cited)
        n_sources = len(sources)

        # Build a set of normalised (doc_id, page) from sources
        source_keys = set()
        for s in sources:
            doc_id = str(s.get("doc_id", "")).strip()
            page = str(s.get("page", "")).strip()
            source_keys.add((doc_id, page))

        # Count valid citations (those that match a source)
        n_valid = 0
        matched_sources: set[tuple] = set()
        for c in cited:
            key = (c["doc_id"], c["page"])
            if key in source_keys:
                n_valid += 1
                matched_sources.add(key)

        citation_precision = n_valid / n_cited if n_cited > 0 else 0.0
        citation_recall = len(matched_sources) / n_sources if n_sources > 0 else 0.0
        if citation_precision + citation_recall > 0:
            citation_f1 = (
                2 * citation_precision * citation_recall
                / (citation_precision + citation_recall)
            )
        else:
            citation_f1 = 0.0

        return {
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "citation_f1": citation_f1,
            "n_cited": n_cited,
            "n_valid": n_valid,
            "n_sources": n_sources,
        }


# ---------------------------------------------------------------------------
# RagasRunner
# ---------------------------------------------------------------------------

class RagasRunner:
    """
    Thin wrapper around the RAGAS 0.2 evaluation API.
    Each metric is run independently so one failure does not block others.
    """

    METRIC_MAP = {
        "faithfulness":        "Faithfulness",
        "context_precision":   "ContextPrecision",
        "context_recall":      "ContextRecall",
        "answer_relevancy":    "AnswerRelevancy",
        "answer_correctness":  "AnswerCorrectness",
    }

    @staticmethod
    def _make_llm():
        try:
            from ragas.llms import LangchainLLMWrapper
            from langchain_openai import ChatOpenAI
            import os
            from dotenv import load_dotenv
            load_dotenv()
            return LangchainLLMWrapper(
                ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            )
        except Exception as exc:
            logger.warning("Could not create RAGAS LLM wrapper: %s", exc)
            return None

    @staticmethod
    def _make_embeddings():
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import OpenAIEmbeddings
            import os
            from dotenv import load_dotenv
            load_dotenv()
            return LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
            )
        except Exception as exc:
            logger.warning("Could not create RAGAS embeddings wrapper: %s", exc)
            return None

    @staticmethod
    def run(
        samples: list[EvalSample],
        metric_names: list[str],
        client=None,
    ) -> dict[str, float]:
        """
        Run RAGAS metrics. Returns dict mapping metric_name -> float score.
        Any metric that fails returns 0.0.
        """
        if not samples:
            return {m: 0.0 for m in metric_names}

        try:
            from ragas import evaluate
            from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
            import ragas.metrics as _rm
        except ImportError as exc:
            logger.warning("RAGAS not installed: %s", exc)
            return {m: 0.0 for m in metric_names}

        ragas_llm = RagasRunner._make_llm()
        ragas_embeddings = RagasRunner._make_embeddings()

        if ragas_llm is None:
            logger.warning("RAGAS LLM unavailable; returning 0.0 for all RAGAS metrics.")
            return {m: 0.0 for m in metric_names}

        # Build dataset
        ragas_samples = []
        for s in samples:
            try:
                ragas_samples.append(
                    SingleTurnSample(
                        user_input=s.query,
                        response=s.answer if s.answer else " ",
                        retrieved_contexts=s.contexts if s.contexts else [""],
                        reference=s.ground_truth if s.ground_truth else " ",
                    )
                )
            except Exception as exc:
                logger.warning("Could not create SingleTurnSample: %s", exc)

        if not ragas_samples:
            return {m: 0.0 for m in metric_names}

        dataset = EvaluationDataset(samples=ragas_samples)

        results: dict[str, float] = {}
        for metric_name in metric_names:
            try:
                cls_name = RagasRunner.METRIC_MAP.get(metric_name)
                if cls_name is None:
                    logger.warning("Unknown RAGAS metric: %s", metric_name)
                    results[metric_name] = 0.0
                    continue

                MetricCls = getattr(_rm, cls_name)
                # Try with embeddings first, fall back to llm-only
                try:
                    if ragas_embeddings is not None:
                        metric_obj = MetricCls(llm=ragas_llm, embeddings=ragas_embeddings)
                    else:
                        metric_obj = MetricCls(llm=ragas_llm)
                except TypeError:
                    try:
                        metric_obj = MetricCls(llm=ragas_llm)
                    except TypeError:
                        metric_obj = MetricCls()

                result = evaluate(dataset=dataset, metrics=[metric_obj])
                # result is a dict-like object; extract scalar
                score = result.get(metric_name, None)
                if score is None:
                    # Try alternate key formats
                    for k, v in result.items():
                        if metric_name in k.lower():
                            score = v
                            break
                if score is None:
                    score = 0.0
                results[metric_name] = float(score) if score == score else 0.0  # NaN guard
            except Exception as exc:
                logger.warning("RAGAS metric '%s' failed: %s", metric_name, exc)
                results[metric_name] = 0.0

        return results
