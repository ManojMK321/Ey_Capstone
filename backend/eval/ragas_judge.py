"""
ragas_judge.py
--------------
Formula-based RAGAS scoring for the live chat interface.

Scores exactly one real user query + the RAG-generated answer + the chunks
retrieved for it. Every metric is computed from OpenAI embedding cosine
similarity — no judge LLM call — since the LLM-as-judge version needed
5-6 sequential model round trips per evaluation and was too slow for
interactive use.

    Faithfulness        — fraction of answer sentences whose embedding is
                           close (>= GROUNDING_THRESHOLD cosine similarity)
                           to at least one retrieved chunk.
    Answer Relevancy    — cosine similarity between the question embedding
                           and the answer embedding.
    Context Precision   — rank-weighted average precision (same formula
                           RAGAS's own non-LLM metric uses) over the
                           retrieved chunks, treating cosine similarity to
                           the question >= GROUNDING_THRESHOLD as "relevant".
    Context Recall       — same grounding computation as Faithfulness. No
                           ground truth exists for a live turn, so the
                           generated answer stands in as the reference —
                           same reference-free spirit as the other metrics.

Requires OPENAI_API_KEY in .env (used only to embed text, no LLM calls).

Calibration note: with text-embedding-3-small, a faithfully *paraphrased*
sentence typically scores only ~0.45-0.65 cosine similarity against its
source chunk — nowhere near 0.75+, which is reserved for near-verbatim
restatements. GROUNDING_THRESHOLD is set below that paraphrase band so
real RAG answers don't read as unfaithful just for being reworded.

This also means these metrics measure topical/semantic grounding, not
factual entailment: a fabricated detail on the same topic as the
retrieved chunk (e.g. a wrong dollar figure) can still land above the
threshold, since cosine similarity can't verify specific facts the way
an NLI/LLM judge can. Use these scores as a fast grounding signal, not
a hallucination detector for precise figures.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GROUNDING_THRESHOLD = 0.40

# Contract/corporate text is full of abbreviations ("Inc.", "Corp.", "vs.",
# "e.g.") and numbered clauses ("1.", "2."). Splitting naively on ". " breaks
# these into meaningless fragments (e.g. "American Water Works Company, Inc."
# as its own "sentence") that can't match any context chunk, which drags
# faithfulness/context_recall down even for fully grounded answers.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "jr", "sr", "st", "vs", "etc", "eg", "ie",
    "e.g", "i.e", "inc", "ltd", "co", "corp", "no", "pp", "al", "fig", "sec",
    "art", "rev", "dept", "approx", "est", "misc", "vol", "u.s", "u.k", "ph.d",
}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TRAILING_TOKEN_RE = re.compile(r"([A-Za-z0-9.]+)$")


def _ends_with_abbreviation(fragment: str) -> bool:
    match = _TRAILING_TOKEN_RE.search(fragment.strip())
    if not match:
        return False
    token = match.group(1).rstrip(".").lower()
    if not token:
        return False
    if token.isdigit():
        return True  # numbered clause marker, e.g. "1."
    return token in _ABBREVIATIONS or len(token) == 1  # incl. single initials


@dataclass
class TurnRagasScore:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


_embedder = None


def _get_embedder():
    """Lazily build (and cache) the OpenAI embedding client."""
    global _embedder
    if _embedder is not None:
        return _embedder

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in .env — required to embed text for RAGAS scoring."
        )

    from openai import OpenAI
    _embedder = OpenAI(api_key=openai_key)
    return _embedder


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return [text]

    parts = [p for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    sentences: list[str] = []
    buffer = ""
    for part in parts:
        buffer = f"{buffer} {part}".strip() if buffer else part
        if _ends_with_abbreviation(buffer):
            continue  # false sentence break (abbreviation/list marker) — keep merging
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)
    return sentences or [text]


def _cosine(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    return float(np.dot(a_arr, b_arr) / denom) if denom else 0.0


def _average_precision(verdicts: list[int]) -> float:
    denominator = sum(verdicts) + 1e-10
    numerator = sum(
        (sum(verdicts[: i + 1]) / (i + 1)) * verdicts[i] for i in range(len(verdicts))
    )
    return numerator / denominator


def _grounding_ratio(sentence_vecs: list, context_vecs: list) -> float | None:
    if not sentence_vecs or not context_vecs:
        return None
    grounded = sum(
        1 for vec in sentence_vecs
        if max(_cosine(vec, ctx_vec) for ctx_vec in context_vecs) >= GROUNDING_THRESHOLD
    )
    return grounded / len(sentence_vecs)


def score_turn(question: str, answer: str, contexts: list[str]) -> TurnRagasScore:
    """Score one live chat turn via embedding-similarity formulas (no judge LLM)."""
    embedder = _get_embedder()
    safe_contexts = contexts or [answer]
    answer_sentences = _split_sentences(answer)

    texts = [question, answer, *answer_sentences, *safe_contexts]
    response = embedder.embeddings.create(model="text-embedding-3-small", input=texts)
    vectors = [item.embedding for item in response.data]

    question_vec = vectors[0]
    answer_vec = vectors[1]
    sentence_vecs = vectors[2: 2 + len(answer_sentences)]
    context_vecs = vectors[2 + len(answer_sentences):]

    faithfulness = _grounding_ratio(sentence_vecs, context_vecs)
    context_recall = faithfulness

    answer_relevancy = _cosine(question_vec, answer_vec)

    if context_vecs:
        verdicts = [
            1 if _cosine(question_vec, ctx_vec) >= GROUNDING_THRESHOLD else 0
            for ctx_vec in context_vecs
        ]
        context_precision = _average_precision(verdicts)
    else:
        context_precision = None

    return TurnRagasScore(
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
    )
