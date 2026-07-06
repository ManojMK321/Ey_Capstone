"""
ragas_judge.py
--------------
Single-turn RAGAS scoring for the live chat interface.

Scores exactly one real user query + the RAG-generated answer + the chunks
retrieved for it

    Faithfulness        — is the answer grounded in the retrieved chunks?
    Answer Relevancy    — does the answer actually address the question?
    Context Precision   — are the retrieved chunks relevant to the question?
    Context Recall      — do the retrieved chunks contain what's needed to
                           support the answer? (No ground truth exists for a
                           live turn, so the generated answer itself stands
                           in as the reference — same reference-free spirit
                           as the other three metrics.)

The judge LLM is OpenAI's gpt-4o-mini, called through the official `openai`
SDK via ragas.llms.llm_factory(provider="openai"). The same OpenAI key also
covers the embedding similarity step Answer Relevancy needs internally.

Requires OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types as _types
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# ragas 0.4.x compatibility shim — must run before any `import ragas`.
# See metrics.py for the full explanation (langchain-community 0.4+ removed
# the vertexai chat model that ragas imports unconditionally at startup).
# ---------------------------------------------------------------------------
_VERTEXAI_KEY = "langchain_community.chat_models.vertexai"
if _VERTEXAI_KEY not in sys.modules:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ImportError:
        _stub = _types.ModuleType(_VERTEXAI_KEY)

        class _ChatVertexAI:  # noqa: N801
            pass

        _stub.ChatVertexAI = _ChatVertexAI
        sys.modules[_VERTEXAI_KEY] = _stub

JUDGE_MODEL = "gpt-4o-mini"

# ContextPrecisionWithoutReference and AnswerRelevancy each call the judge LLM
# in a sequential (non-parallel) loop internally — once per chunk, and once
# per `strictness` sample, respectively. Capping both keeps a single
# evaluation to ~5-6 round trips instead of scaling linearly with retrieval
# depth or sample count.
MAX_CONTEXT_CHUNKS = 3
ANSWER_RELEVANCY_STRICTNESS = 1


@dataclass
class TurnRagasScore:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


_judge_llm = None
_judge_embeddings = None


def _get_judge():
    """Lazily build (and cache) the OpenAI-judge LLM + embeddings."""
    global _judge_llm, _judge_embeddings
    if _judge_llm is not None:
        return _judge_llm, _judge_embeddings

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in .env — required to run the RAGAS judge."
        )

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory

    async_openai_client = AsyncOpenAI(api_key=openai_key)
    _judge_llm = llm_factory(JUDGE_MODEL, provider="openai", client=async_openai_client)
    _judge_embeddings = embedding_factory(
        "openai", "text-embedding-3-small", client=async_openai_client
    )

    return _judge_llm, _judge_embeddings


async def _ascore_turn(question: str, answer: str, contexts: list[str]) -> TurnRagasScore:
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        ContextRecall,
    )

    llm, embeddings = _get_judge()
    safe_contexts = (contexts or [answer])[:MAX_CONTEXT_CHUNKS]

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(
        llm=llm, embeddings=embeddings, strictness=ANSWER_RELEVANCY_STRICTNESS
    )
    context_precision = ContextPrecisionWithoutReference(llm=llm)
    context_recall = ContextRecall(llm=llm)

    faith_result, relevancy_result, precision_result, recall_result = await asyncio.gather(
        faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=safe_contexts),
        answer_relevancy.ascore(user_input=question, response=answer),
        context_precision.ascore(user_input=question, response=answer, retrieved_contexts=safe_contexts),
        context_recall.ascore(user_input=question, retrieved_contexts=safe_contexts, reference=answer),
    )

    return TurnRagasScore(
        faithfulness=faith_result.value,
        answer_relevancy=relevancy_result.value,
        context_precision=precision_result.value,
        context_recall=recall_result.value,
    )


def score_turn(question: str, answer: str, contexts: list[str]) -> TurnRagasScore:
    """Score one live chat turn with gpt-4o-mini as the RAGAS judge."""
    return asyncio.run(_ascore_turn(question, answer, contexts))
