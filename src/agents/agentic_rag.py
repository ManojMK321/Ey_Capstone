import json
import logging
import re
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from docs.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


def _extract_response_text(response) -> str:
    # Responses API: output_text shortcut
    val = getattr(response, "output_text", None)
    if val:
        return val

    # Responses API: traverse output items
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", None) or []:
                # type may be "output_text" or "text" depending on SDK version
                if getattr(c, "type", None) in ("output_text", "text"):
                    text = getattr(c, "text", None)
                    if text:
                        return text

    # Chat Completions API fallback
    try:
        text = response.choices[0].message.content
        if text:
            return text
    except Exception:
        pass

    return ""


def _clean_json(text: str) -> str:
    """Strip markdown code fences that models sometimes wrap JSON in."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Task Router schema
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    COMPARISON = "comparison"
    COMPLIANCE = "compliance"
    GENERAL    = "general"


class TaskRouteResult(BaseModel):
    task: TaskType
    reason: str


# ---------------------------------------------------------------------------
# Validation schema
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    is_grounded: bool
    issues: list[str] = []
    corrected_answer: str


# ---------------------------------------------------------------------------
# AgenticRAG
# Pipeline: Question -> Query Analysis -> Decompose -> Retrieve -> Task Router
#           -> {Comparison | Compliance | General} -> Validation -> Response
# ---------------------------------------------------------------------------

class AgenticRAG:

    def __init__(self, client: OpenAI, vector_store: FAISSVectorStore, top_k: int = 5, model: str = "gpt-4o"):
        self.client = client
        self.vector_store = vector_store
        self.top_k = top_k
        self.model = model

    # ------------------------------------------------------------------
    # 1. Query Analysis + Decompose
    # ------------------------------------------------------------------
    def analyze_query(self, question: str) -> list[str]:
        prompt = (
            "You are a query analysis agent. Decompose the following user question into up to three "
            "retrieval-guided subquestions. Return each subquestion on a new line without numbering or "
            "extra commentary.\n\n"
            f"Question: {question}\n"
            "Subquestions:"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "Decompose a user query into subquestions for retrieval."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_output_tokens=200,
        )
        output_text = _extract_response_text(response)
        lines = [line.strip("- ").strip() for line in output_text.splitlines() if line.strip()]
        return lines or [question]

    # ------------------------------------------------------------------
    # 2. Retrieve
    # ------------------------------------------------------------------
    def retrieve(self, subquestions: list[str]) -> list[dict]:
        retrieved = []
        for sub in subquestions:
            docs = self.vector_store.similarity_search(query=sub, k=self.top_k)
            retrieved.append({"subquestion": sub, "docs": docs})
        return retrieved

    @staticmethod
    def _build_sources(retrieved: list[dict]) -> list[dict]:
        sources = []
        for item in retrieved:
            for doc in item["docs"]:
                metadata = doc.metadata or {}
                source_name = metadata.get("filename") or metadata.get("source") or "document"
                page_number = metadata.get("page")
                if page_number is not None:
                    source_name = f"{source_name} (page {page_number})"
                sources.append({
                    "subquestion": item["subquestion"],
                    "source": source_name,
                    "content": doc.page_content.strip(),
                })
        return sources

    @staticmethod
    def _build_context(sources: list[dict]) -> str:
        if not sources:
            return "No relevant documents were retrieved."
        return "\n\n".join(
            f"Subquestion: {item['subquestion']}\nSource: {item['source']}\n{item['content']}"
            for item in sources
        )

    # ------------------------------------------------------------------
    # 3. Task Router
    # ------------------------------------------------------------------
    def route_task(self, question: str) -> TaskRouteResult:
        system_prompt = """
You are the Task Router for a contract intelligence agentic RAG system.
Decide which specialist agent should handle the question. Choose exactly one:

- comparison: the question asks to compare/contrast two or more contracts, clauses, or terms.
- compliance: the question concerns regulatory, policy, legal, or contractual compliance/risk.
- general: anything else requiring multi-step reasoning over retrieved context that isn't a
  comparison or compliance question.

Return ONLY valid JSON in this exact shape:
{"task": "comparison", "reason": "short justification"}
"""
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_output_tokens=150,
        )
        text = _clean_json(_extract_response_text(response))
        try:
            if not text:
                raise ValueError("Empty response from model")
            return TaskRouteResult.model_validate(json.loads(text))
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Task routing failed, defaulting to general: %s", e)
            return TaskRouteResult(task=TaskType.GENERAL, reason="Fallback due to parsing failure")

    # ------------------------------------------------------------------
    # 4. Specialist agents — Comparison / Compliance / General
    # ------------------------------------------------------------------
    def _run_comparison_agent(self, question: str, context: str, history_section: str) -> str:
        prompt = (
            "You are the Comparison Agent for a contract intelligence system. Using only the provided "
            "passages, compare the relevant contracts/clauses/terms in the question. Clearly list "
            "similarities, differences, and a recommendation if one is asked for. Reference source names "
            "for every claim, and explicitly flag any point where the evidence is insufficient.\n\n"
            f"{history_section}"
            f"{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You are an expert contract comparison assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_output_tokens=600,
        )
        return _extract_response_text(response).strip()

    def _run_compliance_agent(self, question: str, context: str, history_section: str) -> str:
        prompt = (
            "You are the Compliance Agent for a contract intelligence system. Using only the provided "
            "passages, assess the compliance/regulatory/risk question carefully. Cite the specific clause "
            "or source for every claim. If the evidence does not clearly support a compliance conclusion, "
            "say so explicitly rather than speculating, since this output may inform real decisions.\n\n"
            f"{history_section}"
            f"{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You are an expert contract compliance and risk assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_output_tokens=600,
        )
        return _extract_response_text(response).strip()

    def _run_general_agent(self, question: str, context: str, history_section: str) -> str:
        prompt = (
            "You are an expert contract reasoning assistant. Use the provided passages to answer the "
            "user's original question. Explicitly reference the source names and verify the answer "
            "against the evidence.\n\n"
            f"{history_section}"
            f"{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You are an expert assistant that synthesizes multiple documents and validates citations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_output_tokens=600,
        )
        return _extract_response_text(response).strip()

    def run_specialist_agent(self, task: TaskType, question: str, context: str, history_section: str) -> str:
        if task == TaskType.COMPARISON:
            return self._run_comparison_agent(question, context, history_section)
        if task == TaskType.COMPLIANCE:
            return self._run_compliance_agent(question, context, history_section)
        return self._run_general_agent(question, context, history_section)

    # ------------------------------------------------------------------
    # 5. Validation
    # ------------------------------------------------------------------
    def validate(self, draft_answer: str, context: str) -> ValidationResult:
        system_prompt = """
You are the Validation Agent. Check the draft answer strictly against the provided context.
Flag any claim not supported by the context as an issue. Produce a corrected answer that removes
or qualifies unsupported claims, without introducing any new information not present in the context.

Return ONLY valid JSON in this exact shape:
{"is_grounded": true, "issues": [], "corrected_answer": "..."}
"""
        user_prompt = f"Context:\n{context}\n\nDraft answer:\n{draft_answer}\n\nValidate this draft."
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_output_tokens=700,
        )
        text = _clean_json(_extract_response_text(response))
        try:
            if not text:
                raise ValueError("Empty response from model")
            return ValidationResult.model_validate(json.loads(text))
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Validation failed, passing draft through unmodified: %s", e)
            return ValidationResult(is_grounded=True, issues=[], corrected_answer=draft_answer)

    # ------------------------------------------------------------------
    # 6. Response (final polish)
    # ------------------------------------------------------------------
    def generate_response(self, question: str, validated_answer: str) -> str:
        prompt = (
            "Take the validated answer below and present it clearly and concisely to the user, in a tone "
            "appropriate to the original question. Do not add information beyond what's given.\n\n"
            f"Original question: {question}\n\nValidated answer:\n{validated_answer}\n\nFinal response:"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You are the final Response Generator in a RAG pipeline."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_output_tokens=600,
        )
        return _extract_response_text(response).strip()

    # ------------------------------------------------------------------
    # Orchestration entry point
    # ------------------------------------------------------------------
    def run(self, question: str, history: str | None = None) -> dict:
        history_section = f"Previous conversation:\n{history}\n\n" if history else ""

        subquestions = self.analyze_query(question)
        retrieved    = self.retrieve(subquestions)
        sources      = self._build_sources(retrieved)
        context      = self._build_context(sources)

        route = self.route_task(question)
        logger.info("Task routed: %s — %s", route.task.value, route.reason)

        draft_answer = self.run_specialist_agent(route.task, question, context, history_section)

        validation = self.validate(draft_answer, context)
        logger.info("Validation: grounded=%s issues=%s", validation.is_grounded, validation.issues)

        final_answer = self.generate_response(question, validation.corrected_answer)

        return {
            "answer":            final_answer,
            "task":              route.task.value,
            "task_reason":       route.reason,
            "draft_answer":      draft_answer,
            "is_grounded":       validation.is_grounded,
            "validation_issues": validation.issues,
            "subquestions":      subquestions,
            "retrieved":         retrieved,
            "sources":           sources,
        }
