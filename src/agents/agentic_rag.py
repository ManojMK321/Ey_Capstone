import json
import logging
import re
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.observability.langsmith import traceable_operation
from src.observability import metrics

logger = logging.getLogger(__name__)


def _extract_response_text(response) -> str:
    if getattr(response, "output_text", None):
        return response.output_text
    if getattr(response, "output", None):
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for content_item in getattr(item, "content", []) or []:
                    if getattr(content_item, "type", None) == "output_text":
                        return getattr(content_item, "text", "")
    return ""


def _extract_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens":  getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }


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

    def __init__(self, client: OpenAI, vector_store, top_k: int = 5, model: str = "gpt-4o"):
        self.client = client
        self.vector_store = vector_store
        self.top_k = top_k
        self.model = model
        self._usage_log: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Query Analysis + Decompose
    # ------------------------------------------------------------------
    @traceable_operation(
        name="AgenticRAG query analysis",
        tags=["agentic_rag", "llm"],
        metadata={"component": "agentic_rag"},
    )
    def analyze_query(self, question: str) -> list[str]:
        prompt = (
            "You are a query analysis agent. Decompose the following user question into up to three "
            "retrieval-guided subquestions. Return each subquestion on a new line without numbering or "
            "extra commentary.\n\n"
            f"Question: {question}\n"
            "Subquestions:"
        )
        with metrics.timed_llm_call(self.model, operation="analyze_query"):
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "Decompose a user query into subquestions for retrieval."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_output_tokens=200,
            )
        self._usage_log.append(_extract_usage(response))
        output_text = _extract_response_text(response)
        lines = [line.strip("- ").strip() for line in output_text.splitlines() if line.strip()]
        return lines or [question]

    # ------------------------------------------------------------------
    # 2. Retrieve
    # ------------------------------------------------------------------
    def retrieve(self, subquestions: list[str]) -> list[dict]:
        with metrics.timed_agent_step("agentic_rag", "retrieve"):
            retrieved = []
            for sub in subquestions:
                docs = self.vector_store.similarity_search(query=sub, k=self.top_k)
                retrieved.append({"subquestion": sub, "docs": docs})
            return retrieved

    @staticmethod
    def _build_sources(retrieved: list[dict]) -> list[dict]:
        sources = []
        # Each subquestion retrieves independently, so the same passage often
        # comes back more than once across subquestions. Duplicate content
        # adds no new information to the LLM's context, so it's paid for once.
        seen_content: set[str] = set()
        for item in retrieved:
            if isinstance(item, dict) and "docs" in item:
                docs = item["docs"]
                subquestion = item.get("subquestion", "")
            else:
                docs = [item]
                subquestion = ""

            for doc in docs:
                if isinstance(doc, dict):
                    metadata = doc.get("metadata") or {}
                    source_name = metadata.get("filename") or metadata.get("source") or "document"
                    page_number = metadata.get("page")
                    if page_number is not None:
                        source_name = f"{source_name} (page {page_number})"
                    content = str(doc.get("content") or "").strip()
                else:
                    metadata = getattr(doc, "metadata", None) or {}
                    source_name = metadata.get("filename") or metadata.get("source") or "document"
                    page_number = metadata.get("page")
                    if page_number is not None:
                        source_name = f"{source_name} (page {page_number})"
                    content = getattr(doc, "page_content", "").strip()

                if content and content in seen_content:
                    continue
                seen_content.add(content)

                sources.append({
                    "subquestion": subquestion,
                    "source": source_name,
                    "content": content,
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
    @traceable_operation(
        name="AgenticRAG task routing",
        tags=["agentic_rag", "routing"],
        metadata={"component": "agentic_rag"},
    )
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
        with metrics.timed_llm_call(self.model, operation="route_task"):
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_output_tokens=150,
            )
        self._usage_log.append(_extract_usage(response))
        text = _extract_response_text(response).strip()
        if not text:
            logger.warning("Task routing returned empty text, defaulting to general.")
            return TaskRouteResult(task=TaskType.GENERAL, reason="Fallback due to empty router output")

        match = re.search(r"\{.*\}", text, re.DOTALL)
        json_text = match.group(0) if match else text

        try:
            payload = json.loads(json_text)
            return TaskRouteResult.model_validate(payload)
        except Exception as e:
            logger.exception("Task routing failed, defaulting to general: %s", e)
            return TaskRouteResult(task=TaskType.GENERAL, reason="Fallback due to parsing failure")

    # ------------------------------------------------------------------
    # 4. Specialist agents — Comparison / Compliance / General
    # ------------------------------------------------------------------
    @traceable_operation(
        name="AgenticRAG comparison agent",
        tags=["agentic_rag", "llm", "comparison"],
        metadata={"component": "agentic_rag"},
    )
    def _run_comparison_agent(self, question: str, context: str, history_section: str) -> str:
        prompt = (
            "You are the Comparison Agent for a contract intelligence system. Using only the provided "
            "passages, compare the relevant contracts/clauses/terms in the question. Clearly list "
            "similarities, differences, and a recommendation if one is asked for. Reference source names "
            "for every claim, and explicitly flag any point where the evidence is insufficient.\n\n"
            f"{history_section}"
            f"{context}\n\nQuestion: {question}\nAnswer:"
        )
        with metrics.timed_llm_call(self.model, operation="comparison_agent"):
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are an expert contract comparison assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_output_tokens=600,
            )
        self._usage_log.append(_extract_usage(response))
        return _extract_response_text(response).strip()

    @traceable_operation(
        name="AgenticRAG compliance agent",
        tags=["agentic_rag", "llm", "compliance"],
        metadata={"component": "agentic_rag"},
    )
    def _run_compliance_agent(self, question: str, context: str, history_section: str) -> str:
        prompt = (
            "You are the Compliance Agent for a contract intelligence system. Using only the provided "
            "passages, assess the compliance/regulatory/risk question carefully. Cite the specific clause "
            "or source for every claim. If the evidence does not clearly support a compliance conclusion, "
            "say so explicitly rather than speculating, since this output may inform real decisions.\n\n"
            f"{history_section}"
            f"{context}\n\nQuestion: {question}\nAnswer:"
        )
        with metrics.timed_llm_call(self.model, operation="compliance_agent"):
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are an expert contract compliance and risk assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_output_tokens=600,
            )
        self._usage_log.append(_extract_usage(response))
        return _extract_response_text(response).strip()

    @traceable_operation(
        name="AgenticRAG general agent",
        tags=["agentic_rag", "llm", "general"],
        metadata={"component": "agentic_rag"},
    )
    def _run_general_agent(self, question: str, context: str, history_section: str) -> str:
        prompt = (
            "You are an expert contract reasoning assistant. Use the provided passages to answer the "
            "user's original question. Explicitly reference the source names and verify the answer "
            "against the evidence.\n\n"
            f"{history_section}"
            f"{context}\n\nQuestion: {question}\nAnswer:"
        )
        with metrics.timed_llm_call(self.model, operation="general_agent"):
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are an expert assistant that synthesizes multiple documents and validates citations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_output_tokens=600,
            )
        self._usage_log.append(_extract_usage(response))
        return _extract_response_text(response).strip()

    @traceable_operation(
        name="AgenticRAG specialist agent",
        tags=["agentic_rag", "llm"],
        metadata={"component": "agentic_rag"},
    )
    def run_specialist_agent(self, task: TaskType, question: str, context: str, history_section: str) -> str:
        if task == TaskType.COMPARISON:
            return self._run_comparison_agent(question, context, history_section)
        if task == TaskType.COMPLIANCE:
            return self._run_compliance_agent(question, context, history_section)
        return self._run_general_agent(question, context, history_section)

    # ------------------------------------------------------------------
    # 5. Validation
    # ------------------------------------------------------------------
    @traceable_operation(
        name="AgenticRAG validation",
        tags=["agentic_rag", "llm", "validation"],
        metadata={"component": "agentic_rag"},
    )
    def validate(self, draft_answer: str, context: str) -> ValidationResult:
        system_prompt = """
You are the Validation Agent. Check the draft answer strictly against the provided context.
Flag any claim not supported by the context as an issue. Produce a corrected answer that removes
or qualifies unsupported claims, without introducing any new information not present in the context.

Return ONLY valid JSON in this exact shape:
{"is_grounded": true, "issues": [], "corrected_answer": "..."}
"""
        user_prompt = f"Context:\n{context}\n\nDraft answer:\n{draft_answer}\n\nValidate this draft."
        with metrics.timed_llm_call(self.model, operation="validate"):
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_output_tokens=700,
            )
        self._usage_log.append(_extract_usage(response))
        text = _extract_response_text(response).strip()
        try:
            return ValidationResult.model_validate(json.loads(text))
        except (ValidationError, json.JSONDecodeError) as e:
            logger.exception("Validation failed, passing draft through unmodified: %s", e)
            return ValidationResult(is_grounded=True, issues=[], corrected_answer=draft_answer)

    # ------------------------------------------------------------------
    # 6. Response (final polish)
    # ------------------------------------------------------------------
    @traceable_operation(
        name="AgenticRAG final response generation",
        tags=["agentic_rag", "llm", "response_generation"],
        metadata={"component": "agentic_rag"},
    )
    def generate_response(self, question: str, validated_answer: str) -> str:
        prompt = (
            "Take the validated answer below and present it clearly and concisely to the user, in a tone "
            "appropriate to the original question. Do not add information beyond what's given.\n\n"
            f"Original question: {question}\n\nValidated answer:\n{validated_answer}\n\nFinal response:"
        )
        with metrics.timed_llm_call(self.model, operation="generate_response"):
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are the final Response Generator in a RAG pipeline."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_output_tokens=600,
            )
        self._usage_log.append(_extract_usage(response))
        return _extract_response_text(response).strip()

    # ------------------------------------------------------------------
    # Orchestration entry point
    # ------------------------------------------------------------------
    def run(self, question: str, history: str | None = None) -> dict:
        with metrics.timed_agent_step("agentic_rag", "run"):
            self._usage_log = []
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
                "input_tokens":      sum(u["input_tokens"] for u in self._usage_log),
                "output_tokens":     sum(u["output_tokens"] for u in self._usage_log),
            }
