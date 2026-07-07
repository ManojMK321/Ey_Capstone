import json
import logging
import os
import time
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.observability.langsmith import traceable_operation
from src.observability import metrics

logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Workflow(str, Enum):
    KNOWLEDGE_RAG = "KnowledgeRAG"
    AGENTIC_RAG = "AgenticRAG"


class TaskType(str, Enum):
    LOOKUP = "lookup"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    COMPLIANCE = "compliance"
    REASONING = "reasoning"
    RISK_ANALYSIS = "risk_analysis"
    MULTI_STEP = "multi_step"


class IntentResult(BaseModel):
    workflow: Workflow
    task: TaskType
    reason: str
    confidence: float


# JSON Schema passed to the OpenAI API so the model is constrained at
# generation time to only emit valid enum values.
INTENT_JSON_SCHEMA = {
    "name": "intent_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "enum": [w.value for w in Workflow],
            },
            "task": {
                "type": "string",
                "enum": [t.value for t in TaskType],
            },
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["workflow", "task", "reason", "confidence"],
        "additionalProperties": False,
    },
}


class IntentDetector:

    def __init__(self, model: str = "gpt-4.1", temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    @traceable_operation(
        name="Intent detection",
        tags=["intent", "workflow"],
        metadata={"component": "intent_detector"},
    )
    def detect(self, query: str) -> IntentResult:
        system_prompt = """
You are an Intent Detection Agent for a Contract Intelligence System.

Your ONLY responsibility is deciding which workflow should process
the user's question.

Never answer the question.

There are only TWO workflows.

1. KnowledgeRAG

Choose this when the question is

- factual lookup
- clause lookup
- definition
- payment term
- governing law
- notice period
- contract duration
- single document summary
- direct question

Examples

What is the payment term?

Show clause 7.

Explain force majeure.

What is the governing law?


2. AgenticRAG

Choose this when the question requires

- comparison
- compliance
- reasoning
- multiple retrievals
- multi-step reasoning
- cross-document reasoning
- risk analysis

Examples

Compare payment terms across contracts.

Compare termination clauses.

Which contracts expire within 90 days?

Which contracts have unlimited liability?

Find contracts expiring next month and summarize renewal clauses.

Is this contract compliant with procurement policy?

IMPORTANT RULES FOR THE "task" FIELD:

- "task" must be EXACTLY ONE of: lookup, summary, comparison, compliance, reasoning, risk_analysis, multi_step.
- NEVER combine two task labels (e.g. never output "compliance reasoning" or "comparison and reasoning").
- If a question seems to involve more than one of these, choose the SINGLE BEST/PRIMARY one. Use "multi_step" as the
  label for questions that genuinely require several different operations combined, rather than concatenating labels.

Return ONLY valid JSON matching the schema you were given.
"""
        start = time.perf_counter()
        with metrics.timed_llm_call(self.model, operation="intent_detection"):
            response = self.client.responses.create(
                model=self.model,
                temperature=self.temperature,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        **INTENT_JSON_SCHEMA,
                    }
                },
            )
        duration = time.perf_counter() - start

        text = response.output_text.strip()

        try:
            result = IntentResult.model_validate(json.loads(text))
            logger.info(
                "Intent detected | workflow=%s task=%s confidence=%.2f",
                result.workflow, result.task, result.confidence,
            )
            metrics.record_intent_detection(
                workflow_type=result.workflow.value,
                task_type=result.task.value,
                confidence=result.confidence,
                duration=duration,
            )
            return result
        except (ValidationError, json.JSONDecodeError) as e:
            logger.exception("Intent detection failed: %s", e)
            fallback = IntentResult(
                workflow=Workflow.KNOWLEDGE_RAG,
                task=TaskType.LOOKUP,
                reason="Fallback due to parsing failure",
                confidence=0.50,
            )
            metrics.record_intent_detection(
                workflow_type=fallback.workflow.value,
                task_type=fallback.task.value,
                confidence=fallback.confidence,
                duration=duration,
                fallback=True,
            )
            return fallback
