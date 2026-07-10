"""
guardrails/

Layered guardrail pipeline for the Contract Intelligence backend:

    User Query
      -> input_guardrail         (before the query reaches the router)
      -> [Query Router -> KnowledgeRAG / AgenticRAG -> Document Retrieval]
      -> retrieval_guardrail      (after retrieval, before the LLM sees the context)
      -> agent_guardrail          (validates AgenticRAG's plan: subquestions + task route)
      -> tool_guardrail           (wraps AgenticRAG's specialist-agent invocation)
      -> [LLM Response Generation]
      -> output_guardrail         (on the final answer, before it reaches the user)
      -> audit_logger             (records every stage's decision)

document_guardrail is separate — it runs once, at upload time, before a
file enters the ingestion pipeline (parse -> chunk -> embed -> index).

Every check returns a GuardrailResult (status / reason / risk_level / data)
instead of a bare bool — see guardrail_config.py. Each guardrail module is
self-contained: no guardrail imports from another guardrail, so any one of
them can be reused on its own.
"""

from .guardrail_config import (
    GuardrailConfig, GuardrailResult, GuardrailStatus, RiskLevel, config, describe_block,
)
from .audit_logger import AuditLogger, audit_logger
from . import input_guardrail
from . import document_guardrail
from . import retrieval_guardrail
from . import agent_guardrail
from . import tool_guardrail
from . import output_guardrail

__all__ = [
    "GuardrailConfig", "GuardrailResult", "GuardrailStatus", "RiskLevel", "config", "describe_block",
    "AuditLogger", "audit_logger",
    "input_guardrail", "document_guardrail", "retrieval_guardrail",
    "agent_guardrail", "tool_guardrail", "output_guardrail",
]
