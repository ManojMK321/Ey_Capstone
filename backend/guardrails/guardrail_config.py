"""
guardrail_config.py

Shared result type + central configuration for every guardrail stage:

    User Query
      -> input_guardrail
      -> Query Router (intent detection)
      -> KnowledgeRAG / AgenticRAG
      -> Document Retrieval
      -> retrieval_guardrail
      -> agent_guardrail          (AgenticRAG plan validation)
      -> tool_guardrail           (AgenticRAG specialist-agent execution)
      -> LLM Response Generation
      -> output_guardrail
      -> audit_logger
      -> Final Response to User

document_guardrail runs separately, once per uploaded file, before it
enters the ingestion pipeline.

Every guardrail returns a GuardrailResult instead of a bare bool, so
callers (and the audit log) always get a reason, a risk level, and — when
a guardrail cleaned/trimmed/redacted its input rather than blocking it
outright — the modified payload.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# 0-100 score for UI display (e.g. "Risk Score: 92/100") — a coarser,
# human-facing stand-in for the RiskLevel enum, not a separate signal.
RISK_SCORE: dict[RiskLevel, int] = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 25,
    RiskLevel.MEDIUM: 55,
    RiskLevel.HIGH: 85,
    RiskLevel.CRITICAL: 100,
}


_RISK_ORDER = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


def escalate(current: RiskLevel, new: RiskLevel) -> RiskLevel:
    """Return whichever of the two risk levels is more severe."""
    return new if _RISK_ORDER.index(new) > _RISK_ORDER.index(current) else current


class GuardrailStatus(str, Enum):
    ALLOWED = "allowed"      # passed unchanged
    MODIFIED = "modified"    # passed, but the guardrail cleaned/trimmed/redacted the data
    BLOCKED = "blocked"      # rejected — caller must not use `.data`


@dataclass
class GuardrailResult:
    stage: str
    status: GuardrailStatus
    reason: str
    risk_level: RiskLevel = RiskLevel.NONE
    data: Any = None
    metadata: dict = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.status != GuardrailStatus.BLOCKED

    @property
    def risk_score(self) -> int:
        return RISK_SCORE[self.risk_level]

    @classmethod
    def ok(
        cls, stage: str, data: Any = None, reason: str = "passed all checks",
        metadata: Optional[dict] = None,
    ) -> "GuardrailResult":
        return cls(stage=stage, status=GuardrailStatus.ALLOWED, reason=reason,
                   risk_level=RiskLevel.NONE, data=data, metadata=metadata or {})

    @classmethod
    def modified(
        cls, stage: str, data: Any, reason: str,
        risk_level: RiskLevel = RiskLevel.LOW, metadata: Optional[dict] = None,
    ) -> "GuardrailResult":
        return cls(stage=stage, status=GuardrailStatus.MODIFIED, reason=reason,
                   risk_level=risk_level, data=data, metadata=metadata or {})

    @classmethod
    def blocked(
        cls, stage: str, reason: str,
        risk_level: RiskLevel = RiskLevel.MEDIUM, metadata: Optional[dict] = None,
    ) -> "GuardrailResult":
        return cls(stage=stage, status=GuardrailStatus.BLOCKED, reason=reason,
                   risk_level=risk_level, data=None, metadata=metadata or {})


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class GuardrailConfig:
    """Central on/off switches + thresholds. Reads env vars once at import time."""

    def __init__(self) -> None:
        # Master switches — flip a stage off without touching call sites.
        self.enable_input_guardrail = _bool_env("ENABLE_INPUT_GUARDRAIL", True)
        self.enable_document_guardrail = _bool_env("ENABLE_DOCUMENT_GUARDRAIL", True)
        self.enable_retrieval_guardrail = _bool_env("ENABLE_RETRIEVAL_GUARDRAIL", True)
        self.enable_agent_guardrail = _bool_env("ENABLE_AGENT_GUARDRAIL", True)
        self.enable_tool_guardrail = _bool_env("ENABLE_TOOL_GUARDRAIL", True)
        self.enable_output_guardrail = _bool_env("ENABLE_OUTPUT_GUARDRAIL", True)
        self.enable_audit_log = _bool_env("ENABLE_AUDIT_LOG", True)

        # input_guardrail
        self.min_query_len = _int_env("GUARDRAIL_MIN_QUERY_LEN", 3)
        self.max_query_len = _int_env("GUARDRAIL_MAX_QUERY_LEN", 2000)
        self.scope_model = os.getenv("GUARDRAIL_SCOPE_MODEL", "gpt-4o-mini")

        # document_guardrail
        self.allowed_extensions = {".pdf"}
        self.max_file_size_mb = _int_env("GUARDRAIL_MAX_FILE_SIZE_MB", 50)

        # retrieval_guardrail
        self.max_retrieved_chunks = _int_env("GUARDRAIL_MAX_RETRIEVED_CHUNKS", 20)

        # agent_guardrail
        self.max_subquestions = _int_env("GUARDRAIL_MAX_SUBQUESTIONS", 3)

        # tool_guardrail
        self.tool_timeout_seconds = float(_int_env("GUARDRAIL_TOOL_TIMEOUT_SECONDS", 30))
        self.allowed_tools = {"specialist_agent", "vector_search", "reranker"}

        # output_guardrail
        self.groundedness_model = os.getenv("GUARDRAIL_GROUNDEDNESS_MODEL", "gpt-4o-mini")
        self.pii_mode = os.getenv("GUARDRAIL_PII_MODE", "block")  # "block" | "redact"

        # audit_logger
        self.audit_log_path = os.getenv("GUARDRAIL_AUDIT_LOG_PATH", "logs/guardrail_audit.jsonl")


config = GuardrailConfig()


# ---------------------------------------------------------------------------
# UI-facing threat reporting
#
# Every guardrail already tags its BLOCKED results with a `metadata["check"]`
# key (and, for injection hits, `metadata["pattern"]`). describe_block()
# translates that into a human-readable payload so a frontend can show a
# security notice instead of a bare error string.
# ---------------------------------------------------------------------------

_THREAT_TYPE_LABELS: dict[str, str] = {
    "length":         "Invalid Query Length",
    "injection":      "Prompt Injection",
    "scope":          "Off-Topic Request",
    "filename":       "Invalid Upload",
    "extension":      "Unsupported File Type",
    "content_type":   "Unsupported File Type",
    "size":           "File Size Violation",
    "magic_bytes":    "Invalid PDF Signature",
    "active_content": "Malicious PDF Content",
    "empty":          "No Relevant Content",
    "pii":            "Sensitive Data Exposure",
    "groundedness":   "Unverified Claim",
    "allow_list":     "Unauthorized Tool Call",
    "timeout":        "Tool Execution Timeout",
    "exception":      "Tool Execution Failure",
}

# Human-friendly names for the specific regex pattern an injection attempt matched.
_INJECTION_DETAIL_LABELS: dict[str, str] = {
    "instruction override":      "Instruction Override",
    "identity hijack":           "Identity Hijack",
    "restricted role bypass":    "Role Bypass Attempt",
    "identity pretend":          "Identity Spoofing",
    "prompt leakage":            "System Prompt Extraction",
    "system tag injection":      "System Tag Injection",
    "template injection":        "Template Injection",
    "special token injection":   "Special Token Injection",
    "jailbreak keyword":         "Jailbreak Attempt",
    "safety bypass":             "Safety Bypass Attempt",
}


def describe_block(result: GuardrailResult) -> dict:
    """
    Turn a BLOCKED (or MODIFIED-with-risk) GuardrailResult into a UI-ready
    payload: {blocked, stage, threat_type, threat_detail, risk_level,
    risk_score, action, reason}. Meant to be sent to the frontend as-is —
    e.g. as an HTTPException's `detail`, or a ChatResponse field.
    """
    check = result.metadata.get("check", "")
    threat_type = _THREAT_TYPE_LABELS.get(check, result.stage.replace("_", " ").title())

    threat_detail: Optional[str] = None
    if check == "injection":
        pattern = result.metadata.get("pattern", "")
        threat_detail = _INJECTION_DETAIL_LABELS.get(pattern, pattern.title() if pattern else None)
    elif check == "active_content":
        markers = result.metadata.get("markers") or []
        threat_detail = ", ".join(markers) if markers else None
    elif check == "pii":
        labels = result.metadata.get("labels") or []
        threat_detail = ", ".join(labels) if labels else None

    return {
        "blocked":       result.status == GuardrailStatus.BLOCKED,
        "stage":         result.stage,
        "threat_type":   threat_type,
        "threat_detail": threat_detail,
        "risk_level":    result.risk_level.value,
        "risk_score":    result.risk_score,
        "action":        "Request Blocked" if result.status == GuardrailStatus.BLOCKED else "Content Flagged",
        "reason":        result.reason,
    }
