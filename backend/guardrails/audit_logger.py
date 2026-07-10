"""
audit_logger.py

Structured audit trail for every guardrail decision plus the final
request outcome. Appends one JSON object per line to a local log file (so
it survives process restarts) and mirrors each entry to the standard
logger for console/aggregator visibility. Never raises — a broken audit
sink must not take down a request.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .guardrail_config import GuardrailResult, GuardrailStatus, config

logger = logging.getLogger("guardrails.audit")


class AuditLogger:

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path or config.audit_log_path)
        self._lock = threading.Lock()
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Audit log directory could not be created (%s); logging to console only.", exc)

    def _write(self, entry: dict[str, Any]) -> None:
        if not config.enable_audit_log:
            return
        try:
            with self._lock, self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.warning("Failed to write audit log entry: %s", exc)

    def log(
        self,
        stage: str,
        result: GuardrailResult,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record one guardrail stage's decision."""
        log_fn = logger.warning if result.status == GuardrailStatus.BLOCKED else logger.info
        log_fn(
            "guardrail_audit stage=%s status=%s risk=%s reason=%s",
            stage, result.status.value, result.risk_level.value, result.reason,
        )

        entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "stage":      stage,
            "status":     result.status.value,
            "risk_level": result.risk_level.value,
            "reason":     result.reason,
            "session_id": session_id,
            "request_id": request_id,
            "metadata":   result.metadata,
        }
        if extra:
            entry.update(extra)
        self._write(entry)

    def log_final_response(
        self,
        session_id: str,
        query: str,
        workflow: str,
        allowed: bool,
        stage_results: dict[str, GuardrailResult],
        request_id: Optional[str] = None,
    ) -> None:
        """One summary line covering every guardrail stage a request passed through."""
        logger.info("guardrail_audit_summary session=%s workflow=%s allowed=%s", session_id, workflow, allowed)
        entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "stage":      "final_response",
            "session_id": session_id,
            "request_id": request_id,
            "workflow":   workflow,
            "query_len":  len(query or ""),
            "allowed":    allowed,
            "stages": {
                name: {"status": r.status.value, "risk_level": r.risk_level.value, "reason": r.reason}
                for name, r in stage_results.items()
            },
        }
        self._write(entry)


audit_logger = AuditLogger()
