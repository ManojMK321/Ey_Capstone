"""
Single entry point to call once at startup in main.py.

Does NOT add HTTP middleware or mount /metrics — main.py already does that
via `Instrumentator().instrument(app).expose(app)` and
`LangSmithRequestTracingMiddleware`. This just configures logging and
(optionally) OpenTelemetry span tracing for arbitrary function-level spans.
"""

from __future__ import annotations

import logging
import os

from . import tracing
from . import langsmith as ls_module

logger = logging.getLogger(__name__)


def setup_observability(service_name: str = "contract-intelligence-api") -> None:
    """
    Call once, near the top of main.py, before or after creating `app` —
    order relative to Instrumentator/LangSmith middleware doesn't matter.

        from src.observability.telemetry import setup_observability
        setup_observability()
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(log_level)

    tracing.init_tracing(service_name=service_name)

    logger.info(
        "Observability ready: service=%s tracing=%s langsmith=%s prometheus=/metrics",
        service_name,
        tracing.is_available(),
        ls_module.is_enabled(),
    )
