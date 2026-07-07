"""
Minimal OpenTelemetry tracing setup — self-contained (no config.py/context.py
dependency, since this app already has LangSmith for LLM tracing and
prometheus_fastapi_instrumentator for HTTP metrics).

This module only adds distributed *tracing* spans for arbitrary functions
(pipeline steps, retrieval, DB calls) that you want to see as a waterfall in
Jaeger/Tempo/Grafana, correlated by trace id. If you don't have a trace
backend configured yet (no OTEL_EXPORTER_OTLP_ENDPOINT), spans still work and
print to stdout in dev — or you can just skip calling init_tracing() and
everything here becomes a safe no-op.

Safe to import even if `opentelemetry-sdk` isn't installed.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode
    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False

_initialized = False


def is_available() -> bool:
    return _OTEL_AVAILABLE


def init_tracing(service_name: str = "contract-intelligence-api") -> None:
    """
    Call once at app startup. Reads OTEL_EXPORTER_OTLP_ENDPOINT from the
    environment; if unset, spans are printed to stdout instead (useful in
    dev, harmless in prod — just skip calling this if you don't want that).
    """
    global _initialized
    if _initialized or not _OTEL_AVAILABLE:
        if not _OTEL_AVAILABLE:
            logger.info("opentelemetry-sdk not installed; tracing.py running as no-op.")
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
            logger.info("Tracing spans exporting to OTLP collector at %s", otlp_endpoint)
        except ImportError:
            logger.warning(
                "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp-proto-grpc "
                "isn't installed; falling back to console exporter."
            )
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        # No collector configured — print spans to stdout so tracing is still visible in dev.
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("No OTEL_EXPORTER_OTLP_ENDPOINT set; tracing spans will print to stdout.")

    trace.set_tracer_provider(provider)
    _initialized = True


@contextlib.contextmanager
def start_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """
        with start_span("pipeline.chunk", {"doc_id": file_id}):
            chunks = chunker.chunk(...)
    """
    if not _OTEL_AVAILABLE or not _initialized:
        yield None
        return

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))


def traced(name: str | None = None, attributes: dict[str, Any] | None = None):
    """Decorator form of start_span, for sync or async functions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        import asyncio

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with start_span(span_name, attributes):
                    return await func(*args, **kwargs)
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_span(span_name, attributes):
                return func(*args, **kwargs)
        return sync_wrapper

    return decorator
