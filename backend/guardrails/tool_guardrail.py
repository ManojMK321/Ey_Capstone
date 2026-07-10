"""
tool_guardrail.py

"Tool Execution Guardrail" — wraps an internal tool/function call (a
specialist-agent invocation, a vector-store query, a reranker call) with:

    1. An allow-list check on the tool name
    2. A wall-clock timeout
    3. Exception containment — a failing tool becomes a structured
       GuardrailResult instead of an unhandled exception

Usage:
    result = guarded_tool_call("specialist_agent", rag.run_specialist_agent,
                                task, question, context, history_section)
    if not result.allowed:
        ...
    draft_answer = result.data
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable

from .guardrail_config import GuardrailResult, RiskLevel, config

logger = logging.getLogger(__name__)

STAGE = "tool_guardrail"

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-guardrail")


def guarded_tool_call(
    tool_name: str,
    func: Callable[..., Any],
    *args: Any,
    timeout: float | None = None,
    **kwargs: Any,
) -> GuardrailResult:
    """Execute `func(*args, **kwargs)` under the tool-execution guardrail."""
    if not config.enable_tool_guardrail:
        try:
            return GuardrailResult.ok(STAGE, data=func(*args, **kwargs), reason="tool guardrail disabled")
        except Exception as exc:
            return GuardrailResult.blocked(
                STAGE, f"Tool '{tool_name}' raised an unhandled error: {exc}", RiskLevel.HIGH,
                {"tool": tool_name, "error": type(exc).__name__},
            )

    if tool_name not in config.allowed_tools:
        logger.warning("Tool guardrail — '%s' is not on the allow-list, blocking call.", tool_name)
        return GuardrailResult.blocked(
            STAGE, f"Tool '{tool_name}' is not authorized for execution.", RiskLevel.HIGH,
            {"tool": tool_name, "check": "allow_list"},
        )

    effective_timeout = timeout or config.tool_timeout_seconds
    future = _executor.submit(func, *args, **kwargs)
    try:
        result = future.result(timeout=effective_timeout)
    except FutureTimeoutError:
        logger.error("Tool guardrail — '%s' exceeded %.1fs timeout.", tool_name, effective_timeout)
        return GuardrailResult.blocked(
            STAGE, f"Tool '{tool_name}' timed out after {effective_timeout:.0f}s.", RiskLevel.HIGH,
            {"tool": tool_name, "check": "timeout"},
        )
    except Exception as exc:
        logger.exception("Tool guardrail — '%s' raised an error.", tool_name)
        return GuardrailResult.blocked(
            STAGE, f"Tool '{tool_name}' failed: {exc}", RiskLevel.HIGH,
            {"tool": tool_name, "check": "exception", "error": type(exc).__name__},
        )

    return GuardrailResult.ok(STAGE, data=result, metadata={"tool": tool_name})


def guarded_tool(tool_name: str, timeout: float | None = None):
    """Decorator form of `guarded_tool_call` for a tool that's always invoked the same way."""
    def decorator(func: Callable[..., Any]):
        def wrapper(*args: Any, **kwargs: Any) -> GuardrailResult:
            return guarded_tool_call(tool_name, func, *args, timeout=timeout, **kwargs)
        return wrapper
    return decorator
