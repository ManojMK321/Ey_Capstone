import contextlib
import functools
import logging
import os
from typing import Any, Callable


logger = logging.getLogger(__name__)

try:
    import langsmith as ls
    from langsmith import traceable
except ImportError:  # pragma: no cover
    ls = None
    traceable = None


def _noop_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    return func


def _get_client() -> Any | None:
    if ls is None:
        return None

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    api_url = os.getenv("LANGSMITH_ENDPOINT", "").strip() or None
    if not api_key and not api_url:
        return None

    try:
        return ls.Client(api_key=api_key or None, api_url=api_url)
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to create LangSmith client: %s", exc)
        return None


def _get_project_name() -> str | None:
    return os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")


def traceable_operation(name: str | None = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None):
    """Return a LangSmith traceable decorator if available, otherwise no-op."""
    if traceable is None:
        return _noop_decorator

    enabled = is_enabled()
    client = _get_client()
    project_name = _get_project_name()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        traced = traceable(
            func,
            name=name,
            tags=tags or [],
            metadata=metadata or {},
        )

        if client is None:
            return traced

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracing_context(client=client, enabled=enabled, project_name=project_name):
                return traced(*args, **kwargs)

        return wrapper

    return decorator


def tracing_context(**kwargs):
    """Return a LangSmith tracing context if available, otherwise a no-op context."""
    if ls is None:
        return contextlib.nullcontext()

    try:
        return ls.tracing_context(**kwargs)
    except Exception as exc:  # pragma: no cover
        logger.warning("LangSmith tracing_context unavailable: %s", exc)
        return contextlib.nullcontext()


def is_enabled() -> bool:
    if ls is None:
        return False
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}


class LangSmithRequestTracingMiddleware:
    """
    ASGI middleware — wraps every HTTP request in a LangSmith tracing_context
    so nested `@traceable_operation` calls (chat, upload, agent steps, ...)
    share one project scope and get tagged with the route that triggered
    them. No-op pass-through when LangSmith isn't configured.

        app.add_middleware(LangSmithRequestTracingMiddleware)
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not is_enabled():
            await self.app(scope, receive, send)
            return

        client = _get_client()
        if client is None:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        with tracing_context(
            client=client,
            enabled=True,
            project_name=_get_project_name(),
            tags=[method, path],
        ):
            await self.app(scope, receive, send)
