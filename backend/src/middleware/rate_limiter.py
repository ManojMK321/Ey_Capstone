"""
In-memory sliding-window rate limiter.

State lives in a single SlidingWindowStore held by the process, which is
fine for one uvicorn worker. It does not coordinate across multiple
workers/replicas — swap in a Redis-backed store before scaling out, since
each process would otherwise enforce its own independent limit.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from starlette.responses import JSONResponse

DEFAULT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60"))
DEFAULT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Always exempt from limiting: health/metrics scraping isn't client traffic.
EXEMPT_PATHS = {"/metrics"}


class SlidingWindowStore:
    """Tracks request timestamps per client key using a sliding window."""

    def __init__(self, max_requests: int = DEFAULT_MAX_REQUESTS, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> Tuple[bool, int]:
        """Record a hit for `key` if under the limit. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        hits = self._hits[key]

        cutoff = now - self.window_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()

        if len(hits) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - hits[0])) + 1
            return False, retry_after

        hits.append(now)
        return True, 0


class RateLimiterMiddleware:
    """
    ASGI middleware — rejects a client once it exceeds `store.max_requests`
    requests per `store.window_seconds` with HTTP 429. Keyed by client IP
    so one noisy client can't starve others.

        app.add_middleware(RateLimiterMiddleware, store=SlidingWindowStore())
    """

    def __init__(self, app, store: Optional[SlidingWindowStore] = None):
        self.app = app
        self.store = store or SlidingWindowStore()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        key = client[0] if client else "unknown"

        allowed, retry_after = self.store.allow(key)
        if not allowed:
            response = JSONResponse(
                {"detail": "Rate limit exceeded. Please slow down."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
