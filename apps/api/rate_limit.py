from __future__ import annotations

"""Simple in-memory rate limiting for the FastAPI service."""

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from packages.core.config import get_settings


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after: int
    limit: int
    window_seconds: int


class InMemoryRateLimiter:
    """Fixed-window sliding limiter keyed by client and route."""

    def __init__(self) -> None:
        self._events: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def check(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> RateLimitDecision:
        if limit <= 0 or window_seconds <= 0:
            return RateLimitDecision(
                allowed=True,
                remaining=max(limit - 1, 0),
                retry_after=0,
                limit=limit,
                window_seconds=window_seconds,
            )

        now = monotonic()
        with self._lock:
            history = self._events[key]
            while history and now - history[0] >= window_seconds:
                history.popleft()

            if len(history) >= limit:
                retry_after = max(int(window_seconds - (now - history[0])), 1) if history else window_seconds
                return RateLimitDecision(
                    allowed=False,
                    remaining=0,
                    retry_after=retry_after,
                    limit=limit,
                    window_seconds=window_seconds,
                )

            history.append(now)
            remaining = max(limit - len(history), 0)
            if not history:
                self._events.pop(key, None)
            return RateLimitDecision(
                allowed=True,
                remaining=remaining,
                retry_after=0,
                limit=limit,
                window_seconds=window_seconds,
            )


rate_limiter = InMemoryRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply a lightweight per-client rate limit to HTTP API routes."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.scope.get("type") != "http":
            return await call_next(request)

        if request.url.path.endswith("/health"):
            return await call_next(request)

        settings = get_settings()
        client_host = request.client.host if request.client is not None else "unknown"
        key = f"{client_host}:{request.method}:{request.url.path}"
        decision = rate_limiter.check(
            key,
            limit=settings.api_rate_limit_requests,
            window_seconds=settings.api_rate_limit_window_seconds,
        )
        if not decision.allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded."},
                headers={
                    "Retry-After": str(decision.retry_after),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": str(decision.remaining),
                    "X-RateLimit-Window": str(decision.window_seconds),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        response.headers["X-RateLimit-Window"] = str(decision.window_seconds)
        return response
