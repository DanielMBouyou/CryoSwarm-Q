from __future__ import annotations

"""API key authentication helpers for CryoSwarm-Q HTTP endpoints.

When CRYOSWARM_API_KEY is configured, mutating routes require the same key in
the X-API-Key header. When unset, authentication is bypassed for local use.
"""

import secrets

from fastapi import Depends, HTTPException, Security, WebSocket, WebSocketException, status
from fastapi.security import APIKeyHeader

from packages.core.config import Settings, get_settings


_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    api_key: str | None = Security(_API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> None:
    """Verify the X-API-Key header using constant-time comparison."""
    if not settings.has_api_key:
        return

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )


def verify_websocket_api_key(
    websocket: WebSocket,
    settings: Settings | None = None,
) -> None:
    """Verify the API key for WebSocket connections."""
    active_settings = settings or get_settings()
    if not active_settings.has_api_key:
        return

    api_key = websocket.headers.get("X-API-Key") or websocket.query_params.get("api_key")
    if api_key is None:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Missing API key.",
        )

    if not secrets.compare_digest(api_key, active_settings.api_key):
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid API key.",
        )
