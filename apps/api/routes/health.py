from __future__ import annotations

from fastapi import APIRouter

from packages.core.config import get_settings
from packages.db.mongodb import get_mongo_client


router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    payload: dict[str, object] = {
        "status": "ok",
        "app_env": settings.app_env.value,
        "mongodb_configured": settings.has_mongodb,
    }

    if not settings.has_mongodb:
        payload["mongodb_ping"] = "not_configured"
        return payload

    try:
        client = get_mongo_client(settings)
        client.admin.command("ping")
        payload["mongodb_ping"] = "ok"
    except Exception as exc:  # pragma: no cover - depends on external service
        payload["mongodb_ping"] = f"failed: {exc}"
    return payload
