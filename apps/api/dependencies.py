from __future__ import annotations

from fastapi import HTTPException

from packages.core.config import get_settings
from packages.db.repositories import CryoSwarmRepository


def get_repository() -> CryoSwarmRepository:
    """Provide a repository instance after startup-time database initialization."""
    settings = get_settings()
    if not settings.has_mongodb:
        raise HTTPException(status_code=500, detail="MongoDB is not configured.")
    return CryoSwarmRepository(settings)
