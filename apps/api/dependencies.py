from __future__ import annotations

from fastapi import HTTPException

from packages.core.config import get_settings
from packages.db.init_db import initialize_database
from packages.db.repositories import CryoSwarmRepository


def get_repository() -> CryoSwarmRepository:
    settings = get_settings()
    if not settings.has_mongodb:
        raise HTTPException(status_code=500, detail="MongoDB is not configured.")
    initialize_database()
    return CryoSwarmRepository(settings)
