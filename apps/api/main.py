from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.live import CampaignEventBroadcaster
from apps.api.rate_limit import RateLimitMiddleware
from apps.api.routes.campaigns import router as campaigns_router
from apps.api.routes.candidates import router as candidates_router
from apps.api.routes.goals import router as goals_router
from apps.api.routes.health import router as health_router
from apps.api.routes.streaming import router as streaming_router
from packages.core.config import get_settings
from packages.db.init_db import initialize_database
from packages.db.mongodb import close_mongo_client
from packages.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()
API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize MongoDB once on startup and close the client on shutdown."""
    application.state.event_broadcaster = CampaignEventBroadcaster(asyncio.get_running_loop())
    settings = get_settings()
    if settings.has_mongodb:
        try:
            initialize_database()
            logger.info("MongoDB initialized during startup.")
        except Exception as exc:
            logger.warning("MongoDB initialization during startup failed: %s", exc)
    else:
        logger.warning("MongoDB not configured - skipping database initialization.")
    yield
    close_mongo_client()
    logger.info("MongoDB connection closed.")


app = FastAPI(
    title="CryoSwarm-Q API",
    version="0.1.0",
    description=(
        "FastAPI backend for a hardware-aware multi-agent orchestration prototype "
        "for neutral-atom experimentation."
    ),
    lifespan=lifespan,
)

_cors_origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]
if not _cors_origins:
    _cors_origins = ["http://localhost:8501", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)
app.add_middleware(RateLimitMiddleware)

api_v1_router = APIRouter(prefix=API_V1_PREFIX)
api_v1_router.include_router(health_router)
api_v1_router.include_router(goals_router)
api_v1_router.include_router(campaigns_router)
api_v1_router.include_router(candidates_router, prefix="/campaigns")
api_v1_router.include_router(streaming_router)

app.include_router(api_v1_router)


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error."},
    )
