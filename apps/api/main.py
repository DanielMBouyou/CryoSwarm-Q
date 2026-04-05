from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from apps.api.routes.campaigns import router as campaigns_router
from apps.api.routes.candidates import router as candidates_router
from apps.api.routes.goals import router as goals_router
from apps.api.routes.health import router as health_router
from packages.core.logging import get_logger


logger = get_logger(__name__)


app = FastAPI(
    title="CryoSwarm-Q API",
    version="0.1.0",
    description=(
        "FastAPI backend for a hardware-aware multi-agent orchestration prototype "
        "for neutral-atom experimentation."
    ),
)

app.include_router(health_router)
app.include_router(goals_router)
app.include_router(campaigns_router)
app.include_router(candidates_router)


@app.exception_handler(Exception)
async def global_handler(request, exc):  # type: ignore[no-untyped-def]
    logger.error("Unhandled: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc)})
