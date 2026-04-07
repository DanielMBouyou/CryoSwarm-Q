from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from apps.api.auth import verify_api_key
from apps.api.dependencies import get_repository
from packages.core.config import get_settings
from packages.core.enums import AppEnvironment
from packages.core.logging import get_logger
from packages.core.models import CampaignState, DemoGoalRequest, PipelineSummary
from packages.db.repositories import CryoSwarmRepository
from packages.orchestration.events import EventBus
from packages.orchestration.runner import run_demo_campaign


router = APIRouter(prefix="/campaigns", tags=["campaigns"])
logger = get_logger(__name__)


@router.post(
    "/run-demo",
    response_model=PipelineSummary,
    dependencies=[Depends(verify_api_key)],
)
async def run_demo(
    request: Request,
    payload: DemoGoalRequest | None = None,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> PipelineSummary:
    event_bus = EventBus()
    broadcaster = getattr(request.app.state, "event_broadcaster", None)
    if broadcaster is not None:
        event_bus.subscribe("*", broadcaster.publish)

    try:
        return await run_in_threadpool(
            run_demo_campaign,
            request=payload or DemoGoalRequest(),
            repository=repository,
            event_bus=event_bus,
        )
    except Exception as exc:
        logger.error("Demo campaign failed: %s", exc, exc_info=True)
        settings = get_settings()
        detail: dict[str, str] = {"error": "Campaign execution failed."}
        if settings.app_env == AppEnvironment.DEVELOPMENT:
            debug_message = exc.args[0] if exc.args else exc.__class__.__name__
            detail["debug_message"] = debug_message if isinstance(debug_message, str) else exc.__class__.__name__
        raise HTTPException(
            status_code=500,
            detail=detail,
        ) from exc


@router.get("/{campaign_id}", response_model=CampaignState)
def get_campaign(
    campaign_id: str,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> CampaignState:
    campaign = repository.get_campaign(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found.")
    return campaign
