from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_repository
from packages.core.models import CampaignState, DemoGoalRequest, PipelineSummary
from packages.db.repositories import CryoSwarmRepository
from packages.orchestration.runner import run_demo_campaign


router = APIRouter(prefix="/campaigns", tags=["campaigns"])


@router.post("/run-demo", response_model=PipelineSummary)
def run_demo(
    payload: DemoGoalRequest | None = None,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> PipelineSummary:
    return run_demo_campaign(request=payload or DemoGoalRequest(), repository=repository)


@router.get("/{campaign_id}", response_model=CampaignState)
def get_campaign(
    campaign_id: str,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> CampaignState:
    campaign = repository.get_campaign(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found.")
    return campaign
