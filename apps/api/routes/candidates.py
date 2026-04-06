from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_repository
from packages.core.models import EvaluationResult
from packages.db.repositories import CryoSwarmRepository


router = APIRouter(tags=["candidates"])


@router.get("/{campaign_id}/candidates", response_model=list[EvaluationResult])
def list_campaign_candidates(
    campaign_id: str,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> list[EvaluationResult]:
    campaign = repository.get_campaign(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found.")
    return repository.list_candidates_for_campaign(campaign_id)
