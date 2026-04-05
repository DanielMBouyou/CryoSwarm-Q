from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_repository
from packages.core.enums import GoalStatus
from packages.core.models import ExperimentGoal, ExperimentGoalCreate
from packages.db.repositories import CryoSwarmRepository


router = APIRouter(prefix="/goals", tags=["goals"])


@router.post("", response_model=ExperimentGoal)
def create_goal(
    payload: ExperimentGoalCreate,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> ExperimentGoal:
    goal = ExperimentGoal(
        **payload.model_dump(),
        status=GoalStatus.STORED,
    )
    return repository.create_goal(goal)


@router.get("/{goal_id}", response_model=ExperimentGoal)
def get_goal(
    goal_id: str,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> ExperimentGoal:
    goal = repository.get_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found.")
    return goal
