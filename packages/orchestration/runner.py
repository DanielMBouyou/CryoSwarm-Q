from __future__ import annotations

from packages.core.enums import GoalStatus
from packages.core.models import DemoGoalRequest, ExperimentGoal, PipelineSummary
from packages.db.repositories import CryoSwarmRepository
from packages.orchestration.pipeline import CryoSwarmPipeline


def build_demo_goal(request: DemoGoalRequest | None = None) -> ExperimentGoal:
    request = request or DemoGoalRequest()
    return ExperimentGoal(
        title=request.title,
        scientific_objective=request.scientific_objective,
        target_observable=request.target_observable,
        desired_atom_count=request.desired_atom_count,
        preferred_geometry=request.preferred_geometry,
        status=GoalStatus.DRAFT,
    )


def run_demo_campaign(
    request: DemoGoalRequest | None = None,
    repository: CryoSwarmRepository | None = None,
) -> PipelineSummary:
    goal = build_demo_goal(request)
    pipeline = CryoSwarmPipeline(repository=repository)
    return pipeline.run(goal)
