from __future__ import annotations

import pytest

from packages.core.config import Settings
from packages.core.enums import CampaignStatus, GoalStatus, SequenceFamily
from packages.core.models import CampaignState, ExperimentGoal, PipelineSummary, ScoringWeights
from packages.pasqal_adapters.pasqal_cloud_adapter import PasqalCloudAdapter
from packages.pasqal_adapters.qoolqit_adapter import QoolQitAdapter


def test_valid_scoring_weights_are_accepted() -> None:
    weights = ScoringWeights(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)
    assert weights.alpha + weights.beta + weights.gamma + weights.delta == pytest.approx(1.0)


def test_pasqal_cloud_authenticate_reports_unavailable_without_credentials() -> None:
    adapter = PasqalCloudAdapter(Settings())
    status = adapter.authenticate()
    assert status["authenticated"] is False


def test_qoolqit_status_exposes_availability_flag() -> None:
    status = QoolQitAdapter().status()
    assert "available" in status


def test_pipeline_summary_defaults_allow_empty_rankings() -> None:
    goal = ExperimentGoal(
        title="Summary test",
        scientific_objective="Check default summary fields.",
        desired_atom_count=4,
        status=GoalStatus.COMPLETED,
    )
    campaign = CampaignState(goal_id=goal.id, status=CampaignStatus.NO_CANDIDATES)

    summary = PipelineSummary(campaign=campaign, goal=goal, status="NO_CANDIDATES")

    assert summary.top_candidate is None
    assert summary.ranked_count == 0


def test_campaign_state_no_candidates_status_round_trips() -> None:
    campaign = CampaignState(goal_id="goal_test", status=CampaignStatus.NO_CANDIDATES)
    assert campaign.status == CampaignStatus.NO_CANDIDATES
