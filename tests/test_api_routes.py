from __future__ import annotations

"""Nominal API route tests for CryoSwarm-Q without a live database."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

import apps.api.auth as auth_module
import apps.api.routes.campaigns as campaigns_module
import apps.api.routes.health as health_module
from apps.api.dependencies import get_repository
from apps.api.main import app
from packages.core.config import Settings
from packages.core.enums import BackendType, CampaignStatus, CandidateStatus, GoalStatus
from packages.core.models import CampaignState, EvaluationResult, ExperimentGoal, PipelineSummary

API_PREFIX = "/api/v1"


class FakeRepository:
    """In-memory repository stub used to exercise API routes."""

    def __init__(self) -> None:
        self.goals: dict[str, ExperimentGoal] = {}
        self.campaigns: dict[str, CampaignState] = {}
        self.evaluation_results: dict[str, list[EvaluationResult]] = {}

    def create_goal(self, goal: ExperimentGoal) -> ExperimentGoal:
        self.goals[goal.id] = goal
        return goal

    def get_goal(self, goal_id: str) -> ExperimentGoal | None:
        return self.goals.get(goal_id)

    def get_campaign(self, campaign_id: str) -> CampaignState | None:
        return self.campaigns.get(campaign_id)

    def list_candidates_for_campaign(self, campaign_id: str) -> list[EvaluationResult]:
        return self.evaluation_results.get(campaign_id, [])

    def list_latest_campaigns(self, limit: int = 10) -> list[CampaignState]:
        return list(self.campaigns.values())[:limit]

    def list_agent_decisions(self, campaign_id: str) -> list[Any]:
        return []

    def list_robustness_reports(self, campaign_id: str) -> list[Any]:
        return []


@pytest.fixture()
def fake_repo() -> FakeRepository:
    return FakeRepository()


@pytest.fixture()
def client(fake_repo: FakeRepository) -> TestClient:
    app.dependency_overrides[get_repository] = lambda: fake_repo
    app.dependency_overrides[auth_module.get_settings] = lambda: Settings()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestHealthRoute:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get(f"{API_PREFIX}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestGoalRoutes:
    def test_create_goal_returns_200(self, client: TestClient) -> None:
        response = client.post(
            f"{API_PREFIX}/goals",
            json={
                "title": "Test neutral-atom benchmark",
                "scientific_objective": "Evaluate robustness of small configurations.",
                "target_observable": "rydberg_density",
                "desired_atom_count": 6,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["title"] == "Test neutral-atom benchmark"
        assert payload["status"] == GoalStatus.STORED.value
        assert payload["id"].startswith("goal_")

    def test_create_goal_with_invalid_atom_count(self, client: TestClient) -> None:
        response = client.post(
            f"{API_PREFIX}/goals",
            json={
                "title": "Bad atoms",
                "scientific_objective": "Atom count below lower bound.",
                "desired_atom_count": 1,
            },
        )

        assert response.status_code == 422

    def test_create_goal_with_short_title(self, client: TestClient) -> None:
        response = client.post(
            f"{API_PREFIX}/goals",
            json={
                "title": "AB",
                "scientific_objective": "Title is too short.",
                "desired_atom_count": 6,
            },
        )

        assert response.status_code == 422

    def test_get_existing_goal(self, client: TestClient, fake_repo: FakeRepository) -> None:
        goal = ExperimentGoal(
            title="Test goal retrieval",
            scientific_objective="Retrieve goal by ID.",
            status=GoalStatus.STORED,
        )
        fake_repo.goals[goal.id] = goal

        response = client.get(f"{API_PREFIX}/goals/{goal.id}")

        assert response.status_code == 200
        assert response.json()["id"] == goal.id

    def test_get_missing_goal_returns_404(self, client: TestClient) -> None:
        response = client.get(f"{API_PREFIX}/goals/nonexistent_id")
        assert response.status_code == 404


class TestCampaignRoutes:
    def test_get_missing_campaign_returns_404(self, client: TestClient) -> None:
        response = client.get(f"{API_PREFIX}/campaigns/nonexistent_id")
        assert response.status_code == 404

    def test_list_campaign_candidates_404_for_missing(self, client: TestClient) -> None:
        response = client.get(f"{API_PREFIX}/campaigns/nonexistent/candidates")
        assert response.status_code == 404

    def test_list_campaign_candidates_returns_empty(
        self,
        client: TestClient,
        fake_repo: FakeRepository,
    ) -> None:
        campaign = CampaignState(goal_id="goal_test", status=CampaignStatus.COMPLETED)
        fake_repo.campaigns[campaign.id] = campaign

        response = client.get(f"{API_PREFIX}/campaigns/{campaign.id}/candidates")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_campaign_candidates_returns_results(
        self,
        client: TestClient,
        fake_repo: FakeRepository,
    ) -> None:
        campaign = CampaignState(goal_id="goal_test", status=CampaignStatus.COMPLETED)
        fake_repo.campaigns[campaign.id] = campaign
        result = EvaluationResult(
            campaign_id=campaign.id,
            sequence_candidate_id="seq_1",
            register_candidate_id="reg_1",
            nominal_score=0.8,
            robustness_score=0.75,
            worst_case_score=0.6,
            observable_score=0.78,
            objective_score=0.74,
            backend_choice=BackendType.LOCAL_PULSER_SIMULATION,
            estimated_cost=0.12,
            estimated_latency=0.15,
            final_rank=1,
            status=CandidateStatus.RANKED,
            reasoning_summary="Test result.",
        )
        fake_repo.evaluation_results[campaign.id] = [result]

        response = client.get(f"{API_PREFIX}/campaigns/{campaign.id}/candidates")

        assert response.status_code == 200
        payload = response.json()
        assert len(payload) == 1
        assert payload[0]["objective_score"] == 0.74


class TestRunDemo:
    def test_run_demo_default_payload(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_run(request, repository, event_bus=None) -> PipelineSummary:
            goal = ExperimentGoal(
                title=request.title,
                scientific_objective=request.scientific_objective,
                status=GoalStatus.COMPLETED,
            )
            campaign = CampaignState(goal_id=goal.id, status=CampaignStatus.COMPLETED)
            return PipelineSummary(campaign=campaign, goal=goal, status="COMPLETED")

        monkeypatch.setattr(campaigns_module, "run_demo_campaign", fake_run)

        response = client.post(f"{API_PREFIX}/campaigns/run-demo", json={})

        assert response.status_code == 200
        assert response.json()["status"] == "COMPLETED"

    def test_run_demo_custom_payload(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_run(request, repository, event_bus=None) -> PipelineSummary:
            goal = ExperimentGoal(
                title=request.title,
                scientific_objective=request.scientific_objective,
                desired_atom_count=request.desired_atom_count,
                preferred_geometry=request.preferred_geometry,
                status=GoalStatus.COMPLETED,
            )
            campaign = CampaignState(goal_id=goal.id, status=CampaignStatus.COMPLETED)
            return PipelineSummary(campaign=campaign, goal=goal, status="COMPLETED")

        monkeypatch.setattr(campaigns_module, "run_demo_campaign", fake_run)

        response = client.post(
            f"{API_PREFIX}/campaigns/run-demo",
            json={
                "title": "Custom test campaign",
                "scientific_objective": "Custom objective",
                "desired_atom_count": 8,
                "preferred_geometry": "square",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["goal"]["title"] == "Custom test campaign"
        assert payload["goal"]["desired_atom_count"] == 8

    def test_run_demo_pipeline_crash_returns_500(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            campaigns_module,
            "run_demo_campaign",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated crash")),
        )

        response = client.post(f"{API_PREFIX}/campaigns/run-demo", json={})

        assert response.status_code == 500
        assert response.json()["detail"]["error"] == "Campaign execution failed."


class TestGlobalExceptionHandler:
    def test_unhandled_error_does_not_leak_details(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            health_module,
            "get_settings",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        response = client.get(f"{API_PREFIX}/health")

        assert response.status_code == 500
        body = response.json()
        assert body == {"error": "Internal server error."}
        assert "boom" not in str(body)
