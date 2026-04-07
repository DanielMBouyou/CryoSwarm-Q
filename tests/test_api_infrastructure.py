from __future__ import annotations

from fastapi.testclient import TestClient

import apps.api.auth as auth_module
import apps.api.rate_limit as rate_limit_module
from apps.api.dependencies import get_repository
from apps.api.main import app
from packages.core.config import Settings
from packages.core.enums import CampaignStatus
from packages.core.models import CampaignState
from packages.orchestration.events import PipelineEvent


API_PREFIX = "/api/v1"


class _FakeRepository:
    def __init__(self) -> None:
        self.campaign = CampaignState(goal_id="goal_test", status=CampaignStatus.COMPLETED)

    def get_campaign(self, campaign_id: str):
        if campaign_id == self.campaign.id:
            return self.campaign
        return None

    def list_candidates_for_campaign(self, campaign_id: str):
        return []

    def create_goal(self, goal):
        return goal

    def get_goal(self, goal_id: str):
        return None


def test_rate_limit_returns_429_after_budget_exhaustion(monkeypatch) -> None:
    fake_repo = _FakeRepository()
    app.dependency_overrides[get_repository] = lambda: fake_repo
    app.dependency_overrides[auth_module.get_settings] = lambda: Settings(mongodb_uri="mongodb://fake")
    monkeypatch.setattr(
        rate_limit_module,
        "get_settings",
        lambda: Settings(
            mongodb_uri="mongodb://fake",
            api_rate_limit_requests=2,
            api_rate_limit_window_seconds=60,
        ),
    )
    rate_limit_module.rate_limiter.clear()

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.get(f"{API_PREFIX}/campaigns/{fake_repo.campaign.id}")
        second = client.get(f"{API_PREFIX}/campaigns/{fake_repo.campaign.id}")
        third = client.get(f"{API_PREFIX}/campaigns/{fake_repo.campaign.id}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["detail"] == "Rate limit exceeded."
    rate_limit_module.rate_limiter.clear()
    app.dependency_overrides.clear()


def test_websocket_streams_campaign_events() -> None:
    fake_repo = _FakeRepository()
    app.dependency_overrides[get_repository] = lambda: fake_repo
    app.dependency_overrides[auth_module.get_settings] = lambda: Settings(mongodb_uri="mongodb://fake")
    rate_limit_module.rate_limiter.clear()

    with TestClient(app, raise_server_exceptions=False) as client:
        with client.websocket_connect(f"{API_PREFIX}/ws/campaigns") as websocket:
            connected = websocket.receive_json()
            assert connected["event_type"] == "stream.connected"

            client.app.state.event_broadcaster.publish(
                PipelineEvent(
                    event_type="phase.started",
                    payload={"campaign_id": fake_repo.campaign.id, "phase": "evaluation"},
                )
            )
            event = websocket.receive_json()

    assert event["event_type"] == "phase.started"
    assert event["payload"]["campaign_id"] == fake_repo.campaign.id
    assert event["payload"]["phase"] == "evaluation"
    app.dependency_overrides.clear()


def test_legacy_unversioned_route_is_not_registered() -> None:
    fake_repo = _FakeRepository()
    app.dependency_overrides[get_repository] = lambda: fake_repo
    app.dependency_overrides[auth_module.get_settings] = lambda: Settings(mongodb_uri="mongodb://fake")
    rate_limit_module.rate_limiter.clear()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/campaigns/missing")

    assert response.status_code == 404
    app.dependency_overrides.clear()
