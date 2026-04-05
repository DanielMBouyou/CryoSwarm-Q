from __future__ import annotations

from fastapi.testclient import TestClient

import apps.api.routes.campaigns as campaigns_route_module
from apps.api.dependencies import get_repository
from apps.api.main import app


class _FakeRepository:
    def create_goal(self, goal):
        return goal

    def get_goal(self, goal_id: str):
        return None

    def get_campaign(self, campaign_id: str):
        return None

    def list_candidates_for_campaign(self, campaign_id: str):
        return []


def test_get_missing_goal_returns_404() -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    client = TestClient(app)

    response = client.get("/goals/missing")

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_post_invalid_goal_returns_422() -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    client = TestClient(app)

    response = client.post(
        "/goals",
        json={
            "title": "Bad",
            "scientific_objective": "Invalid atom count should fail.",
            "desired_atom_count": 1,
        },
    )

    assert response.status_code == 422
    app.dependency_overrides.clear()


def test_get_missing_campaign_returns_404() -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    client = TestClient(app)

    response = client.get("/campaigns/missing")

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_get_missing_campaign_candidates_returns_404() -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    client = TestClient(app)

    response = client.get("/campaigns/missing/candidates")

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_run_demo_returns_structured_500_when_pipeline_crashes(monkeypatch) -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    client = TestClient(app)
    monkeypatch.setattr(
        campaigns_route_module,
        "run_demo_campaign",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post("/campaigns/run-demo", json={})

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "Campaign execution failed."
    app.dependency_overrides.clear()
