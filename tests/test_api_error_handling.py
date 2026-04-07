from __future__ import annotations

from fastapi.testclient import TestClient

import apps.api.auth as auth_module
import apps.api.routes.campaigns as campaigns_route_module
import apps.api.routes.health as health_route_module
from apps.api.dependencies import get_repository
from apps.api.main import app
from packages.core.config import Settings
from packages.core.enums import AppEnvironment

API_PREFIX = "/api/v1"


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
    app.dependency_overrides[auth_module.get_settings] = lambda: Settings()
    client = TestClient(app)

    response = client.get(f"{API_PREFIX}/goals/missing")

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_post_invalid_goal_returns_422() -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    app.dependency_overrides[auth_module.get_settings] = lambda: Settings()
    client = TestClient(app)

    response = client.post(
        f"{API_PREFIX}/goals",
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
    app.dependency_overrides[auth_module.get_settings] = lambda: Settings()
    client = TestClient(app)

    response = client.get(f"{API_PREFIX}/campaigns/missing")

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_get_missing_campaign_candidates_returns_404() -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    app.dependency_overrides[auth_module.get_settings] = lambda: Settings()
    client = TestClient(app)

    response = client.get(f"{API_PREFIX}/campaigns/missing/candidates")

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_run_demo_returns_structured_500_when_pipeline_crashes(monkeypatch) -> None:
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    app.dependency_overrides[auth_module.get_settings] = lambda: Settings()
    client = TestClient(app)
    monkeypatch.setattr(
        campaigns_route_module,
        "get_settings",
        lambda: Settings(app_env=AppEnvironment.DEVELOPMENT),
    )
    monkeypatch.setattr(
        campaigns_route_module,
        "run_demo_campaign",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post(f"{API_PREFIX}/campaigns/run-demo", json={})

    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["error"] == "Campaign execution failed."
    if "debug_message" in body["detail"]:
        assert "boom" in body["detail"]["debug_message"]
        assert "traceback" not in body["detail"]["debug_message"].lower()
    app.dependency_overrides.clear()


def test_global_handler_hides_internal_exception_details() -> None:
    def _raise_boom() -> None:
        raise RuntimeError("sensitive internal path /etc/mongodb.conf leaked")

    app.add_api_route("/_test/unhandled", _raise_boom, methods=["GET"])
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/_test/unhandled")

    assert response.status_code == 500
    body = response.json()
    assert body == {"error": "Internal server error."}
    assert "sensitive" not in str(body)
    assert "mongodb" not in str(body).lower()
    app.router.routes.pop()


def test_health_does_not_leak_mongodb_uri(monkeypatch) -> None:
    """Ensures the health endpoint does not expose MongoDB connection details."""

    def _failing_client(*args, **kwargs):
        raise ConnectionError("mongodb+srv://admin:s3cret@cluster0.example.net/db?retryWrites=true")

    monkeypatch.setattr(
        health_route_module,
        "get_settings",
        lambda: Settings(
            mongodb_uri="mongodb://fake",
            app_env=AppEnvironment.DEVELOPMENT,
        ),
    )
    monkeypatch.setattr(health_route_module, "get_mongo_client", _failing_client)

    client = TestClient(app)
    response = client.get(f"{API_PREFIX}/health")

    body = response.json()
    assert response.status_code == 200
    assert "s3cret" not in str(body)
    assert "admin" not in str(body)
    assert body.get("mongodb_ping") == "failed"
