from __future__ import annotations

"""API key authentication tests for mutating CryoSwarm-Q routes."""

import inspect

import pytest
from fastapi.testclient import TestClient

import apps.api.auth as auth_module
from apps.api.dependencies import get_repository
from apps.api.main import app
from packages.core.config import Settings


class _FakeRepository:
    def create_goal(self, goal):
        return goal

    def get_goal(self, goal_id: str):
        return None

    def get_campaign(self, campaign_id: str):
        return None

    def list_candidates_for_campaign(self, campaign_id: str):
        return []


@pytest.fixture()
def _fake_repo():
    app.dependency_overrides[get_repository] = lambda: _FakeRepository()
    yield
    app.dependency_overrides.clear()


def _settings_with_key(key: str = "test-secret-key-12345") -> Settings:
    """Create settings with an API key configured."""
    return Settings(
        api_key=key,
        mongodb_uri="mongodb://fake",
    )


def _settings_without_key() -> Settings:
    """Create settings without an API key configured."""
    return Settings(mongodb_uri="mongodb://fake")


class TestApiKeyRequired:
    """Verify auth enforcement when CRYOSWARM_API_KEY is configured."""

    def test_post_goal_without_key_returns_401(
        self,
        _fake_repo,
    ) -> None:
        app.dependency_overrides[auth_module.get_settings] = _settings_with_key
        client = TestClient(app)

        response = client.post(
            "/goals",
            json={
                "title": "Test goal auth",
                "scientific_objective": "Test auth enforcement.",
            },
        )

        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]
        app.dependency_overrides.clear()

    def test_post_goal_with_wrong_key_returns_403(
        self,
        _fake_repo,
    ) -> None:
        app.dependency_overrides[auth_module.get_settings] = _settings_with_key
        client = TestClient(app)

        response = client.post(
            "/goals",
            json={
                "title": "Test goal auth",
                "scientific_objective": "Test auth enforcement.",
            },
            headers={"X-API-Key": "wrong-key"},
        )

        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]
        app.dependency_overrides.clear()

    def test_post_goal_with_correct_key_succeeds(
        self,
        _fake_repo,
    ) -> None:
        app.dependency_overrides[auth_module.get_settings] = _settings_with_key
        client = TestClient(app)

        response = client.post(
            "/goals",
            json={
                "title": "Test goal auth success",
                "scientific_objective": "Test that valid key passes.",
            },
            headers={"X-API-Key": "test-secret-key-12345"},
        )

        assert response.status_code == 200
        app.dependency_overrides.clear()

    def test_get_health_without_key_succeeds(
        self,
        _fake_repo,
    ) -> None:
        app.dependency_overrides[auth_module.get_settings] = _settings_with_key
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        app.dependency_overrides.clear()

    def test_get_goal_without_key_succeeds(
        self,
        _fake_repo,
    ) -> None:
        app.dependency_overrides[auth_module.get_settings] = _settings_with_key
        client = TestClient(app)

        response = client.get("/goals/any-id")

        assert response.status_code == 404
        app.dependency_overrides.clear()


class TestApiKeyDisabled:
    """Verify local development bypass when no API key is configured."""

    def test_post_goal_without_key_succeeds_in_dev(
        self,
        _fake_repo,
    ) -> None:
        app.dependency_overrides[auth_module.get_settings] = _settings_without_key
        client = TestClient(app)

        response = client.post(
            "/goals",
            json={
                "title": "Test goal no auth",
                "scientific_objective": "Auth disabled in dev.",
            },
        )

        assert response.status_code == 200
        app.dependency_overrides.clear()


class TestTimingSafety:
    """Verify a constant-time comparison is used for secrets."""

    def test_compare_digest_is_used(self) -> None:
        source = inspect.getsource(auth_module.verify_api_key)
        assert "compare_digest" in source
