from __future__ import annotations

"""CORS middleware tests for the CryoSwarm-Q API."""

from fastapi.testclient import TestClient

from apps.api.main import app


def test_cors_allows_configured_origin() -> None:
    client = TestClient(app)

    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_cors_blocks_unknown_origin() -> None:
    client = TestClient(app)

    response = client.options(
        "/health",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_cors_does_not_allow_wildcard() -> None:
    client = TestClient(app)

    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:8501"},
    )

    assert response.headers.get("access-control-allow-origin") != "*"


def test_cors_allows_api_key_header() -> None:
    client = TestClient(app)

    response = client.options(
        "/goals",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-API-Key",
        },
    )

    allowed_headers = response.headers.get("access-control-allow-headers", "").lower()
    assert "x-api-key" in allowed_headers
