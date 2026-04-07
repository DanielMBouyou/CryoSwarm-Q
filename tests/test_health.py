from fastapi.testclient import TestClient

from apps.api.main import app

API_PREFIX = "/api/v1"


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)
    response = client.get(f"{API_PREFIX}/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "mongodb_configured" in payload
