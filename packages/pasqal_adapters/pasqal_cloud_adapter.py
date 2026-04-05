from __future__ import annotations

from typing import Any

from packages.core.config import Settings, get_settings


class PasqalCloudAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def authenticate(self) -> dict[str, Any]:
        if not self.settings.has_pasqal_cloud_credentials:
            return {
                "authenticated": False,
                "mode": "mock",
                "message": "Pasqal Cloud credentials are not configured. Live submission is disabled.",
            }
        return {
            "authenticated": True,
            "mode": "credential_ready",
            "message": "Credentials detected. Live submission methods remain conservative by default.",
        }

    def submit_batch(self, batch_payload: dict[str, Any]) -> dict[str, Any]:
        auth_state = self.authenticate()
        if not auth_state["authenticated"]:
            return {
                "submitted": False,
                "mode": "mock",
                "message": "Mock submission only. No live Pasqal Cloud call was made.",
                "payload_preview": batch_payload,
            }
        return {
            "submitted": False,
            "mode": "safe_placeholder",
            "message": "Credentials are present, but real submission is intentionally not implemented yet.",
            "payload_preview": batch_payload,
        }

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "status": "not_available",
            "message": "Job tracking is a placeholder until live Pasqal Cloud integration is implemented.",
        }
