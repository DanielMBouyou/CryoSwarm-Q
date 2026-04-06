from __future__ import annotations

import json
from typing import Any

from packages.core.config import Settings, get_settings

try:
    from pasqal_cloud import BatchStatus, SDK
    from pasqal_cloud.authentication import TokenProvider

    PASQAL_CLOUD_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional dependency
    SDK = None  # type: ignore[assignment]
    BatchStatus = None  # type: ignore[assignment]
    TokenProvider = object  # type: ignore[assignment]
    PASQAL_CLOUD_AVAILABLE = False
    PASQAL_CLOUD_IMPORT_ERROR = str(exc)
else:
    PASQAL_CLOUD_IMPORT_ERROR = None


class _StaticTokenProvider(TokenProvider):
    def __init__(self, token: str) -> None:
        self._token = token

    def get_token(self) -> str:
        return self._token


class PasqalCloudAdapter:
    """Thin Pasqal Cloud adapter with graceful degradation when credentials are absent."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._available = bool(
            PASQAL_CLOUD_AVAILABLE
            and self.settings.pasqal_cloud_project_id
            and (
                self.settings.pasqal_token
                or (self.settings.pasqal_cloud_username and self.settings.pasqal_cloud_password)
            )
        )
        self._sdk = None
        if self._available:
            if self.settings.pasqal_token:
                self._sdk = SDK(
                    project_id=self.settings.pasqal_cloud_project_id,
                    token_provider=_StaticTokenProvider(self.settings.pasqal_token),
                )
            else:
                self._sdk = SDK(
                    project_id=self.settings.pasqal_cloud_project_id,
                    username=self.settings.pasqal_cloud_username,
                    password=self.settings.pasqal_cloud_password,
                )

    @property
    def available(self) -> bool:
        return self._available

    def authenticate(self) -> dict[str, Any]:
        if not PASQAL_CLOUD_AVAILABLE:
            return {
                "authenticated": False,
                "mode": "unavailable",
                "message": PASQAL_CLOUD_IMPORT_ERROR or "pasqal-cloud is not installed.",
            }
        if not self.available:
            return {
                "authenticated": False,
                "mode": "unavailable",
                "message": "No Pasqal Cloud credentials configured.",
            }
        return {
            "authenticated": True,
            "mode": "configured",
            "message": "Pasqal Cloud SDK is configured.",
        }

    def submit_batch(self, serialized_sequence: dict[str, Any], n_runs: int = 100) -> dict[str, Any]:
        """Submit a serialized sequence to Pasqal Cloud when credentials are configured."""
        if not self.available or self._sdk is None:
            return {"status": "unavailable", "reason": "No Pasqal Cloud credentials configured"}

        batch = self._sdk.create_batch(
            serialized_sequence=json.dumps(serialized_sequence),
            jobs=[{"runs": n_runs}],
            emulator=None,
        )
        created_at = getattr(batch, "created_at", None)
        return {
            "batch_id": batch.id,
            "status": batch.status.value,
            "created_at": created_at.isoformat() if created_at else None,
        }

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        if not self.available or self._sdk is None:
            return {"status": "unavailable"}

        batch = self._sdk.get_batch(batch_id)
        results = None
        if getattr(batch, "status", None) == BatchStatus.DONE and getattr(batch, "ordered_jobs", None):
            results = batch.ordered_jobs[0].result
        return {
            "batch_id": batch.id,
            "status": batch.status.value,
            "results": results,
        }

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Backward-compatible alias around batch status lookup."""
        return self.get_batch_status(job_id)
