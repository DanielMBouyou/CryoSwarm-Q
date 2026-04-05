from __future__ import annotations

from packages.core.config import Settings
from packages.pasqal_adapters.pasqal_cloud_adapter import PasqalCloudAdapter


def test_unavailable_without_credentials() -> None:
    adapter = PasqalCloudAdapter(
        Settings(
            pasqal_cloud_project_id="",
            pasqal_token="",
            pasqal_cloud_username="",
            pasqal_cloud_password="",
        )
    )

    result = adapter.submit_batch({"sequence": "test"})

    assert result["status"] == "unavailable"


def test_available_property_without_credentials_is_false() -> None:
    adapter = PasqalCloudAdapter(
        Settings(
            pasqal_cloud_project_id="",
            pasqal_token="",
            pasqal_cloud_username="",
            pasqal_cloud_password="",
        )
    )

    assert adapter.available is False
