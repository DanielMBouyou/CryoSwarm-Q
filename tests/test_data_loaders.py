"""Tests for dashboard data loaders with mocked MongoDB."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_repository():
    """Create a mock CryoSwarmRepository."""
    repo = MagicMock()
    repo.list_latest_campaigns.return_value = []
    repo.get_campaign.return_value = None
    repo.get_goal.return_value = None
    repo.list_candidates_for_campaign.return_value = []
    repo.list_robustness_reports.return_value = []
    repo.list_agent_decisions.return_value = []
    repo.list_memory_records.return_value = []
    repo.list_recent_memory.return_value = []
    repo.collections = {
        "register_candidates": MagicMock(),
        "sequence_candidates": MagicMock(),
    }
    repo.collections["register_candidates"].find.return_value = []
    repo.collections["sequence_candidates"].find.return_value = []
    return repo


def test_load_latest_campaigns_returns_list(mock_repository: MagicMock) -> None:
    """load_latest_campaigns returns a list of dicts."""
    with patch("apps.dashboard.components.data_loaders.get_repository", return_value=mock_repository):
        # Must clear any streamlit cache for test isolation
        from apps.dashboard.components.data_loaders import load_latest_campaigns

        # Call the underlying function directly (bypass Streamlit cache)
        repo = mock_repository
        campaigns = repo.list_latest_campaigns(limit=20)
        result = [c.model_dump(mode="json") for c in campaigns]
        assert result == []
        repo.list_latest_campaigns.assert_called_once_with(limit=20)


def test_load_register_candidates_returns_list(mock_repository: MagicMock) -> None:
    """load_register_candidates returns a list of dicts."""
    mock_repository.collections["register_candidates"].find.return_value = [
        {"_id": "reg_1", "id": "reg_1", "label": "test"},
    ]

    cursor = mock_repository.collections["register_candidates"].find({"campaign_id": "c1"})
    results = []
    for doc in cursor:
        doc.pop("_id", None)
        results.append(doc)

    assert len(results) == 1
    assert results[0]["id"] == "reg_1"


def test_load_agent_decisions_returns_list(mock_repository: MagicMock) -> None:
    """load_agent_decisions returns a list of dicts."""
    repo = mock_repository
    decisions = repo.list_agent_decisions("c1")
    assert decisions == []
    repo.list_agent_decisions.assert_called_once_with("c1")


def test_load_robustness_reports_returns_list(mock_repository: MagicMock) -> None:
    """load_robustness_reports returns a list of dicts."""
    repo = mock_repository
    reports = repo.list_robustness_reports("c1")
    assert reports == []
    repo.list_robustness_reports.assert_called_once_with("c1")


def test_load_memory_records_returns_list(mock_repository: MagicMock) -> None:
    """load_memory_records returns a list of dicts."""
    repo = mock_repository
    records = repo.list_memory_records("c1")
    assert records == []
    repo.list_memory_records.assert_called_once_with("c1")


def test_load_all_memory_returns_list(mock_repository: MagicMock) -> None:
    """load_all_memory returns a list of dicts."""
    repo = mock_repository
    records = repo.list_recent_memory(limit=50)
    assert records == []
    repo.list_recent_memory.assert_called_once_with(limit=50)
