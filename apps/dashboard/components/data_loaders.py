"""Cached data loaders for the CryoSwarm-Q dashboard.

All MongoDB queries pass through this module so pages never touch
the repository directly.  Results are cached with a 60-second TTL.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from packages.core.models import (
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    MemoryRecord,
    RobustnessReport,
)
from packages.db.repositories import CryoSwarmRepository


def get_repository() -> CryoSwarmRepository:
    """Return a shared repository instance stored in session state."""
    if "repository" not in st.session_state:
        from packages.core.config import get_settings
        from packages.db.init_db import initialize_database

        settings = get_settings()
        if not settings.has_mongodb:
            st.error("MONGODB_URI is not configured.")
            st.stop()
        initialize_database()
        st.session_state["repository"] = CryoSwarmRepository(settings)
    return st.session_state["repository"]


@st.cache_data(ttl=60, show_spinner=False)
def load_latest_campaigns(limit: int = 20) -> list[dict[str, Any]]:
    repo = get_repository()
    campaigns = repo.list_latest_campaigns(limit=limit)
    return [c.model_dump(mode="json") for c in campaigns]


@st.cache_data(ttl=60, show_spinner=False)
def load_campaign(campaign_id: str) -> dict[str, Any] | None:
    repo = get_repository()
    campaign = repo.get_campaign(campaign_id)
    return campaign.model_dump(mode="json") if campaign else None


@st.cache_data(ttl=60, show_spinner=False)
def load_goal(goal_id: str) -> dict[str, Any] | None:
    repo = get_repository()
    goal = repo.get_goal(goal_id)
    return goal.model_dump(mode="json") if goal else None


@st.cache_data(ttl=60, show_spinner=False)
def load_ranked_candidates(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    results = repo.list_candidates_for_campaign(campaign_id)
    return [r.model_dump(mode="json") for r in results]


@st.cache_data(ttl=60, show_spinner=False)
def load_robustness_reports(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    reports = repo.list_robustness_reports(campaign_id)
    return [r.model_dump(mode="json") for r in reports]


@st.cache_data(ttl=60, show_spinner=False)
def load_agent_decisions(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    decisions = repo.list_agent_decisions(campaign_id)
    return [d.model_dump(mode="json") for d in decisions]


@st.cache_data(ttl=60, show_spinner=False)
def load_register_candidates(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    cursor = repo.collections["register_candidates"].find({"campaign_id": campaign_id})
    results = []
    for doc in cursor:
        doc.pop("_id", None)
        results.append(doc)
    return results


@st.cache_data(ttl=60, show_spinner=False)
def load_sequence_candidates(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    cursor = repo.collections["sequence_candidates"].find({"campaign_id": campaign_id})
    results = []
    for doc in cursor:
        doc.pop("_id", None)
        results.append(doc)
    return results


@st.cache_data(ttl=60, show_spinner=False)
def load_memory_records(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    records = repo.list_memory_records(campaign_id)
    return [r.model_dump(mode="json") for r in records]


@st.cache_data(ttl=60, show_spinner=False)
def load_all_memory(limit: int = 50) -> list[dict[str, Any]]:
    repo = get_repository()
    records = repo.list_recent_memory(limit=limit)
    return [r.model_dump(mode="json") for r in records]
