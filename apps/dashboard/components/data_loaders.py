"""Cached data loaders for the CryoSwarm-Q dashboard.

All MongoDB queries pass through this module so pages never touch
the repository directly.  Results are cached with a 60-second TTL.

When MongoDB is unavailable every loader returns an empty list (or
``None``) so that dashboard pages can degrade gracefully instead of
crashing.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
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

_log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]
TRACKING_TABLES_DIR = ROOT_DIR / "experiments" / "tracking" / "tables"


def _mongo_available() -> bool:
    """Return True when a MONGODB_URI is configured."""
    from packages.core.config import get_settings
    return get_settings().has_mongodb


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
    if not _mongo_available():
        return []
    try:
        repo = get_repository()
        campaigns = repo.list_latest_campaigns(limit=limit)
        return [c.model_dump(mode="json") for c in campaigns]
    except Exception as exc:
        _log.warning("load_latest_campaigns failed: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def load_campaign(campaign_id: str) -> dict[str, Any] | None:
    if not _mongo_available():
        return None
    try:
        repo = get_repository()
        campaign = repo.get_campaign(campaign_id)
        return campaign.model_dump(mode="json") if campaign else None
    except Exception as exc:
        _log.warning("load_campaign failed: %s", exc)
        return None


@st.cache_data(ttl=60, show_spinner=False)
def load_goal(goal_id: str) -> dict[str, Any] | None:
    if not _mongo_available():
        return None
    try:
        repo = get_repository()
        goal = repo.get_goal(goal_id)
        return goal.model_dump(mode="json") if goal else None
    except Exception as exc:
        _log.warning("load_goal failed: %s", exc)
        return None


@st.cache_data(ttl=60, show_spinner=False)
def load_ranked_candidates(campaign_id: str) -> list[dict[str, Any]]:
    if not _mongo_available():
        return []
    try:
        repo = get_repository()
        results = repo.list_candidates_for_campaign(campaign_id)
        return [r.model_dump(mode="json") for r in results]
    except Exception as exc:
        _log.warning("load_ranked_candidates failed: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def load_robustness_reports(campaign_id: str) -> list[dict[str, Any]]:
    if not _mongo_available():
        return []
    try:
        repo = get_repository()
        reports = repo.list_robustness_reports(campaign_id)
        return [r.model_dump(mode="json") for r in reports]
    except Exception as exc:
        _log.warning("load_robustness_reports failed: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def load_agent_decisions(campaign_id: str) -> list[dict[str, Any]]:
    if not _mongo_available():
        return []
    try:
        repo = get_repository()
        decisions = repo.list_agent_decisions(campaign_id)
        return [d.model_dump(mode="json") for d in decisions]
    except Exception as exc:
        _log.warning("load_agent_decisions failed: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def load_register_candidates(campaign_id: str) -> list[dict[str, Any]]:
    if not _mongo_available():
        return []
    try:
        repo = get_repository()
        cursor = repo.collections["register_candidates"].find({"campaign_id": campaign_id})
        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(doc)
        return results
    except Exception as exc:
        _log.warning("load_register_candidates failed: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def load_sequence_candidates(campaign_id: str) -> list[dict[str, Any]]:
    if not _mongo_available():
        return []
    try:
        repo = get_repository()
        cursor = repo.collections["sequence_candidates"].find({"campaign_id": campaign_id})
        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(doc)
        return results
    except Exception as exc:
        _log.warning("load_sequence_candidates failed: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def load_memory_records(campaign_id: str) -> list[dict[str, Any]]:
    if not _mongo_available():
        return []
    try:
        repo = get_repository()
        records = repo.list_memory_records(campaign_id)
        return [r.model_dump(mode="json") for r in records]
    except Exception as exc:
        _log.warning("load_memory_records failed: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def load_all_memory(limit: int = 50) -> list[dict[str, Any]]:
    if not _mongo_available():
        return []
    try:
        repo = get_repository()
        records = repo.list_recent_memory(limit=limit)
        return [r.model_dump(mode="json") for r in records]
    except Exception as exc:
        _log.warning("load_all_memory failed: %s", exc)
        return []


def _load_tracking_csv(filename: str) -> list[dict[str, Any]]:
    path = TRACKING_TABLES_DIR / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@st.cache_data(ttl=30, show_spinner=False)
def load_tracking_test_catalog() -> list[dict[str, Any]]:
    return _load_tracking_csv("test_catalog.csv")


@st.cache_data(ttl=30, show_spinner=False)
def load_tracking_run_registry() -> list[dict[str, Any]]:
    return _load_tracking_csv("run_registry.csv")


@st.cache_data(ttl=30, show_spinner=False)
def load_tracking_step_metrics() -> list[dict[str, Any]]:
    return _load_tracking_csv("step_metrics.csv")


@st.cache_data(ttl=30, show_spinner=False)
def load_tracking_summary_metrics() -> list[dict[str, Any]]:
    return _load_tracking_csv("summary_metrics.csv")


@st.cache_data(ttl=30, show_spinner=False)
def load_tracking_artifacts() -> list[dict[str, Any]]:
    return _load_tracking_csv("artifact_registry.csv")


@st.cache_data(ttl=30, show_spinner=False)
def load_tracking_test_observations() -> list[dict[str, Any]]:
    return _load_tracking_csv("test_observations.csv")
