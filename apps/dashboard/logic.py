from __future__ import annotations

"""Pure dashboard helper functions for CryoSwarm-Q.

This module extracts data-shaping logic from the Streamlit application so it
can be tested without a live MongoDB connection or Streamlit runtime.
"""

from typing import Any

from packages.core.models import (
    AgentDecision,
    CampaignState,
    EvaluationResult,
    RegisterCandidate,
    RobustnessReport,
)
from packages.orchestration.events import PipelineEvent


def build_campaign_table(campaigns: list[CampaignState]) -> list[dict[str, Any]]:
    """Transform campaign models into table-ready dictionaries."""

    return [
        {
            "campaign_id": campaign.id,
            "goal_id": campaign.goal_id,
            "status": campaign.status.value,
            "candidate_count": campaign.candidate_count,
            "top_candidate_id": campaign.top_candidate_id,
        }
        for campaign in campaigns
    ]


def build_ranked_table(candidates: list[EvaluationResult]) -> list[dict[str, Any]]:
    """Transform ranked candidates into a display table."""

    return [
        {
            "rank": candidate.final_rank,
            "sequence_candidate_id": candidate.sequence_candidate_id,
            "objective_score": candidate.objective_score,
            "robustness_score": candidate.robustness_score,
            "backend_choice": candidate.backend_choice.value,
        }
        for candidate in candidates
    ]


def build_decision_table(decisions: list[AgentDecision]) -> list[dict[str, str]]:
    """Transform agent decisions into a display table."""

    return [
        {
            "agent": decision.agent_name.value,
            "subject_id": decision.subject_id,
            "decision_type": decision.decision_type.value,
            "status": decision.status,
            "reasoning_summary": decision.reasoning_summary,
        }
        for decision in decisions
    ]


def build_event_table(
    events: list[PipelineEvent],
    limit: int = 12,
) -> list[dict[str, str]]:
    """Transform pipeline events into a compact monitoring table."""

    recent_events = events[-limit:] if limit > 0 else events
    return [
        {
            "time": event.created_at.strftime("%H:%M:%S"),
            "event": event.event_type,
            "phase": str(event.payload.get("phase", "")),
            "status": str(event.payload.get("summary_status", event.payload.get("status", ""))),
        }
        for event in recent_events
    ]


def select_robustness_chart_data(
    reports: list[RobustnessReport],
    limit: int = 5,
) -> tuple[list[str], list[float], list[float], list[float]]:
    """Extract chart-ready robustness comparison data."""

    if not reports:
        return [], [], [], []
    top_reports = reports[:limit]
    labels = [report.sequence_candidate_id[-6:] for report in top_reports]
    nominal = [report.nominal_score for report in top_reports]
    average = [report.perturbation_average for report in top_reports]
    worst = [report.worst_case_score for report in top_reports]
    return labels, nominal, average, worst


def select_noise_sensitivity_data(
    report: RobustnessReport,
) -> tuple[list[str], list[float]]:
    """Extract ordered noise-sensitivity plot data from one robustness report."""

    scenario_order = ["low_noise", "medium_noise", "stressed_noise"]
    labels = [label for label in scenario_order if label in report.scenario_scores]
    values = [report.scenario_scores[label] for label in labels]
    return labels, values


def build_register_lookup_from_documents(
    documents: list[dict[str, Any]],
) -> dict[str, RegisterCandidate]:
    """Parse raw repository documents into a register lookup keyed by document id."""

    return {
        document["_id"]: RegisterCandidate.model_validate(
            {key: value for key, value in document.items() if key != "_id"}
        )
        for document in documents
    }
