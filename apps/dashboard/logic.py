from __future__ import annotations

"""Pure dashboard helper functions for CryoSwarm-Q.

This module extracts data-shaping logic from the Streamlit application so it
can be tested without a live MongoDB connection or Streamlit runtime.
"""

from typing import Any

import numpy as np

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


def generate_waveform(
    family: str,
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    n_points: int = 200,
) -> tuple[list[float], list[float], list[float]]:
    """Return (time_ns, omega_values, delta_values) for a given pulse family."""

    t = np.linspace(0, duration_ns, n_points)
    T = duration_ns

    if family == "constant_drive":
        omega = np.full_like(t, omega_max)
        delta = np.full_like(t, delta_start)
    elif family == "global_ramp":
        omega = omega_max * t / T
        delta = delta_start + (delta_end - delta_start) * t / T
    elif family == "detuning_scan":
        omega = np.full_like(t, omega_max)
        delta = delta_start + (delta_end - delta_start) * t / T
    elif family == "adiabatic_sweep":
        omega = omega_max * np.sin(np.pi * t / T) ** 2
        delta = delta_start + (delta_end - delta_start) * t / T
    elif family == "blackman_sweep":
        omega = omega_max * (
            0.42 - 0.50 * np.cos(2 * np.pi * t / T) + 0.08 * np.cos(4 * np.pi * t / T)
        )
        delta = delta_start + (delta_end - delta_start) * t / T
    else:
        omega = np.full_like(t, omega_max)
        delta = np.full_like(t, delta_start)

    return t.tolist(), omega.tolist(), delta.tolist()


def compute_pareto_front(candidates: list[dict[str, Any]]) -> list[int]:
    """Return indices of Pareto-optimal candidates (maximise objective_score and worst_case_score)."""

    pareto_indices: list[int] = []
    for i, c in enumerate(candidates):
        c_obj = c.get("objective_score", 0)
        c_worst = c.get("worst_case_score", 0)
        dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            o_obj = other.get("objective_score", 0)
            o_worst = other.get("worst_case_score", 0)
            if o_obj >= c_obj and o_worst >= c_worst and (o_obj > c_obj or o_worst > c_worst):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    return pareto_indices
