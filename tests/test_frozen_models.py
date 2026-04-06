"""Regression tests for immutable validated domain models.

These tests ensure value objects remain frozen after construction while
still supporting controlled updates through ``model_copy(update=...)``.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from packages.core.enums import AgentName, BackendType, CandidateStatus, DecisionType, GoalStatus
from packages.core.models import (
    AgentDecision,
    BackendChoice,
    EvaluationResult,
    MemoryRecord,
    RobustnessReport,
    ScoringWeights,
)


def test_scoring_weights_are_frozen() -> None:
    weights = ScoringWeights()

    with pytest.raises(ValidationError):
        weights.alpha = 0.99

    updated = weights.model_copy(update={"alpha": 0.45, "beta": 0.35, "gamma": 0.10, "delta": 0.10})
    assert updated.alpha == pytest.approx(0.45)


def test_experiment_spec_is_frozen(sample_spec) -> None:
    with pytest.raises(ValidationError):
        sample_spec.min_atoms = 999

    updated = sample_spec.model_copy(update={"min_atoms": 3})
    assert updated.min_atoms == 3


def test_register_candidate_is_frozen(sample_register) -> None:
    with pytest.raises(ValidationError):
        sample_register.label = "mutated"

    updated = sample_register.model_copy(update={"label": "updated"})
    assert updated.label == "updated"


def test_sequence_candidate_is_frozen(sample_sequence) -> None:
    with pytest.raises(ValidationError):
        sample_sequence.duration_ns = 4000

    updated = sample_sequence.model_copy(update={"duration_ns": 4000})
    assert updated.duration_ns == 4000


def test_robustness_report_is_frozen(sample_report: RobustnessReport) -> None:
    with pytest.raises(ValidationError):
        sample_report.robustness_score = 0.95

    updated = sample_report.model_copy(update={"robustness_score": 0.95})
    assert updated.robustness_score == pytest.approx(0.95)


def test_backend_choice_is_frozen() -> None:
    choice = BackendChoice(
        campaign_id="campaign_test",
        sequence_candidate_id="seq_test",
        recommended_backend=BackendType.LOCAL_PULSER_SIMULATION,
        state_dimension=64,
        estimated_cost=0.1,
        estimated_latency=0.2,
        rationale="Test backend selection.",
    )

    with pytest.raises(ValidationError):
        choice.estimated_cost = 0.5

    updated = choice.model_copy(update={"estimated_cost": 0.5})
    assert updated.estimated_cost == pytest.approx(0.5)


def test_agent_decision_is_frozen() -> None:
    decision = AgentDecision(
        campaign_id="campaign_test",
        agent_name=AgentName.SEQUENCE,
        subject_id="seq_test",
        decision_type=DecisionType.CANDIDATE_GENERATION,
        status="completed",
        reasoning_summary="Generated candidates.",
    )

    with pytest.raises(ValidationError):
        decision.status = "failed"

    updated = decision.model_copy(update={"status": "failed"})
    assert updated.status == "failed"


def test_evaluation_result_is_frozen_and_rank_updates_via_copy() -> None:
    result = EvaluationResult(
        campaign_id="campaign_test",
        sequence_candidate_id="seq_test",
        register_candidate_id="reg_test",
        nominal_score=0.8,
        robustness_score=0.75,
        worst_case_score=0.65,
        observable_score=0.77,
        objective_score=0.76,
        backend_choice=BackendType.LOCAL_PULSER_SIMULATION,
        estimated_cost=0.12,
        estimated_latency=0.18,
        reasoning_summary="Scored candidate.",
    )

    with pytest.raises(ValidationError):
        result.final_rank = 1

    ranked = result.model_copy(update={"final_rank": 1, "status": CandidateStatus.RANKED})
    assert ranked.final_rank == 1
    assert ranked.status == CandidateStatus.RANKED


def test_memory_record_is_frozen() -> None:
    record = MemoryRecord(
        campaign_id="campaign_test",
        source_candidate_id="seq_test",
        lesson_type="candidate_pattern",
        summary="Reusable lesson.",
        signals={"confidence": 0.8},
        reusable_tags=["strong"],
    )

    with pytest.raises(ValidationError):
        record.summary = "mutated"

    updated = record.model_copy(update={"summary": "updated"})
    assert updated.summary == "updated"


def test_state_models_remain_mutable(sample_goal) -> None:
    sample_goal.status = GoalStatus.COMPLETED
    sample_goal.updated_at = datetime.now(UTC)
    assert sample_goal.status == GoalStatus.COMPLETED
