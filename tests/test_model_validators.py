from __future__ import annotations

import pytest
from pydantic import ValidationError

from packages.core.enums import NoiseLevel, SequenceFamily
from packages.core.models import (
    DemoGoalRequest,
    ExperimentSpec,
    NoiseScenario,
    RegisterCandidate,
    RobustnessReport,
    ScoringWeights,
    SequenceCandidate,
)


def test_scoring_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        ScoringWeights(alpha=0.5, beta=0.5, gamma=0.2, delta=0.1)


def test_experiment_spec_rejects_max_less_than_min() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec(
            goal_id="goal_test",
            objective_class="balanced_campaign_search",
            target_observable="rydberg_density",
            min_atoms=6,
            max_atoms=4,
            preferred_layouts=["line"],
            sequence_families=[SequenceFamily.GLOBAL_RAMP],
            reasoning_summary="Invalid atom window.",
        )


def test_register_candidate_rejects_feasibility_above_one() -> None:
    with pytest.raises(ValidationError):
        RegisterCandidate(
            campaign_id="campaign_test",
            spec_id="spec_test",
            label="invalid-register",
            layout_type="line",
            atom_count=4,
            coordinates=[(0.0, 0.0), (7.0, 0.0), (14.0, 0.0), (21.0, 0.0)],
            min_distance_um=7.0,
            blockade_radius_um=6.0,
            blockade_pair_count=0,
            van_der_waals_matrix=[[0.0] * 4 for _ in range(4)],
            feasibility_score=1.1,
            reasoning_summary="Invalid feasibility score.",
        )


def test_register_candidate_rejects_negative_min_distance() -> None:
    with pytest.raises(ValidationError):
        RegisterCandidate(
            campaign_id="campaign_test",
            spec_id="spec_test",
            label="invalid-register",
            layout_type="line",
            atom_count=4,
            coordinates=[(0.0, 0.0), (7.0, 0.0), (14.0, 0.0), (21.0, 0.0)],
            min_distance_um=-1.0,
            blockade_radius_um=6.0,
            blockade_pair_count=0,
            van_der_waals_matrix=[[0.0] * 4 for _ in range(4)],
            feasibility_score=0.7,
            reasoning_summary="Invalid minimum distance.",
        )


def test_sequence_candidate_rejects_detuning_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SequenceCandidate(
            campaign_id="campaign_test",
            spec_id="spec_test",
            register_candidate_id="reg_test",
            label="invalid-detuning",
            sequence_family=SequenceFamily.CONSTANT_DRIVE,
            duration_ns=1000,
            amplitude=5.0,
            detuning=130.0,
            phase=0.0,
            predicted_cost=0.2,
            reasoning_summary="Invalid detuning.",
        )


def test_robustness_report_rejects_nominal_above_one() -> None:
    with pytest.raises(ValidationError):
        RobustnessReport(
            campaign_id="campaign_test",
            sequence_candidate_id="seq_test",
            nominal_score=1.1,
            perturbation_average=0.8,
            robustness_penalty=0.1,
            robustness_score=0.7,
            worst_case_score=0.6,
            score_std=0.05,
            target_observable="rydberg_density",
            reasoning_summary="Invalid report.",
        )


def test_noise_scenario_rejects_negative_temperature() -> None:
    with pytest.raises(ValidationError):
        NoiseScenario(
            label=NoiseLevel.LOW,
            amplitude_jitter=0.03,
            detuning_jitter=0.02,
            dephasing_rate=0.03,
            atom_loss_rate=0.01,
            temperature_uk=-5.0,
        )


def test_demo_goal_request_rejects_short_title() -> None:
    with pytest.raises(ValidationError):
        DemoGoalRequest(title="No", scientific_objective="Too short title", desired_atom_count=4)
