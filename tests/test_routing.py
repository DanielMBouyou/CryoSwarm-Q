"""Tests for the emulator routing logic."""
from __future__ import annotations

import pytest

from packages.core.enums import BackendType, NoiseLevel, SequenceFamily
from packages.core.models import (
    ExperimentSpec,
    RobustnessReport,
    SequenceCandidate,
    ScoringWeights,
)
from packages.pasqal_adapters.emulator_router import recommend_backend


def _make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        goal_id="g1",
        objective_class="balanced",
        target_observable="rydberg_density",
        min_atoms=4,
        max_atoms=8,
        preferred_layouts=["line"],
        sequence_families=[SequenceFamily.ADIABATIC_SWEEP],
        reasoning_summary="test",
    )


def _make_seq(atom_count: int) -> SequenceCandidate:
    return SequenceCandidate(
        campaign_id="c1",
        spec_id="s1",
        register_candidate_id="r1",
        label="test-seq",
        sequence_family=SequenceFamily.ADIABATIC_SWEEP,
        duration_ns=3000,
        amplitude=5.0,
        detuning=-15.0,
        phase=0.0,
        predicted_cost=0.3,
        reasoning_summary="test",
        metadata={"atom_count": atom_count},
    )


def _make_report(robustness: float, worst: float) -> RobustnessReport:
    return RobustnessReport(
        campaign_id="c1",
        sequence_candidate_id="seq1",
        nominal_score=0.8,
        perturbation_average=robustness,
        robustness_penalty=0.1,
        robustness_score=robustness,
        worst_case_score=worst,
        score_std=0.05,
        target_observable="rydberg_density",
        reasoning_summary="test",
    )


class TestRouting:
    def test_small_robust_gets_emulator_sv(self) -> None:
        choice = recommend_backend(_make_spec(), _make_seq(4), _make_report(0.8, 0.7))
        assert choice.recommended_backend == BackendType.EMU_SV_CANDIDATE

    def test_medium_gets_emulator_mps(self) -> None:
        choice = recommend_backend(_make_spec(), _make_seq(12), _make_report(0.6, 0.5))
        assert choice.recommended_backend == BackendType.EMU_MPS_CANDIDATE

    def test_large_gets_local(self) -> None:
        choice = recommend_backend(_make_spec(), _make_seq(20), _make_report(0.3, 0.2))
        assert choice.recommended_backend == BackendType.LOCAL_PULSER_SIMULATION

    def test_state_dimension_populated(self) -> None:
        choice = recommend_backend(_make_spec(), _make_seq(6), _make_report(0.7, 0.6))
        assert choice.state_dimension == 2**6
