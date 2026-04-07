"""Tests for the robustness scoring module."""
from __future__ import annotations

import pytest

from packages.core.parameter_space import PhysicsParameterSpace
from packages.scoring.robustness import (
    clamp_score,
    perturbation_average,
    perturbation_std,
    robustness_penalty,
    robustness_score,
    worst_case_score,
)


class TestClampScore:
    def test_within_bounds(self) -> None:
        assert clamp_score(0.5) == 0.5

    def test_below_zero(self) -> None:
        assert clamp_score(-0.3) == 0.0

    def test_above_one(self) -> None:
        assert clamp_score(1.5) == 1.0

    def test_rounds(self) -> None:
        assert clamp_score(0.12345) == 0.1235


class TestPerturbationAverage:
    def test_empty(self) -> None:
        assert perturbation_average([]) == 0.0

    def test_single(self) -> None:
        assert perturbation_average([0.7]) == 0.7

    def test_average(self) -> None:
        assert perturbation_average([0.6, 0.8]) == 0.7


class TestPerturbationStd:
    def test_single(self) -> None:
        assert perturbation_std([0.5]) == 0.0

    def test_identical(self) -> None:
        assert perturbation_std([0.5, 0.5, 0.5]) == 0.0

    def test_nonzero(self) -> None:
        std = perturbation_std([0.4, 0.6, 0.8])
        assert std > 0


class TestWorstCase:
    def test_returns_minimum(self) -> None:
        assert worst_case_score([0.9, 0.5, 0.7]) == 0.5

    def test_empty(self) -> None:
        assert worst_case_score([]) == 0.0


class TestRobustnessPenalty:
    def test_no_degradation(self) -> None:
        penalty = robustness_penalty(0.8, 0.8, 0.0)
        assert penalty == 0.0

    def test_degradation(self) -> None:
        penalty = robustness_penalty(0.8, 0.5, 0.1)
        assert penalty > 0


class TestRobustnessScore:
    def test_perfect_scores(self) -> None:
        score = robustness_score(1.0, 1.0, 1.0, 0.0)
        assert score > 0.9

    def test_degraded_scores(self) -> None:
        good = robustness_score(0.9, 0.85, 0.80, 0.02)
        bad = robustness_score(0.9, 0.50, 0.30, 0.20)
        assert good > bad

    def test_bounded(self) -> None:
        score = robustness_score(0.5, 0.4, 0.3, 0.1)
        assert 0.0 <= score <= 1.0

    def test_goal_constraints_can_shift_weights_toward_worst_case(self) -> None:
        param_space = PhysicsParameterSpace.default()
        baseline = robustness_score(0.9, 0.85, 0.4, 0.05, param_space=param_space)
        tuned = robustness_score(
            0.9,
            0.85,
            0.4,
            0.05,
            param_space=param_space,
            constraints={
                "robustness_profile": "worst_case_safety",
                "robustness_weight_smoothing": 1.0,
            },
        )
        assert tuned < baseline
