"""Tests for the objective scoring function."""
from __future__ import annotations

import pytest

from packages.core.models import ScoringWeights
from packages.scoring.objective import compute_objective_score


class TestObjectiveScore:
    def test_uses_weighted_terms(self) -> None:
        score = compute_objective_score(
            observable_score=0.8,
            robustness=0.7,
            cost=0.2,
            latency=0.1,
            weights=ScoringWeights(alpha=0.5, beta=0.3, gamma=0.1, delta=0.1),
        )
        expected = 0.5 * 0.8 + 0.3 * 0.7 - 0.1 * 0.2 - 0.1 * 0.1
        assert score == pytest.approx(expected, abs=1e-4)

    def test_higher_fidelity_gives_higher_score(self) -> None:
        low = compute_objective_score(0.3, 0.5, 0.2, 0.1)
        high = compute_objective_score(0.9, 0.5, 0.2, 0.1)
        assert high > low

    def test_higher_cost_gives_lower_score(self) -> None:
        cheap = compute_objective_score(0.8, 0.7, 0.1, 0.1)
        expensive = compute_objective_score(0.8, 0.7, 0.9, 0.1)
        assert cheap > expensive

    def test_default_weights_sum_to_one(self) -> None:
        w = ScoringWeights()
        assert w.alpha + w.beta + w.gamma + w.delta == pytest.approx(1.0)

    def test_score_is_clamped_to_zero(self) -> None:
        score = compute_objective_score(
            observable_score=0.1,
            robustness=0.1,
            cost=1.0,
            latency=1.0,
        )
        assert score == 0.0

    def test_score_is_clamped_to_one(self) -> None:
        score = compute_objective_score(
            observable_score=1.0,
            robustness=1.0,
            cost=0.0,
            latency=0.0,
            weights=ScoringWeights(alpha=0.7, beta=0.3, gamma=0.0, delta=0.0),
        )
        assert score == 1.0
