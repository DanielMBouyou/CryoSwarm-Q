"""Tests for benchmark metric computation."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark import (
    compute_pipeline_benchmark,
    compute_rl_benchmark,
    compute_surrogate_benchmark,
)

pytestmark = pytest.mark.slow


def test_compute_surrogate_benchmark_shapes():
    y_true = np.array(
        [
            [0.8, 0.7, 0.6, 0.5],
            [0.4, 0.5, 0.6, 0.7],
        ],
        dtype=np.float32,
    )
    y_pred = y_true * 0.95
    report = compute_surrogate_benchmark(y_true, y_pred)
    assert report.n_test_samples == 2
    assert set(report.mse_per_target) == {"robustness", "nominal", "worst_case", "observable"}
    assert report.mse_total >= 0.0


def test_compute_rl_benchmark_threshold():
    rewards = [0.1, 0.2, 0.35, 0.4]
    report = compute_rl_benchmark(rewards, threshold=0.3)
    assert report.episodes_to_threshold == 3
    assert report.best_episode_reward == 0.4


def test_compute_pipeline_benchmark_sources():
    report = compute_pipeline_benchmark(
        [0.4, 0.6, 0.8],
        source_scores={"heuristic": [0.4, 0.6], "rl_policy": [0.8]},
    )
    assert report.top_candidate_robustness == 0.8
    assert report.n_candidates_evaluated == 3
    assert report.heuristic_vs_rl_comparison is not None
