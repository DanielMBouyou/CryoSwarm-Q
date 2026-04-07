from __future__ import annotations

from packages.core.models import ScoringWeights
from packages.scoring.robustness import clamp_score


DEFAULT_WEIGHTS = ScoringWeights()


def compute_objective_score(
    observable_score: float,
    robustness: float,
    cost: float,
    latency: float,
    weights: ScoringWeights | None = None,
) -> float:
    active_weights = weights or DEFAULT_WEIGHTS
    score = (
        active_weights.alpha * observable_score
        + active_weights.beta * robustness
        - active_weights.gamma * cost
        - active_weights.delta * latency
    )
    return clamp_score(score)
