from __future__ import annotations

from packages.core.models import ScoringWeights


DEFAULT_WEIGHTS = ScoringWeights()


def compute_objective_score(
    fidelity: float,
    robustness: float,
    cost: float,
    latency: float,
    weights: ScoringWeights | None = None,
) -> float:
    active_weights = weights or DEFAULT_WEIGHTS
    score = (
        active_weights.alpha * fidelity
        + active_weights.beta * robustness
        - active_weights.gamma * cost
        - active_weights.delta * latency
    )
    return round(score, 4)
