from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from packages.core.parameter_space import PhysicsParameterSpace


def clamp_score(value: float) -> float:
    """Project a raw scalar score into the normalized [0, 1] interval."""
    return max(0.0, min(1.0, round(value, 4)))


def nominal_score(base_signal: float, control_penalty: float, hardware_bonus: float) -> float:
    """Combine nominal observable signal with control and hardware regularizers."""
    return clamp_score(base_signal - control_penalty + hardware_bonus)


def perturbation_average(scores: list[float]) -> float:
    """Average perturbed scores across noise scenarios."""
    if not scores:
        return 0.0
    return clamp_score(mean(scores))


def perturbation_std(scores: list[float]) -> float:
    """Measure spread across perturbation scenarios as a stability proxy."""
    if len(scores) <= 1:
        return 0.0
    return round(pstdev(scores), 4)


def worst_case_score(scores: list[float]) -> float:
    """Return the lowest perturbation score as the conservative robustness bound."""
    if not scores:
        return 0.0
    return clamp_score(min(scores))


def robustness_penalty(nominal: float, worst_case: float, std: float) -> float:
    """Penalize candidates that degrade sharply or fluctuate strongly under noise."""
    return clamp_score(max((nominal - worst_case) + std, 0.0))


def robustness_score(
    nominal: float,
    perturbed_average: float,
    worst_case: float,
    std: float,
    param_space: PhysicsParameterSpace | None = None,
    constraints: dict[str, Any] | None = None,
) -> float:
    """Aggregate nominal, average, worst-case, and stability terms into one score.

    Default weights come from ``PhysicsParameterSpace`` and may be adapted from
    ``ExperimentGoal.constraints`` through ``spec.metadata["goal_constraints"]``.
    When a user requests custom weights, the final coefficients are blended with
    the validated defaults to avoid abrupt scoring discontinuities.
    """
    space = param_space or PhysicsParameterSpace.default()
    resolved = space.resolve_robustness_weight_config(constraints)
    stability_bonus = max(0.0, 1.0 - min(std / resolved.stability_std_threshold, 1.0))
    aggregate = (
        resolved.nominal_weight * nominal
        + resolved.average_weight * perturbed_average
        + resolved.worst_case_weight * worst_case
        + resolved.stability_weight * stability_bonus
    )
    return clamp_score(aggregate)
