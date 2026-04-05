from __future__ import annotations

from statistics import mean


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def nominal_score(base_signal: float, control_penalty: float, hardware_bonus: float) -> float:
    return clamp_score(base_signal - control_penalty + hardware_bonus)


def perturbation_average(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return clamp_score(mean(scores))


def robustness_penalty(nominal: float, perturbed_average: float) -> float:
    return clamp_score(max(nominal - perturbed_average, 0.0))


def robustness_score(nominal: float, perturbed_average: float) -> float:
    return clamp_score((nominal + perturbed_average) / 2.0)
