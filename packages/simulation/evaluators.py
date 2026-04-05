from __future__ import annotations

from packages.core.models import NoiseScenario, SequenceCandidate
from packages.scoring.robustness import (
    nominal_score,
    perturbation_average,
    robustness_penalty,
    robustness_score,
)
from packages.simulation.noise_profiles import default_noise_scenarios


FAMILY_BONUS = {
    "global_ramp": 0.04,
    "detuning_scan": 0.02,
    "adiabatic_sweep": 0.05,
}


def _compute_base_signal(candidate: SequenceCandidate) -> float:
    atom_count = int(candidate.metadata.get("atom_count", 4))
    amplitude_term = 0.06 if 1.0 <= candidate.amplitude <= 2.4 else -0.03
    detuning_term = 0.03 if abs(candidate.detuning) <= 1.8 else -0.02
    duration_penalty = min(candidate.duration_ns / 18000.0, 0.18)
    atom_penalty = max(atom_count - 4, 0) * 0.015
    family_bonus = FAMILY_BONUS.get(candidate.sequence_family.value, 0.01)
    return 0.84 + amplitude_term + detuning_term + family_bonus - duration_penalty - atom_penalty


def _compute_control_penalty(candidate: SequenceCandidate) -> float:
    return min((abs(candidate.phase) / 6.28) * 0.05 + candidate.predicted_cost * 0.10, 0.18)


def _compute_hardware_bonus(candidate: SequenceCandidate) -> float:
    layout_type = str(candidate.metadata.get("layout_type", "line"))
    return 0.03 if layout_type in {"square", "triangular"} else 0.01


def evaluate_candidate_nominal(candidate: SequenceCandidate) -> float:
    return nominal_score(
        base_signal=_compute_base_signal(candidate),
        control_penalty=_compute_control_penalty(candidate),
        hardware_bonus=_compute_hardware_bonus(candidate),
    )


def evaluate_candidate_under_noise(
    candidate: SequenceCandidate,
    scenario: NoiseScenario,
    nominal: float,
) -> float:
    atom_count = int(candidate.metadata.get("atom_count", 4))
    severity = (
        scenario.amplitude_jitter * 0.30
        + scenario.detuning_jitter * 0.25
        + scenario.dephasing_rate * 0.35
        + scenario.atom_loss_rate * 0.50
    )
    scale_penalty = max(atom_count - 4, 0) * 0.01
    cost_penalty = candidate.predicted_cost * 0.06
    return max(round(nominal - severity - scale_penalty - cost_penalty, 4), 0.0)


def evaluate_candidate_robustness(
    candidate: SequenceCandidate,
    scenarios: list[NoiseScenario] | None = None,
) -> tuple[float, dict[str, float], float, float, float]:
    active_scenarios = scenarios or default_noise_scenarios()
    nominal = evaluate_candidate_nominal(candidate)
    scenario_scores = {
        scenario.label.value: evaluate_candidate_under_noise(candidate, scenario, nominal)
        for scenario in active_scenarios
    }
    average_score = perturbation_average(list(scenario_scores.values()))
    penalty = robustness_penalty(nominal, average_score)
    aggregate = robustness_score(nominal, average_score)
    return nominal, scenario_scores, average_score, penalty, aggregate
