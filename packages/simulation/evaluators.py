from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

import numpy as np
from pulser.noise_model import NoiseModel
from pulser_simulation import QutipEmulator

from packages.core.models import ExperimentSpec, NoiseScenario, RegisterCandidate, SequenceCandidate
from packages.pasqal_adapters.pulser_adapter import build_sequence_from_candidate
from packages.scoring.robustness import (
    perturbation_average,
    perturbation_std,
    robustness_penalty,
    robustness_score,
    worst_case_score,
)
from packages.simulation.noise_profiles import default_noise_scenarios


def _rydberg_labels(atom_count: int) -> list[str]:
    return [f"q{index}" for index in range(atom_count)]


def _build_noise_model(scenario: NoiseScenario) -> NoiseModel:
    return NoiseModel(
        runs=4,
        samples_per_run=1,
        state_prep_error=scenario.state_prep_error,
        p_false_pos=scenario.false_positive_rate,
        p_false_neg=scenario.false_negative_rate,
        temperature=scenario.temperature_uk,
        amp_sigma=scenario.amplitude_jitter,
        detuning_sigma=scenario.detuning_jitter,
        relaxation_rate=scenario.atom_loss_rate,
        dephasing_rate=scenario.dephasing_rate,
    )


def _target_density_score(target_density: float, observed_density: float) -> float:
    scale = max(target_density, 1.0 - target_density, 0.5)
    return max(0.0, round(1.0 - abs(observed_density - target_density) / scale, 4))


def _blockade_score(blockade_violation: float) -> float:
    return max(0.0, round(1.0 - min(blockade_violation / 0.20, 1.0), 4))


def _top_bitstrings(counter: Counter[str], limit: int = 5) -> dict[str, int]:
    return dict(counter.most_common(limit))


def _extract_observables(
    emulator: QutipEmulator,
    results: Any,
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
) -> tuple[float, dict[str, Any], dict[str, float]]:
    labels = _rydberg_labels(register_candidate.atom_count)
    single_site_ops = [emulator.build_operator([("sigma_rr", [label])]) for label in labels]
    population_traces = [np.asarray(values, dtype=float) for values in results.expect(single_site_ops)]
    final_populations = [float(trace[-1]) for trace in population_traces]
    rydberg_density = float(mean(final_populations)) if final_populations else 0.0

    blockade_pair_values: dict[str, float] = {}
    interaction_energy = 0.0
    max_interaction = 0.0
    for row_index in range(register_candidate.atom_count):
        for col_index in range(row_index + 1, register_candidate.atom_count):
            interaction = register_candidate.van_der_waals_matrix[row_index][col_index]
            if interaction <= 0.0:
                continue
            operator = emulator.build_operator([("sigma_rr", [f"q{row_index}", f"q{col_index}"])])
            pair_value = float(np.asarray(results.expect([operator])[0], dtype=float)[-1])
            interaction_energy += interaction * pair_value
            max_interaction = max(max_interaction, interaction)
            distance = float(np.linalg.norm(np.array(register_candidate.coordinates[row_index]) - np.array(register_candidate.coordinates[col_index])))
            if distance <= register_candidate.blockade_radius_um:
                blockade_pair_values[f"q{row_index}-q{col_index}"] = pair_value

    blockade_violation = float(mean(blockade_pair_values.values())) if blockade_pair_values else 0.0
    interaction_energy_norm = interaction_energy / max_interaction if max_interaction else 0.0
    density_score = _target_density_score(spec.target_density, rydberg_density)
    blockade_score = _blockade_score(blockade_violation)
    observable_score = round(0.7 * density_score + 0.3 * blockade_score, 4)

    sampled = Counter(results.results[-1]) if hasattr(results, "results") else results.sample_final_state()
    observables = {
        "rydberg_density": round(rydberg_density, 6),
        "target_density": spec.target_density,
        "density_score": density_score,
        "blockade_violation": round(blockade_violation, 6),
        "blockade_score": blockade_score,
        "interaction_energy": round(interaction_energy, 6),
        "interaction_energy_norm": round(interaction_energy_norm, 6),
        "final_populations": [round(value, 6) for value in final_populations],
        "top_bitstrings": _top_bitstrings(sampled),
    }
    summary_metrics = {
        "observable_score": observable_score,
        "rydberg_density": round(rydberg_density, 6),
        "blockade_violation": round(blockade_violation, 6),
        "interaction_energy_norm": round(interaction_energy_norm, 6),
    }
    return observable_score, observables, summary_metrics


def simulate_sequence_candidate(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    noise_scenario: NoiseScenario | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    sequence = build_sequence_from_candidate(register_candidate, sequence_candidate)
    noise_model = _build_noise_model(noise_scenario) if noise_scenario else None
    emulator = QutipEmulator.from_sequence(sequence, noise_model=noise_model)
    results = emulator.run()

    observable_score, observables, summary_metrics = _extract_observables(
        emulator,
        results,
        spec,
        register_candidate,
    )

    hamiltonian = emulator.get_hamiltonian(0.0)
    hamiltonian_matrix = np.asarray(hamiltonian.full(), dtype=complex)
    eigvals = np.linalg.eigvalsh(hamiltonian_matrix)
    hamiltonian_metrics = {
        "dimension": int(hamiltonian.shape[0]),
        "frobenius_norm": round(float(np.linalg.norm(hamiltonian_matrix)), 6),
        "spectral_radius": round(float(np.max(np.abs(eigvals))), 6),
    }
    observables["hamiltonian_metrics"] = hamiltonian_metrics
    observables["noise_label"] = noise_scenario.label.value if noise_scenario else "noiseless"
    return observable_score, observables, {**summary_metrics, **hamiltonian_metrics}


def evaluate_candidate_robustness(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    scenarios: list[NoiseScenario] | None = None,
) -> tuple[float, dict[str, float], float, float, float, float, float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    nominal_score, nominal_observables, hamiltonian_metrics = simulate_sequence_candidate(
        spec,
        register_candidate,
        sequence_candidate,
        noise_scenario=None,
    )
    active_scenarios = scenarios or default_noise_scenarios()
    scenario_scores: dict[str, float] = {}
    scenario_observables: dict[str, Any] = {}
    perturbed_scores: list[float] = []

    for scenario in active_scenarios:
        score, observables, _ = simulate_sequence_candidate(
            spec,
            register_candidate,
            sequence_candidate,
            noise_scenario=scenario,
        )
        scenario_scores[scenario.label.value] = score
        scenario_observables[scenario.label.value] = observables
        perturbed_scores.append(score)

    average_score = perturbation_average(perturbed_scores)
    worst_score = worst_case_score(perturbed_scores)
    std_score = perturbation_std(perturbed_scores)
    penalty = robustness_penalty(nominal_score, worst_score, std_score)
    aggregate = robustness_score(nominal_score, average_score, worst_score, std_score)
    return (
        nominal_score,
        scenario_scores,
        average_score,
        worst_score,
        std_score,
        penalty,
        aggregate,
        nominal_observables,
        scenario_observables,
        hamiltonian_metrics,
    )
