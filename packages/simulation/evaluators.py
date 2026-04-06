from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

import numpy as np

from packages.core.logging import get_logger
from packages.core.parameter_space import PhysicsParameterSpace
from packages.core.models import ExperimentSpec, NoiseScenario, RegisterCandidate, SequenceCandidate
from packages.pasqal_adapters.pulser_adapter import build_sequence_from_candidate, PULSER_AVAILABLE

try:
    from pulser.noise_model import NoiseModel
    from pulser_simulation import QutipEmulator
    EMULATOR_AVAILABLE = True
except ImportError:
    EMULATOR_AVAILABLE = False
    NoiseModel = None  # type: ignore[assignment, misc]
    QutipEmulator = None  # type: ignore[assignment, misc]

try:
    from scipy.sparse.linalg import eigsh, norm as sparse_norm

    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    eigsh = None  # type: ignore[assignment]
    sparse_norm = None  # type: ignore[assignment]
    SCIPY_SPARSE_AVAILABLE = False

from packages.simulation.observables import (
    antiferromagnetic_order,
    connected_correlation,
    entanglement_entropy,
    rydberg_density as obs_rydberg_density,
    total_rydberg_fraction as obs_total_rydberg_fraction,
    bitstring_probabilities,
)

logger = get_logger(__name__)
from packages.scoring.robustness import (
    perturbation_average,
    perturbation_std,
    robustness_penalty,
    robustness_score,
    worst_case_score,
)
from packages.simulation.noise_profiles import default_noise_scenarios


def _compute_hamiltonian_metrics(
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
) -> dict[str, float | int]:
    from packages.simulation.hamiltonian import build_hamiltonian_matrix, build_sparse_hamiltonian

    omega = float(sequence_candidate.amplitude)
    delta = float(sequence_candidate.detuning)
    atom_count = register_candidate.atom_count

    if atom_count > 14 and SCIPY_SPARSE_AVAILABLE:
        hamiltonian = build_sparse_hamiltonian(register_candidate.coordinates, omega, delta)
        spectral_radius = 0.0
        if hamiltonian.shape[0] > 1:
            eigval = eigsh(hamiltonian, k=1, which="LM", return_eigenvectors=False)
            spectral_radius = float(np.max(np.abs(eigval)))
        return {
            "dimension": int(hamiltonian.shape[0]),
            "frobenius_norm": round(float(sparse_norm(hamiltonian, "fro")), 6),
            "spectral_radius": round(spectral_radius, 6),
        }

    dense = build_hamiltonian_matrix(register_candidate.coordinates, omega, delta)
    eigvals = np.linalg.eigvalsh(dense)
    return {
        "dimension": int(dense.shape[0]),
        "frobenius_norm": round(float(np.linalg.norm(dense)), 6),
        "spectral_radius": round(float(np.max(np.abs(eigvals))), 6),
    }


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
    param_space: PhysicsParameterSpace | None = None,
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
    scoring = (param_space or PhysicsParameterSpace.default()).scoring
    observable_score = round(
        scoring.density_score_weight.default * density_score
        + scoring.blockade_score_weight.default * blockade_score,
        4,
    )

    sampled = Counter(results.results[-1]) if hasattr(results, "results") else results.sample_final_state()

    # ---- extended observables from state vector ----
    n = register_candidate.atom_count
    ent_entropy = 0.0
    af_order = 0.0
    conn_corr_summary: list[float] = []
    try:
        final_state_qobj = results.get_final_state()
        state_vec = np.array(final_state_qobj.full()).flatten()
        state_vec = state_vec / np.linalg.norm(state_vec)
        ent_entropy = float(entanglement_entropy(state_vec, n))
        af_order = float(antiferromagnetic_order(state_vec, n))
        cc = connected_correlation(state_vec, n)
        conn_corr_summary = [
            round(float(cc[i, j]), 6)
            for i in range(n)
            for j in range(i + 1, n)
        ]
    except Exception:
        pass  # noisy simulations may yield density matrices

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
        "entanglement_entropy": round(ent_entropy, 6),
        "antiferromagnetic_order": round(af_order, 6),
        "connected_correlations": conn_corr_summary,
    }
    summary_metrics = {
        "observable_score": observable_score,
        "rydberg_density": round(rydberg_density, 6),
        "blockade_violation": round(blockade_violation, 6),
        "interaction_energy_norm": round(interaction_energy_norm, 6),
        "entanglement_entropy": round(ent_entropy, 6),
        "antiferromagnetic_order": round(af_order, 6),
    }
    return observable_score, observables, summary_metrics


def simulate_sequence_candidate(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    noise_scenario: NoiseScenario | None = None,
    param_space: PhysicsParameterSpace | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    if register_candidate.atom_count > 14:
        logger.warning(
            "Skipping Pulser/Qutip evaluation for %s atoms and using fallback path.",
            register_candidate.atom_count,
        )
        return _simulate_with_numpy_fallback(spec, register_candidate, sequence_candidate, noise_scenario)

    if not EMULATOR_AVAILABLE or not PULSER_AVAILABLE:
        return _simulate_with_numpy_fallback(spec, register_candidate, sequence_candidate, noise_scenario)

    sequence = build_sequence_from_candidate(register_candidate, sequence_candidate)
    noise_model = _build_noise_model(noise_scenario) if noise_scenario else None
    emulator = QutipEmulator.from_sequence(sequence, noise_model=noise_model)
    results = emulator.run()

    observable_score, observables, summary_metrics = _extract_observables(
        emulator,
        results,
        spec,
        register_candidate,
        param_space,
    )

    hamiltonian_metrics = _compute_hamiltonian_metrics(register_candidate, sequence_candidate)
    observables["hamiltonian_metrics"] = hamiltonian_metrics
    observables["noise_label"] = noise_scenario.label.value if noise_scenario else "noiseless"
    return observable_score, observables, {**summary_metrics, **hamiltonian_metrics}


def evaluate_candidate_robustness(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    scenarios: list[NoiseScenario] | None = None,
    param_space: PhysicsParameterSpace | None = None,
) -> tuple[float, dict[str, float], float, float, float, float, float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    nominal_score, nominal_observables, hamiltonian_metrics = simulate_sequence_candidate(
        spec,
        register_candidate,
        sequence_candidate,
        noise_scenario=None,
        param_space=param_space,
    )
    active_scenarios = scenarios or default_noise_scenarios(param_space)
    scenario_scores: dict[str, float] = {}
    scenario_observables: dict[str, Any] = {}
    perturbed_scores: list[float] = []

    for scenario in active_scenarios:
        score, observables, _ = simulate_sequence_candidate(
            spec,
            register_candidate,
            sequence_candidate,
            noise_scenario=scenario,
            param_space=param_space,
        )
        scenario_scores[scenario.label.value] = score
        scenario_observables[scenario.label.value] = observables
        perturbed_scores.append(score)

    average_score = perturbation_average(perturbed_scores)
    worst_score = worst_case_score(perturbed_scores)
    std_score = perturbation_std(perturbed_scores)
    penalty = robustness_penalty(nominal_score, worst_score, std_score)
    aggregate = robustness_score(
        nominal_score,
        average_score,
        worst_score,
        std_score,
        param_space=param_space,
    )
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


# ---- numpy / scipy fallback when Pulser is unavailable ----


def _simulate_with_numpy_fallback(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    noise_scenario: NoiseScenario | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Run a lightweight exact-diag simulation via numpy/scipy."""
    from packages.simulation.numpy_backend import simulate_rydberg_evolution

    omega_max = float(sequence_candidate.amplitude)
    delta_start = float(sequence_candidate.detuning)
    delta_end = float(sequence_candidate.metadata.get("detuning_end", delta_start * 0.25))
    duration_ns = float(sequence_candidate.duration_ns)
    omega_shape = "constant"
    family = sequence_candidate.sequence_family.value
    if family == "blackman_sweep":
        omega_shape = "blackman"
    elif family == "global_ramp":
        omega_shape = "ramp"

    if noise_scenario is not None:
        rng = np.random.default_rng()
        omega_max *= 1.0 + rng.normal(0.0, noise_scenario.amplitude_jitter)
        delta_start += rng.normal(0.0, noise_scenario.detuning_jitter * abs(delta_start) + 0.1)
        delta_end += rng.normal(0.0, noise_scenario.detuning_jitter * abs(delta_end) + 0.1)
        spatial_inhom = float(noise_scenario.metadata.get("spatial_inhomogeneity", 0.0))
        spatial_factors = np.ones(register_candidate.atom_count, dtype=np.float64)
        if spatial_inhom > 0.0:
            spatial_factors = 1.0 + rng.normal(0.0, spatial_inhom, size=register_candidate.atom_count)
            omega_max *= float(np.mean(spatial_factors))
    else:
        spatial_inhom = 0.0
        spatial_factors = np.ones(register_candidate.atom_count, dtype=np.float64)

    result = simulate_rydberg_evolution(
        coords=register_candidate.coordinates,
        omega_max=max(omega_max, 0.01),
        delta_start=delta_start,
        delta_end=delta_end,
        duration_ns=duration_ns,
        omega_shape=omega_shape,
        n_steps=150,
    )

    ryd_dens = float(result["total_rydberg_fraction"])
    density_score = _target_density_score(spec.target_density, ryd_dens)
    observable_score = round(density_score, 4)

    n = register_candidate.atom_count
    hamiltonian_metrics: dict[str, Any] = _compute_hamiltonian_metrics(register_candidate, sequence_candidate)

    observables: dict[str, Any] = {
        "rydberg_density": round(ryd_dens, 6),
        "target_density": spec.target_density,
        "density_score": density_score,
        "blockade_violation": 0.0,
        "blockade_score": 1.0,
        "interaction_energy": 0.0,
        "interaction_energy_norm": 0.0,
        "final_populations": result["rydberg_densities"],
        "top_bitstrings": {bs: prob for bs, prob in result["top_bitstrings"]},
        "entanglement_entropy": round(result["entanglement_entropy"], 6),
        "antiferromagnetic_order": round(result["antiferromagnetic_order"], 6),
        "connected_correlations": [
            round(float(result["connected_correlations"][i][j]), 6)
            for i in range(n) for j in range(i + 1, n)
        ],
        "spatial_inhomogeneity": round(spatial_inhom, 6),
        "spatial_drive_factors": [round(float(value), 6) for value in spatial_factors],
        "hamiltonian_metrics": hamiltonian_metrics,
        "noise_label": noise_scenario.label.value if noise_scenario else "noiseless",
        "backend": "numpy_exact",
    }
    summary: dict[str, Any] = {
        "observable_score": observable_score,
        "rydberg_density": round(ryd_dens, 6),
        "blockade_violation": 0.0,
        "interaction_energy_norm": 0.0,
        "entanglement_entropy": round(result["entanglement_entropy"], 6),
        "antiferromagnetic_order": round(result["antiferromagnetic_order"], 6),
        "spatial_inhomogeneity": round(spatial_inhom, 6),
        "spatial_drive_std": round(float(np.std(spatial_factors)), 6),
        **hamiltonian_metrics,
    }
    return observable_score, observables, summary
