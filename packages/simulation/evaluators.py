from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, cast

import numpy as np

from packages.core.logging import get_logger
from packages.core.metadata_schemas import NoiseScenarioMetadata, SequenceMetadata
from packages.core.parameter_space import PhysicsParameterSpace
from packages.core.models import ExperimentSpec, NoiseScenario, RegisterCandidate, SequenceCandidate
from packages.pasqal_adapters.pulser_adapter import build_sequence_from_candidate, PULSER_AVAILABLE
from packages.simulation.evaluation_cache import build_simulation_cache_key, get_evaluation_cache

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
    param_space: PhysicsParameterSpace | None = None,
) -> dict[str, float | int]:
    from packages.simulation.hamiltonian import build_hamiltonian_matrix, build_sparse_hamiltonian

    space = param_space or PhysicsParameterSpace.default()
    omega = float(sequence_candidate.amplitude)
    delta = float(sequence_candidate.detuning)
    atom_count = register_candidate.atom_count

    if atom_count > space.max_atoms_evaluator_parallel and SCIPY_SPARSE_AVAILABLE:
        hamiltonian = build_sparse_hamiltonian(
            register_candidate.coordinates,
            omega,
            delta,
            c6=space.c6_coefficient,
        )
        spectral_radius = 0.0
        if hamiltonian.shape[0] > 1:
            eigval = eigsh(hamiltonian, k=1, which="LM", return_eigenvectors=False)
            spectral_radius = float(np.max(np.abs(eigval)))
        return {
            "dimension": int(hamiltonian.shape[0]),
            "frobenius_norm": round(float(sparse_norm(hamiltonian, "fro")), 6),
            "spectral_radius": round(spectral_radius, 6),
        }

    dense = build_hamiltonian_matrix(
        register_candidate.coordinates,
        omega,
        delta,
        c6=space.c6_coefficient,
    )
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


def _build_sequence_schedule(
    sequence_candidate: SequenceCandidate,
    *,
    omega_max: float | None = None,
    delta_start: float | None = None,
    delta_end: float | None = None,
    duration_ns: float | None = None,
):
    from packages.simulation.numpy_backend import PulseSchedule

    sequence_metadata = cast(SequenceMetadata, sequence_candidate.metadata)
    omega_shape = "constant"
    family = sequence_candidate.sequence_family.value
    if family == "blackman_sweep":
        omega_shape = "blackman"
    elif family == "global_ramp":
        omega_shape = "ramp"

    resolved_delta_start = float(sequence_candidate.detuning if delta_start is None else delta_start)
    resolved_delta_end = float(
        sequence_metadata.get("detuning_end", resolved_delta_start * 0.25)
        if delta_end is None else delta_end
    )
    resolved_duration_ns = float(sequence_candidate.duration_ns if duration_ns is None else duration_ns)
    resolved_omega_max = float(sequence_candidate.amplitude if omega_max is None else omega_max)
    return PulseSchedule.from_legacy(
        omega_max=resolved_omega_max,
        delta_start=resolved_delta_start,
        delta_end=resolved_delta_end,
        duration_ns=resolved_duration_ns,
        omega_shape=omega_shape,
    )


def _compute_schedule_diagnostics(
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    param_space: PhysicsParameterSpace | None = None,
    *,
    omega_max: float | None = None,
    delta_start: float | None = None,
    delta_end: float | None = None,
    duration_ns: float | None = None,
) -> dict[str, Any]:
    from packages.simulation.numpy_backend import compute_schedule_diagnostics

    space = param_space or PhysicsParameterSpace.default()
    schedule = _build_sequence_schedule(
        sequence_candidate,
        omega_max=omega_max,
        delta_start=delta_start,
        delta_end=delta_end,
        duration_ns=duration_ns,
    )
    return compute_schedule_diagnostics(
        coords=register_candidate.coordinates,
        n_steps=150,
        schedule=schedule,
        omega_max=float(sequence_candidate.amplitude if omega_max is None else omega_max),
        delta_start=float(sequence_candidate.detuning if delta_start is None else delta_start),
        delta_end=float(
            cast(SequenceMetadata, sequence_candidate.metadata).get(
                "detuning_end",
                sequence_candidate.detuning * 0.25,
            )
            if delta_end is None else delta_end
        ),
        duration_ns=float(sequence_candidate.duration_ns if duration_ns is None else duration_ns),
        c6=space.c6_coefficient,
        max_atoms_dense=space.max_atoms_dense,
        max_atoms_sparse=space.max_atoms_sparse,
    )


def _schedule_summary_fields(schedule_diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "adiabatic_ratio_max": schedule_diagnostics.get("adiabatic_ratio_max"),
        "adiabatic_warning": bool(schedule_diagnostics.get("adiabatic_warning", False)),
        "adiabatic_gap_min": schedule_diagnostics.get("adiabatic_gap_min"),
        "dynamic_blockade_radius_min_um": schedule_diagnostics.get("blockade_radius_min_um"),
        "dynamic_blockade_radius_max_um": schedule_diagnostics.get("blockade_radius_max_um"),
        "integration_order": schedule_diagnostics.get("integration_order", 2),
    }


def _goal_constraints(spec: ExperimentSpec) -> dict[str, Any]:
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    goal_constraints = metadata.get("goal_constraints", {})
    return goal_constraints if isinstance(goal_constraints, dict) else {}


def _extract_observables(
    emulator: QutipEmulator,
    results: Any,
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    param_space: PhysicsParameterSpace | None = None,
) -> tuple[float, dict[str, Any], dict[str, float]]:
    space = param_space or PhysicsParameterSpace.default()
    active_robustness_weights = space.resolve_robustness_weight_config(_goal_constraints(spec)).to_dict()
    schedule_diagnostics = _compute_schedule_diagnostics(
        register_candidate,
        sequence_candidate,
        param_space=space,
    )
    effective_blockade_radius = float(
        schedule_diagnostics.get("blockade_radius_max_um") or register_candidate.blockade_radius_um
    )
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
            if distance <= effective_blockade_radius:
                blockade_pair_values[f"q{row_index}-q{col_index}"] = pair_value

    blockade_violation = float(mean(blockade_pair_values.values())) if blockade_pair_values else 0.0
    interaction_energy_norm = interaction_energy / max_interaction if max_interaction else 0.0
    density_score = _target_density_score(spec.target_density, rydberg_density)
    blockade_score = _blockade_score(blockade_violation)
    scoring = space.scoring
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
    except Exception as exc:
        logger.debug(
            "Extended observable extraction failed for %d atoms: %s",
            n,
            exc,
        )

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
        "robustness_weight_config": active_robustness_weights,
    }
    summary_metrics = {
        "observable_score": observable_score,
        "rydberg_density": round(rydberg_density, 6),
        "blockade_violation": round(blockade_violation, 6),
        "interaction_energy_norm": round(interaction_energy_norm, 6),
        "entanglement_entropy": round(ent_entropy, 6),
        "antiferromagnetic_order": round(af_order, 6),
        "robustness_weight_config": active_robustness_weights,
    }
    observables.update(_schedule_summary_fields(schedule_diagnostics))
    summary_metrics.update(_schedule_summary_fields(schedule_diagnostics))
    return observable_score, observables, summary_metrics


def simulate_sequence_candidate(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    noise_scenario: NoiseScenario | None = None,
    param_space: PhysicsParameterSpace | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    space = param_space or PhysicsParameterSpace.default()
    cache_key = build_simulation_cache_key(
        spec,
        register_candidate,
        sequence_candidate,
        noise_scenario,
        space,
        emulator_available=EMULATOR_AVAILABLE,
        pulser_available=PULSER_AVAILABLE,
        scipy_sparse_available=SCIPY_SPARSE_AVAILABLE,
    )
    cached = get_evaluation_cache().get(cache_key)
    if cached is not None:
        return cached

    if register_candidate.atom_count > space.max_atoms_evaluator_parallel:
        logger.warning(
            "Skipping Pulser/Qutip evaluation for %s atoms and using fallback path.",
            register_candidate.atom_count,
        )
        result = _simulate_with_numpy_fallback(
            spec,
            register_candidate,
            sequence_candidate,
            noise_scenario,
            param_space=space,
            rng_seed=int(cache_key[:16], 16) % (2**32),
        )
        get_evaluation_cache().set(cache_key, result)
        return result

    if not EMULATOR_AVAILABLE or not PULSER_AVAILABLE:
        result = _simulate_with_numpy_fallback(
            spec,
            register_candidate,
            sequence_candidate,
            noise_scenario,
            param_space=space,
            rng_seed=int(cache_key[:16], 16) % (2**32),
        )
        get_evaluation_cache().set(cache_key, result)
        return result

    sequence = build_sequence_from_candidate(
        register_candidate,
        sequence_candidate,
        param_space=space,
    )
    noise_model = _build_noise_model(noise_scenario) if noise_scenario else None
    emulator = QutipEmulator.from_sequence(sequence, noise_model=noise_model)
    results = emulator.run()

    observable_score, observables, summary_metrics = _extract_observables(
        emulator,
        results,
        spec,
        register_candidate,
        sequence_candidate,
        space,
    )

    hamiltonian_metrics = _compute_hamiltonian_metrics(
        register_candidate,
        sequence_candidate,
        param_space=space,
    )
    observables["hamiltonian_metrics"] = hamiltonian_metrics
    observables["noise_label"] = noise_scenario.label.value if noise_scenario else "noiseless"
    result = (observable_score, observables, {**summary_metrics, **hamiltonian_metrics})
    get_evaluation_cache().set(cache_key, result)
    return result


def evaluate_candidate_robustness(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    scenarios: list[NoiseScenario] | None = None,
    param_space: PhysicsParameterSpace | None = None,
) -> tuple[float, dict[str, float], float, float, float, float, float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    goal_constraints = _goal_constraints(spec)
    space = param_space or PhysicsParameterSpace.default()
    nominal_score, nominal_observables, hamiltonian_metrics = simulate_sequence_candidate(
        spec,
        register_candidate,
        sequence_candidate,
        noise_scenario=None,
        param_space=space,
    )
    active_scenarios = scenarios or default_noise_scenarios(space)
    scenario_scores: dict[str, float] = {}
    scenario_observables: dict[str, Any] = {}
    perturbed_scores: list[float] = []

    for scenario in active_scenarios:
        score, observables, _ = simulate_sequence_candidate(
            spec,
            register_candidate,
            sequence_candidate,
            noise_scenario=scenario,
            param_space=space,
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
        param_space=space,
        constraints=goal_constraints,
    )
    active_robustness_weights = space.resolve_robustness_weight_config(goal_constraints).to_dict()
    nominal_observables = {
        **nominal_observables,
        "robustness_weight_config": active_robustness_weights,
    }
    hamiltonian_metrics = {
        **hamiltonian_metrics,
        "robustness_weight_config": active_robustness_weights,
    }
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
    param_space: PhysicsParameterSpace | None = None,
    rng_seed: int | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Run a lightweight exact-diag simulation via numpy/scipy."""
    from packages.simulation.numpy_backend import simulate_rydberg_evolution

    space = param_space or PhysicsParameterSpace.default()
    active_robustness_weights = space.resolve_robustness_weight_config(_goal_constraints(spec)).to_dict()
    omega_max = float(sequence_candidate.amplitude)
    delta_start = float(sequence_candidate.detuning)
    sequence_metadata = cast(SequenceMetadata, sequence_candidate.metadata)
    delta_end = float(sequence_metadata.get("detuning_end", delta_start * 0.25))
    duration_ns = float(sequence_candidate.duration_ns)
    omega_shape = "constant"
    family = sequence_candidate.sequence_family.value
    if family == "blackman_sweep":
        omega_shape = "blackman"
    elif family == "global_ramp":
        omega_shape = "ramp"

    if noise_scenario is not None:
        rng = np.random.default_rng(rng_seed)
        omega_max *= 1.0 + rng.normal(0.0, noise_scenario.amplitude_jitter)
        delta_start += rng.normal(0.0, noise_scenario.detuning_jitter * abs(delta_start) + 0.1)
        delta_end += rng.normal(0.0, noise_scenario.detuning_jitter * abs(delta_end) + 0.1)
        noise_metadata = cast(NoiseScenarioMetadata, noise_scenario.metadata)
        spatial_inhom = float(noise_metadata.get("spatial_inhomogeneity", 0.0))
        spatial_factors = np.ones(register_candidate.atom_count, dtype=np.float64)
        if spatial_inhom > 0.0:
            spatial_factors = 1.0 + rng.normal(0.0, spatial_inhom, size=register_candidate.atom_count)
            omega_max *= float(np.mean(spatial_factors))
    else:
        spatial_inhom = 0.0
        spatial_factors = np.ones(register_candidate.atom_count, dtype=np.float64)

    schedule_diagnostics = _compute_schedule_diagnostics(
        register_candidate,
        sequence_candidate,
        param_space=space,
        omega_max=max(omega_max, 0.01),
        delta_start=delta_start,
        delta_end=delta_end,
        duration_ns=duration_ns,
    )

    result = simulate_rydberg_evolution(
        coords=register_candidate.coordinates,
        omega_max=max(omega_max, 0.01),
        delta_start=delta_start,
        delta_end=delta_end,
        duration_ns=duration_ns,
        omega_shape=omega_shape,
        n_steps=150,
        c6=space.c6_coefficient,
        max_atoms_dense=space.max_atoms_dense,
        max_atoms_sparse=space.max_atoms_sparse,
    )

    ryd_dens = float(result["total_rydberg_fraction"])
    density_score = _target_density_score(spec.target_density, ryd_dens)
    observable_score = round(density_score, 4)

    n = register_candidate.atom_count
    hamiltonian_metrics: dict[str, Any] = _compute_hamiltonian_metrics(
        register_candidate,
        sequence_candidate,
        param_space=space,
    )

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
        "backend": result["backend"],
        "robustness_weight_config": active_robustness_weights,
        **_schedule_summary_fields(schedule_diagnostics),
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
        "robustness_weight_config": active_robustness_weights,
        **_schedule_summary_fields(schedule_diagnostics),
        **hamiltonian_metrics,
    }
    return observable_score, observables, summary
