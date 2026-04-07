from __future__ import annotations

import packages.simulation.evaluators as evaluators_module
import packages.simulation.numpy_backend as numpy_backend_module
from packages.core.enums import SequenceFamily
from packages.core.models import ExperimentSpec, RegisterCandidate, ScoringWeights, SequenceCandidate
from packages.simulation.evaluation_cache import clear_evaluation_cache


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        goal_id="goal_test",
        objective_class="balanced_campaign_search",
        target_observable="rydberg_density",
        min_atoms=3,
        max_atoms=3,
        preferred_layouts=["line"],
        sequence_families=[SequenceFamily.GLOBAL_RAMP],
        target_density=0.5,
        scoring_weights=ScoringWeights(),
        reasoning_summary="cache test spec",
    )


def _register(spec: ExperimentSpec) -> RegisterCandidate:
    return RegisterCandidate(
        campaign_id="campaign_a",
        spec_id=spec.id,
        label="line-3",
        layout_type="line",
        atom_count=3,
        coordinates=[(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)],
        min_distance_um=7.0,
        blockade_radius_um=9.5,
        blockade_pair_count=2,
        van_der_waals_matrix=[
            [0.0, 7.2, 0.9],
            [7.2, 0.0, 7.2],
            [0.9, 7.2, 0.0],
        ],
        feasibility_score=0.9,
        reasoning_summary="cache test register",
        metadata={"spacing_um": 7.0},
    )


def _sequence(spec: ExperimentSpec, register: RegisterCandidate, *, sequence_id: str) -> SequenceCandidate:
    return SequenceCandidate(
        id=sequence_id,
        campaign_id="campaign_x",
        spec_id=spec.id,
        register_candidate_id=register.id,
        label="global-ramp",
        sequence_family=SequenceFamily.GLOBAL_RAMP,
        duration_ns=1800,
        amplitude=1.4,
        detuning=-1.0,
        phase=0.0,
        waveform_kind="global_ramp",
        predicted_cost=0.2,
        reasoning_summary="cache test sequence",
        metadata={"spacing_um": 7.0, "amplitude_start": 0.15},
    )


def test_simulation_cache_reuses_identical_content(monkeypatch) -> None:
    spec = _spec()
    register = _register(spec)
    sequence_a = _sequence(spec, register, sequence_id="seq_a")
    sequence_b = _sequence(spec, register, sequence_id="seq_b")

    call_counter = {"count": 0}

    def _fake_simulate(**kwargs):  # type: ignore[no-untyped-def]
        call_counter["count"] += 1
        return {
            "total_rydberg_fraction": 0.42,
            "rydberg_densities": [0.4, 0.45, 0.41],
            "top_bitstrings": [("101", 0.6), ("010", 0.4)],
            "entanglement_entropy": 0.12,
            "antiferromagnetic_order": 0.33,
            "connected_correlations": [
                [0.0, 0.1, 0.02],
                [0.1, 0.0, 0.1],
                [0.02, 0.1, 0.0],
            ],
            "backend": "numpy_exact",
        }

    monkeypatch.setattr(evaluators_module, "EMULATOR_AVAILABLE", False)
    monkeypatch.setattr(evaluators_module, "PULSER_AVAILABLE", False)
    monkeypatch.setattr(numpy_backend_module, "simulate_rydberg_evolution", _fake_simulate)
    monkeypatch.setattr(
        evaluators_module,
        "_compute_hamiltonian_metrics",
        lambda *args, **kwargs: {"dimension": 8, "frobenius_norm": 1.0, "spectral_radius": 1.0},
    )
    clear_evaluation_cache()

    first = evaluators_module.simulate_sequence_candidate(spec, register, sequence_a)
    second = evaluators_module.simulate_sequence_candidate(spec, register, sequence_b)

    assert call_counter["count"] == 1
    assert first == second
    clear_evaluation_cache()
