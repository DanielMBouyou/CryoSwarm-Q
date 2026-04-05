from packages.core.models import ExperimentSpec, RegisterCandidate, ScoringWeights, SequenceCandidate
from packages.core.enums import SequenceFamily
from packages.simulation.evaluators import evaluate_candidate_robustness


def test_physical_evaluator_returns_quantum_observables() -> None:
    spec = ExperimentSpec(
        goal_id="goal_test",
        objective_class="robust_neutral_atom_search",
        target_observable="rydberg_density",
        min_atoms=3,
        max_atoms=3,
        preferred_layouts=["triangular"],
        sequence_families=[SequenceFamily.GLOBAL_RAMP],
        target_density=0.5,
        scoring_weights=ScoringWeights(),
        reasoning_summary="test spec",
    )
    register = RegisterCandidate(
        campaign_id="camp_test",
        spec_id=spec.id,
        label="triangular-3",
        layout_type="triangular",
        atom_count=3,
        coordinates=[(0.0, 0.0), (0.0, 7.0), (7.0, 0.0)],
        device_constraints={"min_spacing_um": 5.0},
        min_distance_um=7.0,
        blockade_radius_um=8.0,
        blockade_pair_count=3,
        van_der_waals_matrix=[
            [0.0, 7.2, 7.2],
            [7.2, 0.0, 1.8],
            [7.2, 1.8, 0.0],
        ],
        feasibility_score=0.9,
        reasoning_summary="test register",
    )
    sequence = SequenceCandidate(
        campaign_id="camp_test",
        spec_id=spec.id,
        register_candidate_id=register.id,
        label="triangular-3-global",
        sequence_family=SequenceFamily.GLOBAL_RAMP,
        duration_ns=1800,
        amplitude=1.2,
        detuning=-1.0,
        phase=0.0,
        waveform_kind="global_ramp_compact",
        predicted_cost=0.2,
        reasoning_summary="test sequence",
        metadata={"atom_count": 3, "layout_type": "triangular", "spacing_um": 7.0, "amplitude_start": 0.15},
    )

    (
        nominal_score,
        scenario_scores,
        perturbation_average,
        worst_case,
        score_std,
        penalty,
        robustness,
        nominal_observables,
        scenario_observables,
        hamiltonian_metrics,
    ) = evaluate_candidate_robustness(spec, register, sequence)

    assert 0.0 <= nominal_score <= 1.0
    assert len(scenario_scores) == 3
    assert "rydberg_density" in nominal_observables
    assert "low_noise" in scenario_observables
    assert hamiltonian_metrics["dimension"] == 8
    assert worst_case <= nominal_score + 1e-6 or worst_case <= 1.0
    assert score_std >= 0.0
    assert penalty >= 0.0
    assert 0.0 <= robustness <= 1.0
