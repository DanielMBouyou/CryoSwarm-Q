from __future__ import annotations

import pytest
from pydantic import ValidationError

from packages.core.models import ExperimentGoal, SequenceCandidate
from packages.core.enums import SequenceFamily
from packages.orchestration.pipeline import CryoSwarmPipeline
from packages.simulation.hamiltonian import build_hamiltonian_matrix

pytest.importorskip("scipy", reason="scipy required for numpy backend tests")
from packages.simulation.numpy_backend import simulate_rydberg_evolution  # noqa: E402


def test_experiment_goal_rejects_negative_atom_count() -> None:
    with pytest.raises(ValidationError):
        ExperimentGoal(
            title="Invalid goal",
            scientific_objective="Test invalid atom count handling.",
            desired_atom_count=-1,
        )


def test_experiment_goal_rejects_too_many_atoms() -> None:
    with pytest.raises(ValidationError):
        ExperimentGoal(
            title="Oversized goal",
            scientific_objective="Test upper atom count validation.",
            desired_atom_count=100,
        )


def test_sequence_candidate_rejects_excessive_amplitude() -> None:
    with pytest.raises(ValidationError):
        SequenceCandidate(
            campaign_id="campaign_test",
            spec_id="spec_test",
            register_candidate_id="reg_test",
            label="invalid-sequence",
            sequence_family=SequenceFamily.CONSTANT_DRIVE,
            duration_ns=1000,
            amplitude=20.0,
            detuning=0.0,
            phase=0.0,
            predicted_cost=0.1,
            reasoning_summary="Invalid amplitude test.",
        )


def test_dense_hamiltonian_rejects_too_many_atoms() -> None:
    coords = [(float(index) * 7.0, 0.0) for index in range(15)]
    with pytest.raises(ValueError, match="Dense Hamiltonian limited to 14 atoms"):
        build_hamiltonian_matrix(coords, omega=5.0, delta=0.0)


def test_numpy_simulation_rejects_too_many_atoms() -> None:
    coords = [(float(index) * 7.0, 0.0) for index in range(19)]
    with pytest.raises(ValueError, match="NumPy backend limited to 18 atoms"):
        simulate_rydberg_evolution(
            coords=coords,
            omega_max=5.0,
            delta_start=-10.0,
            delta_end=10.0,
            duration_ns=1000,
        )


def test_pipeline_handles_empty_geometry_output_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = CryoSwarmPipeline(repository=None)
    goal = ExperimentGoal(
        title="Graceful geometry failure",
        scientific_objective="Return no register candidates and ensure graceful handling.",
        desired_atom_count=4,
        preferred_geometry="line",
    )

    monkeypatch.setattr(pipeline.geometry_agent, "run", lambda *args, **kwargs: [])

    summary = pipeline.run(goal)

    assert summary.status == "NO_CANDIDATES"
    assert summary.campaign.status.value == "no_candidates"
    assert summary.total_candidates == 0
    assert summary.ranked_candidates == []
