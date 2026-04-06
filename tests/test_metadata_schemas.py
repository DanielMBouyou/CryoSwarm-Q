from __future__ import annotations

"""Tests for typed metadata schemas used across CryoSwarm-Q."""

from packages.core.metadata_schemas import (
    BackendChoiceMetadata,
    CampaignSummaryReport,
    EvaluationMetadata,
    HamiltonianMetrics,
    MemorySignals,
    NoiseScenarioMetadata,
    NominalObservables,
    RegisterMetadata,
    SequenceMetadata,
    SpecMetadata,
)


def test_register_metadata_keys() -> None:
    meta: RegisterMetadata = {"spacing_um": 7.0}
    assert meta["spacing_um"] == 7.0


def test_sequence_metadata_keys() -> None:
    meta: SequenceMetadata = {
        "atom_count": 6,
        "source": "heuristic",
        "detuning_end": 10.0,
    }
    assert meta["source"] == "heuristic"


def test_evaluation_metadata_keys() -> None:
    meta: EvaluationMetadata = {
        "score_std": 0.05,
        "register_label": "square-6",
        "nominal_observables": {"rydberg_density": 0.4},
        "hamiltonian_metrics": {"spectral_radius": 3.2},
    }
    assert meta["score_std"] == 0.05


def test_memory_signals_keys() -> None:
    signals: MemorySignals = {
        "objective_score": 0.72,
        "confidence": 0.85,
        "sequence_family": "adiabatic_sweep",
        "amplitude": 5.0,
        "detuning": -10.0,
    }
    assert signals["confidence"] == 0.85


def test_all_schemas_are_total_false() -> None:
    schemas = [
        RegisterMetadata,
        SequenceMetadata,
        EvaluationMetadata,
        MemorySignals,
        NoiseScenarioMetadata,
        CampaignSummaryReport,
        BackendChoiceMetadata,
        SpecMetadata,
        NominalObservables,
        HamiltonianMetrics,
    ]
    for schema in schemas:
        assert schema.__total__ is False, f"{schema.__name__} should use total=False"
