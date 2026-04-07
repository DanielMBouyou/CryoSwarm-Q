from __future__ import annotations

"""Typed metadata schemas for CryoSwarm-Q domain objects.

These TypedDict contracts document the expected key structure of metadata
payloads exchanged between agents, adapters, simulation, and orchestration.
Runtime Pydantic models still store plain dictionaries for compatibility.
"""

from typing import Any, TypedDict


class RegisterMetadata(TypedDict, total=False):
    """Metadata attached to a RegisterCandidate by geometry generation."""

    spacing_um: float


class SequenceMetadata(TypedDict, total=False):
    """Metadata attached to a SequenceCandidate by sequence generation."""

    atom_count: int
    layout_type: str
    spacing_um: float | None
    source: str
    problem_class: str
    strategy_used: str
    detuning_end: float
    amplitude_start: float


class EvaluationMetadata(TypedDict, total=False):
    """Metadata attached to an EvaluationResult during pipeline scoring."""

    register_label: str
    backend_rationale: str
    nominal_observables: dict[str, Any]
    hamiltonian_metrics: dict[str, Any]
    score_std: float


class MemorySignals(TypedDict, total=False):
    """Structured lessons stored in MemoryRecord.signals."""

    objective_score: float
    robustness_score: float
    worst_case_score: float
    backend_choice: str
    sequence_family: str
    layout_type: str
    atom_count: int
    spacing_um: float | None
    amplitude: float
    detuning: float
    duration_ns: int
    confidence: float
    spectral_gap: float
    noise_degradation: float


class NoiseScenarioMetadata(TypedDict, total=False):
    """Metadata attached to a noise scenario configuration."""

    spatial_inhomogeneity: float
    sampled: bool


class CampaignSummaryReport(TypedDict, total=False):
    """Structured summary payload stored on CampaignState.summary_report."""

    reason: str
    error: str
    backend_mix: dict[str, int]
    results_summary: dict[str, object]
    top_candidate_id: str | None
    top_objective_score: float | None
    ranked_candidate_count: int


class BackendChoiceMetadata(TypedDict, total=False):
    """Metadata emitted by backend routing decisions."""

    objective_class: str
    atom_count: int
    hamiltonian_dimension: int


class SpecMetadata(TypedDict, total=False):
    """Metadata attached to ExperimentSpec by the problem framing stage."""

    priority: str
    goal_constraints: dict[str, Any]
    memory_record_count: int
    remembered_backends: list[str | None]
    max_register_candidates: int


class NominalObservables(TypedDict, total=False):
    """Observables extracted from nominal or perturbed simulation output."""

    rydberg_density: float
    target_density: float
    density_score: float
    blockade_violation: float
    blockade_score: float
    interaction_energy: float
    interaction_energy_norm: float
    final_populations: list[float]
    top_bitstrings: dict[str, int]
    entanglement_entropy: float
    antiferromagnetic_order: float
    connected_correlations: list[float]
    observable_score: float
    spectral_gap: float
    hamiltonian_metrics: dict[str, float | int]
    noise_label: str
    spatial_inhomogeneity: float
    spatial_drive_factors: list[float]
    backend: str
    robustness_weight_config: dict[str, Any]


class HamiltonianMetrics(TypedDict, total=False):
    """Compact Hamiltonian characterization metrics."""

    dimension: int
    frobenius_norm: float
    spectral_radius: float
    spectral_gap: float
    robustness_weight_config: dict[str, Any]
