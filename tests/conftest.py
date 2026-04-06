"""Shared test fixtures and factories for CryoSwarm-Q test suite.

Provides reusable, parameterized factories for core domain objects and
geometry helpers used across the test suite.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from packages.core.enums import (
    BackendType,
    CampaignStatus,
    CandidateStatus,
    NoiseLevel,
    SequenceFamily,
)
from packages.core.models import (
    AgentDecision,
    BackendChoice,
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    ExperimentSpec,
    MemoryRecord,
    RegisterCandidate,
    RobustnessReport,
    ScoringWeights,
    SequenceCandidate,
)
from packages.core.parameter_space import PhysicsParameterSpace


def make_coordinates(
    layout: str = "square",
    atom_count: int = 4,
    spacing_um: float = 7.0,
) -> list[tuple[float, float]]:
    """Generate physically plausible neutral-atom coordinates for a named layout."""
    if layout == "line":
        return [(i * spacing_um, 0.0) for i in range(atom_count)]

    if layout == "square":
        side = math.isqrt(atom_count)
        if side * side < atom_count:
            side += 1
        coords: list[tuple[float, float]] = []
        for idx in range(atom_count):
            row, col = divmod(idx, side)
            coords.append((col * spacing_um, row * spacing_um))
        return coords

    if layout == "triangular":
        coords: list[tuple[float, float]] = []
        side = math.isqrt(atom_count) + 1
        placed = 0
        for row in range(side):
            x_offset = spacing_um * 0.5 if row % 2 else 0.0
            for col in range(side):
                if placed >= atom_count:
                    break
                coords.append(
                    (
                        col * spacing_um + x_offset,
                        row * spacing_um * math.sqrt(3.0) / 2.0,
                    )
                )
                placed += 1
        return coords

    if layout == "ring":
        return [
            (
                spacing_um * math.cos(2.0 * math.pi * i / atom_count),
                spacing_um * math.sin(2.0 * math.pi * i / atom_count),
            )
            for i in range(atom_count)
        ]

    if layout == "zigzag":
        return [(i * spacing_um, spacing_um * 0.5 * (i % 2)) for i in range(atom_count)]

    if layout == "honeycomb":
        coords: list[tuple[float, float]] = []
        placed = 0
        col = 0
        while placed < atom_count:
            for row_offset in (0.0, spacing_um * math.sqrt(3.0) / 3.0):
                if placed >= atom_count:
                    break
                coords.append(
                    (
                        col * spacing_um * 1.5,
                        row_offset + (col % 2) * spacing_um * math.sqrt(3.0) / 6.0,
                    )
                )
                placed += 1
            col += 1
        return coords

    return [(i * spacing_um, 0.0) for i in range(atom_count)]


def compute_blockade_pairs(
    coordinates: list[tuple[float, float]],
    blockade_radius_um: float = 9.5,
) -> int:
    """Count the number of atom pairs separated by at most the blockade radius."""
    count = 0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            if math.sqrt(dx * dx + dy * dy) <= blockade_radius_um:
                count += 1
    return count


def make_goal(
    *,
    title: str = "Test neutral-atom benchmark",
    scientific_objective: str = "Evaluate robustness of small neutral-atom configurations.",
    target_observable: str = "rydberg_density",
    desired_atom_count: int = 6,
    preferred_geometry: str = "mixed",
    priority: str = "balanced",
    constraints: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> ExperimentGoal:
    """Build an ExperimentGoal with defaults aligned with typical benchmark tests."""
    return ExperimentGoal(
        title=title,
        scientific_objective=scientific_objective,
        target_observable=target_observable,
        desired_atom_count=desired_atom_count,
        preferred_geometry=preferred_geometry,
        priority=priority,
        constraints=constraints or {},
        metadata=metadata or {},
        **overrides,
    )


def make_spec(
    *,
    goal_id: str = "goal_test",
    objective_class: str = "balanced_campaign_search",
    target_observable: str = "rydberg_density",
    min_atoms: int = 4,
    max_atoms: int = 8,
    preferred_layouts: list[str] | None = None,
    sequence_families: list[SequenceFamily] | None = None,
    target_density: float = 0.5,
    perturbation_budget: int = 3,
    latency_budget: float = 0.30,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> ExperimentSpec:
    """Build an ExperimentSpec with a broad, balanced default search window."""
    return ExperimentSpec(
        goal_id=goal_id,
        objective_class=objective_class,
        target_observable=target_observable,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        preferred_layouts=preferred_layouts or ["square", "line", "triangular"],
        sequence_families=sequence_families
        or [
            SequenceFamily.GLOBAL_RAMP,
            SequenceFamily.DETUNING_SCAN,
            SequenceFamily.ADIABATIC_SWEEP,
            SequenceFamily.CONSTANT_DRIVE,
            SequenceFamily.BLACKMAN_SWEEP,
        ],
        target_density=target_density,
        perturbation_budget=perturbation_budget,
        latency_budget=latency_budget,
        reasoning_summary="Test experiment specification.",
        metadata=metadata or {},
        **overrides,
    )


def make_register(
    *,
    campaign_id: str = "campaign_test",
    spec_id: str = "spec_test",
    label: str | None = None,
    layout_type: str = "square",
    atom_count: int = 4,
    spacing_um: float = 7.0,
    coordinates: list[tuple[float, float]] | None = None,
    blockade_radius_um: float = 9.5,
    feasibility_score: float = 0.85,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> RegisterCandidate:
    """Build a RegisterCandidate with geometry-consistent derived fields."""
    if coordinates is None:
        coordinates = make_coordinates(layout_type, atom_count, spacing_um)
    if label is None:
        label = f"{layout_type}-{atom_count}"

    min_dist = float("inf")
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < min_dist:
                min_dist = distance
    if min_dist == float("inf"):
        min_dist = spacing_um

    blockade_pair_count = compute_blockade_pairs(coordinates, blockade_radius_um)

    return RegisterCandidate(
        campaign_id=campaign_id,
        spec_id=spec_id,
        label=label,
        layout_type=layout_type,
        atom_count=atom_count,
        coordinates=coordinates,
        min_distance_um=round(min_dist, 4),
        blockade_radius_um=blockade_radius_um,
        blockade_pair_count=blockade_pair_count,
        feasibility_score=feasibility_score,
        reasoning_summary=f"Test {layout_type} register with {atom_count} atoms.",
        metadata={"spacing_um": spacing_um, **(metadata or {})},
        **overrides,
    )


def make_sequence(
    *,
    campaign_id: str = "campaign_test",
    spec_id: str = "spec_test",
    register_candidate_id: str = "reg_test",
    label: str | None = None,
    sequence_family: SequenceFamily = SequenceFamily.ADIABATIC_SWEEP,
    channel_id: str = "rydberg_global",
    duration_ns: int = 3000,
    amplitude: float = 5.0,
    detuning: float = -10.0,
    phase: float = 0.0,
    waveform_kind: str = "adiabatic_conservative",
    predicted_cost: float = 0.15,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> SequenceCandidate:
    """Build a SequenceCandidate with conservative pulse defaults."""
    if label is None:
        label = f"test-{sequence_family.value}-{waveform_kind}"
    return SequenceCandidate(
        campaign_id=campaign_id,
        spec_id=spec_id,
        register_candidate_id=register_candidate_id,
        label=label,
        sequence_family=sequence_family,
        channel_id=channel_id,
        duration_ns=duration_ns,
        amplitude=amplitude,
        detuning=detuning,
        phase=phase,
        waveform_kind=waveform_kind,
        predicted_cost=predicted_cost,
        reasoning_summary=f"Test {sequence_family.value} pulse sequence.",
        metadata=metadata or {},
        **overrides,
    )


def make_robustness_report(
    *,
    campaign_id: str = "campaign_test",
    sequence_candidate_id: str = "seq_test",
    nominal_score: float = 0.75,
    robustness_score: float = 0.70,
    worst_case_score: float = 0.55,
    perturbation_average: float = 0.68,
    robustness_penalty: float = 0.07,
    score_std: float = 0.05,
    target_observable: str = "rydberg_density",
    scenario_scores: dict[str, float] | None = None,
    nominal_observables: dict[str, Any] | None = None,
    hamiltonian_metrics: dict[str, Any] | None = None,
    **overrides: Any,
) -> RobustnessReport:
    """Build a RobustnessReport with balanced nominal and perturbed scores."""
    return RobustnessReport(
        campaign_id=campaign_id,
        sequence_candidate_id=sequence_candidate_id,
        nominal_score=nominal_score,
        perturbation_average=perturbation_average,
        robustness_penalty=robustness_penalty,
        robustness_score=robustness_score,
        worst_case_score=worst_case_score,
        score_std=score_std,
        target_observable=target_observable,
        scenario_scores=scenario_scores
        or {"low_noise": 0.72, "medium_noise": 0.68, "stressed_noise": 0.55},
        nominal_observables=nominal_observables
        or {"rydberg_density": 0.75, "observable_score": 0.75},
        hamiltonian_metrics=hamiltonian_metrics or {"spectral_gap": 1.2, "energy_spread": 3.5},
        reasoning_summary="Test robustness evaluation.",
        **overrides,
    )


def make_evaluation_result(
    *,
    campaign_id: str = "campaign_test",
    sequence_candidate_id: str = "seq_test",
    register_candidate_id: str = "reg_test",
    nominal_score: float = 0.75,
    robustness_score: float = 0.70,
    worst_case_score: float = 0.55,
    observable_score: float = 0.75,
    objective_score: float = 0.72,
    backend_choice: BackendType = BackendType.LOCAL_PULSER_SIMULATION,
    estimated_cost: float = 0.12,
    estimated_latency: float = 0.15,
    **overrides: Any,
) -> EvaluationResult:
    """Build an EvaluationResult with realistic cost and score defaults."""
    return EvaluationResult(
        campaign_id=campaign_id,
        sequence_candidate_id=sequence_candidate_id,
        register_candidate_id=register_candidate_id,
        nominal_score=nominal_score,
        robustness_score=robustness_score,
        worst_case_score=worst_case_score,
        observable_score=observable_score,
        objective_score=objective_score,
        backend_choice=backend_choice,
        estimated_cost=estimated_cost,
        estimated_latency=estimated_latency,
        reasoning_summary="Test evaluation result.",
        **overrides,
    )


def make_memory_record(
    *,
    campaign_id: str = "campaign_test",
    source_candidate_id: str = "seq_test",
    lesson_type: str = "candidate_pattern",
    summary: str = "Test memory record.",
    signals: dict[str, Any] | None = None,
    reusable_tags: list[str] | None = None,
    **overrides: Any,
) -> MemoryRecord:
    """Build a MemoryRecord with reusable signal metadata."""
    return MemoryRecord(
        campaign_id=campaign_id,
        source_candidate_id=source_candidate_id,
        lesson_type=lesson_type,
        summary=summary,
        signals=signals or {"confidence": 0.8, "sequence_family": "adiabatic_sweep"},
        reusable_tags=reusable_tags or ["strong"],
        **overrides,
    )


def make_campaign(
    *,
    goal_id: str = "goal_test",
    status: CampaignStatus = CampaignStatus.CREATED,
    **overrides: Any,
) -> CampaignState:
    """Build a CampaignState with a configurable initial status."""
    return CampaignState(goal_id=goal_id, status=status, **overrides)


@pytest.fixture()
def default_param_space() -> PhysicsParameterSpace:
    """Provide the default physics parameter space used throughout the project."""
    return PhysicsParameterSpace.default()


@pytest.fixture()
def sample_goal() -> ExperimentGoal:
    """Provide a ready-made ExperimentGoal."""
    return make_goal()


@pytest.fixture()
def sample_spec() -> ExperimentSpec:
    """Provide a ready-made ExperimentSpec."""
    return make_spec()


@pytest.fixture()
def sample_register() -> RegisterCandidate:
    """Provide a ready-made RegisterCandidate."""
    return make_register()


@pytest.fixture()
def sample_sequence(sample_register: RegisterCandidate) -> SequenceCandidate:
    """Provide a SequenceCandidate linked to the sample register."""
    return make_sequence(register_candidate_id=sample_register.id)


@pytest.fixture()
def sample_report(sample_sequence: SequenceCandidate) -> RobustnessReport:
    """Provide a RobustnessReport linked to the sample sequence."""
    return make_robustness_report(sequence_candidate_id=sample_sequence.id)
