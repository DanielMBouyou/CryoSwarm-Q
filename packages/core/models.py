from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from packages.core.enums import (
    AgentName,
    BackendType,
    CampaignStatus,
    CandidateStatus,
    DecisionType,
    GoalStatus,
    NoiseLevel,
    SequenceFamily,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


JsonDict = dict[str, Any]
FloatDict = dict[str, float]


class CryoSwarmModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ScoringWeights(CryoSwarmModel):
    alpha: float = 0.45
    beta: float = 0.35
    gamma: float = 0.10
    delta: float = 0.10


class ExperimentGoal(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("goal"))
    title: str
    scientific_objective: str
    target_observable: str = "rydberg_density"
    priority: str = "balanced"
    desired_atom_count: int = 6
    preferred_geometry: str = "mixed"
    constraints: JsonDict = Field(default_factory=dict)
    priority_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    metadata: JsonDict = Field(default_factory=dict)
    status: GoalStatus = GoalStatus.DRAFT
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ExperimentGoalCreate(CryoSwarmModel):
    title: str
    scientific_objective: str
    target_observable: str = "rydberg_density"
    priority: str = "balanced"
    desired_atom_count: int = 6
    preferred_geometry: str = "mixed"
    constraints: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)


class ExperimentSpec(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("spec"))
    goal_id: str
    objective_class: str
    target_observable: str
    min_atoms: int
    max_atoms: int
    preferred_layouts: list[str]
    sequence_families: list[SequenceFamily]
    target_density: float = 0.5
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    perturbation_budget: int = 3
    latency_budget: float = 0.30
    reasoning_summary: str
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RegisterCandidate(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("reg"))
    campaign_id: str
    spec_id: str
    label: str
    layout_type: str
    atom_count: int
    coordinates: list[tuple[float, float]]
    device_constraints: JsonDict = Field(default_factory=dict)
    min_distance_um: float
    blockade_radius_um: float
    blockade_pair_count: int
    van_der_waals_matrix: list[list[float]] = Field(default_factory=list)
    feasibility_score: float
    status: CandidateStatus = CandidateStatus.PROPOSED
    reasoning_summary: str
    pulser_register_summary: JsonDict | None = None
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class SequenceCandidate(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("seq"))
    campaign_id: str
    spec_id: str
    register_candidate_id: str
    label: str
    sequence_family: SequenceFamily
    channel_id: str = "rydberg_global"
    duration_ns: int
    amplitude: float
    detuning: float
    phase: float
    waveform_kind: str = "constant"
    predicted_cost: float
    status: CandidateStatus = CandidateStatus.PROPOSED
    reasoning_summary: str
    serialized_pulser_sequence: JsonDict | None = None
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class NoiseScenario(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("noise"))
    label: NoiseLevel
    amplitude_jitter: float
    detuning_jitter: float
    dephasing_rate: float
    atom_loss_rate: float
    temperature_uk: float = 50.0
    state_prep_error: float = 0.005
    false_positive_rate: float = 0.01
    false_negative_rate: float = 0.05
    metadata: JsonDict = Field(default_factory=dict)


class RobustnessReport(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("robust"))
    campaign_id: str
    sequence_candidate_id: str
    nominal_score: float
    perturbation_average: float
    robustness_penalty: float
    robustness_score: float
    worst_case_score: float
    score_std: float
    target_observable: str
    scenario_scores: FloatDict = Field(default_factory=dict)
    nominal_observables: JsonDict = Field(default_factory=dict)
    scenario_observables: dict[str, JsonDict] = Field(default_factory=dict)
    hamiltonian_metrics: JsonDict = Field(default_factory=dict)
    reasoning_summary: str
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class BackendChoice(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("backend"))
    campaign_id: str
    sequence_candidate_id: str
    recommended_backend: BackendType
    state_dimension: int
    estimated_cost: float
    estimated_latency: float
    rationale: str
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentDecision(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("decision"))
    campaign_id: str
    agent_name: AgentName
    subject_id: str
    decision_type: DecisionType
    status: str
    reasoning_summary: str
    structured_output: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class EvaluationResult(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("eval"))
    campaign_id: str
    sequence_candidate_id: str
    register_candidate_id: str
    nominal_score: float
    robustness_score: float
    worst_case_score: float
    observable_score: float
    objective_score: float
    backend_choice: BackendType
    estimated_cost: float
    estimated_latency: float
    final_rank: int | None = None
    status: CandidateStatus = CandidateStatus.EVALUATED
    reasoning_summary: str
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryRecord(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("memory"))
    campaign_id: str
    source_candidate_id: str
    lesson_type: str
    summary: str
    signals: JsonDict = Field(default_factory=dict)
    reusable_tags: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class CampaignState(CryoSwarmModel):
    id: str = Field(default_factory=lambda: make_id("campaign"))
    goal_id: str
    spec_id: str | None = None
    status: CampaignStatus = CampaignStatus.CREATED
    candidate_count: int = 0
    ranked_candidate_ids: list[str] = Field(default_factory=list)
    top_candidate_id: str | None = None
    summary: str | None = None
    summary_report: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DemoGoalRequest(CryoSwarmModel):
    title: str = "Robust neutral-atom benchmark sweep"
    scientific_objective: str = (
        "Design a small neutral-atom experiment campaign that balances "
        "robustness and execution feasibility."
    )
    target_observable: str = "rydberg_density"
    desired_atom_count: int = 6
    preferred_geometry: str = "mixed"


class PipelineSummary(CryoSwarmModel):
    campaign: CampaignState
    goal: ExperimentGoal
    spec: ExperimentSpec
    top_candidate: EvaluationResult | None = None
    ranked_candidates: list[EvaluationResult] = Field(default_factory=list)
    decisions: list[AgentDecision] = Field(default_factory=list)
    robustness_reports: list[RobustnessReport] = Field(default_factory=list)
    memory_records: list[MemoryRecord] = Field(default_factory=list)
