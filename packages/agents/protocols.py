from __future__ import annotations

"""Agent interface protocols for CryoSwarm-Q orchestration.

These protocols define the minimal audit-trail contract shared by all agents
and the category-specific run signatures used by the pipeline.
"""

from typing import Any, Protocol, runtime_checkable

from packages.core.enums import AgentName, DecisionType
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
    SequenceCandidate,
)


@runtime_checkable
class AgentProtocol(Protocol):
    """Minimal contract shared by every CryoSwarm-Q agent."""

    agent_name: AgentName

    def build_decision(
        self,
        campaign_id: str,
        subject_id: str,
        decision_type: DecisionType,
        status: str,
        reasoning_summary: str,
        structured_output: dict[str, Any],
    ) -> AgentDecision: ...


@runtime_checkable
class ProblemFramingProtocol(AgentProtocol, Protocol):
    """Agent that frames an experiment goal into an immutable specification."""

    def run(
        self,
        goal: ExperimentGoal,
        memory_records: list[MemoryRecord] | None = None,
    ) -> ExperimentSpec: ...


@runtime_checkable
class GeometryProtocol(AgentProtocol, Protocol):
    """Agent that generates hardware-valid atom-register candidates."""

    def run(
        self,
        spec: ExperimentSpec,
        campaign_id: str,
        memory_records: list[MemoryRecord] | None = None,
    ) -> list[RegisterCandidate]: ...


@runtime_checkable
class SequenceProtocol(AgentProtocol, Protocol):
    """Agent that generates pulse-sequence candidates per register."""

    def run(
        self,
        spec: ExperimentSpec,
        register_candidate: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord] | None = None,
    ) -> list[SequenceCandidate]: ...


@runtime_checkable
class NoiseEvaluationProtocol(AgentProtocol, Protocol):
    """Agent that evaluates robustness under modeled noise."""

    def run(
        self,
        spec: ExperimentSpec,
        register_candidate: RegisterCandidate,
        sequence_candidate: SequenceCandidate,
    ) -> RobustnessReport: ...


@runtime_checkable
class RoutingProtocol(AgentProtocol, Protocol):
    """Agent that recommends an execution backend for a candidate."""

    def run(
        self,
        spec: ExperimentSpec,
        sequence_candidate: SequenceCandidate,
        report: RobustnessReport,
    ) -> BackendChoice: ...


@runtime_checkable
class CampaignProtocol(AgentProtocol, Protocol):
    """Agent that ranks candidates and produces campaign state updates."""

    def run(
        self,
        campaign: CampaignState,
        evaluations: list[EvaluationResult],
    ) -> tuple[CampaignState, list[EvaluationResult]]: ...


@runtime_checkable
class ResultsProtocol(AgentProtocol, Protocol):
    """Agent that generates a human-readable results summary."""

    def run(
        self,
        goal: ExperimentGoal,
        spec: ExperimentSpec,
        campaign: CampaignState,
        ranked_candidates: list[EvaluationResult],
    ) -> dict[str, object]: ...


@runtime_checkable
class MemoryCaptureProtocol(AgentProtocol, Protocol):
    """Agent that extracts reusable lessons from ranked campaign outcomes."""

    def run(
        self,
        campaign_id: str,
        ranked_candidates: list[EvaluationResult],
        sequence_lookup: dict[str, SequenceCandidate],
        register_lookup: dict[str, RegisterCandidate],
    ) -> list[MemoryRecord]: ...
