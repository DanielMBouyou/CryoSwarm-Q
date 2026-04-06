from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.core.enums import AgentName, BackendType, CampaignStatus, CandidateStatus
from packages.core.models import (
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    MemoryRecord,
    RegisterCandidate,
    RobustnessReport,
    SequenceCandidate,
)
from packages.orchestration.pipeline import CryoSwarmPipeline

pytestmark = [pytest.mark.slow, pytest.mark.integration]


class InMemoryRepository:
    def __init__(self) -> None:
        self.goals: dict[str, object] = {}
        self.campaigns: dict[str, CampaignState] = {}
        self.register_candidates: dict[str, RegisterCandidate] = {}
        self.sequence_candidates: dict[str, SequenceCandidate] = {}
        self.robustness_reports: dict[str, RobustnessReport] = {}
        self.agent_decisions: dict[str, object] = {}
        self.memory_records: dict[str, MemoryRecord] = {}
        self.evaluation_results: dict[str, EvaluationResult] = {}

    def create_goal(self, goal):
        self.goals[goal.id] = goal
        return goal

    def get_goal(self, goal_id: str):
        return self.goals.get(goal_id)

    def create_campaign(self, campaign: CampaignState) -> CampaignState:
        self.campaigns[campaign.id] = campaign
        return campaign

    def get_campaign(self, campaign_id: str) -> CampaignState | None:
        return self.campaigns.get(campaign_id)

    def update_campaign(self, campaign: CampaignState) -> CampaignState:
        self.campaigns[campaign.id] = campaign
        return campaign

    def insert_register_candidates(self, candidates: list[RegisterCandidate]) -> list[RegisterCandidate]:
        for candidate in candidates:
            self.register_candidates[candidate.id] = candidate
        return candidates

    def insert_sequence_candidates(self, candidates: list[SequenceCandidate]) -> list[SequenceCandidate]:
        for candidate in candidates:
            self.sequence_candidates[candidate.id] = candidate
        return candidates

    def insert_robustness_report(self, report: RobustnessReport) -> RobustnessReport:
        self.robustness_reports[report.id] = report
        return report

    def insert_agent_decision(self, decision):
        self.agent_decisions[decision.id] = decision
        return decision

    def insert_memory_record(self, record: MemoryRecord) -> MemoryRecord:
        self.memory_records[record.id] = record
        return record

    def insert_evaluation_results(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        for result in results:
            self.evaluation_results[result.id] = result
        return results

    def list_candidates_for_campaign(self, campaign_id: str) -> list[EvaluationResult]:
        results = [
            result for result in self.evaluation_results.values() if result.campaign_id == campaign_id
        ]
        return sorted(
            results,
            key=lambda item: (
                item.final_rank if item.final_rank is not None else 999,
                -item.objective_score,
            ),
        )

    def list_recent_memory(self, limit: int = 10) -> list[MemoryRecord]:
        records = sorted(
            self.memory_records.values(),
            key=lambda record: record.created_at,
            reverse=True,
        )
        return records[:limit]


def _stub_noise_agent(pipeline: CryoSwarmPipeline) -> None:
    def _run(spec, register_candidate, sequence_candidate):  # type: ignore[no-untyped-def]
        normalized_amplitude = min(sequence_candidate.amplitude / 10.0, 1.0)
        nominal = round(max(0.1, min(0.95, 0.45 + 0.35 * normalized_amplitude)), 4)
        robustness = round(max(0.1, nominal - 0.08), 4)
        worst_case = round(max(0.05, robustness - 0.06), 4)
        return RobustnessReport(
            campaign_id=sequence_candidate.campaign_id,
            sequence_candidate_id=sequence_candidate.id,
            nominal_score=nominal,
            perturbation_average=robustness,
            robustness_penalty=round(nominal - worst_case, 4),
            robustness_score=robustness,
            worst_case_score=worst_case,
            score_std=0.03,
            target_observable=spec.target_observable,
            scenario_scores={
                "low_noise": robustness,
                "medium_noise": round(max(0.05, robustness - 0.03), 4),
                "stressed_noise": worst_case,
            },
            nominal_observables={
                "observable_score": nominal,
                "spectral_gap": 0.8,
                "rydberg_density": spec.target_density,
            },
            scenario_observables={},
            hamiltonian_metrics={"dimension": 2 ** register_candidate.atom_count, "spectral_gap": 0.8},
            reasoning_summary=f"Stub robustness for {sequence_candidate.label}.",
            metadata={"stubbed": True},
        )

    pipeline.noise_agent.run = _run  # type: ignore[method-assign]


def _build_goal(desired_atom_count: int) -> ExperimentGoal:
    return ExperimentGoal(
        title="Test AF ordering",
        scientific_objective="Build a robust line-register campaign for AF ordering.",
        desired_atom_count=desired_atom_count,
        priority="balanced",
        preferred_geometry="line",
    )


def test_pipeline_run_end_to_end_with_in_memory_repository() -> None:
    repository = InMemoryRepository()
    pipeline = CryoSwarmPipeline(repository=repository)  # type: ignore[arg-type]
    _stub_noise_agent(pipeline)

    summary = pipeline.run(_build_goal(4))

    assert summary.status == "COMPLETED"
    assert summary.campaign.status == CampaignStatus.COMPLETED
    assert summary.total_candidates > 0
    assert summary.top_candidate_id is not None
    assert summary.ranked_count > 0
    assert summary.backend_mix
    assert summary.memory_records
    assert {
        AgentName.PROBLEM_FRAMING,
        AgentName.GEOMETRY,
        AgentName.SEQUENCE,
        AgentName.NOISE,
        AgentName.ROUTING,
        AgentName.CAMPAIGN,
        AgentName.RESULTS,
        AgentName.MEMORY,
    } <= {decision.agent_name for decision in summary.decisions}


def test_pipeline_supports_minimum_viable_goal() -> None:
    repository = InMemoryRepository()
    pipeline = CryoSwarmPipeline(repository=repository)  # type: ignore[arg-type]
    _stub_noise_agent(pipeline)

    summary = pipeline.run(_build_goal(3))

    assert summary.status == "COMPLETED"
    assert summary.total_candidates > 0
    assert summary.ranked_count > 0


def test_pipeline_supports_larger_goal() -> None:
    repository = InMemoryRepository()
    pipeline = CryoSwarmPipeline(repository=repository)  # type: ignore[arg-type]
    _stub_noise_agent(pipeline)

    summary = pipeline.run(_build_goal(8))

    assert summary.status == "COMPLETED"
    assert summary.total_candidates > 0
    assert summary.backend_mix


def test_pipeline_persists_memory_records() -> None:
    repository = InMemoryRepository()
    pipeline = CryoSwarmPipeline(repository=repository)  # type: ignore[arg-type]
    _stub_noise_agent(pipeline)

    summary = pipeline.run(_build_goal(4))

    assert summary.memory_records
    assert repository.memory_records


def test_second_campaign_receives_memory_context() -> None:
    repository = InMemoryRepository()
    pipeline = CryoSwarmPipeline(repository=repository)  # type: ignore[arg-type]
    _stub_noise_agent(pipeline)

    first_summary = pipeline.run(_build_goal(4))
    second_summary = pipeline.run(_build_goal(4))

    assert first_summary.memory_records
    assert second_summary.spec is not None
    assert second_summary.spec.metadata["memory_record_count"] > 0
    assert second_summary.memory_records
