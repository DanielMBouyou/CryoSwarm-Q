from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, TypeVar

from packages.agents.base import BaseAgent
from packages.agents.campaign_agent import CampaignAgent
from packages.agents.geometry_agent import GeometryAgent
from packages.agents.memory_agent import MemoryAgent
from packages.agents.noise_agent import NoiseRobustnessAgent
from packages.agents.problem_agent import ProblemFramingAgent
from packages.agents.results_agent import ResultsAgent
from packages.agents.routing_agent import BackendRoutingAgent
from packages.agents.sequence_agent import SequenceAgent
from packages.core.enums import CampaignStatus, DecisionType, GoalStatus
from packages.core.logging import get_logger
from packages.core.models import (
    AgentDecision,
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    ExperimentSpec,
    MemoryRecord,
    PipelineSummary,
    RegisterCandidate,
    RobustnessReport,
    SequenceCandidate,
)
from packages.db.repositories import CryoSwarmRepository
from packages.scoring.objective import compute_objective_score


ModelT = TypeVar("ModelT")


def _evaluate_noise_task(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
) -> RobustnessReport:
    return NoiseRobustnessAgent().run(spec, register_candidate, sequence_candidate)


class CryoSwarmPipeline:
    """Run the deterministic CryoSwarm-Q orchestration stages end to end."""

    def __init__(self, repository: CryoSwarmRepository | None = None, parallel: bool = False) -> None:
        self.repository = repository
        self.parallel = parallel
        self.logger = get_logger(__name__)
        self.problem_agent = ProblemFramingAgent()
        self.geometry_agent = GeometryAgent()
        self.sequence_agent = SequenceAgent()
        self.noise_agent = NoiseRobustnessAgent()
        self.routing_agent = BackendRoutingAgent()
        self.campaign_agent = CampaignAgent()
        self.results_agent = ResultsAgent()
        self.memory_agent = MemoryAgent()

    def _safe_repository_call(
        self,
        operation: Callable[[], ModelT],
        description: str,
        default: ModelT | None = None,
    ) -> ModelT | None:
        if not self.repository:
            return default
        try:
            return operation()
        except Exception as exc:
            self.logger.error("Repository %s failed: %s", description, exc)
            return default

    def _record_agent_failure(
        self,
        decisions: list[AgentDecision],
        agent: BaseAgent,
        campaign_id: str,
        subject_id: str,
        decision_type: DecisionType,
        exc: Exception,
    ) -> None:
        self.logger.error("Agent %s failed: %s", agent.agent_name.value, exc)
        decisions.append(
            agent.build_decision(
                campaign_id=campaign_id,
                subject_id=subject_id,
                decision_type=decision_type,
                status="failed",
                reasoning_summary=str(exc),
                structured_output={"error": str(exc)},
            )
        )

    def _persist_state(
        self,
        goal: ExperimentGoal,
        campaign: CampaignState,
        registers: list[RegisterCandidate],
        sequences: list[SequenceCandidate],
        reports: list[RobustnessReport],
        ranked_candidates: list[EvaluationResult],
        decisions: list[AgentDecision],
        memory_records: list[MemoryRecord],
    ) -> None:
        if not self.repository:
            return
        self._safe_repository_call(lambda: self.repository.create_goal(goal), "create_goal")
        self._safe_repository_call(lambda: self.repository.update_campaign(campaign), "update_campaign")
        if registers:
            self._safe_repository_call(
                lambda: self.repository.insert_register_candidates(registers),
                "insert_register_candidates",
            )
        if sequences:
            self._safe_repository_call(
                lambda: self.repository.insert_sequence_candidates(sequences),
                "insert_sequence_candidates",
            )
        for report in reports:
            self._safe_repository_call(
                lambda report=report: self.repository.insert_robustness_report(report),
                "insert_robustness_report",
            )
        if ranked_candidates:
            self._safe_repository_call(
                lambda: self.repository.insert_evaluation_results(ranked_candidates),
                "insert_evaluation_results",
            )
        for decision in decisions:
            self._safe_repository_call(
                lambda decision=decision: self.repository.insert_agent_decision(decision),
                "insert_agent_decision",
            )
        for record in memory_records:
            self._safe_repository_call(
                lambda record=record: self.repository.insert_memory_record(record),
                "insert_memory_record",
            )

    def _build_summary(
        self,
        *,
        status: str,
        campaign: CampaignState,
        goal: ExperimentGoal,
        spec: ExperimentSpec | None,
        ranked_candidates: list[EvaluationResult],
        decisions: list[AgentDecision],
        reports: list[RobustnessReport],
        memory_records: list[MemoryRecord],
        backend_mix: dict[str, int],
        error: str | None = None,
    ) -> PipelineSummary:
        return PipelineSummary(
            campaign=campaign,
            goal=goal,
            spec=spec,
            status=status,
            error=error,
            total_candidates=campaign.candidate_count,
            ranked_count=len(ranked_candidates),
            top_candidate_id=campaign.top_candidate_id,
            backend_mix=backend_mix,
            top_candidate=ranked_candidates[0] if ranked_candidates else None,
            ranked_candidates=ranked_candidates,
            decisions=decisions,
            robustness_reports=reports,
            memory_records=memory_records,
        )

    def _evaluate_sequences(
        self,
        spec: ExperimentSpec,
        campaign_id: str,
        sequences: list[SequenceCandidate],
        register_lookup: dict[str, RegisterCandidate],
        decisions: list[AgentDecision],
    ) -> tuple[list[RobustnessReport], list[EvaluationResult], dict[str, int]]:
        reports: list[RobustnessReport] = []
        evaluations: list[EvaluationResult] = []
        backend_counter: Counter[str] = Counter()

        if self.parallel and sequences:
            max_workers = min(os.cpu_count() or 1, 4)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        _evaluate_noise_task,
                        spec,
                        register_lookup[sequence.register_candidate_id],
                        sequence,
                    ): sequence
                    for sequence in sequences
                }
                for future in as_completed(future_map):
                    sequence = future_map[future]
                    register_candidate = register_lookup[sequence.register_candidate_id]
                    try:
                        report = future.result()
                        reports.append(report)
                        decisions.append(
                            self.noise_agent.build_decision(
                                campaign_id=campaign_id,
                                subject_id=sequence.id,
                                decision_type=DecisionType.ROBUSTNESS_EVALUATION,
                                status="completed",
                                reasoning_summary=report.reasoning_summary,
                                structured_output=report.model_dump(mode="json"),
                            )
                        )
                    except Exception as exc:
                        self._record_agent_failure(
                            decisions,
                            self.noise_agent,
                            campaign_id,
                            sequence.id,
                            DecisionType.ROBUSTNESS_EVALUATION,
                            exc,
                        )
                        continue

                    self._route_and_score_sequence(
                        spec,
                        campaign_id,
                        sequence,
                        register_candidate,
                        report,
                        decisions,
                        evaluations,
                        backend_counter,
                    )
            return reports, evaluations, dict(backend_counter)

        for sequence in sequences:
            register_candidate = register_lookup[sequence.register_candidate_id]
            try:
                report = self.noise_agent.run(spec, register_candidate, sequence)
                reports.append(report)
                decisions.append(
                    self.noise_agent.build_decision(
                        campaign_id=campaign_id,
                        subject_id=sequence.id,
                        decision_type=DecisionType.ROBUSTNESS_EVALUATION,
                        status="completed",
                        reasoning_summary=report.reasoning_summary,
                        structured_output=report.model_dump(mode="json"),
                    )
                )
            except Exception as exc:
                self._record_agent_failure(
                    decisions,
                    self.noise_agent,
                    campaign_id,
                    sequence.id,
                    DecisionType.ROBUSTNESS_EVALUATION,
                    exc,
                )
                continue

            self._route_and_score_sequence(
                spec,
                campaign_id,
                sequence,
                register_candidate,
                report,
                decisions,
                evaluations,
                backend_counter,
            )

        return reports, evaluations, dict(backend_counter)

    def _route_and_score_sequence(
        self,
        spec: ExperimentSpec,
        campaign_id: str,
        sequence: SequenceCandidate,
        register_candidate: RegisterCandidate,
        report: RobustnessReport,
        decisions: list[AgentDecision],
        evaluations: list[EvaluationResult],
        backend_counter: Counter[str],
    ) -> None:
        try:
            backend_choice = self.routing_agent.run(spec, sequence, report)
            backend_counter[backend_choice.recommended_backend.value] += 1
            decisions.append(
                self.routing_agent.build_decision(
                    campaign_id=campaign_id,
                    subject_id=sequence.id,
                    decision_type=DecisionType.BACKEND_ROUTING,
                    status="completed",
                    reasoning_summary=backend_choice.rationale,
                    structured_output=backend_choice.model_dump(mode="json"),
                )
            )
        except Exception as exc:
            self._record_agent_failure(
                decisions,
                self.routing_agent,
                campaign_id,
                sequence.id,
                DecisionType.BACKEND_ROUTING,
                exc,
            )
            return

        objective_score = compute_objective_score(
            observable_score=report.nominal_score,
            robustness=report.robustness_score,
            cost=backend_choice.estimated_cost,
            latency=backend_choice.estimated_latency,
            weights=spec.scoring_weights,
        )

        evaluations.append(
            EvaluationResult(
                campaign_id=campaign_id,
                sequence_candidate_id=sequence.id,
                register_candidate_id=sequence.register_candidate_id,
                nominal_score=report.nominal_score,
                robustness_score=report.robustness_score,
                worst_case_score=report.worst_case_score,
                observable_score=float(
                    report.nominal_observables.get("observable_score", report.nominal_score)
                ),
                objective_score=objective_score,
                backend_choice=backend_choice.recommended_backend,
                estimated_cost=backend_choice.estimated_cost,
                estimated_latency=backend_choice.estimated_latency,
                reasoning_summary=(
                    f"Combined robustness score with routing recommendation for "
                    f"{sequence.label}."
                ),
                metadata={
                    "register_label": register_candidate.label,
                    "backend_rationale": backend_choice.rationale,
                    "nominal_observables": report.nominal_observables,
                    "hamiltonian_metrics": report.hamiltonian_metrics,
                    "score_std": report.score_std,
                },
            )
        )

    def run(self, goal: ExperimentGoal) -> PipelineSummary:
        """Execute the campaign pipeline from goal framing to memory capture.

        Stages:
        1. Load recent memory context.
        2. Frame the incoming goal into an experiment specification.
        3. Generate hardware-valid register candidates.
        4. Generate pulse-sequence candidates per register.
        5. Evaluate robustness, route backends, and score candidates.
        6. Rank the campaign, summarize results, and persist reusable lessons.

        The method is defensive by design: agent and repository failures are
        captured as structured decisions so the API receives a `PipelineSummary`
        instead of an uncaught exception.
        """
        stored_goal = goal.model_copy(update={"status": GoalStatus.RUNNING})
        self._safe_repository_call(lambda: self.repository.create_goal(stored_goal), "create_goal")

        memory_context = self._safe_repository_call(
            lambda: self.repository.list_recent_memory(limit=12),
            "list_recent_memory",
            default=[],
        ) or []
        campaign = CampaignState(goal_id=stored_goal.id, status=CampaignStatus.RUNNING)
        self._safe_repository_call(lambda: self.repository.create_campaign(campaign), "create_campaign")

        decisions: list[AgentDecision] = []
        registers: list[RegisterCandidate] = []
        all_sequences: list[SequenceCandidate] = []
        reports: list[RobustnessReport] = []
        ranked_candidates: list[EvaluationResult] = []
        memory_records: list[MemoryRecord] = []
        backend_counter: Counter[str] = Counter()
        spec: ExperimentSpec | None = None

        try:
            try:
                spec = self.problem_agent.run(stored_goal, memory_context)
                campaign = campaign.model_copy(update={"spec_id": spec.id})
                decisions.append(
                    self.problem_agent.build_decision(
                        campaign_id=campaign.id,
                        subject_id=spec.id,
                        decision_type=DecisionType.SPECIFICATION,
                        status="completed",
                        reasoning_summary=spec.reasoning_summary,
                        structured_output=spec.model_dump(mode="json"),
                    )
                )
            except Exception as exc:
                self._record_agent_failure(
                    decisions,
                    self.problem_agent,
                    campaign.id,
                    stored_goal.id,
                    DecisionType.SPECIFICATION,
                    exc,
                )
                stored_goal = stored_goal.model_copy(update={"status": GoalStatus.FAILED})
                campaign = campaign.model_copy(
                    update={
                        "status": CampaignStatus.FAILED,
                        "summary": "Problem framing failed.",
                    }
                )
                self._persist_state(
                    stored_goal,
                    campaign,
                    registers,
                    all_sequences,
                    reports,
                    ranked_candidates,
                    decisions,
                    memory_records,
                )
                return self._build_summary(
                    status="FAILED",
                    campaign=campaign,
                    goal=stored_goal,
                    spec=None,
                    ranked_candidates=ranked_candidates,
                    decisions=decisions,
                    reports=reports,
                    memory_records=memory_records,
                    backend_mix=dict(backend_counter),
                    error="Problem framing failed.",
                )

            try:
                registers = self.geometry_agent.run(spec, campaign.id, memory_context)
                decisions.append(
                    self.geometry_agent.build_decision(
                        campaign_id=campaign.id,
                        subject_id=campaign.id,
                        decision_type=DecisionType.CANDIDATE_GENERATION,
                        status="completed",
                        reasoning_summary=f"Generated {len(registers)} register candidates.",
                        structured_output={"register_candidate_ids": [item.id for item in registers]},
                    )
                )
            except Exception as exc:
                self._record_agent_failure(
                    decisions,
                    self.geometry_agent,
                    campaign.id,
                    campaign.id,
                    DecisionType.CANDIDATE_GENERATION,
                    exc,
                )
                registers = []

            if not registers:
                self.logger.warning("GeometryAgent produced no register candidates.")
                stored_goal = stored_goal.model_copy(update={"status": GoalStatus.COMPLETED})
                campaign = campaign.model_copy(
                    update={
                        "status": CampaignStatus.NO_CANDIDATES,
                        "candidate_count": 0,
                        "summary": "No feasible register candidates generated.",
                        "summary_report": {"reason": "geometry_agent_returned_no_candidates"},
                    }
                )
                self._persist_state(
                    stored_goal,
                    campaign,
                    registers,
                    all_sequences,
                    reports,
                    ranked_candidates,
                    decisions,
                    memory_records,
                )
                return self._build_summary(
                    status="NO_CANDIDATES",
                    campaign=campaign,
                    goal=stored_goal,
                    spec=spec,
                    ranked_candidates=ranked_candidates,
                    decisions=decisions,
                    reports=reports,
                    memory_records=memory_records,
                    backend_mix=dict(backend_counter),
                    error="No feasible register candidates generated.",
                )

            for register_candidate in registers:
                try:
                    generated = self.sequence_agent.run(spec, register_candidate, campaign.id, memory_context)
                    all_sequences.extend(generated)
                except Exception as exc:
                    self._record_agent_failure(
                        decisions,
                        self.sequence_agent,
                        campaign.id,
                        register_candidate.id,
                        DecisionType.CANDIDATE_GENERATION,
                        exc,
                    )

            decisions.append(
                self.sequence_agent.build_decision(
                    campaign_id=campaign.id,
                    subject_id=campaign.id,
                    decision_type=DecisionType.CANDIDATE_GENERATION,
                    status="completed" if all_sequences else "empty",
                    reasoning_summary=f"Generated {len(all_sequences)} sequence candidates.",
                    structured_output={"sequence_candidate_ids": [item.id for item in all_sequences]},
                )
            )

            if not all_sequences:
                self.logger.warning("SequenceAgent produced no sequence candidates.")
                stored_goal = stored_goal.model_copy(update={"status": GoalStatus.COMPLETED})
                campaign = campaign.model_copy(
                    update={
                        "status": CampaignStatus.NO_CANDIDATES,
                        "candidate_count": 0,
                        "summary": "No sequence candidates generated.",
                        "summary_report": {"reason": "sequence_agent_returned_no_candidates"},
                    }
                )
                self._persist_state(
                    stored_goal,
                    campaign,
                    registers,
                    all_sequences,
                    reports,
                    ranked_candidates,
                    decisions,
                    memory_records,
                )
                return self._build_summary(
                    status="NO_CANDIDATES",
                    campaign=campaign,
                    goal=stored_goal,
                    spec=spec,
                    ranked_candidates=ranked_candidates,
                    decisions=decisions,
                    reports=reports,
                    memory_records=memory_records,
                    backend_mix=dict(backend_counter),
                    error="No sequence candidates generated.",
                )

            register_lookup = {item.id: item for item in registers}
            reports, ranked_candidates, backend_mix = self._evaluate_sequences(
                spec,
                campaign.id,
                all_sequences,
                register_lookup,
                decisions,
            )
            backend_counter = Counter(backend_mix)

            if not ranked_candidates:
                self.logger.warning("No evaluation results were produced for campaign %s.", campaign.id)
                stored_goal = stored_goal.model_copy(update={"status": GoalStatus.COMPLETED})
                campaign = campaign.model_copy(
                    update={
                        "status": CampaignStatus.NO_CANDIDATES,
                        "candidate_count": 0,
                        "summary": "No evaluation results were produced.",
                        "summary_report": {"reason": "no_evaluation_results"},
                    }
                )
                self._persist_state(
                    stored_goal,
                    campaign,
                    registers,
                    all_sequences,
                    reports,
                    ranked_candidates,
                    decisions,
                    memory_records,
                )
                return self._build_summary(
                    status="NO_CANDIDATES",
                    campaign=campaign,
                    goal=stored_goal,
                    spec=spec,
                    ranked_candidates=ranked_candidates,
                    decisions=decisions,
                    reports=reports,
                    memory_records=memory_records,
                    backend_mix=dict(backend_counter),
                    error="No evaluation results were produced.",
                )

            try:
                campaign, ranked_candidates = self.campaign_agent.run(campaign, ranked_candidates)
                decisions.append(
                    self.campaign_agent.build_decision(
                        campaign_id=campaign.id,
                        subject_id=campaign.id,
                        decision_type=DecisionType.CAMPAIGN_RANKING,
                        status="completed",
                        reasoning_summary=campaign.summary or "Campaign ranking completed.",
                        structured_output={
                            "ranked_sequence_ids": [item.sequence_candidate_id for item in ranked_candidates],
                            "backend_mix": dict(backend_counter),
                        },
                    )
                )
            except Exception as exc:
                self._record_agent_failure(
                    decisions,
                    self.campaign_agent,
                    campaign.id,
                    campaign.id,
                    DecisionType.CAMPAIGN_RANKING,
                    exc,
                )
                campaign = campaign.model_copy(
                    update={
                        "status": CampaignStatus.FAILED,
                        "candidate_count": len(ranked_candidates),
                        "summary": "Campaign ranking failed.",
                    }
                )

            try:
                results_summary = self.results_agent.run(stored_goal, spec, campaign, ranked_candidates)
                campaign = campaign.model_copy(
                    update={
                        "summary": str(results_summary["summary"]),
                        "summary_report": {
                            **campaign.summary_report,
                            "backend_mix": dict(backend_counter),
                            "results_summary": results_summary,
                        },
                    }
                )
                decisions.append(
                    self.results_agent.build_decision(
                        campaign_id=campaign.id,
                        subject_id=campaign.id,
                        decision_type=DecisionType.RESULTS_SUMMARY,
                        status="completed",
                        reasoning_summary=str(results_summary["summary"]),
                        structured_output=results_summary,
                    )
                )
            except Exception as exc:
                self._record_agent_failure(
                    decisions,
                    self.results_agent,
                    campaign.id,
                    campaign.id,
                    DecisionType.RESULTS_SUMMARY,
                    exc,
                )
                campaign = campaign.model_copy(
                    update={
                        "summary_report": {
                            **campaign.summary_report,
                            "backend_mix": dict(backend_counter),
                        }
                    }
                )

            sequence_lookup = {item.id: item for item in all_sequences}
            try:
                memory_records = self.memory_agent.run(
                    campaign.id,
                    ranked_candidates,
                    sequence_lookup,
                    register_lookup,
                )
                decisions.append(
                    self.memory_agent.build_decision(
                        campaign_id=campaign.id,
                        subject_id=campaign.id,
                        decision_type=DecisionType.MEMORY_CAPTURE,
                        status="completed",
                        reasoning_summary=f"Stored {len(memory_records)} memory records for campaign reuse.",
                        structured_output={"memory_record_ids": [item.id for item in memory_records]},
                    )
                )
            except Exception as exc:
                self._record_agent_failure(
                    decisions,
                    self.memory_agent,
                    campaign.id,
                    campaign.id,
                    DecisionType.MEMORY_CAPTURE,
                    exc,
                )
                memory_records = []

            stored_goal = stored_goal.model_copy(
                update={
                    "status": GoalStatus.COMPLETED
                    if campaign.status != CampaignStatus.FAILED
                    else GoalStatus.FAILED
                }
            )
            self._persist_state(
                stored_goal,
                campaign,
                registers,
                all_sequences,
                reports,
                ranked_candidates,
                decisions,
                memory_records,
            )
            return self._build_summary(
                status="COMPLETED" if campaign.status != CampaignStatus.FAILED else "FAILED",
                campaign=campaign,
                goal=stored_goal,
                spec=spec,
                ranked_candidates=ranked_candidates,
                decisions=decisions,
                reports=reports,
                memory_records=memory_records,
                backend_mix=dict(backend_counter),
                error=campaign.summary if campaign.status == CampaignStatus.FAILED else None,
            )
        except Exception as exc:
            self.logger.error("Unhandled pipeline failure: %s", exc)
            stored_goal = stored_goal.model_copy(update={"status": GoalStatus.FAILED})
            campaign = campaign.model_copy(
                update={
                    "status": CampaignStatus.FAILED,
                    "summary": "Unhandled pipeline failure.",
                    "summary_report": {"error": str(exc)},
                }
            )
            self._persist_state(
                stored_goal,
                campaign,
                registers,
                all_sequences,
                reports,
                ranked_candidates,
                decisions,
                memory_records,
            )
            return self._build_summary(
                status="FAILED",
                campaign=campaign,
                goal=stored_goal,
                spec=spec,
                ranked_candidates=ranked_candidates,
                decisions=decisions,
                reports=reports,
                memory_records=memory_records,
                backend_mix=dict(backend_counter),
                error=str(exc),
            )
