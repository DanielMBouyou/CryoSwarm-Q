from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, TypeVar

from packages.agents.campaign_agent import CampaignAgent
from packages.agents.geometry_agent import GeometryAgent
from packages.agents.memory_agent import MemoryAgent
from packages.agents.noise_agent import NoiseRobustnessAgent
from packages.agents.problem_agent import ProblemFramingAgent
from packages.agents.protocols import (
    AgentProtocol,
    CampaignProtocol,
    GeometryProtocol,
    MemoryCaptureProtocol,
    NoiseEvaluationProtocol,
    ProblemFramingProtocol,
    ResultsProtocol,
    RoutingProtocol,
    SequenceProtocol,
)
from packages.agents.results_agent import ResultsAgent
from packages.agents.routing_agent import BackendRoutingAgent
from packages.agents.sequence_agent import SequenceAgent
from packages.agents.sequence_strategy import SequenceStrategy, SequenceStrategyMode
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
from packages.core.parameter_space import PhysicsParameterSpace
from packages.db.repositories import CryoSwarmRepository
from packages.ml.surrogate_filter import SurrogateFilter
from packages.orchestration.events import EventBus
from packages.orchestration.phases import (
    EvaluationPhase,
    GeometryGenerationPhase,
    MemoryCapturePhase,
    PipelinePhase,
    ProblemFramingPhase,
    RankingPhase,
    ResultsSummaryPhase,
    SequenceGenerationPhase,
    SurrogateFilterPhase,
)
from packages.scoring.objective import compute_objective_score


ModelT = TypeVar("ModelT")


def _evaluate_noise_task(
    spec: ExperimentSpec,
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
) -> RobustnessReport:
    return NoiseRobustnessAgent().run(spec, register_candidate, sequence_candidate)


@dataclass
class PipelineContext:
    """Mutable pipeline state threaded through the orchestration phases."""

    goal: ExperimentGoal
    campaign: CampaignState
    spec: ExperimentSpec | None = None
    registers: list[RegisterCandidate] = field(default_factory=list)
    sequences: list[SequenceCandidate] = field(default_factory=list)
    reports: list[RobustnessReport] = field(default_factory=list)
    ranked_candidates: list[EvaluationResult] = field(default_factory=list)
    decisions: list[AgentDecision] = field(default_factory=list)
    memory_records: list[MemoryRecord] = field(default_factory=list)
    memory_context: list[MemoryRecord] = field(default_factory=list)
    backend_counter: Counter[str] = field(default_factory=Counter)
    surrogate_filter_report: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    status: str = "RUNNING"

    @property
    def failed(self) -> bool:
        return self.status == "FAILED"

    @property
    def has_no_candidates(self) -> bool:
        return self.status == "NO_CANDIDATES"

    @property
    def should_stop(self) -> bool:
        return self.failed or self.has_no_candidates

    def fail(self, reason: str) -> None:
        self.status = "FAILED"
        self.error = reason

    def no_candidates(self, reason: str) -> None:
        self.status = "NO_CANDIDATES"
        self.error = reason


class CryoSwarmPipeline:
    """Run the deterministic CryoSwarm-Q orchestration stages end to end."""

    def __init__(
        self,
        repository: CryoSwarmRepository | None = None,
        parallel: bool = False,
        sequence_strategy_mode: str = "adaptive",
        rl_checkpoint_path: str | None = None,
        param_space: PhysicsParameterSpace | None = None,
        problem_agent: ProblemFramingProtocol | None = None,
        geometry_agent: GeometryProtocol | None = None,
        sequence_agent: SequenceProtocol | None = None,
        noise_agent: NoiseEvaluationProtocol | None = None,
        routing_agent: RoutingProtocol | None = None,
        campaign_agent: CampaignProtocol | None = None,
        results_agent: ResultsProtocol | None = None,
        memory_agent: MemoryCaptureProtocol | None = None,
        event_bus: EventBus | None = None,
        surrogate_filter: SurrogateFilter | None = None,
        surrogate_filter_enabled: bool = False,
        surrogate_model_path: str | None = None,
        surrogate_use_ensemble: bool = False,
        surrogate_filter_top_k: int = 20,
        surrogate_filter_min_score: float = 0.1,
        surrogate_filter_max_uncertainty: float = 0.15,
    ) -> None:
        self.repository = repository
        self.parallel = parallel
        self.param_space = param_space or PhysicsParameterSpace.default()
        self.event_bus = event_bus or EventBus()
        self.logger = get_logger(__name__)
        self.problem_agent: ProblemFramingProtocol = problem_agent or ProblemFramingAgent()
        self.geometry_agent: GeometryProtocol = geometry_agent or GeometryAgent(param_space=self.param_space)
        self.sequence_agent: SequenceProtocol = sequence_agent or SequenceAgent(param_space=self.param_space)
        self.noise_agent: NoiseEvaluationProtocol = noise_agent or NoiseRobustnessAgent(param_space=self.param_space)
        self.routing_agent: RoutingProtocol = routing_agent or BackendRoutingAgent()
        self.campaign_agent: CampaignProtocol = campaign_agent or CampaignAgent()
        self.results_agent: ResultsProtocol = results_agent or ResultsAgent()
        self.memory_agent: MemoryCaptureProtocol = memory_agent or MemoryAgent()
        self.surrogate_filter = surrogate_filter or SurrogateFilter(
            model_path=surrogate_model_path,
            top_k=surrogate_filter_top_k,
            min_score=surrogate_filter_min_score,
            max_uncertainty=surrogate_filter_max_uncertainty,
            enabled=surrogate_filter_enabled or bool(surrogate_model_path),
            use_ensemble=surrogate_use_ensemble,
        )
        self.sequence_strategy = SequenceStrategy(
            mode=SequenceStrategyMode(sequence_strategy_mode),
            param_space=self.param_space,
            rl_checkpoint_path=rl_checkpoint_path,
            heuristic_agent=self.sequence_agent,
        )
        self.phases: tuple[PipelinePhase, ...] = tuple(self._build_phases())

    @property
    def phase_names(self) -> list[str]:
        return [phase.name for phase in self.phases]

    def _build_phases(self) -> list[PipelinePhase]:
        return [
            ProblemFramingPhase(self),
            GeometryGenerationPhase(self),
            SequenceGenerationPhase(self),
            SurrogateFilterPhase(self),
            EvaluationPhase(self),
            RankingPhase(self),
            ResultsSummaryPhase(self),
            MemoryCapturePhase(self),
        ]

    def _publish_event(
        self,
        event_type: str,
        ctx: PipelineContext | None = None,
        **payload: Any,
    ) -> None:
        base_payload: dict[str, Any] = {}
        if ctx is not None:
            base_payload.update(
                {
                    "goal_id": ctx.goal.id,
                    "campaign_id": ctx.campaign.id,
                    "status": ctx.status,
                }
            )
            if ctx.error:
                base_payload["error"] = ctx.error
        base_payload.update(payload)
        self.event_bus.publish(event_type, **base_payload)

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
            self.logger.warning(
                "Repository %s failed with %s: %s",
                description,
                type(exc).__name__,
                exc,
            )
            return default

    def _record_agent_failure(
        self,
        decisions: list[AgentDecision],
        agent: AgentProtocol,
        campaign_id: str,
        subject_id: str,
        decision_type: DecisionType,
        exc: Exception,
    ) -> None:
        self.logger.error(
            "Agent %s failed with %s: %s",
            agent.agent_name.value,
            type(exc).__name__,
            exc,
        )
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
        registers: list[RegisterCandidate],
        sequences: list[SequenceCandidate],
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
            registers=registers,
            sequences=sequences,
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

    def _init_context(self, goal: ExperimentGoal) -> PipelineContext:
        """Create the mutable pipeline context and persist the running campaign shell."""

        stored_goal = goal.model_copy(update={"status": GoalStatus.RUNNING})
        self._safe_repository_call(lambda: self.repository.create_goal(stored_goal), "create_goal")

        memory_context = self._safe_repository_call(
            lambda: self.repository.list_recent_memory(limit=12),
            "list_recent_memory",
            default=[],
        ) or []
        campaign = CampaignState(goal_id=stored_goal.id, status=CampaignStatus.RUNNING)
        self._safe_repository_call(lambda: self.repository.create_campaign(campaign), "create_campaign")
        ctx = PipelineContext(goal=stored_goal, campaign=campaign, memory_context=memory_context)
        self._publish_event(
            "pipeline.initialized",
            ctx,
            memory_context_count=len(memory_context),
        )
        return ctx

    def _run_phase(self, phase: PipelinePhase, ctx: PipelineContext) -> PipelineContext:
        self._publish_event("phase.started", ctx, phase=phase.name)
        updated = phase.execute(ctx)
        self._publish_event("phase.completed", updated, phase=phase.name)
        return updated

    def _finalize(self, ctx: PipelineContext) -> PipelineSummary:
        """Persist terminal state and materialize the observable pipeline summary."""

        final_status = ctx.status
        if not ctx.should_stop:
            final_status = "FAILED" if ctx.campaign.status == CampaignStatus.FAILED else "COMPLETED"

        final_goal_status = GoalStatus.FAILED if final_status == "FAILED" else GoalStatus.COMPLETED
        ctx.goal = ctx.goal.model_copy(update={"status": final_goal_status})

        if ctx.has_no_candidates and ctx.campaign.status == CampaignStatus.RUNNING:
            ctx.campaign = ctx.campaign.model_copy(
                update={
                    "status": CampaignStatus.NO_CANDIDATES,
                    "candidate_count": 0,
                    "summary": ctx.error,
                    "summary_report": {"reason": ctx.error},
                }
            )
        elif final_status == "FAILED" and ctx.campaign.status == CampaignStatus.RUNNING:
            ctx.campaign = ctx.campaign.model_copy(
                update={
                    "status": CampaignStatus.FAILED,
                    "summary": ctx.error,
                }
            )

        self._persist_state(
            ctx.goal,
            ctx.campaign,
            ctx.registers,
            ctx.sequences,
            ctx.reports,
            ctx.ranked_candidates,
            ctx.decisions,
            ctx.memory_records,
        )

        summary_error = ctx.error if ctx.should_stop else (
            ctx.campaign.summary if final_status == "FAILED" else None
        )
        summary = self._build_summary(
            status=final_status,
            campaign=ctx.campaign,
            goal=ctx.goal,
            spec=ctx.spec,
            ranked_candidates=ctx.ranked_candidates,
            decisions=ctx.decisions,
            reports=ctx.reports,
            memory_records=ctx.memory_records,
            backend_mix=dict(ctx.backend_counter),
            registers=ctx.registers,
            sequences=ctx.sequences,
            error=summary_error,
        )
        final_event = "pipeline.failed" if summary.status == "FAILED" else "pipeline.completed"
        self._publish_event(
            final_event,
            ctx,
            summary_status=summary.status,
            ranked_count=summary.ranked_count,
        )
        return summary

    def run(self, goal: ExperimentGoal) -> PipelineSummary:
        """Execute the full orchestration pipeline through composable phase objects."""

        ctx = self._init_context(goal)
        self._publish_event("pipeline.started", ctx, phase_count=len(self.phases))
        try:
            for phase in self.phases:
                ctx = self._run_phase(phase, ctx)
                if ctx.should_stop:
                    break
            return self._finalize(ctx)
        except Exception as exc:
            self.logger.error("Unhandled pipeline failure: %s", exc, exc_info=True)
            self._publish_event(
                "pipeline.unhandled_error",
                ctx,
                error=str(exc),
            )
            ctx.campaign = ctx.campaign.model_copy(
                update={
                    "status": CampaignStatus.FAILED,
                    "summary": "Unhandled pipeline failure.",
                    "summary_report": {"error": str(exc)},
                }
            )
            ctx.fail(str(exc))
            return self._finalize(ctx)
