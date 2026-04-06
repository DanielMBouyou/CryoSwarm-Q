from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

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
from packages.core.metadata_schemas import SequenceMetadata
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
    error: str | None = None
    status: str = "COMPLETED"

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
    ) -> None:
        self.repository = repository
        self.parallel = parallel
        self.param_space = param_space or PhysicsParameterSpace.default()
        self.logger = get_logger(__name__)
        self.problem_agent: ProblemFramingProtocol = problem_agent or ProblemFramingAgent()
        self.geometry_agent: GeometryProtocol = geometry_agent or GeometryAgent(param_space=self.param_space)
        self.sequence_agent: SequenceProtocol = sequence_agent or SequenceAgent(param_space=self.param_space)
        self.noise_agent: NoiseEvaluationProtocol = noise_agent or NoiseRobustnessAgent(param_space=self.param_space)
        self.routing_agent: RoutingProtocol = routing_agent or BackendRoutingAgent()
        self.campaign_agent: CampaignProtocol = campaign_agent or CampaignAgent()
        self.results_agent: ResultsProtocol = results_agent or ResultsAgent()
        self.memory_agent: MemoryCaptureProtocol = memory_agent or MemoryAgent()
        self.sequence_strategy = SequenceStrategy(
            mode=SequenceStrategyMode(sequence_strategy_mode),
            param_space=self.param_space,
            rl_checkpoint_path=rl_checkpoint_path,
            heuristic_agent=self.sequence_agent,
        )

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
        return PipelineContext(goal=stored_goal, campaign=campaign, memory_context=memory_context)

    def _phase_problem_framing(self, ctx: PipelineContext) -> None:
        """Frame the goal into an immutable experiment specification."""
        try:
            ctx.spec = self.problem_agent.run(ctx.goal, ctx.memory_context)
            ctx.campaign = ctx.campaign.model_copy(update={"spec_id": ctx.spec.id})
            ctx.decisions.append(
                self.problem_agent.build_decision(
                    campaign_id=ctx.campaign.id,
                    subject_id=ctx.spec.id,
                    decision_type=DecisionType.SPECIFICATION,
                    status="completed",
                    reasoning_summary=ctx.spec.reasoning_summary,
                    structured_output=ctx.spec.model_dump(mode="json"),
                )
            )
        except Exception as exc:
            self._record_agent_failure(
                ctx.decisions,
                self.problem_agent,
                ctx.campaign.id,
                ctx.goal.id,
                DecisionType.SPECIFICATION,
                exc,
            )
            ctx.goal = ctx.goal.model_copy(update={"status": GoalStatus.FAILED})
            ctx.campaign = ctx.campaign.model_copy(
                update={
                    "status": CampaignStatus.FAILED,
                    "summary": "Problem framing failed.",
                }
            )
            ctx.fail("Problem framing failed.")

    def _phase_geometry_generation(self, ctx: PipelineContext) -> None:
        """Generate hardware-valid register candidates from the framed specification."""
        if ctx.spec is None:
            raise RuntimeError("Pipeline context is missing an experiment specification.")

        try:
            ctx.registers = self.geometry_agent.run(ctx.spec, ctx.campaign.id, ctx.memory_context)
            ctx.decisions.append(
                self.geometry_agent.build_decision(
                    campaign_id=ctx.campaign.id,
                    subject_id=ctx.campaign.id,
                    decision_type=DecisionType.CANDIDATE_GENERATION,
                    status="completed",
                    reasoning_summary=f"Generated {len(ctx.registers)} register candidates.",
                    structured_output={"register_candidate_ids": [item.id for item in ctx.registers]},
                )
            )
        except Exception as exc:
            self._record_agent_failure(
                ctx.decisions,
                self.geometry_agent,
                ctx.campaign.id,
                ctx.campaign.id,
                DecisionType.CANDIDATE_GENERATION,
                exc,
            )
            ctx.registers = []

        if ctx.registers:
            return

        self.logger.warning("GeometryAgent produced no register candidates.")
        ctx.goal = ctx.goal.model_copy(update={"status": GoalStatus.COMPLETED})
        ctx.campaign = ctx.campaign.model_copy(
            update={
                "status": CampaignStatus.NO_CANDIDATES,
                "candidate_count": 0,
                "summary": "No feasible register candidates generated.",
                "summary_report": {"reason": "geometry_agent_returned_no_candidates"},
            }
        )
        ctx.no_candidates("No feasible register candidates generated.")

    def _phase_sequence_generation(self, ctx: PipelineContext) -> None:
        """Generate sequence candidates for each register and annotate strategy metadata."""
        if ctx.spec is None:
            raise RuntimeError("Pipeline context is missing an experiment specification.")

        for register_candidate in ctx.registers:
            try:
                generated, strategy_meta = self.sequence_strategy.generate_candidates(
                    ctx.spec,
                    register_candidate,
                    ctx.campaign.id,
                    ctx.memory_context,
                )
                enriched_sequences: list[SequenceCandidate] = []
                for sequence in generated:
                    metadata = cast(SequenceMetadata, dict(sequence.metadata))
                    metadata.setdefault("problem_class", strategy_meta["problem_class"])
                    metadata.setdefault("strategy_used", strategy_meta["strategy_used"])
                    enriched_sequences.append(sequence.model_copy(update={"metadata": metadata}))
                generated = enriched_sequences
                ctx.sequences.extend(generated)
                ctx.decisions.append(
                    self.sequence_agent.build_decision(
                        campaign_id=ctx.campaign.id,
                        subject_id=register_candidate.id,
                        decision_type=DecisionType.CANDIDATE_GENERATION,
                        status="completed" if generated else "empty",
                        reasoning_summary=(
                            f"Generated {len(generated)} sequence candidates for {register_candidate.label} "
                            f"using {strategy_meta['strategy_used']}."
                        ),
                        structured_output={
                            "register_candidate_id": register_candidate.id,
                            "sequence_candidate_ids": [item.id for item in generated],
                            "strategy_used": strategy_meta["strategy_used"],
                            "strategy_reason": strategy_meta["strategy_reason"],
                            "rl_candidates": strategy_meta.get("rl_candidates_count", 0),
                            "heuristic_candidates": strategy_meta.get("heuristic_candidates_count", 0),
                        },
                    )
                )
            except Exception as exc:
                self._record_agent_failure(
                    ctx.decisions,
                    self.sequence_agent,
                    ctx.campaign.id,
                    register_candidate.id,
                    DecisionType.CANDIDATE_GENERATION,
                    exc,
                )

        ctx.decisions.append(
            self.sequence_agent.build_decision(
                campaign_id=ctx.campaign.id,
                subject_id=ctx.campaign.id,
                decision_type=DecisionType.CANDIDATE_GENERATION,
                status="completed" if ctx.sequences else "empty",
                reasoning_summary=f"Generated {len(ctx.sequences)} sequence candidates.",
                structured_output={
                    "sequence_candidate_ids": [item.id for item in ctx.sequences],
                    "strategy_report": self.sequence_strategy.get_strategy_report(),
                },
            )
        )

        if ctx.sequences:
            return

        self.logger.warning("SequenceAgent produced no sequence candidates.")
        ctx.goal = ctx.goal.model_copy(update={"status": GoalStatus.COMPLETED})
        ctx.campaign = ctx.campaign.model_copy(
            update={
                "status": CampaignStatus.NO_CANDIDATES,
                "candidate_count": 0,
                "summary": "No sequence candidates generated.",
                "summary_report": {"reason": "sequence_agent_returned_no_candidates"},
            }
        )
        ctx.no_candidates("No sequence candidates generated.")

    def _phase_evaluation(self, ctx: PipelineContext) -> None:
        """Evaluate robustness, route execution backends, and score each sequence."""
        if ctx.spec is None:
            raise RuntimeError("Pipeline context is missing an experiment specification.")

        register_lookup = {item.id: item for item in ctx.registers}
        reports, ranked_candidates, backend_mix = self._evaluate_sequences(
            ctx.spec,
            ctx.campaign.id,
            ctx.sequences,
            register_lookup,
            ctx.decisions,
        )
        ctx.reports = reports
        ctx.ranked_candidates = ranked_candidates
        ctx.backend_counter = Counter(backend_mix)

        sequence_lookup = {item.id: item for item in ctx.sequences}
        report_groups: dict[tuple[str, str], list[float]] = {}
        for report in ctx.reports:
            sequence = sequence_lookup.get(report.sequence_candidate_id)
            if sequence is None:
                continue
            sequence_metadata = cast(SequenceMetadata, sequence.metadata)
            key = (
                str(sequence_metadata.get("problem_class", "unknown_problem_class")),
                str(
                    sequence_metadata.get(
                        "strategy_used",
                        SequenceStrategyMode.HEURISTIC_ONLY.value,
                    )
                ),
            )
            report_groups.setdefault(key, []).append(report.robustness_score)
        for (problem_class, strategy_used), scores in report_groups.items():
            self.sequence_strategy.update_performance(problem_class, strategy_used, scores)

        if ctx.ranked_candidates:
            return

        self.logger.warning("No evaluation results were produced for campaign %s.", ctx.campaign.id)
        ctx.goal = ctx.goal.model_copy(update={"status": GoalStatus.COMPLETED})
        ctx.campaign = ctx.campaign.model_copy(
            update={
                "status": CampaignStatus.NO_CANDIDATES,
                "candidate_count": 0,
                "summary": "No evaluation results were produced.",
                "summary_report": {"reason": "no_evaluation_results"},
            }
        )
        ctx.no_candidates("No evaluation results were produced.")

    def _phase_ranking(self, ctx: PipelineContext) -> None:
        """Rank campaign candidates and keep processing even if ranking fails."""
        try:
            ctx.campaign, ctx.ranked_candidates = self.campaign_agent.run(
                ctx.campaign,
                ctx.ranked_candidates,
            )
            ctx.decisions.append(
                self.campaign_agent.build_decision(
                    campaign_id=ctx.campaign.id,
                    subject_id=ctx.campaign.id,
                    decision_type=DecisionType.CAMPAIGN_RANKING,
                    status="completed",
                    reasoning_summary=ctx.campaign.summary or "Campaign ranking completed.",
                    structured_output={
                        "ranked_sequence_ids": [
                            item.sequence_candidate_id for item in ctx.ranked_candidates
                        ],
                        "backend_mix": dict(ctx.backend_counter),
                    },
                )
            )
        except Exception as exc:
            self._record_agent_failure(
                ctx.decisions,
                self.campaign_agent,
                ctx.campaign.id,
                ctx.campaign.id,
                DecisionType.CAMPAIGN_RANKING,
                exc,
            )
            ctx.campaign = ctx.campaign.model_copy(
                update={
                    "status": CampaignStatus.FAILED,
                    "candidate_count": len(ctx.ranked_candidates),
                    "summary": "Campaign ranking failed.",
                }
            )

    def _phase_results_summary(self, ctx: PipelineContext) -> None:
        """Build the final campaign summary and preserve backend mix metadata."""
        if ctx.spec is None:
            raise RuntimeError("Pipeline context is missing an experiment specification.")

        try:
            results_summary = self.results_agent.run(
                ctx.goal,
                ctx.spec,
                ctx.campaign,
                ctx.ranked_candidates,
            )
            ctx.campaign = ctx.campaign.model_copy(
                update={
                    "summary": str(results_summary["summary"]),
                    "summary_report": {
                        **ctx.campaign.summary_report,
                        "backend_mix": dict(ctx.backend_counter),
                        "results_summary": results_summary,
                    },
                }
            )
            ctx.decisions.append(
                self.results_agent.build_decision(
                    campaign_id=ctx.campaign.id,
                    subject_id=ctx.campaign.id,
                    decision_type=DecisionType.RESULTS_SUMMARY,
                    status="completed",
                    reasoning_summary=str(results_summary["summary"]),
                    structured_output=results_summary,
                )
            )
        except Exception as exc:
            self._record_agent_failure(
                ctx.decisions,
                self.results_agent,
                ctx.campaign.id,
                ctx.campaign.id,
                DecisionType.RESULTS_SUMMARY,
                exc,
            )
            ctx.campaign = ctx.campaign.model_copy(
                update={
                    "summary_report": {
                        **ctx.campaign.summary_report,
                        "backend_mix": dict(ctx.backend_counter),
                    }
                }
            )

    def _phase_memory_capture(self, ctx: PipelineContext) -> None:
        """Extract reusable memory records from the ranked campaign outcome."""
        register_lookup = {item.id: item for item in ctx.registers}
        sequence_lookup = {item.id: item for item in ctx.sequences}
        try:
            ctx.memory_records = self.memory_agent.run(
                ctx.campaign.id,
                ctx.ranked_candidates,
                sequence_lookup,
                register_lookup,
            )
            ctx.decisions.append(
                self.memory_agent.build_decision(
                    campaign_id=ctx.campaign.id,
                    subject_id=ctx.campaign.id,
                    decision_type=DecisionType.MEMORY_CAPTURE,
                    status="completed",
                    reasoning_summary=(
                        f"Stored {len(ctx.memory_records)} memory records for campaign reuse."
                    ),
                    structured_output={
                        "memory_record_ids": [item.id for item in ctx.memory_records]
                    },
                )
            )
        except Exception as exc:
            self._record_agent_failure(
                ctx.decisions,
                self.memory_agent,
                ctx.campaign.id,
                ctx.campaign.id,
                DecisionType.MEMORY_CAPTURE,
                exc,
            )
            ctx.memory_records = []

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
        return self._build_summary(
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

    def run(self, goal: ExperimentGoal) -> PipelineSummary:
        """Execute the full orchestration pipeline with composable internal phases."""
        ctx = self._init_context(goal)
        try:
            self._phase_problem_framing(ctx)
            if ctx.should_stop:
                return self._finalize(ctx)

            self._phase_geometry_generation(ctx)
            if ctx.should_stop:
                return self._finalize(ctx)

            self._phase_sequence_generation(ctx)
            if ctx.should_stop:
                return self._finalize(ctx)

            self._phase_evaluation(ctx)
            if ctx.should_stop:
                return self._finalize(ctx)

            self._phase_ranking(ctx)
            self._phase_results_summary(ctx)
            self._phase_memory_capture(ctx)
            return self._finalize(ctx)
        except Exception as exc:
            self.logger.error("Unhandled pipeline failure: %s", exc, exc_info=True)
            ctx.campaign = ctx.campaign.model_copy(
                update={
                    "status": CampaignStatus.FAILED,
                    "summary": "Unhandled pipeline failure.",
                    "summary_report": {"error": str(exc)},
                }
            )
            ctx.fail(str(exc))
            return self._finalize(ctx)
