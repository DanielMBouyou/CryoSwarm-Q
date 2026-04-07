from __future__ import annotations

"""Composable pipeline phases for CryoSwarm orchestration."""

from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any, cast

from packages.agents.sequence_strategy import SequenceStrategyMode
from packages.core.enums import CampaignStatus, DecisionType, GoalStatus
from packages.core.metadata_schemas import SequenceMetadata

if TYPE_CHECKING:
    from packages.orchestration.pipeline import CryoSwarmPipeline, PipelineContext


class PipelinePhase(ABC):
    """A single pipeline stage operating on a shared mutable context."""

    name: str

    def __init__(self, pipeline: CryoSwarmPipeline) -> None:
        self.pipeline = pipeline

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Apply the phase to the current pipeline context."""

    def _publish(self, event_type: str, ctx: PipelineContext, **payload: Any) -> None:
        self.pipeline._publish_event(
            event_type,
            ctx,
            phase=self.name,
            **payload,
        )


class ProblemFramingPhase(PipelinePhase):
    name = "problem_framing"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            ctx.spec = self.pipeline.problem_agent.run(ctx.goal, ctx.memory_context)
            ctx.campaign = ctx.campaign.model_copy(update={"spec_id": ctx.spec.id})
            ctx.decisions.append(
                self.pipeline.problem_agent.build_decision(
                    campaign_id=ctx.campaign.id,
                    subject_id=ctx.spec.id,
                    decision_type=DecisionType.SPECIFICATION,
                    status="completed",
                    reasoning_summary=ctx.spec.reasoning_summary,
                    structured_output=ctx.spec.model_dump(mode="json"),
                )
            )
            self._publish("problem_framing.completed", ctx, spec_id=ctx.spec.id)
            return ctx
        except Exception as exc:
            self.pipeline._record_agent_failure(
                ctx.decisions,
                self.pipeline.problem_agent,
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
            self._publish("problem_framing.failed", ctx, error=str(exc))
            return ctx


class GeometryGenerationPhase(PipelinePhase):
    name = "geometry_generation"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.spec is None:
            raise RuntimeError("Pipeline context is missing an experiment specification.")

        try:
            ctx.registers = self.pipeline.geometry_agent.run(
                ctx.spec,
                ctx.campaign.id,
                ctx.memory_context,
            )
            ctx.decisions.append(
                self.pipeline.geometry_agent.build_decision(
                    campaign_id=ctx.campaign.id,
                    subject_id=ctx.campaign.id,
                    decision_type=DecisionType.CANDIDATE_GENERATION,
                    status="completed",
                    reasoning_summary=f"Generated {len(ctx.registers)} register candidates.",
                    structured_output={"register_candidate_ids": [item.id for item in ctx.registers]},
                )
            )
        except Exception as exc:
            self.pipeline._record_agent_failure(
                ctx.decisions,
                self.pipeline.geometry_agent,
                ctx.campaign.id,
                ctx.campaign.id,
                DecisionType.CANDIDATE_GENERATION,
                exc,
            )
            ctx.registers = []

        if ctx.registers:
            self._publish(
                "geometry.completed",
                ctx,
                register_count=len(ctx.registers),
            )
            return ctx

        self.pipeline.logger.warning("GeometryAgent produced no register candidates.")
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
        self._publish("geometry.empty", ctx, register_count=0)
        return ctx


class SequenceGenerationPhase(PipelinePhase):
    name = "sequence_generation"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.spec is None:
            raise RuntimeError("Pipeline context is missing an experiment specification.")

        for register_candidate in ctx.registers:
            try:
                generated, strategy_meta = self.pipeline.sequence_strategy.generate_candidates(
                    ctx.spec,
                    register_candidate,
                    ctx.campaign.id,
                    ctx.memory_context,
                )
                enriched_sequences = self._enrich_sequences(generated, strategy_meta)
                ctx.sequences.extend(enriched_sequences)
                ctx.decisions.append(
                    self.pipeline.sequence_agent.build_decision(
                        campaign_id=ctx.campaign.id,
                        subject_id=register_candidate.id,
                        decision_type=DecisionType.CANDIDATE_GENERATION,
                        status="completed" if enriched_sequences else "empty",
                        reasoning_summary=(
                            f"Generated {len(enriched_sequences)} sequence candidates for "
                            f"{register_candidate.label} using {strategy_meta['strategy_used']}."
                        ),
                        structured_output={
                            "register_candidate_id": register_candidate.id,
                            "sequence_candidate_ids": [item.id for item in enriched_sequences],
                            "strategy_used": strategy_meta["strategy_used"],
                            "strategy_reason": strategy_meta["strategy_reason"],
                            "rl_candidates": strategy_meta.get("rl_candidates_count", 0),
                            "heuristic_candidates": strategy_meta.get("heuristic_candidates_count", 0),
                        },
                    )
                )
                self._publish(
                    "sequence.register_completed",
                    ctx,
                    register_candidate_id=register_candidate.id,
                    generated_count=len(enriched_sequences),
                    strategy_used=strategy_meta["strategy_used"],
                )
            except Exception as exc:
                self.pipeline._record_agent_failure(
                    ctx.decisions,
                    self.pipeline.sequence_agent,
                    ctx.campaign.id,
                    register_candidate.id,
                    DecisionType.CANDIDATE_GENERATION,
                    exc,
                )

        ctx.decisions.append(
            self.pipeline.sequence_agent.build_decision(
                campaign_id=ctx.campaign.id,
                subject_id=ctx.campaign.id,
                decision_type=DecisionType.CANDIDATE_GENERATION,
                status="completed" if ctx.sequences else "empty",
                reasoning_summary=f"Generated {len(ctx.sequences)} sequence candidates.",
                structured_output={
                    "sequence_candidate_ids": [item.id for item in ctx.sequences],
                    "strategy_report": self.pipeline.sequence_strategy.get_strategy_report(),
                },
            )
        )

        if ctx.sequences:
            self._publish(
                "sequence.completed",
                ctx,
                sequence_count=len(ctx.sequences),
            )
            return ctx

        self.pipeline.logger.warning("SequenceAgent produced no sequence candidates.")
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
        self._publish("sequence.empty", ctx, sequence_count=0)
        return ctx

    @staticmethod
    def _enrich_sequences(
        generated: list,
        strategy_meta: dict[str, Any],
    ) -> list:
        enriched_sequences = []
        for sequence in generated:
            metadata = cast(SequenceMetadata, dict(sequence.metadata))
            metadata.setdefault("problem_class", strategy_meta["problem_class"])
            metadata.setdefault("strategy_used", strategy_meta["strategy_used"])
            enriched_sequences.append(sequence.model_copy(update={"metadata": metadata}))
        return enriched_sequences


class SurrogateFilterPhase(PipelinePhase):
    name = "surrogate_filter"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        register_lookup = {item.id: item for item in ctx.registers}
        filtered_sequences, filter_report = self.pipeline.surrogate_filter.filter_with_report(
            ctx.sequences,
            register_lookup,
        )
        ctx.surrogate_filter_report = filter_report
        ctx.campaign = ctx.campaign.model_copy(
            update={
                "summary_report": {
                    **ctx.campaign.summary_report,
                    "surrogate_filter": filter_report,
                }
            }
        )

        if not filter_report.get("applied", False):
            self._publish(
                "surrogate_filter.skipped",
                ctx,
                input_count=filter_report.get("input_count", len(ctx.sequences)),
                reason=str(filter_report.get("reason", "not_applied")),
            )
            return ctx

        ctx.sequences = filtered_sequences
        if ctx.sequences:
            self._publish(
                "surrogate_filter.completed",
                ctx,
                input_count=filter_report.get("input_count", len(filtered_sequences)),
                kept_count=filter_report.get("kept_count", len(filtered_sequences)),
                rejected_count=filter_report.get("rejected_count", 0),
                use_ensemble=filter_report.get("use_ensemble", False),
            )
            return ctx

        self.pipeline.logger.warning(
            "Surrogate filter rejected all sequence candidates for campaign %s.",
            ctx.campaign.id,
        )
        ctx.goal = ctx.goal.model_copy(update={"status": GoalStatus.COMPLETED})
        ctx.campaign = ctx.campaign.model_copy(
            update={
                "status": CampaignStatus.NO_CANDIDATES,
                "candidate_count": 0,
                "summary": "Surrogate pre-filter rejected all sequence candidates.",
                "summary_report": {
                    **ctx.campaign.summary_report,
                    "reason": "surrogate_filter_rejected_all_candidates",
                    "surrogate_filter": filter_report,
                },
            }
        )
        ctx.no_candidates("Surrogate pre-filter rejected all sequence candidates.")
        self._publish("surrogate_filter.empty", ctx, candidate_count=0)
        return ctx


class EvaluationPhase(PipelinePhase):
    name = "evaluation"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.spec is None:
            raise RuntimeError("Pipeline context is missing an experiment specification.")

        register_lookup = {item.id: item for item in ctx.registers}
        reports, ranked_candidates, backend_mix = self.pipeline._evaluate_sequences(
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
            self.pipeline.sequence_strategy.update_performance(problem_class, strategy_used, scores)

        if ctx.ranked_candidates:
            self._publish(
                "evaluation.completed",
                ctx,
                report_count=len(ctx.reports),
                candidate_count=len(ctx.ranked_candidates),
                backend_mix=dict(ctx.backend_counter),
            )
            return ctx

        self.pipeline.logger.warning(
            "No evaluation results were produced for campaign %s.",
            ctx.campaign.id,
        )
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
        self._publish("evaluation.empty", ctx, candidate_count=0)
        return ctx


class RankingPhase(PipelinePhase):
    name = "ranking"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        try:
            ctx.campaign, ctx.ranked_candidates = self.pipeline.campaign_agent.run(
                ctx.campaign,
                ctx.ranked_candidates,
            )
            ctx.decisions.append(
                self.pipeline.campaign_agent.build_decision(
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
            self._publish(
                "ranking.completed",
                ctx,
                ranked_count=len(ctx.ranked_candidates),
            )
        except Exception as exc:
            self.pipeline._record_agent_failure(
                ctx.decisions,
                self.pipeline.campaign_agent,
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
            self._publish("ranking.failed", ctx, error=str(exc))
        return ctx


class ResultsSummaryPhase(PipelinePhase):
    name = "results_summary"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.spec is None:
            raise RuntimeError("Pipeline context is missing an experiment specification.")

        try:
            results_summary = self.pipeline.results_agent.run(
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
                self.pipeline.results_agent.build_decision(
                    campaign_id=ctx.campaign.id,
                    subject_id=ctx.campaign.id,
                    decision_type=DecisionType.RESULTS_SUMMARY,
                    status="completed",
                    reasoning_summary=str(results_summary["summary"]),
                    structured_output=results_summary,
                )
            )
            self._publish("results_summary.completed", ctx)
        except Exception as exc:
            self.pipeline._record_agent_failure(
                ctx.decisions,
                self.pipeline.results_agent,
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
            self._publish("results_summary.failed", ctx, error=str(exc))
        return ctx


class MemoryCapturePhase(PipelinePhase):
    name = "memory_capture"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        register_lookup = {item.id: item for item in ctx.registers}
        sequence_lookup = {item.id: item for item in ctx.sequences}
        try:
            ctx.memory_records = self.pipeline.memory_agent.run(
                ctx.campaign.id,
                ctx.ranked_candidates,
                sequence_lookup,
                register_lookup,
            )
            ctx.decisions.append(
                self.pipeline.memory_agent.build_decision(
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
            self._publish(
                "memory_capture.completed",
                ctx,
                memory_record_count=len(ctx.memory_records),
            )
        except Exception as exc:
            self.pipeline._record_agent_failure(
                ctx.decisions,
                self.pipeline.memory_agent,
                ctx.campaign.id,
                ctx.campaign.id,
                DecisionType.MEMORY_CAPTURE,
                exc,
            )
            ctx.memory_records = []
            self._publish("memory_capture.failed", ctx, error=str(exc))
        return ctx
