from __future__ import annotations

from collections import Counter

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
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    PipelineSummary,
)
from packages.db.repositories import CryoSwarmRepository
from packages.scoring.objective import compute_objective_score


class CryoSwarmPipeline:
    def __init__(self, repository: CryoSwarmRepository | None = None) -> None:
        self.repository = repository
        self.logger = get_logger(__name__)
        self.problem_agent = ProblemFramingAgent()
        self.geometry_agent = GeometryAgent()
        self.sequence_agent = SequenceAgent()
        self.noise_agent = NoiseRobustnessAgent()
        self.routing_agent = BackendRoutingAgent()
        self.campaign_agent = CampaignAgent()
        self.results_agent = ResultsAgent()
        self.memory_agent = MemoryAgent()

    def run(self, goal: ExperimentGoal) -> PipelineSummary:
        stored_goal = goal.model_copy(update={"status": GoalStatus.RUNNING})
        if self.repository:
            self.repository.create_goal(stored_goal)

        memory_context = self.repository.list_recent_memory(limit=12) if self.repository else []
        campaign = CampaignState(goal_id=stored_goal.id, status=CampaignStatus.RUNNING)
        if self.repository:
            self.repository.create_campaign(campaign)

        decisions = []

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

        all_sequences = []
        for register_candidate in registers:
            all_sequences.extend(
                self.sequence_agent.run(spec, register_candidate, campaign.id, memory_context)
            )
        decisions.append(
            self.sequence_agent.build_decision(
                campaign_id=campaign.id,
                subject_id=campaign.id,
                decision_type=DecisionType.CANDIDATE_GENERATION,
                status="completed",
                reasoning_summary=f"Generated {len(all_sequences)} sequence candidates.",
                structured_output={"sequence_candidate_ids": [item.id for item in all_sequences]},
            )
        )

        reports = []
        evaluations = []
        backend_counter: Counter[str] = Counter()
        register_lookup = {item.id: item for item in registers}

        for sequence in all_sequences:
            register_candidate = register_lookup[sequence.register_candidate_id]
            report = self.noise_agent.run(spec, register_candidate, sequence)
            reports.append(report)
            decisions.append(
                self.noise_agent.build_decision(
                    campaign_id=campaign.id,
                    subject_id=sequence.id,
                    decision_type=DecisionType.ROBUSTNESS_EVALUATION,
                    status="completed",
                    reasoning_summary=report.reasoning_summary,
                    structured_output=report.model_dump(mode="json"),
                )
            )

            backend_choice = self.routing_agent.run(spec, sequence, report)
            backend_counter[backend_choice.recommended_backend.value] += 1
            decisions.append(
                self.routing_agent.build_decision(
                    campaign_id=campaign.id,
                    subject_id=sequence.id,
                    decision_type=DecisionType.BACKEND_ROUTING,
                    status="completed",
                    reasoning_summary=backend_choice.rationale,
                    structured_output=backend_choice.model_dump(mode="json"),
                )
            )

            objective_score = compute_objective_score(
                observable_score=report.nominal_score,
                robustness=report.robustness_score,
                cost=backend_choice.estimated_cost,
                latency=backend_choice.estimated_latency,
                weights=spec.scoring_weights,
            )

            evaluations.append(
                EvaluationResult(
                    campaign_id=campaign.id,
                    sequence_candidate_id=sequence.id,
                    register_candidate_id=sequence.register_candidate_id,
                    nominal_score=report.nominal_score,
                    robustness_score=report.robustness_score,
                    worst_case_score=report.worst_case_score,
                    observable_score=float(report.nominal_observables.get("observable_score", report.nominal_score)),
                    objective_score=objective_score,
                    backend_choice=backend_choice.recommended_backend,
                    estimated_cost=backend_choice.estimated_cost,
                    estimated_latency=backend_choice.estimated_latency,
                    reasoning_summary=(
                        f"Combined robustness score with routing recommendation for "
                        f"{sequence.label}."
                    ),
                    metadata={
                        "register_label": register_lookup[sequence.register_candidate_id].label,
                        "backend_rationale": backend_choice.rationale,
                        "nominal_observables": report.nominal_observables,
                    },
                )
            )

        campaign, ranked_candidates = self.campaign_agent.run(campaign, evaluations)
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

        sequence_lookup = {item.id: item for item in all_sequences}
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

        stored_goal = stored_goal.model_copy(update={"status": GoalStatus.COMPLETED})

        if self.repository:
            self.repository.create_goal(stored_goal)
            self.repository.insert_register_candidates(registers)
            self.repository.insert_sequence_candidates(all_sequences)
            for report in reports:
                self.repository.insert_robustness_report(report)
            self.repository.insert_evaluation_results(ranked_candidates)
            for decision in decisions:
                self.repository.insert_agent_decision(decision)
            for record in memory_records:
                self.repository.insert_memory_record(record)
            self.repository.update_campaign(campaign)

        return PipelineSummary(
            campaign=campaign,
            goal=stored_goal,
            spec=spec,
            top_candidate=ranked_candidates[0] if ranked_candidates else None,
            ranked_candidates=ranked_candidates,
            decisions=decisions,
            robustness_reports=reports,
            memory_records=memory_records,
        )
