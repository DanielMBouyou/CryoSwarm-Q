from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName
from packages.core.models import CampaignState, EvaluationResult, ExperimentGoal, ExperimentSpec


class ResultsAgent(BaseAgent):
    agent_name = AgentName.RESULTS

    def run(
        self,
        goal: ExperimentGoal,
        spec: ExperimentSpec,
        campaign: CampaignState,
        ranked_candidates: list[EvaluationResult],
    ) -> dict[str, object]:
        top_candidate = ranked_candidates[0] if ranked_candidates else None
        return {
            "goal_title": goal.title,
            "objective_class": spec.objective_class,
            "campaign_id": campaign.id,
            "candidate_count": len(ranked_candidates),
            "top_candidate_id": top_candidate.sequence_candidate_id if top_candidate else None,
            "top_backend": top_candidate.backend_choice.value if top_candidate else None,
            "top_objective_score": top_candidate.objective_score if top_candidate else None,
            "summary": (
                f"Campaign {campaign.id} completed for '{goal.title}'. "
                f"Top backend recommendation: "
                f"{top_candidate.backend_choice.value if top_candidate else 'n/a'}."
            ),
        }
