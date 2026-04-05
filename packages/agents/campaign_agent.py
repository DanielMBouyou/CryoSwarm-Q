from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName, CampaignStatus
from packages.core.models import CampaignState, EvaluationResult
from packages.scoring.ranking import rank_evaluations


class CampaignAgent(BaseAgent):
    agent_name = AgentName.CAMPAIGN

    def run(
        self,
        campaign: CampaignState,
        evaluations: list[EvaluationResult],
    ) -> tuple[CampaignState, list[EvaluationResult]]:
        ranked = rank_evaluations(evaluations)
        top_candidate = ranked[0] if ranked else None
        updated_campaign = campaign.model_copy(
            update={
                "status": CampaignStatus.COMPLETED,
                "candidate_count": len(ranked),
                "ranked_candidate_ids": [item.sequence_candidate_id for item in ranked],
                "top_candidate_id": top_candidate.sequence_candidate_id if top_candidate else None,
                "summary": (
                    f"Ranked {len(ranked)} sequence candidates and selected "
                    f"{top_candidate.sequence_candidate_id if top_candidate else 'none'} as top candidate."
                ),
                "summary_report": {
                    "top_candidate_id": top_candidate.sequence_candidate_id if top_candidate else None,
                    "top_objective_score": top_candidate.objective_score if top_candidate else None,
                    "ranked_candidate_count": len(ranked),
                },
            }
        )
        return updated_campaign, ranked
