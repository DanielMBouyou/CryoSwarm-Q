from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName
from packages.core.models import EvaluationResult, MemoryRecord


class MemoryAgent(BaseAgent):
    agent_name = AgentName.MEMORY

    def run(self, campaign_id: str, ranked_candidates: list[EvaluationResult]) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for result in ranked_candidates[:3]:
            robustness_band = "strong" if result.robustness_score >= 0.70 else "moderate"
            records.append(
                MemoryRecord(
                    campaign_id=campaign_id,
                    source_candidate_id=result.sequence_candidate_id,
                    lesson_type="candidate_pattern",
                    summary=(
                        f"Candidate {result.sequence_candidate_id} showed {robustness_band} "
                        f"robustness with backend {result.backend_choice.value}."
                    ),
                    signals={
                        "objective_score": result.objective_score,
                        "robustness_score": result.robustness_score,
                        "backend_choice": result.backend_choice.value,
                    },
                    reusable_tags=[
                        robustness_band,
                        result.backend_choice.value,
                    ],
                )
            )
        return records
