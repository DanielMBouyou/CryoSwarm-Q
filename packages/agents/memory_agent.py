from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName
from packages.core.models import EvaluationResult, MemoryRecord, RegisterCandidate, SequenceCandidate


class MemoryAgent(BaseAgent):
    agent_name = AgentName.MEMORY

    def run(
        self,
        campaign_id: str,
        ranked_candidates: list[EvaluationResult],
        sequence_lookup: dict[str, SequenceCandidate],
        register_lookup: dict[str, RegisterCandidate],
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for result in ranked_candidates[:3]:
            sequence = sequence_lookup[result.sequence_candidate_id]
            register = register_lookup[result.register_candidate_id]
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
                        "worst_case_score": result.worst_case_score,
                        "backend_choice": result.backend_choice.value,
                        "sequence_family": sequence.sequence_family.value,
                        "layout_type": register.layout_type,
                        "atom_count": register.atom_count,
                        "spacing_um": register.metadata.get("spacing_um"),
                        "amplitude": sequence.amplitude,
                        "detuning": sequence.detuning,
                        "duration_ns": sequence.duration_ns,
                    },
                    reusable_tags=[
                        robustness_band,
                        result.backend_choice.value,
                        sequence.sequence_family.value,
                        register.layout_type,
                    ],
                )
            )
        return records
