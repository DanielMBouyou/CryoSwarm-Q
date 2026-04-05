from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName
from packages.core.models import RobustnessReport, SequenceCandidate
from packages.simulation.evaluators import evaluate_candidate_robustness


class NoiseRobustnessAgent(BaseAgent):
    agent_name = AgentName.NOISE

    def run(self, sequence_candidate: SequenceCandidate) -> RobustnessReport:
        nominal, scenario_scores, average, penalty, robustness = evaluate_candidate_robustness(
            sequence_candidate
        )
        return RobustnessReport(
            campaign_id=sequence_candidate.campaign_id,
            sequence_candidate_id=sequence_candidate.id,
            nominal_score=nominal,
            perturbation_average=average,
            robustness_penalty=penalty,
            robustness_score=robustness,
            scenario_scores=scenario_scores,
            reasoning_summary=(
                f"Evaluated nominal={nominal:.3f} with perturbation average={average:.3f} "
                f"for sequence {sequence_candidate.label}."
            ),
            metadata={
                "sequence_family": sequence_candidate.sequence_family.value,
                "predicted_cost": sequence_candidate.predicted_cost,
            },
        )
