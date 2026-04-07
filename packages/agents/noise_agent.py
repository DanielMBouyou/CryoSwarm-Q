from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName
from packages.core.models import ExperimentSpec, RegisterCandidate, RobustnessReport, SequenceCandidate
from packages.core.parameter_space import PhysicsParameterSpace
from packages.simulation.evaluators import evaluate_candidate_robustness


class NoiseRobustnessAgent(BaseAgent):
    agent_name = AgentName.NOISE

    def __init__(self, param_space: PhysicsParameterSpace | None = None) -> None:
        super().__init__()
        self.param_space = param_space or PhysicsParameterSpace.default()

    def run(
        self,
        spec: ExperimentSpec,
        register_candidate: RegisterCandidate,
        sequence_candidate: SequenceCandidate,
    ) -> RobustnessReport:
        (
            nominal,
            scenario_scores,
            average,
            worst_score,
            score_std,
            penalty,
            robustness,
            nominal_observables,
            scenario_observables,
            hamiltonian_metrics,
        ) = evaluate_candidate_robustness(
            spec,
            register_candidate,
            sequence_candidate,
            param_space=self.param_space,
        )
        return RobustnessReport(
            campaign_id=sequence_candidate.campaign_id,
            sequence_candidate_id=sequence_candidate.id,
            nominal_score=nominal,
            perturbation_average=average,
            robustness_penalty=penalty,
            robustness_score=robustness,
            worst_case_score=worst_score,
            score_std=score_std,
            target_observable=spec.target_observable,
            scenario_scores=scenario_scores,
            nominal_observables=nominal_observables,
            scenario_observables=scenario_observables,
            hamiltonian_metrics=hamiltonian_metrics,
            reasoning_summary=(
                f"Evaluated {sequence_candidate.label} with actual Pulser emulation. "
                f"Nominal observable score={nominal:.3f}, worst-case={worst_score:.3f}, "
                f"robustness={robustness:.3f}."
            ),
            metadata={
                "sequence_family": sequence_candidate.sequence_family.value,
                "predicted_cost": sequence_candidate.predicted_cost,
                "robustness_weight_config": nominal_observables.get("robustness_weight_config", {}),
            },
        )
