from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName
from packages.core.models import BackendChoice, ExperimentSpec, RobustnessReport, SequenceCandidate
from packages.pasqal_adapters.emulator_router import recommend_backend


class BackendRoutingAgent(BaseAgent):
    agent_name = AgentName.ROUTING

    def run(
        self,
        spec: ExperimentSpec,
        sequence_candidate: SequenceCandidate,
        report: RobustnessReport,
    ) -> BackendChoice:
        return recommend_backend(spec, sequence_candidate, report)
