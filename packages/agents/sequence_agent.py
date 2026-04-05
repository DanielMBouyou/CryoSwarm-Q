from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName, SequenceFamily
from packages.core.models import ExperimentSpec, RegisterCandidate, SequenceCandidate
from packages.pasqal_adapters.pulser_adapter import build_simple_sequence_summary


class SequenceAgent(BaseAgent):
    agent_name = AgentName.SEQUENCE

    def run(
        self,
        spec: ExperimentSpec,
        register_candidate: RegisterCandidate,
        campaign_id: str,
    ) -> list[SequenceCandidate]:
        candidates: list[SequenceCandidate] = []
        for family in spec.sequence_families:
            duration_ns, amplitude, detuning, phase = self._family_parameters(
                family,
                register_candidate.atom_count,
            )
            predicted_cost = round(
                min(0.95, register_candidate.atom_count * duration_ns / 100000.0),
                4,
            )
            payload = build_simple_sequence_summary(
                coordinates=register_candidate.coordinates,
                duration_ns=duration_ns,
                amplitude=amplitude,
                detuning=detuning,
                phase=phase,
            )
            candidates.append(
                SequenceCandidate(
                    campaign_id=campaign_id,
                    spec_id=spec.id,
                    register_candidate_id=register_candidate.id,
                    label=f"{register_candidate.label}-{family.value}",
                    sequence_family=family,
                    duration_ns=duration_ns,
                    amplitude=amplitude,
                    detuning=detuning,
                    phase=phase,
                    predicted_cost=predicted_cost,
                    reasoning_summary=(
                        f"Generated a {family.value} pulse candidate for register "
                        f"{register_candidate.label}."
                    ),
                    serialized_pulser_sequence=payload,
                    metadata={
                        "atom_count": register_candidate.atom_count,
                        "layout_type": register_candidate.layout_type,
                    },
                )
            )
        return candidates

    def _family_parameters(
        self,
        family: SequenceFamily,
        atom_count: int,
    ) -> tuple[int, float, float, float]:
        if family == SequenceFamily.ADIABATIC_SWEEP:
            return 2600 + atom_count * 180, 1.8, -0.9, 0.0
        if family == SequenceFamily.DETUNING_SCAN:
            return 2100 + atom_count * 160, 1.5, -1.2, 0.25
        return 1800 + atom_count * 140, 1.2, -0.6, 0.10
