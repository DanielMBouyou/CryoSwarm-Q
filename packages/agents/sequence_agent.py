from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName, SequenceFamily
from packages.core.models import ExperimentSpec, MemoryRecord, RegisterCandidate, SequenceCandidate
from packages.pasqal_adapters.pulser_adapter import build_simple_sequence_summary


class SequenceAgent(BaseAgent):
    agent_name = AgentName.SEQUENCE

    def run(
        self,
        spec: ExperimentSpec,
        register_candidate: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord] | None = None,
    ) -> list[SequenceCandidate]:
        candidates: list[SequenceCandidate] = []
        for family in spec.sequence_families:
            for variant in self._family_variants(family, register_candidate.atom_count, memory_records or []):
                duration_ns, amplitude, detuning, phase, waveform_kind, metadata = variant
                predicted_cost = round(
                    min(1.0, (2 ** register_candidate.atom_count) * duration_ns / 500000.0),
                    4,
                )
                sequence_candidate = SequenceCandidate(
                    campaign_id=campaign_id,
                    spec_id=spec.id,
                    register_candidate_id=register_candidate.id,
                    label=f"{register_candidate.label}-{family.value}-{waveform_kind}",
                    sequence_family=family,
                    channel_id="rydberg_global",
                    duration_ns=duration_ns,
                    amplitude=amplitude,
                    detuning=detuning,
                    phase=phase,
                    waveform_kind=waveform_kind,
                    predicted_cost=predicted_cost,
                    reasoning_summary=(
                        f"Generated a {family.value} pulse candidate with {waveform_kind} waveform "
                        f"for register {register_candidate.label}."
                    ),
                    metadata={
                        "atom_count": register_candidate.atom_count,
                        "layout_type": register_candidate.layout_type,
                        "spacing_um": register_candidate.metadata.get("spacing_um"),
                        **metadata,
                    },
                )
                sequence_candidate.serialized_pulser_sequence = build_simple_sequence_summary(
                    register_candidate,
                    sequence_candidate,
                )
                candidates.append(sequence_candidate)
        return candidates

    def _family_variants(
        self,
        family: SequenceFamily,
        atom_count: int,
        memory_records: list[MemoryRecord],
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        exploit_bonus = any("strong" in record.reusable_tags for record in memory_records)
        duration_offset = atom_count * 120
        shared_phase = 0.0

        if family == SequenceFamily.ADIABATIC_SWEEP:
            variants = [
                (
                    2200 + duration_offset,
                    1.2,
                    -1.4,
                    shared_phase,
                    "adiabatic_conservative",
                    {"amplitude_start": 0.2, "detuning_end": -0.35},
                ),
                (
                    2800 + duration_offset,
                    1.6,
                    -1.8,
                    shared_phase,
                    "adiabatic_extended",
                    {"amplitude_start": 0.25, "detuning_end": -0.25},
                ),
            ]
            if exploit_bonus:
                variants.append(
                    (
                        2500 + duration_offset,
                        1.4,
                        -1.6,
                        shared_phase,
                        "adiabatic_memory",
                        {"amplitude_start": 0.22, "detuning_end": -0.30},
                    )
                )
            return variants
        if family == SequenceFamily.DETUNING_SCAN:
            variants = [
                (
                    1800 + duration_offset,
                    1.0,
                    -1.8,
                    0.15,
                    "detuning_scan_fast",
                    {"detuning_end": 0.3},
                ),
                (
                    2400 + duration_offset,
                    1.25,
                    -2.2,
                    0.15,
                    "detuning_scan_wide",
                    {"detuning_end": 0.6},
                ),
            ]
            if exploit_bonus:
                variants.append(
                    (
                        2100 + duration_offset,
                        1.15,
                        -2.0,
                        0.15,
                        "detuning_scan_memory",
                        {"detuning_end": 0.4},
                    )
                )
            return variants
        variants = [
            (
                1600 + duration_offset,
                1.0,
                -0.8,
                0.05,
                "global_ramp_compact",
                {"amplitude_start": 0.15},
            ),
            (
                2200 + duration_offset,
                1.35,
                -1.0,
                0.05,
                "global_ramp_extended",
                {"amplitude_start": 0.20},
            ),
        ]
        if exploit_bonus:
            variants.append(
                (
                    2000 + duration_offset,
                    1.2,
                    -0.9,
                    0.05,
                    "global_ramp_memory",
                    {"amplitude_start": 0.18},
                )
            )
        return variants
