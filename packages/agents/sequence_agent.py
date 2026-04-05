from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName, SequenceFamily
from packages.core.models import ExperimentSpec, MemoryRecord, RegisterCandidate, SequenceCandidate
from packages.pasqal_adapters.pulser_adapter import build_simple_sequence_summary


class SequenceAgent(BaseAgent):
    """Generate pulse-sequence candidates with physically realistic parameters.

    All amplitudes are Rabi frequencies in rad/us compatible with
    ``AnalogDevice`` (max ~15.7 rad/us). Detunings sweep through the
    interaction energy scale so that the system crosses the quantum phase
    transition between disordered and ordered Rydberg phases.
    """

    agent_name = AgentName.SEQUENCE

    def run(
        self,
        spec: ExperimentSpec,
        register_candidate: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord] | None = None,
    ) -> list[SequenceCandidate]:
        memory_records = memory_records or []
        candidates: list[SequenceCandidate] = []
        for family in spec.sequence_families:
            for variant in self._family_variants(
                family,
                register_candidate.atom_count,
                memory_records,
            ):
                duration_ns, amplitude, detuning, phase, waveform_kind, metadata = variant
                predicted_cost = round(
                    min(1.0, (2**register_candidate.atom_count) * duration_ns / 500000.0),
                    4,
                )
                seq = SequenceCandidate(
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
                        f"Generated a {family.value} pulse ({waveform_kind}) "
                        f"with Omega={amplitude:.1f} rad/us, delta={detuning:.1f} rad/us, "
                        f"T={duration_ns} ns for register {register_candidate.label}."
                    ),
                    metadata={
                        "atom_count": register_candidate.atom_count,
                        "layout_type": register_candidate.layout_type,
                        "spacing_um": register_candidate.metadata.get("spacing_um"),
                        **metadata,
                    },
                )
                seq.serialized_pulser_sequence = build_simple_sequence_summary(
                    register_candidate,
                    seq,
                )
                candidates.append(seq)
        return candidates

    def _family_variants(
        self,
        family: SequenceFamily,
        atom_count: int,
        memory_records: list[MemoryRecord],
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        exploit_bonus = any("strong" in record.reusable_tags for record in memory_records)
        duration_offset = atom_count * 150

        if family == SequenceFamily.ADIABATIC_SWEEP:
            variants = self._adiabatic_variants(duration_offset, exploit_bonus)
        elif family == SequenceFamily.DETUNING_SCAN:
            variants = self._detuning_scan_variants(duration_offset, exploit_bonus)
        elif family == SequenceFamily.CONSTANT_DRIVE:
            variants = self._constant_drive_variants(duration_offset)
        elif family == SequenceFamily.BLACKMAN_SWEEP:
            variants = self._blackman_sweep_variants(duration_offset, exploit_bonus)
        else:
            variants = self._global_ramp_variants(duration_offset, exploit_bonus)

        if self._family_has_weak_failure(family, memory_records) and len(variants) > 1:
            variants = variants[:-1]

        variants.extend(self._refined_variants_from_memory(family, atom_count, memory_records))
        return variants

    def _family_has_weak_failure(
        self,
        family: SequenceFamily,
        memory_records: list[MemoryRecord],
    ) -> bool:
        return any(
            record.lesson_type == "failure_pattern"
            and "weak" in record.reusable_tags
            and record.signals.get("sequence_family") == family.value
            for record in memory_records
        )

    def _refined_variants_from_memory(
        self,
        family: SequenceFamily,
        atom_count: int,
        memory_records: list[MemoryRecord],
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        matching_records = [
            record
            for record in memory_records
            if record.lesson_type == "candidate_pattern"
            and record.signals.get("sequence_family") == family.value
            and float(record.signals.get("confidence", 0.0)) > 0.7
            and "amplitude" in record.signals
            and "detuning" in record.signals
        ]
        if not matching_records:
            return []

        best_record = max(matching_records, key=lambda item: float(item.signals.get("confidence", 0.0)))
        base_duration = int(best_record.signals.get("duration_ns", 2000 + atom_count * 150))
        base_amplitude = float(best_record.signals["amplitude"])
        base_detuning = float(best_record.signals["detuning"])

        refined: list[tuple[int, float, float, float, str, dict[str, float]]] = []
        for label_suffix, scale in (("minus", 0.9), ("plus", 1.1)):
            refined.append(
                (
                    max(16, base_duration),
                    round(max(0.0, min(15.8, base_amplitude * scale)), 4),
                    round(max(-126.0, min(126.0, base_detuning * scale)), 4),
                    0.0,
                    f"refined_memory_{label_suffix}",
                    self._memory_metadata(family, base_amplitude, base_detuning),
                )
            )
        return refined

    def _memory_metadata(
        self,
        family: SequenceFamily,
        amplitude: float,
        detuning: float,
    ) -> dict[str, float]:
        if family == SequenceFamily.GLOBAL_RAMP:
            return {"amplitude_start": round(max(0.2, amplitude * 0.15), 4)}
        if family == SequenceFamily.DETUNING_SCAN:
            return {"detuning_end": round(-detuning * 0.35 if detuning else 5.0, 4)}
        if family == SequenceFamily.BLACKMAN_SWEEP:
            return {"detuning_end": round(-detuning * 0.45 if detuning else 8.0, 4)}
        if family == SequenceFamily.ADIABATIC_SWEEP:
            return {
                "amplitude_start": round(max(0.2, amplitude * 0.12), 4),
                "detuning_end": round(-detuning * 0.5 if detuning else 10.0, 4),
            }
        return {}

    @staticmethod
    def _adiabatic_variants(
        offset: int, exploit: bool
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        """Sweep detuning through the phase transition with an adiabatic protocol."""
        variants: list[tuple[int, float, float, float, str, dict[str, float]]] = [
            (
                3000 + offset,
                5.0,
                -20.0,
                0.0,
                "adiabatic_conservative",
                {"amplitude_start": 0.5, "detuning_end": 10.0},
            ),
            (
                4000 + offset,
                7.5,
                -25.0,
                0.0,
                "adiabatic_extended",
                {"amplitude_start": 0.8, "detuning_end": 15.0},
            ),
        ]
        if exploit:
            variants.append(
                (
                    3500 + offset,
                    6.0,
                    -22.0,
                    0.0,
                    "adiabatic_memory",
                    {"amplitude_start": 0.6, "detuning_end": 12.0},
                )
            )
        return variants

    @staticmethod
    def _detuning_scan_variants(
        offset: int, exploit: bool
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        """Constant amplitude with a detuning ramp to probe the phase diagram."""
        variants: list[tuple[int, float, float, float, str, dict[str, float]]] = [
            (
                2000 + offset,
                4.0,
                -15.0,
                0.0,
                "detuning_scan_fast",
                {"detuning_end": 5.0},
            ),
            (
                3000 + offset,
                6.0,
                -20.0,
                0.0,
                "detuning_scan_wide",
                {"detuning_end": 10.0},
            ),
        ]
        if exploit:
            variants.append(
                (
                    2500 + offset,
                    5.0,
                    -18.0,
                    0.0,
                    "detuning_scan_memory",
                    {"detuning_end": 8.0},
                )
            )
        return variants

    @staticmethod
    def _global_ramp_variants(
        offset: int, exploit: bool
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        """Ramp amplitude first, then sweep detuning across the ordered regime."""
        variants: list[tuple[int, float, float, float, str, dict[str, float]]] = [
            (
                2000 + offset,
                6.0,
                -12.0,
                0.0,
                "global_ramp_compact",
                {"amplitude_start": 0.5},
            ),
            (
                3000 + offset,
                8.0,
                -18.0,
                0.0,
                "global_ramp_extended",
                {"amplitude_start": 1.0},
            ),
        ]
        if exploit:
            variants.append(
                (
                    2500 + offset,
                    7.0,
                    -15.0,
                    0.0,
                    "global_ramp_memory",
                    {"amplitude_start": 0.8},
                )
            )
        return variants

    @staticmethod
    def _constant_drive_variants(
        offset: int,
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        """Constant Omega and delta for Rabi oscillations and calibration-style runs."""
        return [
            (
                1200 + offset,
                5.0,
                0.0,
                0.0,
                "constant_resonant",
                {},
            ),
            (
                1500 + offset,
                5.0,
                -5.0,
                0.0,
                "constant_detuned",
                {},
            ),
        ]

    @staticmethod
    def _blackman_sweep_variants(
        offset: int, exploit: bool
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        """Blackman-envelope sweeps that suppress leakage during the ramp."""
        variants: list[tuple[int, float, float, float, str, dict[str, float]]] = [
            (
                3000 + offset,
                7.0,
                -20.0,
                0.0,
                "blackman_sweep_standard",
                {"detuning_end": 10.0},
            ),
            (
                4000 + offset,
                8.5,
                -30.0,
                0.0,
                "blackman_sweep_wide",
                {"detuning_end": 15.0},
            ),
        ]
        if exploit:
            variants.append(
                (
                    3500 + offset,
                    7.5,
                    -25.0,
                    0.0,
                    "blackman_sweep_memory",
                    {"detuning_end": 12.0},
                )
            )
        return variants
