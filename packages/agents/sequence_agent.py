from __future__ import annotations

from typing import cast

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName, SequenceFamily
from packages.core.metadata_schemas import RegisterMetadata
from packages.core.models import ExperimentSpec, MemoryRecord, RegisterCandidate, SequenceCandidate
from packages.core.parameter_space import ParameterRange, PhysicsParameterSpace
from packages.pasqal_adapters.pulser_adapter import build_simple_sequence_summary


class SequenceAgent(BaseAgent):
    """Generate pulse-sequence candidates with physically realistic parameters."""

    agent_name = AgentName.SEQUENCE

    def __init__(self, param_space: PhysicsParameterSpace | None = None) -> None:
        super().__init__()
        self.param_space = param_space or PhysicsParameterSpace.default()

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
                register_metadata = cast(RegisterMetadata, register_candidate.metadata)
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
                    predicted_cost=self.param_space.cost_for(register_candidate.atom_count, duration_ns),
                    reasoning_summary=(
                        f"Generated a {family.value} pulse ({waveform_kind}) "
                        f"with Omega={amplitude:.1f} rad/us, delta={detuning:.1f} rad/us, "
                        f"T={duration_ns} ns for register {register_candidate.label}."
                    ),
                    metadata={
                        "atom_count": register_candidate.atom_count,
                        "layout_type": register_candidate.layout_type,
                        "spacing_um": register_metadata.get("spacing_um"),
                        "source": "heuristic",
                        **metadata,
                    },
                )
                seq = seq.model_copy(
                    update={
                        "serialized_pulser_sequence": build_simple_sequence_summary(
                            register_candidate,
                            seq,
                            param_space=self.param_space,
                        )
                    }
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
        duration_offset = self.param_space.duration_offset(atom_count)

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
            self.logger.debug(
                "Dropped trailing %s variant due to weak failure memory for family %s.",
                family.value,
                family.value,
            )
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

        pulse_space = self.param_space.pulse[family]
        best_record = max(matching_records, key=lambda item: float(item.signals.get("confidence", 0.0)))
        base_duration = int(
            best_record.signals.get(
                "duration_ns",
                pulse_space.duration_ns.default + self.param_space.duration_offset(atom_count),
            )
        )
        base_amplitude = float(best_record.signals["amplitude"])
        base_detuning = float(best_record.signals["detuning"])

        refined: list[tuple[int, float, float, float, str, dict[str, float]]] = []
        for label_suffix, scale in (("minus", 0.9), ("plus", 1.1)):
            refined.append(
                (
                    self._clip_duration(pulse_space.duration_ns, base_duration, 0),
                    self._clip_value(pulse_space.amplitude, base_amplitude * scale),
                    self._clip_value(pulse_space.detuning_start, base_detuning * scale),
                    pulse_space.phase.default,
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
        pulse_space = self.param_space.pulse[family]
        if family == SequenceFamily.GLOBAL_RAMP and pulse_space.amplitude_start is not None:
            return {"amplitude_start": self._clip_value(pulse_space.amplitude_start, max(0.2, amplitude * 0.15))}
        if family == SequenceFamily.DETUNING_SCAN and pulse_space.detuning_end is not None:
            return {"detuning_end": self._clip_value(pulse_space.detuning_end, -detuning * 0.35 if detuning else 5.0)}
        if family == SequenceFamily.BLACKMAN_SWEEP and pulse_space.detuning_end is not None:
            return {"detuning_end": self._clip_value(pulse_space.detuning_end, -detuning * 0.45 if detuning else 8.0)}
        if family == SequenceFamily.ADIABATIC_SWEEP:
            metadata: dict[str, float] = {}
            if pulse_space.amplitude_start is not None:
                metadata["amplitude_start"] = self._clip_value(pulse_space.amplitude_start, max(0.2, amplitude * 0.12))
            if pulse_space.detuning_end is not None:
                metadata["detuning_end"] = self._clip_value(pulse_space.detuning_end, -detuning * 0.5 if detuning else 10.0)
            return metadata
        return {}

    def _clip_value(self, parameter_range: ParameterRange, value: float) -> float:
        return round(parameter_range.clip(value), 4)

    def _clip_duration(self, parameter_range: ParameterRange, base_duration: float, offset: int) -> int:
        quantized = parameter_range.clip(base_duration)
        return max(16, int(round(quantized + offset)))

    def _adiabatic_variants(
        self,
        offset: int,
        exploit: bool,
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        pulse_space = self.param_space.pulse[SequenceFamily.ADIABATIC_SWEEP]
        variants = [
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default),
                pulse_space.phase.default,
                "adiabatic_conservative",
                {
                    "amplitude_start": self._clip_value(pulse_space.amplitude_start, pulse_space.amplitude_start.default),  # type: ignore[arg-type]
                    "detuning_end": self._clip_value(pulse_space.detuning_end, pulse_space.detuning_end.default),  # type: ignore[arg-type]
                },
            ),
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default + 1000.0, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default * 1.5),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default * 1.25),
                pulse_space.phase.default,
                "adiabatic_extended",
                {
                    "amplitude_start": self._clip_value(pulse_space.amplitude_start, pulse_space.amplitude_start.default * 1.6),  # type: ignore[arg-type]
                    "detuning_end": self._clip_value(pulse_space.detuning_end, pulse_space.detuning_end.default * 1.5),  # type: ignore[arg-type]
                },
            ),
        ]
        if exploit:
            variants.append(
                (
                    self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default + 500.0, offset),
                    self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default * 1.2),
                    self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default * 1.1),
                    pulse_space.phase.default,
                    "adiabatic_memory",
                    {
                        "amplitude_start": self._clip_value(pulse_space.amplitude_start, pulse_space.amplitude_start.default * 1.2),  # type: ignore[arg-type]
                        "detuning_end": self._clip_value(pulse_space.detuning_end, pulse_space.detuning_end.default * 1.2),  # type: ignore[arg-type]
                    },
                )
            )
        return variants

    def _detuning_scan_variants(
        self,
        offset: int,
        exploit: bool,
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        pulse_space = self.param_space.pulse[SequenceFamily.DETUNING_SCAN]
        variants = [
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default),
                pulse_space.phase.default,
                "detuning_scan_fast",
                {"detuning_end": self._clip_value(pulse_space.detuning_end, pulse_space.detuning_end.default)},  # type: ignore[arg-type]
            ),
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default + 1000.0, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default * 1.5),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default * (4.0 / 3.0)),
                pulse_space.phase.default,
                "detuning_scan_wide",
                {"detuning_end": self._clip_value(pulse_space.detuning_end, pulse_space.detuning_end.default * 2.0)},  # type: ignore[arg-type]
            ),
        ]
        if exploit:
            variants.append(
                (
                    self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default + 500.0, offset),
                    self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default * 1.25),
                    self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default * 1.2),
                    pulse_space.phase.default,
                    "detuning_scan_memory",
                    {"detuning_end": self._clip_value(pulse_space.detuning_end, pulse_space.detuning_end.default * 1.6)},  # type: ignore[arg-type]
                )
            )
        return variants

    def _global_ramp_variants(
        self,
        offset: int,
        exploit: bool,
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        pulse_space = self.param_space.pulse[SequenceFamily.GLOBAL_RAMP]
        variants = [
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default),
                pulse_space.phase.default,
                "global_ramp_compact",
                {"amplitude_start": self._clip_value(pulse_space.amplitude_start, pulse_space.amplitude_start.default)},  # type: ignore[arg-type]
            ),
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default + 1000.0, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default * (4.0 / 3.0)),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default * 1.5),
                pulse_space.phase.default,
                "global_ramp_extended",
                {"amplitude_start": self._clip_value(pulse_space.amplitude_start, pulse_space.amplitude_start.default * 2.0)},  # type: ignore[arg-type]
            ),
        ]
        if exploit:
            variants.append(
                (
                    self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default + 500.0, offset),
                    self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default * (7.0 / 6.0)),
                    self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default * 1.25),
                    pulse_space.phase.default,
                    "global_ramp_memory",
                    {"amplitude_start": self._clip_value(pulse_space.amplitude_start, pulse_space.amplitude_start.default * 1.6)},  # type: ignore[arg-type]
                )
            )
        return variants

    def _constant_drive_variants(
        self,
        offset: int,
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        pulse_space = self.param_space.pulse[SequenceFamily.CONSTANT_DRIVE]
        return [
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default),
                pulse_space.phase.default,
                "constant_resonant",
                {},
            ),
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default + 300.0, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.min_val * 0.5),
                pulse_space.phase.default,
                "constant_detuned",
                {},
            ),
        ]

    def _blackman_sweep_variants(
        self,
        offset: int,
        exploit: bool,
    ) -> list[tuple[int, float, float, float, str, dict[str, float]]]:
        pulse_space = self.param_space.pulse[SequenceFamily.BLACKMAN_SWEEP]
        variants = [
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default),
                pulse_space.phase.default,
                "blackman_sweep_standard",
                {"detuning_end": self._clip_value(pulse_space.detuning_end, pulse_space.detuning_end.default)},  # type: ignore[arg-type]
            ),
            (
                self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default + 1000.0, offset),
                self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default + 1.5),
                self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default * 1.5),
                pulse_space.phase.default,
                "blackman_sweep_wide",
                {"detuning_end": self._clip_value(pulse_space.detuning_end, pulse_space.detuning_end.default * 1.5)},  # type: ignore[arg-type]
            ),
        ]
        if exploit:
            variants.append(
                (
                    self._clip_duration(pulse_space.duration_ns, pulse_space.duration_ns.default + 500.0, offset),
                    self._clip_value(pulse_space.amplitude, pulse_space.amplitude.default + 0.5),
                    self._clip_value(pulse_space.detuning_start, pulse_space.detuning_start.default * 1.25),
                    pulse_space.phase.default,
                    "blackman_sweep_memory",
                    {"detuning_end": self._clip_value(pulse_space.detuning_end, pulse_space.detuning_end.default * 1.2)},  # type: ignore[arg-type]
                )
            )
        return variants
