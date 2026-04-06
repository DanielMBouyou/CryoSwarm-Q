from __future__ import annotations

from dataclasses import dataclass
from math import dist
from typing import Any, cast

import numpy as np

from packages.core.logging import get_logger
from packages.core.metadata_schemas import SequenceMetadata
from packages.core.parameter_space import PhysicsParameterSpace
from packages.core.models import RegisterCandidate, SequenceCandidate


try:
    from pulser import Pulse, Register, Sequence
    from pulser.devices import AnalogDevice
    from pulser.waveforms import BlackmanWaveform, ConstantWaveform, RampWaveform

    PULSER_AVAILABLE = True
    PULSER_IMPORT_ERROR: str | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - depends on external package
    Pulse = None
    Register = None
    Sequence = None
    AnalogDevice = None
    RampWaveform = None
    PULSER_AVAILABLE = False
    PULSER_IMPORT_ERROR = str(exc)


logger = get_logger(__name__)


@dataclass(slots=True)
class RegisterPhysicsSummary:
    valid: bool
    validation_error: str | None
    min_distance_um: float
    blockade_radius_um: float
    blockade_pair_count: int
    van_der_waals_matrix: list[list[float]]


def get_public_device() -> Any:
    if not PULSER_AVAILABLE:
        raise RuntimeError(PULSER_IMPORT_ERROR or "Pulser is not available.")
    return AnalogDevice


def _channel() -> Any:
    return get_public_device().channels["rydberg_global"]


def _safety_margin(param_space: PhysicsParameterSpace | None = None) -> float:
    return (param_space or PhysicsParameterSpace.default()).amplitude_safety_margin


def _quantize_duration(duration_ns: int) -> int:
    channel = _channel()
    duration = max(int(duration_ns), int(channel.min_duration))
    remainder = duration % int(channel.clock_period)
    if remainder:
        duration += int(channel.clock_period) - remainder
    return duration


def _blackman_with_max_amplitude(duration_ns: int, max_amplitude: float) -> Any:
    """Build a fixed-duration Blackman waveform whose peak does not exceed max_amplitude."""
    if not PULSER_AVAILABLE:
        raise RuntimeError(PULSER_IMPORT_ERROR or "Pulser is not available.")
    normalized = np.clip(np.blackman(int(duration_ns)), 0.0, np.inf)
    area = float(max_amplitude) * float(np.sum(normalized)) / 1000.0
    return BlackmanWaveform(int(duration_ns), area)


def _clip_amplitude(
    amplitude: float,
    param_space: PhysicsParameterSpace | None = None,
) -> float:
    return min(float(amplitude), float(_channel().max_amp) * _safety_margin(param_space))


def _clip_detuning(
    detuning: float,
    param_space: PhysicsParameterSpace | None = None,
) -> float:
    max_detuning = float(_channel().max_abs_detuning) * _safety_margin(param_space)
    return max(-max_detuning, min(float(detuning), max_detuning))


def pairwise_distance_matrix(coordinates: list[tuple[float, float]]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for point_i in coordinates:
        matrix.append([round(dist(point_i, point_j), 6) for point_j in coordinates])
    return matrix


def interaction_matrix(coordinates: list[tuple[float, float]]) -> list[list[float]]:
    device = get_public_device()
    coeff = float(device.interaction_coeff)
    distances = pairwise_distance_matrix(coordinates)
    matrix: list[list[float]] = []
    for row_index, row in enumerate(distances):
        interaction_row: list[float] = []
        for col_index, distance_um in enumerate(row):
            if row_index == col_index or distance_um == 0.0:
                interaction_row.append(0.0)
            else:
                interaction_row.append(round(coeff / (distance_um**6), 6))
        matrix.append(interaction_row)
    return matrix


def summarize_register_physics(
    coordinates: list[tuple[float, float]],
    reference_amplitude: float = 5.0,
    param_space: PhysicsParameterSpace | None = None,
) -> RegisterPhysicsSummary:
    if not PULSER_AVAILABLE:
        return RegisterPhysicsSummary(
            valid=False,
            validation_error=PULSER_IMPORT_ERROR or "Pulser is not available.",
            min_distance_um=0.0,
            blockade_radius_um=0.0,
            blockade_pair_count=0,
            van_der_waals_matrix=[],
        )

    register = Register.from_coordinates(coordinates, prefix="q")
    device = get_public_device()
    validation_error: str | None = None
    valid = True
    try:
        device.validate_register(register)
    except (ValueError, TypeError) as exc:
        valid = False
        validation_error = str(exc)

    distance_matrix = pairwise_distance_matrix(coordinates)
    positive_distances = [
        value
        for row_index, row in enumerate(distance_matrix)
        for col_index, value in enumerate(row)
        if row_index != col_index and value > 0.0
    ]
    min_distance_um = round(min(positive_distances), 6) if positive_distances else 0.0
    blockade_radius_um = round(
        device.rydberg_blockade_radius(_clip_amplitude(reference_amplitude, param_space)),
        6,
    )
    blockade_pairs = 0
    for row_index, row in enumerate(distance_matrix):
        for col_index, value in enumerate(row):
            if col_index > row_index and value <= blockade_radius_um:
                blockade_pairs += 1

    return RegisterPhysicsSummary(
        valid=valid,
        validation_error=validation_error,
        min_distance_um=min_distance_um,
        blockade_radius_um=blockade_radius_um,
        blockade_pair_count=blockade_pairs,
        van_der_waals_matrix=interaction_matrix(coordinates),
    )


def create_simple_register(coordinates: list[tuple[float, float]]) -> dict[str, Any]:
    summary = summarize_register_physics(coordinates)
    payload: dict[str, Any] = {
        "pulser_available": PULSER_AVAILABLE,
        "coordinate_count": len(coordinates),
        "coordinates": coordinates,
        "register_valid": summary.valid,
        "validation_error": summary.validation_error,
        "min_distance_um": summary.min_distance_um,
        "blockade_radius_um": summary.blockade_radius_um,
        "blockade_pair_count": summary.blockade_pair_count,
        "van_der_waals_matrix": summary.van_der_waals_matrix,
    }

    if not PULSER_AVAILABLE:
        payload["warning"] = PULSER_IMPORT_ERROR or "Pulser is not installed."
        return payload

    try:  # pragma: no cover - requires pulser runtime
        register = Register.from_coordinates(coordinates, prefix="q")
        payload["pulser_register_created"] = True
        payload["register_repr"] = str(register)
    except (ValueError, RuntimeError) as exc:
        payload["pulser_register_created"] = False
        payload["warning"] = f"Pulser register creation failed: {exc}"
    return payload


def build_sequence_from_candidate(
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    param_space: PhysicsParameterSpace | None = None,
) -> Any:
    if not PULSER_AVAILABLE:
        raise RuntimeError(PULSER_IMPORT_ERROR or "Pulser is not available.")

    register = Register.from_coordinates(register_candidate.coordinates, prefix="q")
    sequence = Sequence(register, get_public_device())
    channel_name = "ryd"
    sequence.declare_channel(channel_name, sequence_candidate.channel_id)

    duration_ns = _quantize_duration(sequence_candidate.duration_ns)
    amplitude = _clip_amplitude(sequence_candidate.amplitude, param_space)
    detuning = _clip_detuning(sequence_candidate.detuning, param_space)
    phase = float(sequence_candidate.phase)
    family = sequence_candidate.sequence_family.value
    sequence_metadata = cast(SequenceMetadata, sequence_candidate.metadata)

    if family == "global_ramp":
        amplitude_start = _clip_amplitude(
            float(sequence_metadata.get("amplitude_start", max(0.0, amplitude * 0.15))),
            param_space,
        )
        pulse = Pulse.ConstantDetuning(
            RampWaveform(duration_ns, amplitude_start, amplitude),
            detuning,
            phase,
        )
        sequence.add(pulse, channel_name)
    elif family == "detuning_scan":
        detuning_end = _clip_detuning(
            float(sequence_metadata.get("detuning_end", -detuning * 0.25)),
            param_space,
        )
        pulse = Pulse.ConstantAmplitude(
            amplitude,
            RampWaveform(duration_ns, detuning, detuning_end),
            phase,
        )
        sequence.add(pulse, channel_name)
    elif family == "constant_drive":
        pulse = Pulse.ConstantPulse(
            duration_ns,
            amplitude,
            detuning,
            phase,
        )
        sequence.add(pulse, channel_name)
    elif family == "blackman_sweep":
        detuning_end = _clip_detuning(
            float(sequence_metadata.get("detuning_end", -detuning * 0.25)),
            param_space,
        )
        pulse = Pulse(
            _blackman_with_max_amplitude(duration_ns, amplitude),
            RampWaveform(duration_ns, detuning, detuning_end),
            phase,
        )
        sequence.add(pulse, channel_name)
    else:
        ramp_duration = _quantize_duration(max(duration_ns // 2, 16))
        hold_duration = _quantize_duration(max(duration_ns - ramp_duration, 16))
        ramp_pulse = Pulse.ConstantDetuning(
            RampWaveform(
                ramp_duration,
                _clip_amplitude(max(0.0, amplitude * 0.2), param_space),
                amplitude,
            ),
            detuning,
            phase,
        )
        sweep_pulse = Pulse.ConstantAmplitude(
            amplitude,
            RampWaveform(hold_duration, detuning, _clip_detuning(detuning * 0.25, param_space)),
            phase,
        )
        sequence.add(ramp_pulse, channel_name)
        sequence.add(sweep_pulse, channel_name)

    return sequence


def build_simple_sequence_summary(
    register_candidate: RegisterCandidate,
    sequence_candidate: SequenceCandidate,
    param_space: PhysicsParameterSpace | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pulser_available": PULSER_AVAILABLE,
        "register_summary": create_simple_register(register_candidate.coordinates),
        "duration_ns": _quantize_duration(sequence_candidate.duration_ns) if PULSER_AVAILABLE else sequence_candidate.duration_ns,
        "controls": {
            "Omega_max": (
                _clip_amplitude(sequence_candidate.amplitude, param_space)
                if PULSER_AVAILABLE
                else sequence_candidate.amplitude
            ),
            "delta_start": (
                _clip_detuning(sequence_candidate.detuning, param_space)
                if PULSER_AVAILABLE
                else sequence_candidate.detuning
            ),
            "phi": sequence_candidate.phase,
        },
        "waveform_kind": sequence_candidate.waveform_kind,
        "sequence_family": sequence_candidate.sequence_family.value,
        "serialization_mode": "structured_summary",
    }
    if not PULSER_AVAILABLE:
        return payload

    sequence = build_sequence_from_candidate(
        register_candidate,
        sequence_candidate,
        param_space=param_space,
    )
    payload["device"] = "AnalogDevice"
    payload["channel_id"] = sequence_candidate.channel_id
    payload["sequence_duration_ns"] = sequence.get_duration()
    if hasattr(sequence, "to_abstract_repr"):
        try:
            payload["abstract_sequence"] = sequence.to_abstract_repr()
        except Exception as exc:
            logger.debug("Could not generate abstract sequence representation: %s", exc)
    return payload


def summarize_sequence_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"available": False}
    return {
        "pulser_available": payload.get("pulser_available", False),
        "duration_ns": payload.get("duration_ns"),
        "controls": payload.get("controls", {}),
        "waveform_kind": payload.get("waveform_kind"),
        "sequence_family": payload.get("sequence_family"),
        "serialization_mode": payload.get("serialization_mode", "structured_summary"),
    }
