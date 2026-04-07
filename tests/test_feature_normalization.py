from __future__ import annotations

import math

import numpy as np
import pytest

from packages.core.enums import SequenceFamily
from packages.core.models import RegisterCandidate, SequenceCandidate
from packages.core.parameter_space import PhysicsParameterSpace
from packages.ml.dataset import (
    FEATURE_NORMALIZATION_DEFAULTS,
    INPUT_DIM_V2,
    LAYOUT_ENCODING,
    build_feature_vector_v2,
    feature_normalization_constants,
)


def _expected_feature_normalization(space: PhysicsParameterSpace) -> dict[str, float]:
    amplitude_scale = max(float(pulse_space.amplitude.max_val) for pulse_space in space.pulse.values())
    detuning_mins = [float(pulse_space.detuning_start.min_val) for pulse_space in space.pulse.values()]
    detuning_maxs = [float(pulse_space.detuning_start.max_val) for pulse_space in space.pulse.values()]
    for pulse_space in space.pulse.values():
        if pulse_space.detuning_end is None:
            continue
        detuning_mins.append(float(pulse_space.detuning_end.min_val))
        detuning_maxs.append(float(pulse_space.detuning_end.max_val))

    detuning_min = min(detuning_mins)
    detuning_max = max(detuning_maxs)
    duration_ns_scale = max(float(pulse_space.duration_ns.max_val) for pulse_space in space.pulse.values())
    min_amplitude = min(float(pulse_space.amplitude.min_val) for pulse_space in space.pulse.values())
    blockade_radius_scale = float((space.c6_coefficient / min_amplitude) ** (1.0 / 6.0))
    min_spacing = float(space.geometry.spacing_um.min_val)
    max_spacing = float(space.geometry.spacing_um.max_val)

    sweep_span_scale = 0.0
    for pulse_space in space.pulse.values():
        if pulse_space.detuning_end is None:
            continue
        for detuning_start in (pulse_space.detuning_start.min_val, pulse_space.detuning_start.max_val):
            for detuning_end in (pulse_space.detuning_end.min_val, pulse_space.detuning_end.max_val):
                sweep_span_scale = max(sweep_span_scale, abs(float(detuning_end) - float(detuning_start)))

    return {
        "amplitude_scale": amplitude_scale,
        "detuning_offset": abs(detuning_min),
        "detuning_range": detuning_max - detuning_min,
        "duration_ns_scale": duration_ns_scale,
        "layout_scale": float(len(LAYOUT_ENCODING) - 1),
        "family_scale": float(len(SequenceFamily) - 1),
        "omega_over_interaction_scale": amplitude_scale * max_spacing**6 / space.c6_coefficient,
        "detuning_over_omega_scale": max(abs(detuning_min), abs(detuning_max)) / min_amplitude,
        "adiabaticity_scale": max(
            float(pulse_space.duration_ns.max_val) / 1000.0 * float(pulse_space.amplitude.max_val)
            for pulse_space in space.pulse.values()
        ),
        "atom_count_scale": float(space.geometry.atom_count.max_val),
        "blockade_radius_scale": blockade_radius_scale,
        "blockade_over_spacing_scale": blockade_radius_scale / min_spacing,
        "min_distance_scale": float(space.geometry.spacing_um.max_val),
        "sweep_span_scale": sweep_span_scale,
    }


def _coordinates(layout: str, atom_count: int, spacing_um: float) -> list[tuple[float, float]]:
    if layout == "line":
        return [(float(index) * spacing_um, 0.0) for index in range(atom_count)]
    if layout == "square":
        side = int(math.ceil(math.sqrt(atom_count)))
        return [
            (float(index % side) * spacing_um, float(index // side) * spacing_um)
            for index in range(atom_count)
        ]
    if layout == "triangular":
        return [
            (
                float(index % 3) * spacing_um + (0.5 * spacing_um if (index // 3) % 2 else 0.0),
                float(index // 3) * spacing_um * math.sqrt(3.0) / 2.0,
            )
            for index in range(atom_count)
        ]
    if layout == "ring":
        radius = spacing_um / (2.0 * math.sin(math.pi / max(atom_count, 2)))
        return [
            (
                radius * math.cos(2.0 * math.pi * index / atom_count),
                radius * math.sin(2.0 * math.pi * index / atom_count),
            )
            for index in range(atom_count)
        ]
    if layout == "zigzag":
        return [
            (float(index) * spacing_um * 0.75, float(index % 2) * spacing_um * 0.5)
            for index in range(atom_count)
        ]
    if layout == "honeycomb":
        return [
            (
                float(index % 4) * spacing_um * 0.75,
                float(index // 4) * spacing_um * math.sqrt(3.0) / 2.0,
            )
            for index in range(atom_count)
        ]
    raise AssertionError(f"Unsupported layout: {layout}")


def _min_distance_um(coordinates: list[tuple[float, float]]) -> float:
    min_distance = float("inf")
    for row_index in range(len(coordinates)):
        for col_index in range(row_index + 1, len(coordinates)):
            dx = coordinates[row_index][0] - coordinates[col_index][0]
            dy = coordinates[row_index][1] - coordinates[col_index][1]
            min_distance = min(min_distance, math.sqrt(dx * dx + dy * dy))
    return min_distance if min_distance != float("inf") else 0.0


def _blockade_pairs(
    coordinates: list[tuple[float, float]],
    blockade_radius_um: float,
) -> int:
    pairs = 0
    for row_index in range(len(coordinates)):
        for col_index in range(row_index + 1, len(coordinates)):
            dx = coordinates[row_index][0] - coordinates[col_index][0]
            dy = coordinates[row_index][1] - coordinates[col_index][1]
            if math.sqrt(dx * dx + dy * dy) <= blockade_radius_um:
                pairs += 1
    return pairs


def _sample_register_and_sequence(
    space: PhysicsParameterSpace,
    rng: np.random.Generator,
) -> tuple[RegisterCandidate, SequenceCandidate]:
    family = list(SequenceFamily)[int(rng.integers(0, len(SequenceFamily)))]
    atom_count = int(space.geometry.atom_count.sample(rng))
    spacing_um = float(space.geometry.spacing_um.sample(rng))
    config = space.sample_pulse_config(family, atom_count, rng)
    layout = list(LAYOUT_ENCODING)[int(rng.integers(0, len(LAYOUT_ENCODING)))]
    coordinates = _coordinates(layout, atom_count, spacing_um)
    blockade_radius_um = float((space.c6_coefficient / max(float(config["amplitude"]), 1e-6)) ** (1.0 / 6.0))
    min_distance_um = _min_distance_um(coordinates)
    register = RegisterCandidate(
        campaign_id="campaign_norm",
        spec_id="spec_norm",
        label=f"{layout}-{atom_count}",
        layout_type=layout,
        atom_count=atom_count,
        coordinates=coordinates,
        min_distance_um=min_distance_um,
        blockade_radius_um=blockade_radius_um,
        blockade_pair_count=_blockade_pairs(coordinates, blockade_radius_um),
        feasibility_score=float(rng.uniform(0.35, 1.0)),
        reasoning_summary="Feature normalization sample register.",
        metadata={"spacing_um": spacing_um},
    )
    sequence_metadata: dict[str, float] = {}
    if "detuning_end" in config:
        sequence_metadata["detuning_end"] = float(config["detuning_end"])
    if "amplitude_start" in config:
        sequence_metadata["amplitude_start"] = float(config["amplitude_start"])
    sequence = SequenceCandidate(
        campaign_id="campaign_norm",
        spec_id="spec_norm",
        register_candidate_id=register.id,
        label=f"{family.value}-sample",
        sequence_family=family,
        duration_ns=int(config["duration_ns"]),
        amplitude=float(config["amplitude"]),
        detuning=float(config["detuning"]),
        phase=float(config["phase"]),
        waveform_kind=family.value,
        predicted_cost=space.cost_for(atom_count, int(config["duration_ns"])),
        reasoning_summary="Feature normalization sample sequence.",
        metadata=sequence_metadata,
    )
    return register, sequence


def test_feature_normalization_constants_match_parameter_space(default_param_space: PhysicsParameterSpace) -> None:
    expected = _expected_feature_normalization(default_param_space)
    actual = feature_normalization_constants(default_param_space)

    for name, expected_value in expected.items():
        assert actual[name] >= expected_value - 1e-9, (
            f"{name}={actual[name]!r} is smaller than its derived physical maximum {expected_value!r}."
        )
        assert actual[name] == pytest.approx(expected_value, rel=1e-9, abs=1e-9), (
            f"{name}={actual[name]!r} does not match its derivation source {expected_value!r}."
        )

    assert FEATURE_NORMALIZATION_DEFAULTS == pytest.approx(actual)


def test_build_feature_vector_v2_stays_in_unit_interval_for_sampled_parameter_space() -> None:
    space = PhysicsParameterSpace.default()
    rng = np.random.default_rng(7)

    for _ in range(1000):
        register, sequence = _sample_register_and_sequence(space, rng)
        features = build_feature_vector_v2(register, sequence, space)

        assert features.shape == (INPUT_DIM_V2,)
        assert np.all(np.isfinite(features))
        assert np.all(features >= -1e-6), features
        assert np.all(features <= 1.0 + 1e-6), features
