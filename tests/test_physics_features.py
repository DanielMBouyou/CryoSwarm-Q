"""Validate that physics-informed features are physically meaningful."""
from __future__ import annotations

import math

import numpy as np

from packages.core.enums import SequenceFamily
from packages.core.models import RegisterCandidate, SequenceCandidate
from packages.ml.dataset import build_feature_vector_v2


def _coordinates(layout: str, atom_count: int, spacing_um: float) -> list[tuple[float, float]]:
    if layout == "ring":
        radius = spacing_um / (2.0 * math.sin(math.pi / max(atom_count, 3)))
        return [
            (
                radius * math.cos(2 * math.pi * index / atom_count),
                radius * math.sin(2 * math.pi * index / atom_count),
            )
            for index in range(atom_count)
        ]
    if layout == "triangular":
        return [
            (float(index % 3) * spacing_um, float(index // 3) * spacing_um * math.sqrt(3) / 2.0)
            for index in range(atom_count)
        ]
    if layout == "zigzag":
        return [
            (float(index) * spacing_um * 0.75, float(index % 2) * spacing_um * 0.5)
            for index in range(atom_count)
        ]
    if layout == "honeycomb":
        return [
            (float(index % 4) * spacing_um * 0.75, float(index // 4) * spacing_um * 0.65)
            for index in range(atom_count)
        ]
    if layout == "square":
        side = int(math.ceil(math.sqrt(atom_count)))
        return [
            (float(index % side) * spacing_um, float(index // side) * spacing_um)
            for index in range(atom_count)
        ]
    return [(float(index) * spacing_um, 0.0) for index in range(atom_count)]


def _make_register(layout: str = "square", spacing_um: float = 7.0, atom_count: int = 4) -> RegisterCandidate:
    blockade_pairs = atom_count if layout == "ring" else max(atom_count - 1, 1)
    return RegisterCandidate(
        campaign_id="test",
        spec_id="spec",
        label=f"{layout}-{atom_count}",
        layout_type=layout,
        atom_count=atom_count,
        coordinates=_coordinates(layout, atom_count, spacing_um),
        min_distance_um=spacing_um,
        blockade_radius_um=9.5,
        blockade_pair_count=blockade_pairs,
        feasibility_score=0.8,
        reasoning_summary="Test register.",
        metadata={"spacing_um": spacing_um},
    )


def _make_sequence(
    amplitude: float = 5.0,
    detuning: float = -10.0,
    duration_ns: int = 2000,
) -> SequenceCandidate:
    return SequenceCandidate(
        campaign_id="test",
        spec_id="spec",
        register_candidate_id="reg",
        label="seq",
        sequence_family=SequenceFamily.ADIABATIC_SWEEP,
        duration_ns=duration_ns,
        amplitude=amplitude,
        detuning=detuning,
        phase=0.0,
        waveform_kind="constant",
        predicted_cost=0.2,
        reasoning_summary="Test sequence.",
    )


def test_omega_over_interaction_scales_correctly():
    register = _make_register(spacing_um=5.0, atom_count=4)
    seq_high = _make_sequence(amplitude=10.0)
    seq_low = _make_sequence(amplitude=1.0)
    f_high = build_feature_vector_v2(register, seq_high)
    f_low = build_feature_vector_v2(register, seq_low)
    assert f_high[8] > f_low[8]


def test_adiabaticity_scales_with_duration():
    register = _make_register(spacing_um=7.0, atom_count=6)
    seq_long = _make_sequence(duration_ns=5000, amplitude=5.0)
    seq_short = _make_sequence(duration_ns=1000, amplitude=5.0)
    f_long = build_feature_vector_v2(register, seq_long)
    f_short = build_feature_vector_v2(register, seq_short)
    assert f_long[10] > f_short[10]


def test_blockade_fraction_ring_vs_line():
    reg_ring = _make_register(layout="ring", spacing_um=7.0, atom_count=6)
    reg_line = _make_register(layout="line", spacing_um=7.0, atom_count=6)
    seq = _make_sequence()
    f_ring = build_feature_vector_v2(reg_ring, seq)
    f_line = build_feature_vector_v2(reg_line, seq)
    assert f_ring[7] >= f_line[7]


def test_feature_v2_dimension():
    register = _make_register()
    sequence = _make_sequence()
    features = build_feature_vector_v2(register, sequence)
    assert features.shape == (18,)
    assert features.dtype == np.float32


def test_feature_v2_all_finite():
    for layout in ["square", "line", "triangular", "ring", "zigzag", "honeycomb"]:
        for atoms in [3, 6, 10, 15]:
            register = _make_register(layout=layout, atom_count=atoms)
            sequence = _make_sequence()
            features = build_feature_vector_v2(register, sequence)
            assert np.all(np.isfinite(features)), f"Non-finite features for {layout}/{atoms}"
