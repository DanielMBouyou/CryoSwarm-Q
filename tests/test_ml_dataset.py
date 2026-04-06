"""Tests for Phase 1 — dataset builder and feature extraction."""
from __future__ import annotations

import numpy as np
import pytest

from packages.core.enums import SequenceFamily
from packages.core.models import (
    EvaluationResult,
    RegisterCandidate,
    RobustnessReport,
    SequenceCandidate,
)
from packages.ml.dataset import (
    INPUT_DIM,
    OUTPUT_DIM,
    CandidateDatasetBuilder,
    build_feature_vector,
    build_target_vector,
    _encode_family,
    _encode_layout,
)


# ---- fixtures ----


def _make_register(layout: str = "square", atom_count: int = 4) -> RegisterCandidate:
    coords = [(float(i) * 7.0, 0.0) for i in range(atom_count)]
    return RegisterCandidate(
        campaign_id="test",
        spec_id="spec_1",
        label=f"reg-{layout}-{atom_count}",
        layout_type=layout,
        atom_count=atom_count,
        coordinates=coords,
        min_distance_um=7.0,
        blockade_radius_um=9.5,
        blockade_pair_count=atom_count - 1,
        feasibility_score=0.85,
        reasoning_summary="Test register.",
        metadata={"spacing_um": 7.0},
    )


def _make_sequence(
    register_id: str = "reg_1",
    family: SequenceFamily = SequenceFamily.ADIABATIC_SWEEP,
) -> SequenceCandidate:
    return SequenceCandidate(
        campaign_id="test",
        spec_id="spec_1",
        register_candidate_id=register_id,
        label="seq-test",
        sequence_family=family,
        duration_ns=3000,
        amplitude=5.0,
        detuning=-15.0,
        phase=0.0,
        waveform_kind="constant",
        predicted_cost=0.2,
        reasoning_summary="Test sequence.",
    )


def _make_report(seq_id: str = "seq_1") -> RobustnessReport:
    return RobustnessReport(
        campaign_id="test",
        sequence_candidate_id=seq_id,
        nominal_score=0.75,
        perturbation_average=0.60,
        robustness_penalty=0.15,
        robustness_score=0.65,
        worst_case_score=0.50,
        score_std=0.08,
        target_observable="rydberg_density",
        reasoning_summary="Test report.",
    )


def _make_evaluation(seq_id: str = "seq_1", reg_id: str = "reg_1") -> EvaluationResult:
    return EvaluationResult(
        campaign_id="test",
        sequence_candidate_id=seq_id,
        register_candidate_id=reg_id,
        nominal_score=0.75,
        robustness_score=0.65,
        worst_case_score=0.50,
        observable_score=0.70,
        objective_score=0.55,
        backend_choice="local_pulser_simulation",
        estimated_cost=0.2,
        estimated_latency=0.1,
        reasoning_summary="Test eval.",
    )


# ---- encoding tests ----


class TestEncoding:
    def test_layout_encoding_known(self):
        assert _encode_layout("square") == 0
        assert _encode_layout("triangular") == 2
        assert _encode_layout("honeycomb") == 5

    def test_layout_encoding_unknown(self):
        assert _encode_layout("unknown_layout") == len({"square": 0, "line": 1, "triangular": 2, "ring": 3, "zigzag": 4, "honeycomb": 5})

    def test_family_encoding(self):
        for i, fam in enumerate(SequenceFamily):
            assert _encode_family(fam.value) == i


# ---- feature vector tests ----


class TestFeatureVector:
    def test_shape(self):
        reg = _make_register()
        seq = _make_sequence(reg.id)
        vec = build_feature_vector(reg, seq)
        assert vec.shape == (INPUT_DIM,)
        assert vec.dtype == np.float32

    def test_values(self):
        reg = _make_register("line", 6)
        seq = _make_sequence(reg.id, SequenceFamily.DETUNING_SCAN)
        vec = build_feature_vector(reg, seq)
        assert vec[0] == 6.0  # atom_count
        assert vec[1] == 7.0  # spacing
        assert vec[2] == 5.0  # amplitude
        assert vec[3] == -15.0  # detuning
        assert vec[4] == 3000.0  # duration
        assert vec[5] == float(_encode_layout("line"))
        assert vec[6] == float(_encode_family("detuning_scan"))

    def test_target_shape(self):
        report = _make_report()
        evaluation = _make_evaluation()
        vec = build_target_vector(report, evaluation)
        assert vec.shape == (OUTPUT_DIM,)
        assert vec[0] == pytest.approx(0.65)  # robustness_score
        assert vec[1] == pytest.approx(0.75)  # nominal_score


# ---- dataset builder tests ----


class TestDatasetBuilder:
    def test_empty(self):
        builder = CandidateDatasetBuilder()
        assert builder.size == 0
        X, Y = builder.to_numpy()
        assert X.shape == (0, INPUT_DIM)
        assert Y.shape == (0, OUTPUT_DIM)

    def test_add_sample(self):
        builder = CandidateDatasetBuilder()
        reg = _make_register()
        seq = _make_sequence(reg.id)
        report = _make_report(seq.id)
        evaluation = _make_evaluation(seq.id, reg.id)
        builder.add_sample(reg, seq, report, evaluation)
        assert builder.size == 1

    def test_add_from_pipeline(self):
        reg = _make_register()
        seq = _make_sequence(reg.id)
        report = _make_report(seq.id)
        evaluation = _make_evaluation(seq.id, reg.id)

        builder = CandidateDatasetBuilder()
        added = builder.add_from_pipeline([reg], [seq], [report], [evaluation])
        assert added == 1
        assert builder.size == 1

        X, Y = builder.to_numpy()
        assert X.shape == (1, INPUT_DIM)
        assert Y.shape == (1, OUTPUT_DIM)

    def test_save_load(self, tmp_path):
        reg = _make_register()
        seq = _make_sequence(reg.id)
        report = _make_report(seq.id)
        evaluation = _make_evaluation(seq.id, reg.id)

        builder = CandidateDatasetBuilder()
        builder.add_sample(reg, seq, report, evaluation)
        path = str(tmp_path / "test_data.npz")
        builder.save(path)

        builder2 = CandidateDatasetBuilder()
        builder2.load(path)
        assert builder2.size == 1

        X1, Y1 = builder.to_numpy()
        X2, Y2 = builder2.to_numpy()
        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_almost_equal(Y1, Y2)
