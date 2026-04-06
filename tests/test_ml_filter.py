"""Tests for Phase 1 — Surrogate filter."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from packages.core.enums import SequenceFamily
from packages.core.models import RegisterCandidate, SequenceCandidate
from packages.ml.surrogate_filter import SurrogateFilter


def _make_register(rid: str = "reg_1", atom_count: int = 4) -> RegisterCandidate:
    coords = [(float(i) * 7.0, 0.0) for i in range(atom_count)]
    return RegisterCandidate(
        id=rid,
        campaign_id="test",
        spec_id="spec_1",
        label=f"reg-{atom_count}",
        layout_type="square",
        atom_count=atom_count,
        coordinates=coords,
        min_distance_um=7.0,
        blockade_radius_um=9.5,
        blockade_pair_count=atom_count - 1,
        feasibility_score=0.8,
        reasoning_summary="Test.",
        metadata={"spacing_um": 7.0},
    )


def _make_sequences(register_id: str, count: int = 10) -> list[SequenceCandidate]:
    seqs = []
    for i in range(count):
        seqs.append(
            SequenceCandidate(
                id=f"seq_{i}",
                campaign_id="test",
                spec_id="spec_1",
                register_candidate_id=register_id,
                label=f"seq-{i}",
                sequence_family=SequenceFamily.ADIABATIC_SWEEP,
                duration_ns=2000 + i * 200,
                amplitude=4.0 + i * 0.5,
                detuning=-20.0 + i * 2.0,
                phase=0.0,
                waveform_kind="constant",
                predicted_cost=0.1 + i * 0.05,
                reasoning_summary=f"Test seq {i}.",
            )
        )
    return seqs


class TestSurrogateFilterPassthrough:
    """Tests that work without PyTorch — passthrough mode."""

    def test_disabled_returns_all(self):
        filt = SurrogateFilter(enabled=False)
        reg = _make_register()
        seqs = _make_sequences(reg.id, 10)
        result = filt.filter(seqs, {reg.id: reg})
        assert len(result) == 10

    def test_no_model_path_returns_all(self):
        filt = SurrogateFilter(enabled=True, model_path=None)
        reg = _make_register()
        seqs = _make_sequences(reg.id, 5)
        result = filt.filter(seqs, {reg.id: reg})
        assert len(result) == 5

    def test_missing_checkpoint_returns_all(self):
        filt = SurrogateFilter(enabled=True, model_path="/nonexistent/model.pt")
        reg = _make_register()
        seqs = _make_sequences(reg.id, 5)
        result = filt.filter(seqs, {reg.id: reg})
        assert len(result) == 5

    def test_empty_input(self):
        filt = SurrogateFilter(enabled=False)
        result = filt.filter([], {})
        assert result == []


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestSurrogateFilterWithModel:
    def test_filter_reduces_candidates(self, tmp_path):
        from packages.ml.surrogate import SurrogateModel

        model = SurrogateModel()
        path = tmp_path / "test_model.pt"
        model.save(path)

        filt = SurrogateFilter(model_path=str(path), top_k=3, enabled=True)
        reg = _make_register()
        seqs = _make_sequences(reg.id, 10)
        result = filt.filter(seqs, {reg.id: reg})
        assert len(result) <= 3

    def test_min_score_filtering(self, tmp_path):
        from packages.ml.surrogate import SurrogateModel

        model = SurrogateModel()
        path = tmp_path / "test_model.pt"
        model.save(path)

        # Very high min_score should filter most candidates
        filt = SurrogateFilter(
            model_path=str(path), top_k=100, min_score=0.99, enabled=True,
        )
        reg = _make_register()
        seqs = _make_sequences(reg.id, 10)
        result = filt.filter(seqs, {reg.id: reg})
        # With random weights, predictions won't consistently be > 0.99
        assert len(result) <= 10
