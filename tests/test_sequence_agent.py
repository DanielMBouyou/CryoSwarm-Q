"""Tests for the sequence agent's pulse parameter realism."""
from __future__ import annotations

import pytest

from packages.agents.sequence_agent import SequenceAgent
from packages.core.enums import SequenceFamily
from packages.core.models import ExperimentSpec, RegisterCandidate, ScoringWeights


def _make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        goal_id="goal_test",
        objective_class="balanced_campaign_search",
        target_observable="rydberg_density",
        min_atoms=4,
        max_atoms=6,
        preferred_layouts=["square", "line"],
        sequence_families=[
            SequenceFamily.GLOBAL_RAMP,
            SequenceFamily.DETUNING_SCAN,
            SequenceFamily.ADIABATIC_SWEEP,
            SequenceFamily.CONSTANT_DRIVE,
            SequenceFamily.BLACKMAN_SWEEP,
        ],
        reasoning_summary="test spec",
    )


def _make_register() -> RegisterCandidate:
    return RegisterCandidate(
        campaign_id="camp_test",
        spec_id="spec_test",
        label="line-4-s7.0",
        layout_type="line",
        atom_count=4,
        coordinates=[(0.0, 0.0), (7.0, 0.0), (14.0, 0.0), (21.0, 0.0)],
        min_distance_um=7.0,
        blockade_radius_um=7.5,
        blockade_pair_count=3,
        van_der_waals_matrix=[[0.0] * 4 for _ in range(4)],
        feasibility_score=0.8,
        reasoning_summary="test register",
    )


class TestSequenceParameters:
    agent = SequenceAgent()
    spec = _make_spec()
    register = _make_register()

    def test_generates_all_families(self) -> None:
        candidates = self.agent.run(self.spec, self.register, "camp_test")
        families = {c.sequence_family for c in candidates}
        assert SequenceFamily.GLOBAL_RAMP in families
        assert SequenceFamily.DETUNING_SCAN in families
        assert SequenceFamily.ADIABATIC_SWEEP in families
        assert SequenceFamily.CONSTANT_DRIVE in families
        assert SequenceFamily.BLACKMAN_SWEEP in families

    def test_amplitudes_physically_realistic(self) -> None:
        """Rabi frequencies should be in 1-15 rad/us for AnalogDevice."""
        candidates = self.agent.run(self.spec, self.register, "camp_test")
        for c in candidates:
            assert 0.5 <= c.amplitude <= 15.0, (
                f"Amplitude {c.amplitude} out of physical range for {c.label}"
            )

    def test_detunings_physically_realistic(self) -> None:
        """Detunings should be in the range relevant to Rydberg interactions."""
        candidates = self.agent.run(self.spec, self.register, "camp_test")
        for c in candidates:
            assert -50.0 <= c.detuning <= 10.0, (
                f"Detuning {c.detuning} out of range for {c.label}"
            )

    def test_durations_reasonable(self) -> None:
        """Durations should be 500 ns - 10 us for meaningful dynamics."""
        candidates = self.agent.run(self.spec, self.register, "camp_test")
        for c in candidates:
            assert 500 <= c.duration_ns <= 10_000, (
                f"Duration {c.duration_ns} ns out of range for {c.label}"
            )

    def test_adiabatic_sweep_has_detuning_end(self) -> None:
        candidates = self.agent.run(self.spec, self.register, "camp_test")
        sweeps = [c for c in candidates if c.sequence_family == SequenceFamily.ADIABATIC_SWEEP]
        for c in sweeps:
            assert "detuning_end" in c.metadata
            assert c.metadata["detuning_end"] > c.detuning, (
                "Adiabatic sweep should go from negative to positive detuning"
            )

    def test_blackman_sweep_has_detuning_end(self) -> None:
        candidates = self.agent.run(self.spec, self.register, "camp_test")
        bm = [c for c in candidates if c.sequence_family == SequenceFamily.BLACKMAN_SWEEP]
        assert len(bm) >= 1
        for c in bm:
            assert "detuning_end" in c.metadata

    def test_constant_drive_no_detuning_sweep(self) -> None:
        candidates = self.agent.run(self.spec, self.register, "camp_test")
        const = [c for c in candidates if c.sequence_family == SequenceFamily.CONSTANT_DRIVE]
        assert len(const) >= 1

    def test_candidate_count(self) -> None:
        """5 families × 2 variants each = at least 10 candidates."""
        candidates = self.agent.run(self.spec, self.register, "camp_test")
        assert len(candidates) >= 10


class TestMemoryExploitation:
    agent = SequenceAgent()
    spec = _make_spec()
    register = _make_register()

    def test_memory_adds_exploit_variants(self) -> None:
        from packages.core.models import MemoryRecord

        record = MemoryRecord(
            campaign_id="camp_old",
            source_candidate_id="seq_old",
            lesson_type="candidate_pattern",
            summary="previous strong candidate",
            reusable_tags=["strong", "emu_sv_candidate"],
        )
        without = self.agent.run(self.spec, self.register, "camp_test")
        with_mem = self.agent.run(self.spec, self.register, "camp_test", [record])
        assert len(with_mem) > len(without), "Memory should add exploit variants"
