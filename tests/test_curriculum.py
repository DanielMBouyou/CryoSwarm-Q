"""Tests for curriculum learning scheduler."""
from __future__ import annotations

from packages.core.models import RegisterCandidate
from packages.ml.curriculum import CurriculumScheduler


def _make_register(atom_count: int, layout: str) -> RegisterCandidate:
    return RegisterCandidate(
        campaign_id="test",
        spec_id="spec",
        label=f"{layout}-{atom_count}",
        layout_type=layout,
        atom_count=atom_count,
        coordinates=[(float(index) * 7.0, 0.0) for index in range(atom_count)],
        min_distance_um=7.0,
        blockade_radius_um=9.5,
        blockade_pair_count=max(1, atom_count - 1),
        feasibility_score=0.8,
        reasoning_summary="Test register.",
        metadata={"spacing_um": 7.0},
    )


def test_curriculum_stage_progression():
    scheduler = CurriculumScheduler(mode="adaptive", total_updates=100)
    assert scheduler.current_stage.name == "warm-up"
    for _ in range(60):
        scheduler.record_episode(0.4)
    scheduler.step_update()
    assert scheduler.current_stage.name == "expansion"


def test_curriculum_filters_registers():
    scheduler = CurriculumScheduler()
    registers = [
        _make_register(atom_count=4, layout="square"),
        _make_register(atom_count=10, layout="ring"),
        _make_register(atom_count=3, layout="line"),
    ]
    filtered = scheduler.filter_registers(registers)
    assert len(filtered) == 2
    assert all(register.atom_count <= 5 for register in filtered)


def test_curriculum_fallback_on_no_match():
    scheduler = CurriculumScheduler()
    registers = [_make_register(atom_count=12, layout="honeycomb")]
    filtered = scheduler.filter_registers(registers)
    assert len(filtered) == 1
