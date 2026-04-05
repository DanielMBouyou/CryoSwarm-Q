"""Tests for the geometry agent's physics-aware register generation."""
from __future__ import annotations

import math

import pytest

from packages.agents.geometry_agent import GeometryAgent
from packages.core.enums import SequenceFamily
from packages.core.models import ExperimentSpec, ScoringWeights


def _make_spec(
    min_atoms: int = 4,
    max_atoms: int = 6,
    layouts: list[str] | None = None,
    max_candidates: int = 8,
) -> ExperimentSpec:
    return ExperimentSpec(
        goal_id="goal_test",
        objective_class="balanced_campaign_search",
        target_observable="rydberg_density",
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        preferred_layouts=layouts or ["square", "line", "triangular"],
        sequence_families=[SequenceFamily.ADIABATIC_SWEEP],
        reasoning_summary="test spec",
        metadata={"max_register_candidates": max_candidates},
    )


class TestGeometryLayouts:
    agent = GeometryAgent()

    def test_square_coordinate_count(self) -> None:
        coords = self.agent._coordinates_for_layout("square", 4, 7.0)
        assert len(coords) == 4

    def test_triangular_coordinate_count(self) -> None:
        coords = self.agent._coordinates_for_layout("triangular", 6, 7.0)
        assert len(coords) == 6

    def test_line_is_1d(self) -> None:
        coords = self.agent._coordinates_for_layout("line", 5, 7.0)
        ys = {c[1] for c in coords}
        assert len(ys) == 1, "Line layout should be along one axis"

    def test_ring_forms_circle(self) -> None:
        coords = self.agent._coordinates_for_layout("ring", 6, 7.0)
        assert len(coords) == 6
        cx = sum(x for x, _ in coords) / 6
        cy = sum(y for _, y in coords) / 6
        radii = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in coords]
        assert max(radii) - min(radii) < 0.01, "Ring atoms should be equidistant from centre"

    def test_honeycomb_coordinate_count(self) -> None:
        coords = self.agent._coordinates_for_layout("honeycomb", 6, 7.0)
        assert len(coords) == 6

    def test_zigzag_alternates_y(self) -> None:
        coords = self.agent._coordinates_for_layout("zigzag", 4, 7.0)
        ys = [c[1] for c in coords]
        assert ys[0] != ys[1], "Zigzag should alternate y-positions"

    def test_spacing_preserved(self) -> None:
        """Nearest-neighbour distance in a line should match spacing."""
        coords = self.agent._coordinates_for_layout("line", 3, 8.0)
        d01 = math.dist(coords[0], coords[1])
        assert d01 == pytest.approx(8.0, abs=0.01)


class TestCandidateGeneration:
    agent = GeometryAgent()

    def test_returns_candidates(self) -> None:
        spec = _make_spec()
        candidates = self.agent.run(spec, "camp_test")
        assert len(candidates) >= 1

    def test_candidate_fields_populated(self) -> None:
        spec = _make_spec(max_candidates=2)
        candidates = self.agent.run(spec, "camp_test")
        for c in candidates:
            assert c.atom_count >= spec.min_atoms
            assert c.atom_count <= spec.max_atoms
            assert len(c.coordinates) == c.atom_count
            assert c.blockade_radius_um > 0
            assert len(c.van_der_waals_matrix) == c.atom_count
            assert 0.0 <= c.feasibility_score <= 1.0

    def test_respects_max_candidates(self) -> None:
        spec = _make_spec(max_candidates=3)
        candidates = self.agent.run(spec, "camp_test")
        assert len(candidates) <= 3
