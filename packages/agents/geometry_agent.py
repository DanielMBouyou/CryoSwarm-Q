from __future__ import annotations

from math import ceil

import numpy as np

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName
from packages.core.models import ExperimentSpec, MemoryRecord, RegisterCandidate
from packages.core.parameter_space import PhysicsParameterSpace
from packages.pasqal_adapters.pulser_adapter import create_simple_register, summarize_register_physics


class GeometryAgent(BaseAgent):
    agent_name = AgentName.GEOMETRY

    def __init__(self, param_space: PhysicsParameterSpace | None = None) -> None:
        super().__init__()
        self.param_space = param_space or PhysicsParameterSpace.default()

    def run(
        self,
        spec: ExperimentSpec,
        campaign_id: str,
        memory_records: list[MemoryRecord] | None = None,
    ) -> list[RegisterCandidate]:
        atom_targets = self._atom_targets(spec.min_atoms, spec.max_atoms)
        layouts = self._layout_order(spec.preferred_layouts, memory_records or [])
        spacing_values = self._spacing_values(memory_records or [])
        candidates: list[RegisterCandidate] = []
        max_candidates = int(spec.metadata.get("max_register_candidates", 4))

        for atom_count in atom_targets:
            for layout in layouts:
                for spacing in spacing_values:
                    coordinates = self._coordinates_for_layout(layout, atom_count, spacing)
                    physics = summarize_register_physics(coordinates)
                    if not physics.valid:
                        continue
                    feasibility = self._feasibility_score(atom_count, physics.min_distance_um, physics.blockade_pair_count)
                    candidates.append(
                        RegisterCandidate(
                            campaign_id=campaign_id,
                            spec_id=spec.id,
                            label=f"{layout}-{atom_count}-s{spacing:.1f}",
                            layout_type=layout,
                            atom_count=atom_count,
                            coordinates=coordinates,
                            device_constraints={
                                "min_spacing_um": self.param_space.geometry.min_spacing_um.default,
                                "max_atoms_assumed": 80,
                            },
                            min_distance_um=physics.min_distance_um,
                            blockade_radius_um=physics.blockade_radius_um,
                            blockade_pair_count=physics.blockade_pair_count,
                            van_der_waals_matrix=physics.van_der_waals_matrix,
                            feasibility_score=feasibility,
                            reasoning_summary=(
                                f"Proposed a {layout} layout with {atom_count} atoms and {spacing:.1f} um spacing. "
                                f"Minimum spacing={physics.min_distance_um:.2f} um, blockade radius={physics.blockade_radius_um:.2f} um."
                            ),
                            pulser_register_summary=create_simple_register(coordinates),
                            metadata={"spacing_um": spacing},
                        )
                    )
                    if len(candidates) >= max_candidates:
                        return candidates
        return candidates

    def _atom_targets(self, min_atoms: int, max_atoms: int) -> list[int]:
        midpoint = ceil((min_atoms + max_atoms) / 2)
        min_bound = int(self.param_space.geometry.atom_count.min_val)
        max_bound = int(self.param_space.geometry.atom_count.max_val)
        targets = [min_atoms, midpoint, max_atoms]
        return list(dict.fromkeys([min(max(target, min_bound), max_bound) for target in targets]))

    def _layout_order(self, preferred_layouts: list[str], memory_records: list[MemoryRecord]) -> list[str]:
        remembered_layouts = [
            str(record.signals.get("layout_type"))
            for record in memory_records
            if record.signals.get("layout_type")
        ]
        ordered = remembered_layouts + preferred_layouts
        return list(dict.fromkeys(ordered))[:3]

    def _spacing_values(self, memory_records: list[MemoryRecord]) -> list[float]:
        remembered_spacings = []
        avoided_spacings = set()
        spacing_range = self.param_space.geometry.spacing_um
        min_spacing = self.param_space.geometry.min_spacing_um.default
        delta = 0.5
        for record in memory_records:
            spacing = record.signals.get("spacing_um")
            if spacing is None:
                continue
            spacing_value = round(spacing_range.clip(float(spacing)), 2)
            confidence = float(record.signals.get("confidence", 0.0))
            if record.lesson_type == "failure_pattern" and "high_noise_sensitivity" in record.reusable_tags:
                avoided_spacings.add(spacing_value)
                continue
            remembered_spacings.append(spacing_value)
            if confidence > 0.7:
                remembered_spacings.extend(
                    [
                        round(max(min_spacing, spacing_value - delta), 2),
                        round(spacing_range.clip(spacing_value + delta), 2),
                    ]
                )
        baseline = [
            round(spacing_range.default, 2),
            round(spacing_range.clip(spacing_range.default + 1.5), 2),
        ]
        ordered = remembered_spacings + baseline
        unique = [round(value, 2) for value in ordered if round(value, 2) not in avoided_spacings]
        return list(dict.fromkeys(unique))[:4]

    def _feasibility_score(self, atom_count: int, min_distance_um: float, blockade_pair_count: int) -> float:
        geometry = self.param_space.geometry
        spacing_margin = min(min_distance_um / max(geometry.min_spacing_um.default, 1e-6), 1.5)
        blockade_bonus = min(blockade_pair_count / max(atom_count - 1, 1), 1.0)
        atom_penalty = max(atom_count - int(geometry.atom_count.default), 0) * geometry.feasibility_atom_penalty.default
        score = (
            geometry.feasibility_base.default
            + geometry.feasibility_spacing_weight.default * spacing_margin
            + geometry.feasibility_blockade_weight.default * blockade_bonus
            - atom_penalty
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _coordinates_for_layout(self, layout: str, atom_count: int, spacing: float) -> list[tuple[float, float]]:
        if layout == "square":
            side = ceil(atom_count**0.5)
            coordinates: list[tuple[float, float]] = []
            for row in range(side):
                for col in range(side):
                    if len(coordinates) == atom_count:
                        return coordinates
                    coordinates.append((float(col) * spacing, float(row) * spacing))
            return coordinates

        if layout == "triangular":
            coordinates = []
            row = 0
            while len(coordinates) < atom_count:
                for col in range(row + 1):
                    if len(coordinates) == atom_count:
                        break
                    coordinates.append((col * spacing + row * 0.5, row * spacing * 0.86))
                row += 1
            return coordinates

        if layout == "zigzag":
            return [
                (float(index) * spacing, 0.0 if index % 2 == 0 else spacing * 0.5)
                for index in range(atom_count)
            ]

        if layout == "ring":
            radius = spacing / (2.0 * np.sin(np.pi / max(atom_count, 2)))
            import math

            return [
                (
                    round(radius * math.cos(2 * math.pi * k / atom_count), 6),
                    round(radius * math.sin(2 * math.pi * k / atom_count), 6),
                )
                for k in range(atom_count)
            ]

        if layout == "honeycomb":
            coordinates = []
            cols = max(2, int(atom_count**0.5) + 1)
            for row_idx in range(atom_count):
                r = row_idx // cols
                c = row_idx % cols
                x = c * spacing + (0.5 * spacing if r % 2 else 0.0)
                y = r * spacing * 0.866
                coordinates.append((round(x, 6), round(y, 6)))
                if len(coordinates) == atom_count:
                    return coordinates
            return coordinates

        return [(float(index) * spacing, 0.0) for index in range(atom_count)]
