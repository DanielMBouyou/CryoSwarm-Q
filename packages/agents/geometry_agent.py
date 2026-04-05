from __future__ import annotations

from math import ceil

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName
from packages.core.models import ExperimentSpec, RegisterCandidate
from packages.pasqal_adapters.pulser_adapter import create_simple_register


class GeometryAgent(BaseAgent):
    agent_name = AgentName.GEOMETRY

    def run(self, spec: ExperimentSpec, campaign_id: str) -> list[RegisterCandidate]:
        atom_targets = self._atom_targets(spec.min_atoms, spec.max_atoms)
        layouts = spec.preferred_layouts[:3]
        candidates: list[RegisterCandidate] = []

        for index, atom_count in enumerate(atom_targets):
            layout = layouts[index % len(layouts)]
            coordinates = self._coordinates_for_layout(layout, atom_count)
            feasibility = round(max(0.55, 0.90 - atom_count * 0.025), 4)
            candidates.append(
                RegisterCandidate(
                    campaign_id=campaign_id,
                    spec_id=spec.id,
                    label=f"{layout}-{atom_count}",
                    layout_type=layout,
                    atom_count=atom_count,
                    coordinates=coordinates,
                    device_constraints={
                        "min_spacing_um": 5.0,
                        "max_atoms_assumed": 24,
                    },
                    feasibility_score=feasibility,
                    reasoning_summary=(
                        f"Proposed a {layout} layout with {atom_count} atoms as a plausible "
                        "neutral-atom register candidate."
                    ),
                    pulser_register_summary=create_simple_register(coordinates),
                    metadata={"layout_rank": index + 1},
                )
            )
        return candidates

    def _atom_targets(self, min_atoms: int, max_atoms: int) -> list[int]:
        midpoint = ceil((min_atoms + max_atoms) / 2)
        return list(dict.fromkeys([min_atoms, midpoint, max_atoms]))

    def _coordinates_for_layout(self, layout: str, atom_count: int) -> list[tuple[float, float]]:
        if layout == "square":
            side = ceil(atom_count ** 0.5)
            coordinates: list[tuple[float, float]] = []
            for row in range(side):
                for col in range(side):
                    if len(coordinates) == atom_count:
                        return coordinates
                    coordinates.append((float(col) * 6.0, float(row) * 6.0))
            return coordinates

        if layout == "triangular":
            coordinates = []
            row = 0
            spacing = 5.5
            while len(coordinates) < atom_count:
                for col in range(row + 1):
                    if len(coordinates) == atom_count:
                        break
                    coordinates.append((col * spacing + row * 0.5, row * spacing * 0.86))
                row += 1
            return coordinates

        if layout == "zigzag":
            return [
                (float(index) * 5.5, 0.0 if index % 2 == 0 else 3.0)
                for index in range(atom_count)
            ]

        return [(float(index) * 5.5, 0.0) for index in range(atom_count)]
