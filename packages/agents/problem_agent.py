from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.models import MemoryRecord
from packages.core.enums import AgentName, SequenceFamily
from packages.core.models import ExperimentGoal, ExperimentSpec


class ProblemFramingAgent(BaseAgent):
    agent_name = AgentName.PROBLEM_FRAMING

    def run(self, goal: ExperimentGoal, memory_records: list[MemoryRecord] | None = None) -> ExperimentSpec:
        objective_text = f"{goal.title} {goal.scientific_objective}".lower()

        if "robust" in objective_text or "noise" in objective_text:
            objective_class = "robust_neutral_atom_search"
        elif "geometry" in objective_text or "layout" in objective_text:
            objective_class = "geometry_exploration"
        else:
            objective_class = "balanced_campaign_search"

        preferred_layouts = self._resolve_layouts(goal.preferred_geometry)
        desired_atoms = max(goal.desired_atom_count, 3)
        min_atoms = max(3, desired_atoms - 2)
        max_atoms = desired_atoms + 2
        target_density = 0.5
        if "single excitation" in objective_text:
            target_density = 1.0 / max(desired_atoms, 1)

        recent_memory = memory_records or []
        remembered_backends = [record.signals.get("backend_choice") for record in recent_memory]

        reasoning_summary = (
            f"Framed '{goal.title}' as {objective_class} with {min_atoms}-{max_atoms} atoms, "
            f"target observable '{goal.target_observable}', and layouts {preferred_layouts}."
        )

        return ExperimentSpec(
            goal_id=goal.id,
            objective_class=objective_class,
            target_observable=goal.target_observable,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            preferred_layouts=preferred_layouts,
            sequence_families=[
                SequenceFamily.GLOBAL_RAMP,
                SequenceFamily.DETUNING_SCAN,
                SequenceFamily.ADIABATIC_SWEEP,
            ],
            target_density=target_density,
            scoring_weights=goal.priority_weights,
            perturbation_budget=3,
            latency_budget=0.30 if goal.priority == "balanced" else 0.20,
            reasoning_summary=reasoning_summary,
            metadata={
                "priority": goal.priority,
                "goal_constraints": goal.constraints,
                "memory_record_count": len(recent_memory),
                "remembered_backends": remembered_backends,
                "max_register_candidates": 2,
            },
        )

    def _resolve_layouts(self, preferred_geometry: str) -> list[str]:
        geometry = preferred_geometry.lower()
        if geometry in {"square", "2d"}:
            return ["square", "triangular", "line"]
        if geometry in {"line", "1d"}:
            return ["line", "zigzag", "square"]
        return ["square", "line", "triangular"]
