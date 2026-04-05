from packages.agents.problem_agent import ProblemFramingAgent
from packages.core.models import ExperimentGoal


def test_problem_agent_returns_structured_spec() -> None:
    agent = ProblemFramingAgent()
    goal = ExperimentGoal(
        title="Robust geometry exploration",
        scientific_objective="Search for a robust neutral-atom layout under perturbations.",
        desired_atom_count=7,
        preferred_geometry="square",
    )

    spec = agent.run(goal)

    assert spec.goal_id == goal.id
    assert spec.objective_class == "robust_neutral_atom_search"
    assert spec.min_atoms <= spec.max_atoms
    assert "square" in spec.preferred_layouts
