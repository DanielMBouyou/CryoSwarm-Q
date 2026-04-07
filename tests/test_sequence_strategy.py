from __future__ import annotations

from dataclasses import dataclass

from packages.agents.sequence_agent import SequenceAgent
from packages.agents.sequence_strategy import (
    BanditSelector,
    HeuristicGenerator,
    RLCandidateGenerator,
    SequenceStrategy,
    SequenceStrategyMode,
)
from packages.core.enums import SequenceFamily
from packages.core.models import ExperimentSpec, MemoryRecord, RegisterCandidate, SequenceCandidate
from packages.core.parameter_space import PhysicsParameterSpace


def _make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        goal_id="goal_test",
        objective_class="balanced_campaign_search",
        target_observable="rydberg_density",
        min_atoms=4,
        max_atoms=6,
        preferred_layouts=["square"],
        sequence_families=[SequenceFamily.ADIABATIC_SWEEP],
        reasoning_summary="test spec",
    )


def _make_register() -> RegisterCandidate:
    return RegisterCandidate(
        campaign_id="camp_test",
        spec_id="spec_test",
        label="square-4-s7.0",
        layout_type="square",
        atom_count=4,
        coordinates=[(0.0, 0.0), (7.0, 0.0), (0.0, 7.0), (7.0, 7.0)],
        min_distance_um=7.0,
        blockade_radius_um=9.0,
        blockade_pair_count=4,
        van_der_waals_matrix=[[0.0] * 4 for _ in range(4)],
        feasibility_score=0.8,
        reasoning_summary="register",
        metadata={"spacing_um": 7.0},
    )


@dataclass
class StubRLAgent:
    ready: bool = True

    def run(self, spec, register, campaign_id, memory_records=None):  # type: ignore[no-untyped-def]
        return [
            SequenceCandidate(
                campaign_id=campaign_id,
                spec_id=spec.id,
                register_candidate_id=register.id,
                label="rl-seq",
                sequence_family=SequenceFamily.ADIABATIC_SWEEP,
                duration_ns=3000,
                amplitude=6.0,
                detuning=-12.0,
                phase=0.0,
                waveform_kind="constant",
                predicted_cost=0.2,
                reasoning_summary="rl",
                metadata={"source": "rl_policy"},
            )
        ]


def test_strategy_falls_back_to_heuristic_without_ready_rl() -> None:
    strategy = SequenceStrategy(
        mode=SequenceStrategyMode.ADAPTIVE,
        heuristic_agent=SequenceAgent(),
        rl_agent=StubRLAgent(ready=False),  # type: ignore[arg-type]
    )
    selected = strategy.select_strategy(_make_spec(), _make_register(), [])
    assert selected == SequenceStrategyMode.HEURISTIC_ONLY


def test_hybrid_mode_merges_rl_and_heuristic_candidates() -> None:
    strategy = SequenceStrategy(
        mode=SequenceStrategyMode.HYBRID,
        heuristic_agent=SequenceAgent(),
        rl_agent=StubRLAgent(ready=True),  # type: ignore[arg-type]
    )
    candidates, meta = strategy.generate_candidates(_make_spec(), _make_register(), "camp_test", [])
    sources = {candidate.metadata.get("source") for candidate in candidates}
    assert "heuristic" in sources
    assert "rl_policy" in sources
    assert meta["strategy_used"] == SequenceStrategyMode.HYBRID.value


def test_rl_only_mode_uses_rl_candidates_when_ready() -> None:
    strategy = SequenceStrategy(
        mode=SequenceStrategyMode.RL_ONLY,
        heuristic_agent=SequenceAgent(),
        rl_agent=StubRLAgent(ready=True),  # type: ignore[arg-type]
    )
    candidates, meta = strategy.generate_candidates(_make_spec(), _make_register(), "camp_test", [])
    assert candidates
    assert all(candidate.metadata.get("source") == "rl_policy" for candidate in candidates)
    assert meta["strategy_used"] == SequenceStrategyMode.RL_ONLY.value


def test_bandit_updates_metrics_and_selects_known_strategies() -> None:
    strategy = SequenceStrategy(
        mode=SequenceStrategyMode.BANDIT,
        heuristic_agent=SequenceAgent(),
        rl_agent=StubRLAgent(ready=True),  # type: ignore[arg-type]
    )
    problem_class = strategy._compute_problem_class(_make_spec(), _make_register())
    strategy.update_performance(problem_class, SequenceStrategyMode.HEURISTIC_ONLY.value, [0.5, 0.55])
    strategy.update_performance(problem_class, SequenceStrategyMode.RL_ONLY.value, [0.8, 0.82])
    selected = strategy.select_strategy(_make_spec(), _make_register(), [])

    assert selected in {
        SequenceStrategyMode.HEURISTIC_ONLY,
        SequenceStrategyMode.RL_ONLY,
        SequenceStrategyMode.HYBRID,
    }
    report = strategy.get_strategy_report()
    assert report["metrics"]


def test_problem_class_uses_objective_atoms_and_layout() -> None:
    strategy = SequenceStrategy(
        heuristic_agent=SequenceAgent(),
        rl_agent=StubRLAgent(ready=False),  # type: ignore[arg-type]
    )
    problem_class = strategy._compute_problem_class(_make_spec(), _make_register())
    assert problem_class == "balanced_campaign_search_4atoms_square"


def test_component_generators_share_candidate_generator_contract() -> None:
    heuristic = HeuristicGenerator(SequenceAgent())
    rl = RLCandidateGenerator(StubRLAgent(ready=True))  # type: ignore[arg-type]

    heuristic_candidates = heuristic.generate(_make_spec(), _make_register(), "camp_test", [])
    rl_candidates = rl.generate(_make_spec(), _make_register(), "camp_test", [])

    assert heuristic.available is True
    assert rl.available is True
    assert heuristic_candidates
    assert rl_candidates
    assert all(candidate.metadata.get("source") == "heuristic" for candidate in heuristic_candidates)
    assert all(candidate.metadata.get("source") == "rl_policy" for candidate in rl_candidates)


def test_bandit_selector_prefers_highest_ucb_strategy() -> None:
    selector = BanditSelector()
    problem_class = "balanced_campaign_search_4atoms_square"
    selector.update_performance(problem_class, SequenceStrategyMode.HEURISTIC_ONLY.value, [0.4, 0.45])
    selector.update_performance(problem_class, SequenceStrategyMode.RL_ONLY.value, [0.8, 0.82])

    selected = selector.select(
        problem_class,
        [
            SequenceStrategyMode.HEURISTIC_ONLY,
            SequenceStrategyMode.RL_ONLY,
            SequenceStrategyMode.HYBRID,
        ],
    )

    assert selected in {
        SequenceStrategyMode.HEURISTIC_ONLY,
        SequenceStrategyMode.RL_ONLY,
        SequenceStrategyMode.HYBRID,
    }
    assert selector.build_report()
