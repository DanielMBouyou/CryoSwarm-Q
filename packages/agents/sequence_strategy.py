"""Intelligent sequence generation strategy."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from packages.agents.protocols import SequenceProtocol
from packages.agents.sequence_agent import SequenceAgent
from packages.core.logging import get_logger
from packages.core.models import ExperimentSpec, MemoryRecord, RegisterCandidate, SequenceCandidate
from packages.core.parameter_space import PhysicsParameterSpace
from packages.ml.rl_sequence_agent import RLSequenceAgent
from packages.ml.surrogate import SurrogateModel


class SequenceStrategyMode(StrEnum):
    HEURISTIC_ONLY = "heuristic_only"
    RL_ONLY = "rl_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    BANDIT = "bandit"


@dataclass
class StrategyMetrics:
    """Performance tracking for a strategy on a problem class."""

    strategy: str
    problem_class: str
    n_trials: int = 0
    total_reward: float = 0.0
    best_reward: float = 0.0
    avg_reward: float = 0.0
    total_trials_context: int = 0

    def update(self, reward: float) -> None:
        self.n_trials += 1
        self.total_reward += reward
        self.best_reward = max(self.best_reward, reward)
        self.avg_reward = self.total_reward / max(self.n_trials, 1)

    @property
    def ucb1_score(self) -> float:
        if self.n_trials == 0:
            return float("inf")
        total_trials = max(self.total_trials_context, self.n_trials)
        return self.avg_reward + math.sqrt(2.0 * math.log(total_trials) / self.n_trials)


@runtime_checkable
class CandidateGenerator(Protocol):
    """Common interface for sequence candidate generators."""

    name: str

    @property
    def available(self) -> bool: ...

    def generate(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord],
    ) -> list[SequenceCandidate]: ...


def _with_metadata_defaults(
    candidate: SequenceCandidate,
    **defaults: Any,
) -> SequenceCandidate:
    metadata = dict(candidate.metadata)
    changed = False
    for key, value in defaults.items():
        if key not in metadata:
            metadata[key] = value
            changed = True
    if not changed:
        return candidate
    return candidate.model_copy(update={"metadata": metadata})


class HeuristicGenerator:
    """Generate candidates via the heuristic sequence agent."""

    name = "heuristic"

    def __init__(self, agent: SequenceProtocol) -> None:
        self.agent = agent

    @property
    def available(self) -> bool:
        return True

    def generate(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord],
    ) -> list[SequenceCandidate]:
        return [
            _with_metadata_defaults(candidate, source="heuristic")
            for candidate in self.agent.run(spec, register, campaign_id, memory_records)
        ]


class RLCandidateGenerator:
    """Generate candidates via the RL policy when a checkpoint is available."""

    name = "rl_policy"

    def __init__(self, agent: RLSequenceAgent) -> None:
        self.agent = agent

    @property
    def available(self) -> bool:
        return self.agent.ready

    def generate(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord],
    ) -> list[SequenceCandidate]:
        if not self.available:
            return []
        return [
            _with_metadata_defaults(candidate, source="rl_policy")
            for candidate in self.agent.run(spec, register, campaign_id, memory_records)
        ]


class BanditSelector:
    """Track per-problem-class rewards and select strategies via UCB1."""

    def __init__(self) -> None:
        self.metrics: dict[tuple[str, str], StrategyMetrics] = {}

    def update_performance(
        self,
        problem_class: str,
        strategy_used: str,
        robustness_scores: list[float],
    ) -> None:
        if not robustness_scores:
            return
        key = (problem_class, strategy_used)
        metric = self.metrics.setdefault(key, StrategyMetrics(strategy=strategy_used, problem_class=problem_class))
        metric.update(sum(robustness_scores) / len(robustness_scores))

    def select(
        self,
        problem_class: str,
        available: list[SequenceStrategyMode],
    ) -> SequenceStrategyMode:
        total_trials = sum(
            metric.n_trials for (klass, _), metric in self.metrics.items() if klass == problem_class
        )
        scored: list[tuple[SequenceStrategyMode, float]] = []
        for strategy in available:
            metric = self.metrics.setdefault(
                (problem_class, strategy.value),
                StrategyMetrics(strategy=strategy.value, problem_class=problem_class),
            )
            metric.total_trials_context = max(total_trials, 1)
            scored.append((strategy, metric.ucb1_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[0][0]

    def build_report(self) -> dict[str, Any]:
        return {
            f"{problem_class}:{strategy}": {
                "n_trials": metric.n_trials,
                "avg_reward": metric.avg_reward,
                "best_reward": metric.best_reward,
                "ucb1_score": metric.ucb1_score,
            }
            for (problem_class, strategy), metric in self.metrics.items()
        }


class SequenceStrategy:
    """Research-grade strategy for sequence generation method selection."""

    def __init__(
        self,
        mode: SequenceStrategyMode = SequenceStrategyMode.ADAPTIVE,
        param_space: PhysicsParameterSpace | None = None,
        rl_checkpoint_path: str | None = None,
        rl_temperature: float = 0.3,
        rl_n_candidates: int = 5,
        heuristic_enabled: bool = True,
        min_rl_confidence: float = 0.4,
        hybrid_rl_fraction: float = 0.5,
        surrogate_model: SurrogateModel | None = None,
        heuristic_agent: SequenceProtocol | None = None,
        rl_agent: RLSequenceAgent | None = None,
        heuristic_generator: CandidateGenerator | None = None,
        rl_generator: CandidateGenerator | None = None,
        bandit_selector: BanditSelector | None = None,
    ) -> None:
        self.mode = mode
        self.param_space = param_space or PhysicsParameterSpace.default()
        self.rl_checkpoint_path = rl_checkpoint_path
        self.heuristic_enabled = heuristic_enabled
        self.min_rl_confidence = min_rl_confidence
        self.hybrid_rl_fraction = hybrid_rl_fraction
        self.surrogate_model = surrogate_model
        self.logger = get_logger(__name__)

        self.heuristic_agent: SequenceProtocol = heuristic_agent or SequenceAgent(param_space=self.param_space)
        self.rl_agent = rl_agent or RLSequenceAgent(
            param_space=self.param_space,
            checkpoint_path=rl_checkpoint_path,
            n_candidates=rl_n_candidates,
            temperature=rl_temperature,
            enabled=bool(rl_checkpoint_path),
            heuristic_agent=self.heuristic_agent,
        )

        self.heuristic_generator = heuristic_generator or HeuristicGenerator(self.heuristic_agent)
        self.rl_generator = rl_generator or RLCandidateGenerator(self.rl_agent)
        self.bandit_selector = bandit_selector or BanditSelector()

    @property
    def metrics(self) -> dict[tuple[str, str], StrategyMetrics]:
        return self.bandit_selector.metrics

    def select_strategy(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
        memory_records: list[MemoryRecord],
    ) -> SequenceStrategyMode:
        if not self.rl_generator.available:
            return SequenceStrategyMode.HEURISTIC_ONLY
        if self.mode in {
            SequenceStrategyMode.HEURISTIC_ONLY,
            SequenceStrategyMode.RL_ONLY,
            SequenceStrategyMode.HYBRID,
        }:
            return self.mode

        problem_class = self._compute_problem_class(spec, register)
        if self.mode == SequenceStrategyMode.BANDIT:
            return self.bandit_selector.select(problem_class, self._available_bandit_strategies())

        rl_quality = self._assess_rl_checkpoint_quality()
        if rl_quality < self.min_rl_confidence:
            return SequenceStrategyMode.HEURISTIC_ONLY

        heuristic_metrics = self.metrics.get((problem_class, SequenceStrategyMode.HEURISTIC_ONLY.value))
        rl_metrics = self.metrics.get((problem_class, SequenceStrategyMode.RL_ONLY.value))

        if heuristic_metrics and rl_metrics and heuristic_metrics.n_trials >= 3 and rl_metrics.n_trials >= 3:
            if rl_metrics.avg_reward > heuristic_metrics.avg_reward + 0.02:
                return SequenceStrategyMode.RL_ONLY
            if heuristic_metrics.avg_reward > rl_metrics.avg_reward + 0.02:
                return SequenceStrategyMode.HEURISTIC_ONLY

        related_memory = [
            record
            for record in memory_records
            if str(record.signals.get("layout_type", register.layout_type)) == register.layout_type
            and int(record.signals.get("atom_count", register.atom_count)) == register.atom_count
        ]
        if len(related_memory) < 3:
            return SequenceStrategyMode.HYBRID

        return SequenceStrategyMode.HYBRID

    def generate_candidates(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord],
    ) -> tuple[list[SequenceCandidate], dict[str, Any]]:
        problem_class = self._compute_problem_class(spec, register)
        selected_mode = self.select_strategy(spec, register, memory_records)

        heuristic_candidates: list[SequenceCandidate] = []
        rl_candidates: list[SequenceCandidate] = []
        strategy_used = selected_mode
        reason = ""

        if selected_mode == SequenceStrategyMode.HEURISTIC_ONLY:
            heuristic_candidates = self.heuristic_generator.generate(
                spec,
                register,
                campaign_id,
                memory_records,
            )
            reason = "RL checkpoint unavailable or below confidence threshold."
        elif selected_mode == SequenceStrategyMode.RL_ONLY:
            rl_candidates = self.rl_generator.generate(spec, register, campaign_id, memory_records)
            reason = "RL selected explicitly or by adaptive policy."
        elif selected_mode == SequenceStrategyMode.HYBRID:
            heuristic_candidates = self.heuristic_generator.generate(
                spec,
                register,
                campaign_id,
                memory_records,
            )
            rl_candidates = self.rl_generator.generate(spec, register, campaign_id, memory_records)
            reason = "Hybrid exploration enabled for this problem class."
        else:
            heuristic_candidates = self.heuristic_generator.generate(
                spec,
                register,
                campaign_id,
                memory_records,
            )
            reason = "Fallback to heuristic execution."
            strategy_used = SequenceStrategyMode.HEURISTIC_ONLY

        if strategy_used == SequenceStrategyMode.RL_ONLY and not rl_candidates:
            heuristic_candidates = self.heuristic_generator.generate(
                spec,
                register,
                campaign_id,
                memory_records,
            )
            strategy_used = SequenceStrategyMode.HEURISTIC_ONLY
            reason = "RL generation returned no candidates; fallback to heuristic execution."

        candidates = self._merge_candidates(
            strategy_used,
            heuristic_candidates,
            rl_candidates,
        )
        metadata = {
            "problem_class": problem_class,
            "strategy_used": strategy_used.value,
            "strategy_reason": reason,
            "rl_candidates_count": len([c for c in candidates if c.metadata.get("source") == "rl_policy"]),
            "heuristic_candidates_count": len([c for c in candidates if c.metadata.get("source") == "heuristic"]),
            "confidence": self._assess_rl_checkpoint_quality() if self.rl_generator.available else 0.0,
        }
        return candidates, metadata

    def update_performance(
        self,
        problem_class: str,
        strategy_used: str,
        robustness_scores: list[float],
    ) -> None:
        self.bandit_selector.update_performance(problem_class, strategy_used, robustness_scores)

    @staticmethod
    def _compute_problem_class(
        spec: ExperimentSpec,
        register: RegisterCandidate,
    ) -> str:
        return f"{spec.objective_class}_{register.atom_count}atoms_{register.layout_type}"

    def _assess_rl_checkpoint_quality(self) -> float:
        if not self.rl_generator.available:
            return 0.0
        if not self.rl_checkpoint_path:
            return 0.6
        checkpoint = Path(self.rl_checkpoint_path)
        if not checkpoint.exists():
            return 0.0
        sidecar = checkpoint.with_suffix(".json")
        if sidecar.exists():
            try:
                import json

                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return float(payload.get("validation_score", payload.get("quality", 0.75)))
            except Exception as exc:
                self.logger.debug(
                    "Checkpoint sidecar %s could not be parsed: %s",
                    sidecar,
                    exc,
                )
                return 0.75
        return 0.75

    def _available_bandit_strategies(self) -> list[SequenceStrategyMode]:
        available = [SequenceStrategyMode.HEURISTIC_ONLY]
        if self.rl_generator.available:
            available.extend(
                [
                    SequenceStrategyMode.RL_ONLY,
                    SequenceStrategyMode.HYBRID,
                ]
            )
        return available

    def _merge_candidates(
        self,
        strategy_used: SequenceStrategyMode,
        heuristic_candidates: list[SequenceCandidate],
        rl_candidates: list[SequenceCandidate],
    ) -> list[SequenceCandidate]:
        if strategy_used == SequenceStrategyMode.HYBRID and self.hybrid_rl_fraction < 1.0:
            combined = heuristic_candidates + rl_candidates
            target_rl = max(1, int(round(len(combined) * self.hybrid_rl_fraction))) if rl_candidates else 0
            target_heuristic = max(1, len(combined) - target_rl) if heuristic_candidates else 0
            return heuristic_candidates[:target_heuristic] + rl_candidates[:target_rl]
        return heuristic_candidates + rl_candidates

    def get_strategy_report(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "rl_ready": self.rl_generator.available,
            "rl_confidence": self._assess_rl_checkpoint_quality(),
            "metrics": self.bandit_selector.build_report(),
        }
