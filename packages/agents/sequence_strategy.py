"""Intelligent sequence generation strategy."""
from __future__ import annotations

from dataclasses import dataclass
import math
from enum import StrEnum
from pathlib import Path
from typing import Any

from packages.agents.sequence_agent import SequenceAgent
from packages.core.enums import SequenceFamily
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
        heuristic_agent: SequenceAgent | None = None,
        rl_agent: RLSequenceAgent | None = None,
    ) -> None:
        self.mode = mode
        self.param_space = param_space or PhysicsParameterSpace.default()
        self.rl_checkpoint_path = rl_checkpoint_path
        self.heuristic_enabled = heuristic_enabled
        self.min_rl_confidence = min_rl_confidence
        self.hybrid_rl_fraction = hybrid_rl_fraction
        self.surrogate_model = surrogate_model
        self.heuristic_agent = heuristic_agent or SequenceAgent(param_space=self.param_space)
        self.rl_agent = rl_agent or RLSequenceAgent(
            param_space=self.param_space,
            checkpoint_path=rl_checkpoint_path,
            n_candidates=rl_n_candidates,
            temperature=rl_temperature,
            enabled=bool(rl_checkpoint_path),
            heuristic_agent=self.heuristic_agent,
        )
        self.metrics: dict[tuple[str, str], StrategyMetrics] = {}

    def select_strategy(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
        memory_records: list[MemoryRecord],
    ) -> SequenceStrategyMode:
        if not self.rl_agent.ready:
            return SequenceStrategyMode.HEURISTIC_ONLY
        if self.mode in {
            SequenceStrategyMode.HEURISTIC_ONLY,
            SequenceStrategyMode.RL_ONLY,
            SequenceStrategyMode.HYBRID,
        }:
            return self.mode

        problem_class = self._compute_problem_class(spec, register)
        if self.mode == SequenceStrategyMode.BANDIT:
            return self._select_bandit(problem_class)

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
        strategy_used = self.select_strategy(spec, register, memory_records)
        heuristic_candidates: list[SequenceCandidate] = []
        rl_candidates: list[SequenceCandidate] = []
        reason = ""

        if strategy_used == SequenceStrategyMode.HEURISTIC_ONLY:
            heuristic_candidates = self.heuristic_agent.run(spec, register, campaign_id, memory_records)
            reason = "RL checkpoint unavailable or below confidence threshold."
        elif strategy_used == SequenceStrategyMode.RL_ONLY:
            rl_candidates = self.rl_agent.run(spec, register, campaign_id, memory_records)
            reason = "RL selected explicitly or by adaptive policy."
        elif strategy_used == SequenceStrategyMode.HYBRID:
            heuristic_candidates = self.heuristic_agent.run(spec, register, campaign_id, memory_records)
            rl_candidates = self.rl_agent.run(spec, register, campaign_id, memory_records)
            reason = "Hybrid exploration enabled for this problem class."
        else:
            heuristic_candidates = self.heuristic_agent.run(spec, register, campaign_id, memory_records)
            reason = "Fallback to heuristic execution."
            strategy_used = SequenceStrategyMode.HEURISTIC_ONLY

        candidates = heuristic_candidates + rl_candidates
        for candidate in heuristic_candidates:
            candidate.metadata.setdefault("source", "heuristic")
        for candidate in rl_candidates:
            candidate.metadata.setdefault("source", "rl_policy")

        if strategy_used == SequenceStrategyMode.HYBRID and self.hybrid_rl_fraction < 1.0:
            target_rl = max(1, int(round(len(candidates) * self.hybrid_rl_fraction))) if rl_candidates else 0
            target_heuristic = max(1, len(candidates) - target_rl) if heuristic_candidates else 0
            candidates = heuristic_candidates[:target_heuristic] + rl_candidates[:target_rl]

        metadata = {
            "problem_class": problem_class,
            "strategy_used": strategy_used.value,
            "strategy_reason": reason,
            "rl_candidates_count": len([c for c in candidates if c.metadata.get("source") == "rl_policy"]),
            "heuristic_candidates_count": len([c for c in candidates if c.metadata.get("source") == "heuristic"]),
            "confidence": self._assess_rl_checkpoint_quality() if self.rl_agent.ready else 0.0,
        }
        return candidates, metadata

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

    def _compute_problem_class(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
    ) -> str:
        return f"{spec.objective_class}_{register.atom_count}atoms_{register.layout_type}"

    def _assess_rl_checkpoint_quality(self) -> float:
        if not self.rl_agent.ready:
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
            except Exception:
                return 0.75
        return 0.75

    def _select_bandit(self, problem_class: str) -> SequenceStrategyMode:
        available = [
            SequenceStrategyMode.HEURISTIC_ONLY,
            SequenceStrategyMode.RL_ONLY,
            SequenceStrategyMode.HYBRID,
        ]
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

    def get_strategy_report(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "rl_ready": self.rl_agent.ready,
            "rl_confidence": self._assess_rl_checkpoint_quality(),
            "metrics": {
                f"{problem_class}:{strategy}": {
                    "n_trials": metric.n_trials,
                    "avg_reward": metric.avg_reward,
                    "best_reward": metric.best_reward,
                    "ucb1_score": metric.ucb1_score,
                }
                for (problem_class, strategy), metric in self.metrics.items()
            },
        }
