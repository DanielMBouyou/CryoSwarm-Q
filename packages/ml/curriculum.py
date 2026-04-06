"""Curriculum learning for progressive RL training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.core.logging import get_logger
from packages.core.models import RegisterCandidate

logger = get_logger(__name__)


@dataclass
class CurriculumStage:
    """Definition of a single curriculum stage."""

    name: str
    min_atoms: int
    max_atoms: int
    allowed_layouts: list[str]
    min_performance: float
    min_episodes: int

    def accepts(self, register: RegisterCandidate) -> bool:
        if register.atom_count < self.min_atoms or register.atom_count > self.max_atoms:
            return False
        if register.layout_type not in self.allowed_layouts:
            return False
        return True


class CurriculumScheduler:
    """Manage progression through curriculum stages."""

    def __init__(
        self,
        stages: list[CurriculumStage] | None = None,
        mode: str = "adaptive",
        total_updates: int = 500,
    ) -> None:
        self.stages = stages or self._default_stages()
        self.mode = mode
        self.total_updates = total_updates
        self.current_stage_idx = 0
        self._episode_rewards: list[float] = []
        self._stage_episode_count = 0
        self._update_count = 0

    @staticmethod
    def _default_stages() -> list[CurriculumStage]:
        return [
            CurriculumStage(
                name="warm-up",
                min_atoms=3,
                max_atoms=5,
                allowed_layouts=["square", "line"],
                min_performance=0.3,
                min_episodes=50,
            ),
            CurriculumStage(
                name="expansion",
                min_atoms=4,
                max_atoms=8,
                allowed_layouts=["square", "line", "triangular", "ring"],
                min_performance=0.35,
                min_episodes=100,
            ),
            CurriculumStage(
                name="full",
                min_atoms=3,
                max_atoms=15,
                allowed_layouts=["square", "line", "triangular", "ring", "zigzag", "honeycomb"],
                min_performance=0.0,
                min_episodes=0,
            ),
        ]

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[min(self.current_stage_idx, len(self.stages) - 1)]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    def filter_registers(self, candidates: list[RegisterCandidate]) -> list[RegisterCandidate]:
        stage = self.current_stage
        valid = [candidate for candidate in candidates if stage.accepts(candidate)]
        if valid:
            return valid
        logger.warning(
            "No registers match curriculum stage '%s' (atoms %d-%d). Using all candidates as fallback.",
            stage.name,
            stage.min_atoms,
            stage.max_atoms,
        )
        return candidates

    def record_episode(self, reward: float) -> None:
        self._episode_rewards.append(float(reward))
        self._stage_episode_count += 1

    def step_update(self) -> bool:
        self._update_count += 1
        if self.is_final_stage:
            return False

        if self.mode == "linear":
            boundaries = [
                self.total_updates * (index + 1) / len(self.stages)
                for index in range(len(self.stages) - 1)
            ]
            if self._update_count >= boundaries[self.current_stage_idx]:
                return self._advance()
            return False

        if self.mode == "cycling":
            next_index = self._update_count % len(self.stages)
            if next_index != self.current_stage_idx:
                self.current_stage_idx = next_index
                self._stage_episode_count = 0
                logger.info("Curriculum: now at stage '%s'", self.current_stage.name)
                return True
            return False

        stage = self.current_stage
        if self._stage_episode_count >= stage.min_episodes and stage.min_episodes > 0:
            recent = self._episode_rewards[-stage.min_episodes:]
            avg_reward = sum(recent) / len(recent)
            if avg_reward >= stage.min_performance:
                logger.info(
                    "Curriculum: advancing from '%s' (avg_reward=%.3f >= %.3f)",
                    stage.name,
                    avg_reward,
                    stage.min_performance,
                )
                return self._advance()

        linear_boundaries = [
            self.total_updates * (index + 1) / len(self.stages)
            for index in range(len(self.stages) - 1)
        ]
        if self._update_count >= linear_boundaries[self.current_stage_idx]:
            logger.info("Curriculum: advancing from '%s' (linear fallback)", self.current_stage.name)
            return self._advance()
        return False

    def _advance(self) -> bool:
        if self.is_final_stage:
            return False
        self.current_stage_idx += 1
        self._stage_episode_count = 0
        logger.info("Curriculum: now at stage '%s'", self.current_stage.name)
        return True

    def get_report(self) -> dict[str, Any]:
        recent = self._episode_rewards[-20:]
        avg_recent_reward = sum(recent) / len(recent) if recent else 0.0
        return {
            "current_stage": self.current_stage.name,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_episodes": self._stage_episode_count,
            "total_episodes": len(self._episode_rewards),
            "update_count": self._update_count,
            "avg_recent_reward": avg_recent_reward,
        }
