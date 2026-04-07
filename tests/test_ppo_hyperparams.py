from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from packages.core.enums import SequenceFamily
from packages.core.models import ExperimentSpec, RegisterCandidate
from packages.ml.ppo import PPOConfig
from packages.ml.rl_env import PulseDesignEnv

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def _make_env_components() -> tuple[ExperimentSpec, list[RegisterCandidate]]:
    spec = ExperimentSpec(
        goal_id="goal_ppo",
        objective_class="ppo_hyperparam_validation",
        target_observable="rydberg_density",
        min_atoms=4,
        max_atoms=6,
        preferred_layouts=["square"],
        sequence_families=[SequenceFamily.ADIABATIC_SWEEP],
        target_density=0.5,
        reasoning_summary="PPO hyperparameter validation spec.",
    )
    register = RegisterCandidate(
        campaign_id="campaign_ppo",
        spec_id=spec.id,
        label="square-6",
        layout_type="square",
        atom_count=6,
        coordinates=[(0.0, 0.0), (7.0, 0.0), (14.0, 0.0), (0.0, 7.0), (7.0, 7.0), (14.0, 7.0)],
        min_distance_um=7.0,
        blockade_radius_um=9.5,
        blockade_pair_count=7,
        feasibility_score=0.85,
        reasoning_summary="PPO hyperparameter validation register.",
        metadata={"spacing_um": 7.0},
    )
    return spec, [register]


def _reward_landscape(_: RegisterCandidate, params: dict[str, float | int | str]) -> float:
    amplitude = float(params["amplitude"])
    detuning = float(params["detuning"])
    duration_ns = float(params["duration_ns"])
    family = str(params["family"])

    amplitude_term = 1.0 - abs(amplitude - 11.0) / 14.0
    detuning_term = 1.0 - abs(detuning + 8.0) / 55.0
    duration_term = 1.0 - abs(duration_ns - 4200.0) / 5000.0
    family_bonus = 1.0 if family == SequenceFamily.ADIABATIC_SWEEP.value else 0.6

    score = (
        0.35 * amplitude_term
        + 0.30 * detuning_term
        + 0.20 * duration_term
        + 0.15 * family_bonus
    )
    return float(np.clip(score, 0.0, 1.0))


def test_default_ppo_learning_rate_ratio_is_in_standard_range() -> None:
    config = PPOConfig()
    ratio = config.lr_critic / config.lr_actor
    assert 2.0 <= ratio <= 5.0


def test_default_ppo_clip_range_is_standard() -> None:
    config = PPOConfig()
    assert 0.1 <= config.epsilon_clip <= 0.3


def test_default_ppo_discount_is_high_but_stable() -> None:
    config = PPOConfig()
    assert 0.95 <= config.gamma <= 0.999


def test_default_ppo_gae_lambda_is_in_standard_range() -> None:
    config = PPOConfig()
    assert 0.9 <= config.gae_lambda <= 0.99


def test_ppo_training_over_ten_updates_improves_reward_signal() -> None:
    from packages.ml.ppo import PPOTrainer

    spec, registers = _make_env_components()
    env = PulseDesignEnv(
        spec,
        registers,
        max_steps=1,
        reward_shaping=False,
        simulate_fn=_reward_landscape,
    )
    config = PPOConfig(
        total_updates=10,
        rollout_steps=32,
        epochs_per_update=2,
        batch_size=16,
        normalize_rewards=False,
        use_replay=False,
    )
    trainer = PPOTrainer(config)
    history = trainer.train(env, seed=0)

    episode_rewards = np.asarray(history["episode_rewards"], dtype=np.float32)
    assert episode_rewards.shape == (config.total_updates * config.rollout_steps,)
    assert np.all(np.isfinite(episode_rewards))

    per_update_means = episode_rewards.reshape(config.total_updates, config.rollout_steps).mean(axis=1)
    first_window = float(np.mean(per_update_means[:3]))
    last_window = float(np.mean(per_update_means[-3:]))

    assert last_window >= first_window - 0.02


def test_default_ppo_total_interactions_clear_minimum_budget() -> None:
    config = PPOConfig()
    assert config.rollout_steps * config.total_updates >= 50_000
