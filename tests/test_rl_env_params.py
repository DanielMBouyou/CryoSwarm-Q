from __future__ import annotations

import numpy as np

from packages.core.enums import SequenceFamily
from packages.core.models import ExperimentSpec, RegisterCandidate
from packages.ml.rl_env import (
    ACT_DIM,
    DEFAULT_RL_ENV_IMPROVEMENT_SCALE,
    DEFAULT_RL_ENV_MAX_STEPS,
    PulseDesignEnv,
)


def _make_env_components(
    *,
    atom_count: int = 10,
    spacing_um: float = 15.0,
    layout_type: str = "honeycomb",
) -> tuple[ExperimentSpec, list[RegisterCandidate]]:
    spec = ExperimentSpec(
        goal_id="goal_rl_env",
        objective_class="rl_env_hyperparam_validation",
        target_observable="rydberg_density",
        min_atoms=4,
        max_atoms=10,
        preferred_layouts=[layout_type],
        sequence_families=[SequenceFamily.ADIABATIC_SWEEP],
        target_density=0.5,
        reasoning_summary="RL environment hyperparameter validation spec.",
    )
    register = RegisterCandidate(
        campaign_id="campaign_rl_env",
        spec_id=spec.id,
        label=f"{layout_type}-{atom_count}",
        layout_type=layout_type,
        atom_count=atom_count,
        coordinates=[(float(index) * spacing_um, 0.0) for index in range(atom_count)],
        min_distance_um=spacing_um,
        blockade_radius_um=9.5,
        blockade_pair_count=max(atom_count - 1, 1),
        feasibility_score=0.85,
        reasoning_summary="RL environment hyperparameter validation register.",
        metadata={"spacing_um": spacing_um},
    )
    return spec, [register]


def _reward_landscape(_: RegisterCandidate, params: dict[str, float | int | str]) -> float:
    amplitude = float(params["amplitude"])
    detuning = float(params["detuning"])
    duration_ns = float(params["duration_ns"])

    amplitude_term = 1.0 - abs(amplitude - 11.0) / 14.0
    detuning_term = 1.0 - abs(detuning + 8.0) / 55.0
    duration_term = 1.0 - abs(duration_ns - 4200.0) / 5000.0

    score = 0.50 * amplitude_term + 0.30 * detuning_term + 0.20 * duration_term
    return float(np.clip(score, 0.0, 1.0))


def test_default_max_steps_is_in_reasonable_exploration_range() -> None:
    assert 3 <= DEFAULT_RL_ENV_MAX_STEPS <= 10


def test_reward_stays_bounded_for_valid_actions() -> None:
    spec, registers = _make_env_components()
    env = PulseDesignEnv(spec, registers, simulate_fn=_reward_landscape)
    rng = np.random.default_rng(11)

    rewards: list[float] = []
    for _ in range(64):
        env.reset(seed=0)
        episode_done = False
        while not episode_done:
            action = rng.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            episode_done = terminated or truncated

    reward_array = np.asarray(rewards, dtype=np.float32)
    assert np.all(reward_array >= -1.0)
    assert np.all(reward_array <= 10.0)


def test_improvement_scale_does_not_dominate_raw_reward_on_first_improvement() -> None:
    spec, registers = _make_env_components()
    env = PulseDesignEnv(spec, registers, simulate_fn=_reward_landscape)
    rng = np.random.default_rng(23)

    assert DEFAULT_RL_ENV_IMPROVEMENT_SCALE < 2.0

    for _ in range(128):
        env.reset(seed=0)
        action = rng.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)
        _, _, _, _, info = env.step(action)
        raw_robustness = float(info["raw_robustness"])
        improvement_bonus = float(info["improvement_bonus"])
        assert improvement_bonus < 2.0 * raw_robustness


def test_terminal_bonus_stays_within_half_of_episode_reward_budget() -> None:
    spec, registers = _make_env_components()
    env = PulseDesignEnv(spec, registers, simulate_fn=_reward_landscape)
    rng = np.random.default_rng(31)

    for episode_seed in range(16):
        env.reset(seed=episode_seed)
        cumulative_reward_without_terminal = 0.0
        episode_done = False
        while not episode_done:
            action = rng.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            terminal_bonus = float(info["terminal_bonus"])
            cumulative_reward_without_terminal += float(reward) - terminal_bonus
            episode_done = terminated or truncated
            if episode_done:
                assert terminal_bonus <= 0.5 * cumulative_reward_without_terminal + 1e-6


def test_observations_are_normalized_for_reasonable_registers() -> None:
    spec, registers = _make_env_components(atom_count=10, spacing_um=15.0, layout_type="honeycomb")
    env = PulseDesignEnv(spec, registers, simulate_fn=_reward_landscape)
    rng = np.random.default_rng(41)

    observation, _ = env.reset(seed=0)
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 2.0)

    for _ in range(8):
        action = rng.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        assert np.all(observation >= -1.0)
        assert np.all(observation <= 2.0)
        if terminated or truncated:
            observation, _ = env.reset(seed=0)
