"""Tests for multi-step RL environment behavior."""
from __future__ import annotations

import numpy as np

from packages.core.enums import SequenceFamily
from packages.core.models import ExperimentSpec, RegisterCandidate
from packages.ml.rl_env import ACT_DIM, OBS_DIM, PulseDesignEnv


def _make_components():
    spec = ExperimentSpec(
        goal_id="goal_rl_multi",
        objective_class="rydberg_density",
        target_observable="rydberg_density",
        min_atoms=3,
        max_atoms=6,
        preferred_layouts=["square"],
        sequence_families=[SequenceFamily.ADIABATIC_SWEEP],
        reasoning_summary="Test spec.",
    )
    register = RegisterCandidate(
        campaign_id="test",
        spec_id=spec.id,
        label="reg-square-4",
        layout_type="square",
        atom_count=4,
        coordinates=[(0.0, 0.0), (7.0, 0.0), (0.0, 7.0), (7.0, 7.0)],
        min_distance_um=7.0,
        blockade_radius_um=9.5,
        blockade_pair_count=4,
        feasibility_score=0.8,
        reasoning_summary="Test register.",
        metadata={"spacing_um": 7.0},
    )
    return spec, [register]


def test_multi_step_episode():
    spec, registers = _make_components()
    env = PulseDesignEnv(spec, registers, max_steps=5, simulate_fn=lambda r, p: 0.2)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,)
    for step in range(5):
        action = np.random.uniform(-1, 1, ACT_DIM).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert not truncated
        if step < 4:
            assert not terminated
        else:
            assert terminated


def test_reward_shaping_improvement():
    spec, registers = _make_components()
    env = PulseDesignEnv(
        spec,
        registers,
        max_steps=3,
        reward_shaping=True,
        simulate_fn=lambda r, p: p["amplitude"] / 15.0,
    )
    env.reset(seed=42)
    low_action = np.array([-0.8, 0.0, 0.0, 0.0], dtype=np.float32)
    _, r1, _, _, _ = env.step(low_action)

    high_action = np.array([0.8, 0.0, 0.0, 0.0], dtype=np.float32)
    _, r2, _, _, info2 = env.step(high_action)

    assert info2["improvement"] > 0.0
    assert r2 > r1


def test_obs_dim_16():
    spec, registers = _make_components()
    env = PulseDesignEnv(spec, registers, max_steps=5, simulate_fn=lambda r, p: 0.1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (16,)


def test_backward_compat_single_step():
    spec, registers = _make_components()
    env = PulseDesignEnv(
        spec,
        registers,
        max_steps=1,
        reward_shaping=False,
        simulate_fn=lambda r, p: 0.4,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (16,)
    _, reward, terminated, _, _ = env.step(np.zeros(ACT_DIM, dtype=np.float32))
    assert terminated
    assert reward == 0.4
