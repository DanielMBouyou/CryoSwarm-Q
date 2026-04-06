"""Tests for Phase 2 — RL environment and PPO components."""
from __future__ import annotations

import numpy as np
import pytest

from packages.ml.rl_env import (
    ACT_DIM,
    OBS_DIM,
    PulseDesignEnv,
    inverse_rescale,
    rescale_action,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---- action rescaling tests (no PyTorch needed) ----


class TestActionRescaling:
    def test_center_action(self):
        action = np.zeros(ACT_DIM, dtype=np.float32)
        params = rescale_action(action)
        assert 1.0 <= params["amplitude"] <= 15.0
        assert -30.0 <= params["detuning"] <= 15.0
        assert params["duration_ns"] >= 16

    def test_min_action(self):
        action = np.full(ACT_DIM, -1.0, dtype=np.float32)
        params = rescale_action(action)
        assert params["amplitude"] == pytest.approx(1.0, abs=0.01)
        assert params["detuning"] == pytest.approx(-30.0, abs=0.01)
        assert params["duration_ns"] >= 16

    def test_max_action(self):
        action = np.full(ACT_DIM, 1.0, dtype=np.float32)
        params = rescale_action(action)
        assert params["amplitude"] == pytest.approx(15.0, abs=0.01)
        assert params["detuning"] == pytest.approx(15.0, abs=0.01)

    def test_clipping(self):
        action = np.array([5.0, -5.0, 3.0, -3.0], dtype=np.float32)
        params = rescale_action(action)
        assert 1.0 <= params["amplitude"] <= 15.0
        assert -30.0 <= params["detuning"] <= 15.0

    def test_duration_quantized(self):
        action = np.array([0.0, 0.0, 0.3, 0.0], dtype=np.float32)
        params = rescale_action(action)
        assert params["duration_ns"] % 4 == 0

    def test_family_always_valid(self):
        from packages.core.enums import SequenceFamily

        for val in np.linspace(-1, 1, 20):
            action = np.array([0.0, 0.0, 0.0, val], dtype=np.float32)
            params = rescale_action(action)
            assert params["family"] in [f.value for f in SequenceFamily]

    def test_inverse_roundtrip(self):
        params = {
            "amplitude": 7.5,
            "detuning": -10.0,
            "duration_ns": 3000,
            "family": "adiabatic_sweep",
        }
        action = inverse_rescale(params)
        recovered = rescale_action(action)
        assert recovered["amplitude"] == pytest.approx(7.5, abs=0.5)
        assert recovered["detuning"] == pytest.approx(-10.0, abs=1.0)


# ---- environment tests ----


def _make_env_components():
    from packages.core.enums import SequenceFamily
    from packages.core.models import ExperimentSpec, RegisterCandidate, ScoringWeights

    spec = ExperimentSpec(
        goal_id="goal_1",
        objective_class="rydberg_density",
        target_observable="rydberg_density",
        min_atoms=4,
        max_atoms=6,
        preferred_layouts=["square"],
        sequence_families=[SequenceFamily.ADIABATIC_SWEEP],
        reasoning_summary="Test spec.",
    )
    reg = RegisterCandidate(
        campaign_id="test",
        spec_id=spec.id,
        label="reg-square-4",
        layout_type="square",
        atom_count=4,
        coordinates=[(0, 0), (7, 0), (0, 7), (7, 7)],
        min_distance_um=7.0,
        blockade_radius_um=9.5,
        blockade_pair_count=3,
        feasibility_score=0.8,
        reasoning_summary="Test.",
        metadata={"spacing_um": 7.0},
    )
    return spec, [reg]


class TestPulseDesignEnv:
    def test_reset_shapes(self):
        spec, regs = _make_env_components()
        env = PulseDesignEnv(spec, regs, simulate_fn=lambda r, p: 0.5)
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBS_DIM,)
        assert "register_id" in info

    def test_step_shapes(self):
        spec, regs = _make_env_components()
        env = PulseDesignEnv(spec, regs, max_steps=1, reward_shaping=False, simulate_fn=lambda r, p: 0.6)
        env.reset(seed=0)
        action = np.zeros(ACT_DIM, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert reward == pytest.approx(0.6)
        assert terminated is True  # max_steps=1
        assert "params" in info

    def test_multi_step_episode(self):
        spec, regs = _make_env_components()
        step_count = [0]

        def sim_fn(r, p):
            step_count[0] += 1
            return 0.3 * step_count[0]

        env = PulseDesignEnv(spec, regs, max_steps=3, simulate_fn=sim_fn)
        obs, _ = env.reset(seed=0)
        for i in range(3):
            action = np.random.randn(ACT_DIM).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if i < 2:
                assert not terminated
            else:
                assert terminated


# ---- PPO components tests ----


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPPOComponents:
    def test_actor_critic_forward(self):
        from packages.ml.ppo import ActorCritic

        ac = ActorCritic()
        obs = torch.randn(4, OBS_DIM)
        mean, std, value = ac(obs)
        assert mean.shape == (4, ACT_DIM)
        assert std.shape == (ACT_DIM,)
        assert value.shape == (4,)

    def test_get_action(self):
        from packages.ml.ppo import ActorCritic

        ac = ActorCritic()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, log_prob, value = ac.get_action(obs)
        assert action.shape == (ACT_DIM,)
        assert -1.0 <= action.max() <= 1.0

    def test_rollout_buffer_gae(self):
        from packages.ml.ppo import RolloutBuffer

        buf = RolloutBuffer()
        for i in range(10):
            buf.add(
                obs=np.zeros(OBS_DIM, dtype=np.float32),
                action=np.zeros(ACT_DIM, dtype=np.float32),
                log_prob=-1.0,
                reward=float(i) * 0.1,
                value=0.5,
                done=(i == 9),
            )
        advantages, returns = buf.compute_gae(last_value=0.0)
        assert advantages.shape == (10,)
        assert returns.shape == (10,)

    def test_save_load_policy(self, tmp_path):
        from packages.ml.ppo import ActorCritic

        ac = ActorCritic()
        path = tmp_path / "policy.pt"
        ac.save(path)

        ac2 = ActorCritic()
        ac2.load(path)

        obs = torch.randn(2, OBS_DIM)
        m1, s1, v1 = ac(obs)
        m2, s2, v2 = ac2(obs)
        torch.testing.assert_close(m1, m2)

    def test_ppo_save_load_weights_only(self, tmp_path):
        from packages.ml.ppo import ActorCritic

        model = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=32)
        path = tmp_path / "weights_only_policy.pt"
        model.save(path)

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        assert "model" in checkpoint
        assert checkpoint["version"] == "ppo_v2"

    def test_ppo_short_training(self):
        from packages.ml.ppo import PPOConfig, PPOTrainer

        spec, regs = _make_env_components()
        env = PulseDesignEnv(spec, regs, simulate_fn=lambda r, p: np.random.random())

        config = PPOConfig(
            total_updates=2,
            rollout_steps=8,
            epochs_per_update=1,
            batch_size=4,
        )
        trainer = PPOTrainer(config)
        history = trainer.train(env, seed=0)

        assert len(history["policy_loss"]) == 2
        assert len(history["value_loss"]) == 2


# ---- RL sequence agent tests ----


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestRLSequenceAgent:
    def test_fallback_when_disabled(self):
        from packages.ml.rl_sequence_agent import RLSequenceAgent

        agent = RLSequenceAgent(enabled=False)
        spec, regs = _make_env_components()
        candidates = agent.run(spec[0] if isinstance(spec, tuple) else spec, regs[0], "test_campaign")
        assert len(candidates) > 0

    def test_with_checkpoint(self, tmp_path):
        from packages.ml.ppo import ActorCritic
        from packages.ml.rl_sequence_agent import RLSequenceAgent

        ac = ActorCritic()
        path = tmp_path / "test_policy.pt"
        ac.save(path)

        agent = RLSequenceAgent(checkpoint_path=str(path), n_candidates=3, enabled=True)
        spec, regs = _make_env_components()
        candidates = agent.run(spec, regs[0], "rl_campaign")
        assert len(candidates) == 3
        for c in candidates:
            assert "rl_policy" in c.metadata.get("source", "")
            assert 0.0 <= c.amplitude <= 15.8
            assert -126.0 <= c.detuning <= 126.0
