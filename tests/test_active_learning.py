"""Tests for active learning loop and diversity selection."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from packages.core.enums import SequenceFamily
from packages.core.models import ExperimentSpec, RegisterCandidate
from packages.ml.active_learning import ActiveLearningConfig, ActiveLearningLoop

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def _make_spec_and_registers():
    spec = ExperimentSpec(
        goal_id="goal_active",
        objective_class="rydberg_density",
        target_observable="rydberg_density",
        min_atoms=3,
        max_atoms=6,
        preferred_layouts=["square"],
        sequence_families=[SequenceFamily.ADIABATIC_SWEEP],
        reasoning_summary="Active learning test.",
    )
    registers = [
        RegisterCandidate(
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
    ]
    return spec, registers


def test_active_learning_basic(tmp_path):
    x = np.random.randn(100, 10).astype(np.float32)
    y = np.random.rand(100, 4).astype(np.float32)
    spec, registers = _make_spec_and_registers()

    config = ActiveLearningConfig(
        n_iterations=2,
        top_k_per_iteration=10,
        rl_updates=2,
        rl_rollout_episodes=4,
        surrogate_epochs=1,
        rl_max_steps=2,
        checkpoint_dir=str(tmp_path / "active"),
    )

    def mock_sim(register, params):
        return min(1.0, float(params["amplitude"]) / 15.0)

    loop = ActiveLearningLoop(
        config=config,
        initial_features=x,
        initial_targets=y,
        spec=spec,
        register_candidates=registers,
        simulate_fn=mock_sim,
    )
    results = loop.run()
    assert len(loop.features) > 100
    assert len(results["iterations"]) == 2


def test_diversity_selection():
    x = np.random.randn(20, 10).astype(np.float32)
    y = np.random.rand(20, 4).astype(np.float32)
    spec, registers = _make_spec_and_registers()
    loop = ActiveLearningLoop(
        config=ActiveLearningConfig(n_iterations=1, top_k_per_iteration=2, checkpoint_dir="checkpoints/test_active_diversity"),
        initial_features=x,
        initial_targets=y,
        spec=spec,
        register_candidates=registers,
        simulate_fn=lambda register, params: min(1.0, params["amplitude"] / 15.0),
    )
    reg = registers[0]
    configs = [
        {
            "params": {"amplitude": 5.0, "detuning": -10.0, "duration_ns": 2000, "family_enum": SequenceFamily.ADIABATIC_SWEEP},
            "reward": 0.8,
            "register": reg,
        },
        {
            "params": {"amplitude": 5.1, "detuning": -10.1, "duration_ns": 2004, "family_enum": SequenceFamily.ADIABATIC_SWEEP},
            "reward": 0.79,
            "register": reg,
        },
        {
            "params": {"amplitude": 12.0, "detuning": 5.0, "duration_ns": 4000, "family_enum": SequenceFamily.ADIABATIC_SWEEP},
            "reward": 0.75,
            "register": reg,
        },
    ]
    selected = loop._select_diverse_configs(configs)
    assert len(selected) <= loop.config.top_k_per_iteration
