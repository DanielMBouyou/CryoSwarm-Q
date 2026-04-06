"""Phase 2 - Gymnasium-style environment for pulse-sequence design."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from packages.core.enums import SequenceFamily
from packages.core.models import ExperimentSpec, RegisterCandidate, SequenceCandidate
from packages.core.parameter_space import PhysicsParameterSpace

ACTION_LOW = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
FAMILY_LIST = list(SequenceFamily)
OBS_DIM_V1 = 14
OBS_DIM = 16
ACT_DIM = 4


def _active_param_space(param_space: PhysicsParameterSpace | None = None) -> PhysicsParameterSpace:
    return param_space or PhysicsParameterSpace.default()


def rescale_action(
    action: NDArray[np.float32],
    param_space: PhysicsParameterSpace | None = None,
) -> dict[str, float | int | str]:
    """Map normalized [-1, 1] actions to physical pulse parameters."""
    space = _active_param_space(param_space)
    ranges = space.rl_action_ranges()
    clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    amplitude = ranges["amplitude"][0] + (clipped[0] + 1.0) / 2.0 * (ranges["amplitude"][1] - ranges["amplitude"][0])
    detuning = ranges["detuning"][0] + (clipped[1] + 1.0) / 2.0 * (ranges["detuning"][1] - ranges["detuning"][0])
    duration_raw = ranges["duration_ns"][0] + (clipped[2] + 1.0) / 2.0 * (ranges["duration_ns"][1] - ranges["duration_ns"][0])
    duration_ns = max(16, int(round(duration_raw / 4.0) * 4))
    family_idx = int(np.clip(round((clipped[3] + 1.0) / 2.0 * (len(FAMILY_LIST) - 1)), 0, len(FAMILY_LIST) - 1))
    family = FAMILY_LIST[family_idx]

    return {
        "amplitude": round(float(amplitude), 4),
        "detuning": round(float(detuning), 4),
        "duration_ns": duration_ns,
        "family": family.value,
        "family_enum": family,
    }


def inverse_rescale(
    params: dict[str, Any],
    param_space: PhysicsParameterSpace | None = None,
) -> NDArray[np.float32]:
    """Map physical parameters back to [-1, 1] normalized actions."""
    space = _active_param_space(param_space)
    ranges = space.rl_action_ranges()

    amp = float(params.get("amplitude", 5.0))
    det = float(params.get("detuning", -10.0))
    dur = float(params.get("duration_ns", 3000.0))
    fam = str(params.get("family", SequenceFamily.ADIABATIC_SWEEP.value))

    a0 = (amp - ranges["amplitude"][0]) / max(ranges["amplitude"][1] - ranges["amplitude"][0], 1e-6) * 2.0 - 1.0
    a1 = (det - ranges["detuning"][0]) / max(ranges["detuning"][1] - ranges["detuning"][0], 1e-6) * 2.0 - 1.0
    a2 = (dur - ranges["duration_ns"][0]) / max(ranges["duration_ns"][1] - ranges["duration_ns"][0], 1e-6) * 2.0 - 1.0

    family_index = 0
    for index, family in enumerate(FAMILY_LIST):
        if family.value == fam:
            family_index = index
            break
    a3 = family_index / max(len(FAMILY_LIST) - 1, 1) * 2.0 - 1.0
    return np.array([a0, a1, a2, a3], dtype=np.float32)


class PulseDesignEnv:
    """Gymnasium-compatible environment for pulse-sequence parameter selection."""

    def __init__(
        self,
        spec: ExperimentSpec,
        register_candidates: list[RegisterCandidate],
        param_space: PhysicsParameterSpace | None = None,
        max_steps: int = 5,
        simulate_fn: Any | None = None,
        reward_shaping: bool = True,
        improvement_weight: float = 0.5,
        improvement_scale: float = 5.0,
    ) -> None:
        self.spec = spec
        self.param_space = param_space or PhysicsParameterSpace.default()
        self._all_register_candidates = list(register_candidates)
        self.register_candidates = list(register_candidates)
        self.max_steps = max_steps
        self._simulate_fn = simulate_fn
        self._reward_shaping = reward_shaping
        self._improvement_weight = improvement_weight
        self._improvement_scale = improvement_scale

        self._rng = np.random.default_rng(42)
        self._step_count = 0
        self._current_register: RegisterCandidate | None = None
        self._best_robustness = 0.0
        self._best_nominal = 0.0
        self._best_observable = 0.0
        self._best_params: dict[str, Any] = {}
        self._episode_history: list[dict[str, Any]] = []

    @property
    def observation_space_shape(self) -> tuple[int]:
        return (OBS_DIM,)

    @property
    def action_space_shape(self) -> tuple[int]:
        return (ACT_DIM,)

    def _build_observation(self) -> NDArray[np.float32]:
        register = self._current_register
        if register is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        from packages.ml.dataset import _encode_layout

        spacing = float(register.metadata.get("spacing_um", self.param_space.geometry.spacing_um.default))
        return np.array(
            [
                float(register.atom_count),
                spacing,
                float(register.blockade_radius_um),
                float(register.feasibility_score),
                float(self.spec.target_density),
                float(self.spec.min_atoms),
                float(self.spec.max_atoms),
                float(_encode_layout(register.layout_type)),
                float(self._best_robustness),
                float(self._best_params.get("amplitude", 0.0)) / 15.0,
                float(self._best_params.get("detuning", 0.0)) / 30.0,
                float(self._best_params.get("duration_ns", 0.0)) / 5500.0,
                float(self._step_count) / max(self.max_steps, 1),
                float(max(0, self.max_steps - self._step_count)) / max(self.max_steps, 1),
                float(self._best_nominal),
                float(self._best_observable),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        pool = self.register_candidates or self._all_register_candidates
        if not pool:
            self._current_register = None
            self._step_count = 0
            self._best_robustness = 0.0
            self._best_nominal = 0.0
            self._best_observable = 0.0
            self._best_params = {}
            self._episode_history = []
            return self._build_observation(), {"register_id": None}

        index = int(self._rng.integers(0, len(pool)))
        self._current_register = pool[index]
        self._step_count = 0
        self._best_robustness = 0.0
        self._best_nominal = 0.0
        self._best_observable = 0.0
        self._best_params = {}
        self._episode_history = []
        return self._build_observation(), {"register_id": self._current_register.id}

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        params = rescale_action(action, self.param_space)
        register = self._current_register

        raw_reward = 0.0
        reward = 0.0
        info: dict[str, Any] = {
            "params": params,
            "step": self._step_count,
        }

        if register is not None:
            raw_reward, details = self._evaluate(register, params)
            info["raw_robustness"] = raw_reward

            improvement = max(0.0, raw_reward - self._best_robustness)
            if self._reward_shaping:
                base_weight = 1.0 - self._improvement_weight
                reward = (
                    base_weight * raw_reward
                    + self._improvement_weight * improvement * self._improvement_scale
                )
            else:
                reward = raw_reward

            info["shaped_reward"] = reward
            info["improvement"] = improvement

            if raw_reward > self._best_robustness:
                self._best_robustness = raw_reward
                self._best_nominal = float(details.get("nominal_score", raw_reward))
                self._best_observable = float(details.get("observable_score", raw_reward))
                self._best_params = params

            self._episode_history.append(
                {
                    "step": self._step_count,
                    "params": dict(params),
                    "raw_reward": raw_reward,
                    "shaped_reward": reward,
                    "best_so_far": self._best_robustness,
                }
            )
        else:
            info["raw_robustness"] = raw_reward
            info["shaped_reward"] = reward
            info["improvement"] = 0.0

        terminated = self._step_count >= self.max_steps
        if terminated and self._reward_shaping:
            reward += self._best_robustness * 0.5

        info["robustness_score"] = raw_reward
        info["episode_best"] = self._best_robustness
        info["episode_history"] = list(self._episode_history) if terminated else []

        return self._build_observation(), float(reward), terminated, False, info

    def _evaluate(
        self,
        register: RegisterCandidate,
        params: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        if self._simulate_fn is not None:
            reward = float(self._simulate_fn(register, params))
            return reward, {
                "nominal_score": reward,
                "observable_score": reward,
            }

        from packages.simulation.evaluators import evaluate_candidate_robustness

        family_enum = params.get("family_enum", SequenceFamily.ADIABATIC_SWEEP)
        sequence = SequenceCandidate(
            campaign_id="rl_episode",
            spec_id=self.spec.id,
            register_candidate_id=register.id,
            label=f"rl-{params['family']}",
            sequence_family=family_enum,
            channel_id="rydberg_global",
            duration_ns=int(params["duration_ns"]),
            amplitude=float(params["amplitude"]),
            detuning=float(params["detuning"]),
            phase=0.0,
            waveform_kind="constant",
            predicted_cost=self.param_space.cost_for(register.atom_count, int(params["duration_ns"])),
            reasoning_summary="RL-generated candidate.",
        )

        try:
            result = evaluate_candidate_robustness(
                self.spec,
                register,
                sequence,
                param_space=self.param_space,
            )
            nominal_score = float(result[0])
            observables = result[7]
            observable_score = float(observables.get("observable_score", nominal_score))
            return float(result[6]), {
                "nominal_score": nominal_score,
                "observable_score": observable_score,
            }
        except Exception:
            return 0.0, {
                "nominal_score": 0.0,
                "observable_score": 0.0,
            }
