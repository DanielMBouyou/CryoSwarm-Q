"""Active learning loop for iterative surrogate-RL co-training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from packages.core.logging import get_logger
from packages.core.models import ExperimentSpec, RegisterCandidate, SequenceCandidate

logger = get_logger(__name__)


@dataclass
class ActiveLearningConfig:
    """Configuration for the active learning loop."""

    n_iterations: int = 5
    top_k_per_iteration: int = 200
    diversity_fraction: float = 0.5
    uncertainty_weight: float = 0.3
    surrogate_epochs: int = 50
    surrogate_lr: float = 5e-4
    rl_updates: int = 100
    rl_max_steps: int = 5
    rl_rollout_episodes: int = 200
    max_atoms_for_resim: int = 12
    n_workers: int = 4
    checkpoint_dir: str = "checkpoints/active_learning"
    seed: int = 42


class ActiveLearningLoop:
    """Iterative surrogate-RL co-training with simulation-based augmentation."""

    def __init__(
        self,
        config: ActiveLearningConfig,
        initial_features: NDArray[np.float32],
        initial_targets: NDArray[np.float32],
        spec: ExperimentSpec,
        register_candidates: list[RegisterCandidate],
        param_space: Any | None = None,
        simulate_fn: Callable | None = None,
    ) -> None:
        self.config = config
        self.features = np.asarray(initial_features, dtype=np.float32).copy()
        self.targets = np.asarray(initial_targets, dtype=np.float32).copy()
        self.spec = spec
        self.register_candidates = list(register_candidates)
        self.param_space = param_space
        self.simulate_fn = simulate_fn
        self._rng = np.random.default_rng(config.seed)

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "iterations": [],
            "dataset_growth": [len(self.features)],
        }

        for iteration in range(self.config.n_iterations):
            logger.info(
                "=== Active Learning iteration %d/%d (dataset: %d samples) ===",
                iteration + 1,
                self.config.n_iterations,
                len(self.features),
            )
            iter_result = self._run_iteration(iteration)
            results["iterations"].append(iter_result)
            results["dataset_growth"].append(len(self.features))

        results["final_surrogate_path"] = str(Path(self.config.checkpoint_dir) / "surrogate_final.pt")
        results["final_rl_path"] = str(Path(self.config.checkpoint_dir) / "ppo_final.pt")
        return results

    def _feature_builder(self) -> Callable[[RegisterCandidate, SequenceCandidate], NDArray[np.float32]]:
        from packages.ml.dataset import INPUT_DIM_V2, build_feature_vector, build_feature_vector_v2

        if self.features.shape[1] >= INPUT_DIM_V2:
            return lambda register, sequence: build_feature_vector_v2(register, sequence, self.param_space)
        return build_feature_vector

    def _run_iteration(self, iteration: int) -> dict[str, Any]:
        import shutil
        import torch
        from torch.utils.data import TensorDataset

        from packages.ml.ppo import ActorCritic
        from packages.ml.surrogate import SurrogateModelV2, SurrogateTrainer
        from packages.ml.training_runner import TrainingConfig, TrainingRunner

        torch.manual_seed(self.config.seed + iteration)
        checkpoint_dir = Path(self.config.checkpoint_dir) / f"iter_{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        x_tensor = torch.from_numpy(self.features)
        y_tensor = torch.from_numpy(self.targets)
        n_val = max(1, len(self.features) // 10)
        indices = torch.randperm(len(self.features))
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
        val_ds = TensorDataset(x_tensor[val_idx], y_tensor[val_idx])

        surrogate = SurrogateModelV2(input_dim=self.features.shape[1])
        prev_checkpoint = checkpoint_dir.parent / f"iter_{iteration - 1}" / "surrogate.pt"
        if iteration > 0 and prev_checkpoint.exists():
            surrogate.load(prev_checkpoint)
            logger.info("Warm-starting surrogate from iteration %d", iteration - 1)

        trainer = SurrogateTrainer(surrogate, lr=self.config.surrogate_lr)
        history = trainer.fit(
            train_ds,
            val_ds,
            epochs=self.config.surrogate_epochs,
            batch_size=64,
        )
        surrogate_path = checkpoint_dir / "surrogate.pt"
        surrogate.save(surrogate_path)
        surrogate.save(Path(self.config.checkpoint_dir) / "surrogate_final.pt")
        surrogate.eval()

        feature_builder = self._feature_builder()

        def surrogate_sim(register: RegisterCandidate, params: dict[str, Any]) -> float:
            sequence = self._params_to_seq(register, params)
            features = feature_builder(register, sequence)
            return float(surrogate.predict_robustness(features))

        runner = TrainingRunner(
            TrainingConfig(
                checkpoint_dir=str(checkpoint_dir),
                rl_total_updates=self.config.rl_updates,
                rl_max_steps=self.config.rl_max_steps,
                rl_rollout_steps=max(32, self.config.rl_max_steps * 32),
                use_curriculum=True,
                seed=self.config.seed + iteration,
            )
        )
        rl_result = runner.run_rl(
            self.spec,
            self.register_candidates,
            simulate_fn=surrogate_sim,
        )

        rl_checkpoint = Path(rl_result["checkpoint"])
        if rl_checkpoint.exists():
            shutil.copy2(rl_checkpoint, Path(self.config.checkpoint_dir) / "ppo_final.pt")

        policy = ActorCritic.from_checkpoint(rl_checkpoint)
        policy.eval()

        collected_configs = self._collect_rl_configurations(policy, surrogate)
        selected_configs = self._select_diverse_configs(collected_configs, surrogate)
        new_samples = self._resimulate_configs(selected_configs)

        if new_samples:
            new_features = np.stack([sample[0] for sample in new_samples]).astype(np.float32)
            new_targets = np.stack([sample[1] for sample in new_samples]).astype(np.float32)
            self.features = np.concatenate([self.features, new_features], axis=0)
            self.targets = np.concatenate([self.targets, new_targets], axis=0)

        return {
            "iteration": iteration,
            "surrogate_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "surrogate_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "rl_episodes": len(rl_result["history"]["episode_rewards"]),
            "rl_avg_reward": (
                float(np.mean(rl_result["history"]["episode_rewards"][-20:]))
                if rl_result["history"]["episode_rewards"]
                else 0.0
            ),
            "configs_collected": len(collected_configs),
            "configs_selected": len(selected_configs),
            "new_samples": len(new_samples),
            "dataset_size": len(self.features),
        }

    def _collect_rl_configurations(
        self,
        policy: Any,
        surrogate: Any,
    ) -> list[dict[str, Any]]:
        from packages.ml.rl_env import PulseDesignEnv

        feature_builder = self._feature_builder()

        def surrogate_sim(register: RegisterCandidate, params: dict[str, Any]) -> float:
            sequence = self._params_to_seq(register, params)
            features = feature_builder(register, sequence)
            return float(surrogate.predict_robustness(features))

        env = PulseDesignEnv(
            spec=self.spec,
            register_candidates=self.register_candidates,
            max_steps=self.config.rl_max_steps,
            simulate_fn=surrogate_sim,
            reward_shaping=True,
        )

        configs: list[dict[str, Any]] = []
        for episode in range(self.config.rl_rollout_episodes):
            obs, _ = env.reset(seed=self.config.seed + episode)
            episode_best: dict[str, Any] | None = None

            for _ in range(self.config.rl_max_steps):
                action, _, _ = policy.get_action(obs, deterministic=False)
                obs, _, terminated, truncated, info = env.step(action)
                score = float(info.get("raw_robustness", info.get("robustness_score", 0.0)))
                if episode_best is None or score > episode_best["reward"]:
                    episode_best = {
                        "params": dict(info["params"]),
                        "reward": score,
                        "register_id": env._current_register.id if env._current_register else None,
                        "register": env._current_register,
                    }
                if terminated or truncated:
                    break

            if episode_best and episode_best["register"] is not None:
                configs.append(episode_best)
        return configs

    def _select_diverse_configs(
        self,
        configs: list[dict[str, Any]],
        surrogate: Any | None = None,
    ) -> list[dict[str, Any]]:
        if not configs:
            return []

        feature_builder = self._feature_builder()
        k = min(self.config.top_k_per_iteration, len(configs))

        acquisition_weight = self.config.uncertainty_weight
        scored_configs: list[tuple[dict[str, Any], float]] = []
        for config in configs:
            acquisition_score = float(config["reward"])
            if surrogate is not None and hasattr(surrogate, "predict_with_uncertainty"):
                sequence = self._params_to_seq(config["register"], config["params"])
                features = feature_builder(config["register"], sequence)
                mean_pred, uncertainty = surrogate.predict_with_uncertainty(features)
                acquisition_score = (
                    (1.0 - acquisition_weight) * float(mean_pred[0, 0])
                    + acquisition_weight * float(uncertainty[0, 0])
                )
            scored_configs.append((config, acquisition_score))

        sorted_configs = [item[0] for item in sorted(scored_configs, key=lambda item: item[1], reverse=True)]
        reward_pool_size = max(k, len(sorted_configs) // 2)
        reward_pool = sorted_configs[:reward_pool_size]

        if len(reward_pool) <= k:
            return reward_pool[:k]

        feature_matrix = np.stack(
            [
                feature_builder(config["register"], self._params_to_seq(config["register"], config["params"]))
                for config in reward_pool
            ]
        )
        feature_stds = feature_matrix.std(axis=0)
        feature_stds[feature_stds < 1e-8] = 1.0
        normalized = (feature_matrix - feature_matrix.mean(axis=0)) / feature_stds

        selected_indices = [0]
        while len(selected_indices) < k:
            min_dists = np.full(len(normalized), np.inf)
            for selected_idx in selected_indices:
                dists = np.linalg.norm(normalized - normalized[selected_idx], axis=1)
                min_dists = np.minimum(min_dists, dists)
            min_dists[selected_indices] = -np.inf
            selected_indices.append(int(np.argmax(min_dists)))

        return [reward_pool[index] for index in selected_indices]

    def _params_to_seq(
        self,
        register: RegisterCandidate,
        params: dict[str, Any],
    ) -> SequenceCandidate:
        from packages.core.enums import SequenceFamily

        family_enum = params.get("family_enum", SequenceFamily.ADIABATIC_SWEEP)
        metadata: dict[str, Any] = {}
        if "detuning_end" in params:
            metadata["detuning_end"] = params["detuning_end"]
        if "amplitude_start" in params:
            metadata["amplitude_start"] = params["amplitude_start"]

        return SequenceCandidate(
            campaign_id="active_learning",
            spec_id=self.spec.id,
            register_candidate_id=register.id,
            label="al-sequence",
            sequence_family=family_enum,
            duration_ns=int(params["duration_ns"]),
            amplitude=float(params["amplitude"]),
            detuning=float(params["detuning"]),
            phase=0.0,
            waveform_kind="constant",
            predicted_cost=0.0,
            reasoning_summary="Active learning candidate.",
            metadata=metadata,
        )

    def _resimulate_configs(
        self,
        configs: list[dict[str, Any]],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        from packages.simulation.evaluators import evaluate_candidate_robustness

        feature_builder = self._feature_builder()
        samples: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
        for config in configs:
            register = config["register"]
            params = config["params"]
            if register.atom_count > self.config.max_atoms_for_resim:
                continue

            sequence = self._params_to_seq(register, params)
            try:
                if self.simulate_fn is not None:
                    reward = float(self.simulate_fn(register, params))
                    target = np.array([reward, reward, reward * 0.8, reward * 0.9], dtype=np.float32)
                else:
                    result = evaluate_candidate_robustness(
                        self.spec,
                        register,
                        sequence,
                        param_space=self.param_space,
                    )
                    robustness = float(result[6])
                    nominal = float(result[0])
                    worst_case = float(result[3])
                    observable = float(result[7].get("observable_score", nominal))
                    target = np.array([robustness, nominal, worst_case, observable], dtype=np.float32)

                samples.append((feature_builder(register, sequence), target))
            except Exception as exc:
                logger.warning("Re-simulation failed for config: %s", exc)
                continue

        logger.info("Re-simulated %d/%d configurations successfully", len(samples), len(configs))
        return samples
