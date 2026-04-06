"""Phase 3 - Training runner for surrogate, RL, and active learning."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packages.core.logging import get_logger

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training runs."""

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    surrogate_epochs: int = 100
    surrogate_batch_size: int = 64
    surrogate_lr: float = 1e-3
    rl_total_updates: int = 500
    rl_rollout_steps: int = 256
    rl_max_steps: int = 5
    rl_reward_shaping: bool = True
    use_curriculum: bool = True
    distributed: bool = False
    device: str = "auto"
    seed: int = 42

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        from packages.ml.gpu_backend import get_device

        return get_device()


class TrainingRunner:
    """Unified training runner for surrogate, RL, and active learning."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required: pip install 'cryoswarm-q[ml]'")
        self.config = config or TrainingConfig()
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _setup_distributed(self) -> int:
        if not self.config.distributed:
            return 0
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank

    def _cleanup_distributed(self) -> None:
        if self.config.distributed and dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def _build_surrogate_model(input_dim: int) -> tuple[Any, str]:
        from packages.ml.dataset import INPUT_DIM_V2
        from packages.ml.surrogate import SurrogateModel, SurrogateModelV2

        if input_dim >= INPUT_DIM_V2:
            return SurrogateModelV2(input_dim=input_dim), "v2"
        return SurrogateModel(input_dim=input_dim), "v1"

    def run_surrogate(
        self,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset | None = None,
    ) -> dict[str, Any]:
        from packages.ml.surrogate import SurrogateTrainer

        local_rank = self._setup_distributed()
        device = self.config.resolve_device()
        input_dim = int(train_dataset[0][0].shape[0]) if len(train_dataset) else 10
        model, version = self._build_surrogate_model(input_dim)
        model.to(device)

        if self.config.distributed and torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank])

        raw_model = model.module if isinstance(model, DDP) else model
        trainer = SurrogateTrainer(raw_model, lr=self.config.surrogate_lr)
        history = trainer.fit(
            train_dataset,
            val_dataset,
            epochs=self.config.surrogate_epochs,
            batch_size=self.config.surrogate_batch_size,
            log_dir=os.path.join(self.config.log_dir, "surrogate"),
        )

        checkpoint_path = os.path.join(self.config.checkpoint_dir, "surrogate_latest.pt")
        if local_rank == 0:
            raw_model.save(checkpoint_path)
            logger.info("Surrogate checkpoint saved to %s", checkpoint_path)

        self._cleanup_distributed()
        return {
            "history": history,
            "checkpoint": checkpoint_path,
            "input_dim": input_dim,
            "model_version": version,
        }

    def run_rl(
        self,
        spec: Any,
        register_candidates: list[Any],
        simulate_fn: Any | None = None,
    ) -> dict[str, Any]:
        from packages.ml.curriculum import CurriculumScheduler
        from packages.ml.ppo import PPOConfig, PPOTrainer
        from packages.ml.rl_env import PulseDesignEnv

        curriculum = None
        if self.config.use_curriculum:
            curriculum = CurriculumScheduler(
                mode="adaptive",
                total_updates=self.config.rl_total_updates,
            )

        env = PulseDesignEnv(
            spec=spec,
            register_candidates=register_candidates,
            max_steps=self.config.rl_max_steps,
            simulate_fn=simulate_fn,
            reward_shaping=self.config.rl_reward_shaping,
        )

        ppo_config = PPOConfig(
            total_updates=self.config.rl_total_updates,
            rollout_steps=self.config.rl_rollout_steps,
            log_dir=os.path.join(self.config.log_dir, "ppo"),
        )
        trainer = PPOTrainer(ppo_config, curriculum=curriculum)
        history = trainer.train(env, seed=self.config.seed)

        checkpoint_path = os.path.join(self.config.checkpoint_dir, "ppo_latest.pt")
        trainer.policy.save(checkpoint_path)
        logger.info("PPO policy saved to %s", checkpoint_path)

        if curriculum is not None:
            history["curriculum"] = curriculum.get_report()

        return {
            "history": history,
            "checkpoint": checkpoint_path,
        }

    def run_full_pipeline(
        self,
        train_dataset: TensorDataset,
        spec: Any,
        register_candidates: list[Any],
        val_dataset: TensorDataset | None = None,
        simulate_fn: Any | None = None,
    ) -> dict[str, Any]:
        from packages.core.models import RegisterCandidate, SequenceCandidate
        from packages.ml.dataset import INPUT_DIM_V2, build_feature_vector, build_feature_vector_v2

        logger.info("=== Phase 1: Training surrogate model ===")
        surrogate_result = self.run_surrogate(train_dataset, val_dataset)

        input_dim = surrogate_result["input_dim"]
        surrogate_model, _ = self._build_surrogate_model(input_dim)
        surrogate_model.load(surrogate_result["checkpoint"])
        surrogate_model.eval()

        def surrogate_simulate(register: RegisterCandidate, params: dict[str, Any]) -> float:
            from packages.core.enums import SequenceFamily

            family_enum = params.get("family_enum", SequenceFamily.ADIABATIC_SWEEP)
            sequence = SequenceCandidate(
                campaign_id="rl_surrogate",
                spec_id=spec.id,
                register_candidate_id=register.id,
                label="rl-surrogate",
                sequence_family=family_enum,
                duration_ns=params["duration_ns"],
                amplitude=params["amplitude"],
                detuning=params["detuning"],
                phase=0.0,
                waveform_kind="constant",
                predicted_cost=0.0,
                reasoning_summary="Surrogate-evaluated RL candidate.",
            )
            features = (
                build_feature_vector_v2(register, sequence)
                if input_dim >= INPUT_DIM_V2
                else build_feature_vector(register, sequence)
            )
            return float(surrogate_model.predict_robustness(features))

        logger.info("=== Phase 2: Training RL policy with surrogate ===")
        rl_result = self.run_rl(
            spec,
            register_candidates,
            simulate_fn=simulate_fn or surrogate_simulate,
        )

        return {
            "surrogate": surrogate_result,
            "rl": rl_result,
        }

    def run_active_learning(
        self,
        initial_features: Any,
        initial_targets: Any,
        spec: Any,
        register_candidates: list[Any],
        n_iterations: int = 5,
        top_k: int = 200,
    ) -> dict[str, Any]:
        from packages.ml.active_learning import ActiveLearningConfig, ActiveLearningLoop

        config = ActiveLearningConfig(
            n_iterations=n_iterations,
            top_k_per_iteration=top_k,
            checkpoint_dir=self.config.checkpoint_dir,
            seed=self.config.seed,
            surrogate_epochs=max(1, self.config.surrogate_epochs // 2),
            rl_updates=max(1, self.config.rl_total_updates // 5),
            rl_max_steps=self.config.rl_max_steps,
        )
        loop = ActiveLearningLoop(
            config=config,
            initial_features=initial_features,
            initial_targets=initial_targets,
            spec=spec,
            register_candidates=register_candidates,
        )
        results = loop.run()
        results["final_features"] = loop.features
        results["final_targets"] = loop.targets
        return results
