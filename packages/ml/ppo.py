"""Phase 2 - PPO agent for pulse-sequence parameter optimization."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from packages.ml.rl_env import ACT_DIM, OBS_DIM

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore[assignment]


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required: pip install 'cryoswarm-q[ml]'")


@dataclass
class PPOConfig:
    """PPO training configuration."""

    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon_clip: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    epochs_per_update: int = 4
    batch_size: int = 64
    rollout_steps: int = 256
    total_updates: int = 500
    log_dir: str | None = None
    normalize_rewards: bool = True
    reward_normalizer_epsilon: float = 1e-4
    reward_clip_value: float = 5.0
    use_replay: bool = True
    replay_capacity: int = 4096
    replay_batch_size: int = 64
    replay_updates_per_update: int = 1
    replay_warmup_updates: int = 2
    max_replay_policy_lag: int = 2


class ActorCritic(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Shared-backbone actor-critic for continuous pulse parameter control."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden: int = 128,
    ) -> None:
        _check_torch()
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _checkpoint_config(checkpoint: Any) -> dict[str, int]:
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            config = checkpoint.get("config", {})
            return {
                "obs_dim": int(config.get("obs_dim", OBS_DIM)),
                "act_dim": int(config.get("act_dim", ACT_DIM)),
                "hidden": int(config.get("hidden", 128)),
            }

        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        backbone_weight = state.get("backbone.0.weight")
        actor_weight = state.get("actor_mean.weight")
        return {
            "obs_dim": int(backbone_weight.shape[1]) if backbone_weight is not None else OBS_DIM,
            "act_dim": int(actor_weight.shape[0]) if actor_weight is not None else ACT_DIM,
            "hidden": int(backbone_weight.shape[0]) if backbone_weight is not None else 128,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "ActorCritic":
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        config = cls._checkpoint_config(checkpoint)
        model = cls(
            obs_dim=config["obs_dim"],
            act_dim=config["act_dim"],
            hidden=config["hidden"],
        )
        model.load(path)
        return model

    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs.float())
        mean = torch.tanh(self.actor_mean(features))
        std = torch.exp(self.actor_log_std.clamp(-2.0, 0.5))
        value = self.critic(features).squeeze(-1)
        return mean, std, value

    def get_action(
        self,
        obs: NDArray[np.float32],
        deterministic: bool = False,
    ) -> tuple[NDArray[np.float32], float, float]:
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        mean, std, value = self.forward(obs_t)
        dist = Normal(mean, std)

        action = mean if deterministic else dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            action.squeeze(0).detach().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, values = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values, entropy

    def save(self, path: str | Path) -> None:
        state = {
            "model": self.state_dict(),
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden": self.hidden,
            },
            "version": "ppo_v2",
        }
        torch.save(state, str(path))

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.load_state_dict(checkpoint["model"])
            return
        self.load_state_dict(checkpoint)


class RolloutBuffer:
    """Stores transitions from environment interaction for PPO updates."""

    def __init__(self) -> None:
        self.observations: list[NDArray[np.float32]] = []
        self.actions: list[NDArray[np.float32]] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.raw_rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        obs: NDArray[np.float32],
        action: NDArray[np.float32],
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        raw_reward: float | None = None,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.raw_rewards.append(float(raw_reward if raw_reward is not None else reward))
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_torch()
        count = len(self.rewards)
        advantages = np.zeros(count, dtype=np.float32)
        gae = 0.0
        for index in reversed(range(count)):
            next_value = last_value if index == count - 1 else self.values[index + 1]
            next_non_terminal = 0.0 if self.dones[index] else 1.0
            delta = self.rewards[index] + gamma * next_value * next_non_terminal - self.values[index]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[index] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return torch.from_numpy(advantages), torch.from_numpy(returns)

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_torch()
        obs = torch.from_numpy(np.stack(self.observations).astype(np.float32))
        actions = torch.from_numpy(np.stack(self.actions).astype(np.float32))
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        return obs, actions, old_log_probs

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.raw_rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class RunningMeanStd:
    """Numerically stable running mean/std estimator for reward normalization."""

    def __init__(
        self,
        epsilon: float = 1e-4,
        clip_value: float | None = 5.0,
    ) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)
        self.clip_value = clip_value

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 1e-8))

    def update(self, values: NDArray[np.float32] | list[float] | tuple[float, ...]) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return

        batch_mean = float(np.mean(array))
        batch_var = float(np.var(array))
        batch_count = float(array.size)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / max(total_count, 1e-8)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / max(total_count, 1e-8)

        self.mean = new_mean
        self.var = m2 / max(total_count, 1e-8)
        self.count = total_count

    def normalize(self, value: float) -> float:
        normalized = (float(value) - self.mean) / self.std
        if self.clip_value is not None:
            normalized = float(np.clip(normalized, -self.clip_value, self.clip_value))
        return float(normalized)


@dataclass
class ReplaySample:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ExperienceReplayBuffer:
    """Recent-rollout replay for PPO auxiliary updates."""

    def __init__(
        self,
        capacity: int = 4096,
        seed: int = 42,
    ) -> None:
        self.capacity = max(int(capacity), 1)
        self._rng = np.random.default_rng(seed)
        self._batches: list[dict[str, Any]] = []
        self._size = 0

    def add_rollout(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        policy_version: int,
    ) -> None:
        batch = {
            "observations": observations.detach().cpu().numpy().astype(np.float32, copy=False),
            "actions": actions.detach().cpu().numpy().astype(np.float32, copy=False),
            "old_log_probs": old_log_probs.detach().cpu().numpy().astype(np.float32, copy=False),
            "advantages": advantages.detach().cpu().numpy().astype(np.float32, copy=False),
            "returns": returns.detach().cpu().numpy().astype(np.float32, copy=False),
            "policy_version": int(policy_version),
        }
        self._batches.append(batch)
        self._size += int(batch["observations"].shape[0])

        while self._size > self.capacity and self._batches:
            removed = self._batches.pop(0)
            self._size -= int(removed["observations"].shape[0])

    def sample(
        self,
        batch_size: int,
        current_policy_version: int,
        max_policy_lag: int,
    ) -> ReplaySample | None:
        eligible = [
            batch
            for batch in self._batches
            if current_policy_version - int(batch["policy_version"]) <= max_policy_lag
        ]
        if not eligible:
            return None

        observations = np.concatenate([batch["observations"] for batch in eligible], axis=0)
        actions = np.concatenate([batch["actions"] for batch in eligible], axis=0)
        old_log_probs = np.concatenate([batch["old_log_probs"] for batch in eligible], axis=0)
        advantages = np.concatenate([batch["advantages"] for batch in eligible], axis=0)
        returns = np.concatenate([batch["returns"] for batch in eligible], axis=0)

        sample_size = min(max(int(batch_size), 1), observations.shape[0])
        indices = self._rng.choice(observations.shape[0], size=sample_size, replace=False)

        return ReplaySample(
            observations=torch.from_numpy(observations[indices].astype(np.float32, copy=False)),
            actions=torch.from_numpy(actions[indices].astype(np.float32, copy=False)),
            old_log_probs=torch.from_numpy(old_log_probs[indices].astype(np.float32, copy=False)),
            advantages=torch.from_numpy(advantages[indices].astype(np.float32, copy=False)),
            returns=torch.from_numpy(returns[indices].astype(np.float32, copy=False)),
        )

    def __len__(self) -> int:
        return self._size


class PPOTrainer:
    """Proximal Policy Optimization training loop."""

    def __init__(
        self,
        config: PPOConfig | None = None,
        curriculum: Any | None = None,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden: int = 128,
    ) -> None:
        _check_torch()
        self.config = config or PPOConfig()
        self.curriculum = curriculum
        self.policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.backbone.parameters(), "lr": self.config.lr_actor},
                {"params": self.policy.actor_mean.parameters(), "lr": self.config.lr_actor},
                {"params": [self.policy.actor_log_std], "lr": self.config.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": self.config.lr_critic},
            ]
        )

    def _compute_losses(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, values, entropy = self.policy.evaluate(observations, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.epsilon_clip,
            1.0 + self.config.epsilon_clip,
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((returns - values) ** 2).mean()
        entropy_loss = -entropy.mean()
        return policy_loss, value_loss, entropy_loss

    def _optimize_minibatch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[float, float, float]:
        policy_loss, value_loss, entropy_loss = self._compute_losses(
            observations,
            actions,
            old_log_probs,
            advantages,
            returns,
        )
        loss = (
            policy_loss
            + self.config.value_coeff * value_loss
            + self.config.entropy_coeff * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return float(policy_loss.item()), float(value_loss.item()), float(loss.item())

    def train(
        self,
        env: Any,
        seed: int = 42,
    ) -> dict[str, list[float]]:
        writer = None
        if self.config.log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(self.config.log_dir)
            except ImportError:
                pass

        history: dict[str, list[float]] = {
            "episode_rewards": [],
            "normalized_episode_rewards": [],
            "policy_loss": [],
            "value_loss": [],
            "replay_loss": [],
            "reward_mean": [],
            "reward_std": [],
        }

        obs, _ = env.reset(seed=seed)
        buffer = RolloutBuffer()
        reward_normalizer = (
            RunningMeanStd(
                epsilon=self.config.reward_normalizer_epsilon,
                clip_value=self.config.reward_clip_value,
            )
            if self.config.normalize_rewards
            else None
        )
        replay_buffer = (
            ExperienceReplayBuffer(capacity=self.config.replay_capacity, seed=seed)
            if self.config.use_replay and self.config.replay_updates_per_update > 0
            else None
        )
        episode_reward_raw = 0.0
        episode_reward_normalized = 0.0

        if self.curriculum is not None:
            env.register_candidates = self.curriculum.filter_registers(env._all_register_candidates)

        for update in range(self.config.total_updates):
            for _ in range(self.config.rollout_steps):
                action, log_prob, value = self.policy.get_action(obs)
                next_obs, raw_reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                normalized_reward = float(raw_reward)
                if reward_normalizer is not None:
                    reward_normalizer.update([float(raw_reward)])
                    normalized_reward = reward_normalizer.normalize(float(raw_reward))

                buffer.add(
                    obs,
                    action,
                    log_prob,
                    normalized_reward,
                    value,
                    done,
                    raw_reward=float(raw_reward),
                )
                episode_reward_raw += float(raw_reward)
                episode_reward_normalized += float(normalized_reward)
                obs = next_obs

                if done:
                    history["episode_rewards"].append(episode_reward_raw)
                    history["normalized_episode_rewards"].append(episode_reward_normalized)
                    if self.curriculum is not None:
                        self.curriculum.record_episode(episode_reward_raw)
                        env.register_candidates = self.curriculum.filter_registers(env._all_register_candidates)
                    episode_reward_raw = 0.0
                    episode_reward_normalized = 0.0
                    obs, _ = env.reset()

            _, _, last_value = self.policy.get_action(obs, deterministic=True)
            advantages, returns = buffer.compute_gae(
                last_value,
                self.config.gamma,
                self.config.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            obs_t, actions_t, old_log_probs_t = buffer.to_tensors()

            total_policy_loss = 0.0
            total_value_loss = 0.0
            minibatch_updates = 0

            for _ in range(self.config.epochs_per_update):
                indices = torch.randperm(len(buffer))
                for start in range(0, len(buffer), self.config.batch_size):
                    end = min(start + self.config.batch_size, len(buffer))
                    idx = indices[start:end]
                    policy_loss, value_loss, _ = self._optimize_minibatch(
                        obs_t[idx],
                        actions_t[idx],
                        old_log_probs_t[idx],
                        advantages[idx],
                        returns[idx],
                    )
                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
                    minibatch_updates += 1

            history["policy_loss"].append(total_policy_loss / max(minibatch_updates, 1))
            history["value_loss"].append(total_value_loss / max(minibatch_updates, 1))
            history["reward_mean"].append(reward_normalizer.mean if reward_normalizer is not None else 0.0)
            history["reward_std"].append(reward_normalizer.std if reward_normalizer is not None else 1.0)

            replay_loss = 0.0
            replay_updates = 0
            if replay_buffer is not None:
                replay_buffer.add_rollout(
                    obs_t,
                    actions_t,
                    old_log_probs_t,
                    advantages,
                    returns,
                    policy_version=update,
                )
                if update + 1 >= self.config.replay_warmup_updates:
                    for _ in range(self.config.replay_updates_per_update):
                        sample = replay_buffer.sample(
                            batch_size=self.config.replay_batch_size,
                            current_policy_version=update,
                            max_policy_lag=self.config.max_replay_policy_lag,
                        )
                        if sample is None:
                            break
                        _, _, total_loss = self._optimize_minibatch(
                            sample.observations,
                            sample.actions,
                            sample.old_log_probs,
                            sample.advantages,
                            sample.returns,
                        )
                        replay_loss += total_loss
                        replay_updates += 1
            history["replay_loss"].append(replay_loss / max(replay_updates, 1) if replay_updates else 0.0)

            if self.curriculum is not None:
                stage_changed = self.curriculum.step_update()
                if stage_changed:
                    env.register_candidates = self.curriculum.filter_registers(env._all_register_candidates)

            if writer:
                writer.add_scalar("loss/policy", history["policy_loss"][-1], update)
                writer.add_scalar("loss/value", history["value_loss"][-1], update)
                writer.add_scalar("loss/replay", history["replay_loss"][-1], update)
                writer.add_scalar("reward/running_mean", history["reward_mean"][-1], update)
                writer.add_scalar("reward/running_std", history["reward_std"][-1], update)
                if history["episode_rewards"]:
                    writer.add_scalar("reward/episode", history["episode_rewards"][-1], update)
                if history["normalized_episode_rewards"]:
                    writer.add_scalar(
                        "reward/episode_normalized",
                        history["normalized_episode_rewards"][-1],
                        update,
                    )

            buffer.clear()

        if writer:
            writer.close()

        return history
