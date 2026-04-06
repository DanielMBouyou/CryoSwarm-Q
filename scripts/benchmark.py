"""Standardized benchmark suite for CryoSwarm-Q ML models."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from packages.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SurrogateBenchmark:
    mse_total: float
    mae_total: float
    r2_total: float
    mse_per_target: dict[str, float]
    mae_per_target: dict[str, float]
    r2_per_target: dict[str, float]
    calibration_error: float
    worst_prediction_error: float
    n_test_samples: int


@dataclass
class RLBenchmark:
    avg_reward_last_50: float
    best_episode_reward: float
    episodes_to_threshold: int | None
    reward_std: float
    n_episodes: int


@dataclass
class PipelineBenchmark:
    top_candidate_robustness: float
    mean_ranked_robustness: float
    n_candidates_evaluated: int
    heuristic_vs_rl_comparison: dict[str, float] | None


@dataclass
class FullBenchmark:
    surrogate: SurrogateBenchmark | None
    rl: RLBenchmark | None
    pipeline: PipelineBenchmark | None
    timestamp: str
    git_hash: str | None
    config: dict[str, Any]


TARGET_NAMES = ["robustness", "nominal", "worst_case", "observable"]


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_surrogate_benchmark(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> SurrogateBenchmark:
    mse_per_target = {
        name: float(np.mean((y_true[:, index] - y_pred[:, index]) ** 2))
        for index, name in enumerate(TARGET_NAMES)
    }
    mae_per_target = {
        name: float(np.mean(np.abs(y_true[:, index] - y_pred[:, index])))
        for index, name in enumerate(TARGET_NAMES)
    }
    r2_per_target = {
        name: _r2_score(y_true[:, index], y_pred[:, index])
        for index, name in enumerate(TARGET_NAMES)
    }
    return SurrogateBenchmark(
        mse_total=float(np.mean((y_true - y_pred) ** 2)),
        mae_total=float(np.mean(np.abs(y_true - y_pred))),
        r2_total=_r2_score(y_true.reshape(-1), y_pred.reshape(-1)),
        mse_per_target=mse_per_target,
        mae_per_target=mae_per_target,
        r2_per_target=r2_per_target,
        calibration_error=float(abs(y_pred.mean() - y_true.mean())),
        worst_prediction_error=float(np.max(np.abs(y_true - y_pred))),
        n_test_samples=int(y_true.shape[0]),
    )


def compute_rl_benchmark(
    episode_rewards: list[float],
    threshold: float = 0.3,
) -> RLBenchmark:
    episodes_to_threshold = next(
        (index + 1 for index, reward in enumerate(episode_rewards) if reward > threshold),
        None,
    )
    tail = episode_rewards[-50:] if episode_rewards else []
    return RLBenchmark(
        avg_reward_last_50=float(np.mean(tail)) if tail else 0.0,
        best_episode_reward=float(max(episode_rewards)) if episode_rewards else 0.0,
        episodes_to_threshold=episodes_to_threshold,
        reward_std=float(np.std(episode_rewards)) if episode_rewards else 0.0,
        n_episodes=len(episode_rewards),
    )


def compute_pipeline_benchmark(
    robustness_scores: list[float],
    source_scores: dict[str, list[float]] | None = None,
) -> PipelineBenchmark:
    comparison = None
    if source_scores:
        comparison = {
            source: float(np.mean(scores))
            for source, scores in source_scores.items()
            if scores
        }
    return PipelineBenchmark(
        top_candidate_robustness=float(max(robustness_scores)) if robustness_scores else 0.0,
        mean_ranked_robustness=float(np.mean(robustness_scores)) if robustness_scores else 0.0,
        n_candidates_evaluated=len(robustness_scores),
        heuristic_vs_rl_comparison=comparison,
    )


def _load_surrogate_predictor(model_path: Path):
    from packages.ml.surrogate import SurrogateEnsemble, SurrogateModel, SurrogateModelV2

    if model_path.is_dir():
        meta = json.loads((model_path / "ensemble_meta.json").read_text(encoding="utf-8"))
        model_class = SurrogateModelV2 if meta.get("model_class", "SurrogateModelV2") == "SurrogateModelV2" else SurrogateModel
        ensemble = SurrogateEnsemble(
            n_models=int(meta.get("n_models", 3)),
            model_class=model_class,
            **meta.get("model_kwargs", {}),
        )
        ensemble.load(model_path)
        ensemble.eval()
        return ensemble, "ensemble"

    import torch

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    version = checkpoint.get("version") if isinstance(checkpoint, dict) else None
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if version == "v2":
        model = SurrogateModelV2(
            input_dim=int(config.get("input_dim", 18)),
            output_dim=int(config.get("output_dim", 4)),
            hidden=int(config.get("hidden", 128)),
            n_blocks=int(config.get("n_blocks", 3)),
            dropout=float(config.get("dropout", 0.1)),
        )
    else:
        model = SurrogateModel(
            input_dim=int(config.get("input_dim", 10)),
            output_dim=int(config.get("output_dim", 4)),
            hidden=int(config.get("hidden", 64)),
        )
    model.load(model_path)
    model.eval()
    return model, version or "v1"


def benchmark_surrogate(model_path: str, test_data: str) -> SurrogateBenchmark:
    predictor, kind = _load_surrogate_predictor(Path(model_path))
    data = np.load(test_data)
    features = np.asarray(data["features"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    if kind == "ensemble":
        predictions, _ = predictor.predict_with_uncertainty(features)
    else:
        predictions = predictor.predict_numpy(features)
    return compute_surrogate_benchmark(targets, predictions)


def benchmark_rl_checkpoint(
    checkpoint_path: str,
    env_atoms: int = 6,
    n_episodes: int = 50,
) -> RLBenchmark:
    from packages.agents.geometry_agent import GeometryAgent
    from packages.agents.problem_agent import ProblemFramingAgent
    from packages.core.models import ExperimentGoal
    from packages.ml.ppo import ActorCritic
    from packages.ml.rl_env import PulseDesignEnv

    goal = ExperimentGoal(
        title="RL benchmark",
        scientific_objective="Benchmark PPO policy.",
        target_observable="rydberg_density",
        desired_atom_count=env_atoms,
        preferred_geometry="mixed",
    )
    spec = ProblemFramingAgent().run(goal)
    registers = GeometryAgent().run(spec, "benchmark_rl", memory_records=[])
    env = PulseDesignEnv(spec=spec, register_candidates=registers, max_steps=5, reward_shaping=True)
    policy = ActorCritic.from_checkpoint(checkpoint_path)
    policy.eval()

    episode_rewards: list[float] = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        total_reward = 0.0
        for _ in range(env.max_steps):
            action, _, _ = policy.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
    return compute_rl_benchmark(episode_rewards)


def benchmark_pipeline_run(
    rl_checkpoint_path: str | None = None,
    sequence_strategy_mode: str = "adaptive",
) -> PipelineBenchmark:
    from packages.core.models import ExperimentGoal
    from packages.orchestration.pipeline import CryoSwarmPipeline

    pipeline = CryoSwarmPipeline(
        repository=None,
        parallel=False,
        sequence_strategy_mode=sequence_strategy_mode,
        rl_checkpoint_path=rl_checkpoint_path,
    )
    goal = ExperimentGoal(
        title="Pipeline benchmark",
        scientific_objective="Benchmark ranking quality.",
        target_observable="rydberg_density",
        desired_atom_count=6,
        preferred_geometry="mixed",
    )
    summary = pipeline.run(goal)
    source_scores: dict[str, list[float]] = {}
    report_lookup = {report.sequence_candidate_id: report for report in summary.robustness_reports}
    sequence_lookup = {sequence.id: sequence for sequence in summary.sequences}
    for evaluation in summary.ranked_candidates:
        sequence = sequence_lookup.get(evaluation.sequence_candidate_id)
        report = report_lookup.get(evaluation.sequence_candidate_id)
        if sequence is None or report is None:
            continue
        source = str(sequence.metadata.get("source", "unknown"))
        source_scores.setdefault(source, []).append(float(report.robustness_score))
    return compute_pipeline_benchmark(
        [float(item.robustness_score) for item in summary.ranked_candidates],
        source_scores=source_scores if source_scores else None,
    )


def _git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def save_benchmark_report(report: FullBenchmark, output_dir: str = "benchmarks") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report.timestamp.replace(":", "-")
    path = out_dir / f"results_{timestamp}.json"
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="CryoSwarm-Q benchmark suite")
    parser.add_argument("--model", help="Surrogate checkpoint path or ensemble directory.")
    parser.add_argument("--test-data", help="Fixed test dataset (.npz).")
    parser.add_argument("--rl-checkpoint", help="RL checkpoint path.")
    parser.add_argument("--env-atoms", type=int, default=6)
    parser.add_argument("--full", action="store_true", help="Run all available benchmarks.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint dir for --full.")
    args = parser.parse_args()

    surrogate_report = None
    rl_report = None
    pipeline_report = None

    if args.model and args.test_data:
        surrogate_report = benchmark_surrogate(args.model, args.test_data)
    if args.rl_checkpoint:
        rl_report = benchmark_rl_checkpoint(args.rl_checkpoint, env_atoms=args.env_atoms)
    if args.full:
        checkpoint_dir = Path(args.checkpoint_dir)
        surrogate_path = checkpoint_dir / "surrogate_latest.pt"
        rl_path = checkpoint_dir / "ppo_latest.pt"
        if args.test_data and surrogate_path.exists():
            surrogate_report = benchmark_surrogate(str(surrogate_path), args.test_data)
        if rl_path.exists():
            rl_report = benchmark_rl_checkpoint(str(rl_path), env_atoms=args.env_atoms)
            pipeline_report = benchmark_pipeline_run(
                rl_checkpoint_path=str(rl_path),
                sequence_strategy_mode="adaptive",
            )

    report = FullBenchmark(
        surrogate=surrogate_report,
        rl=rl_report,
        pipeline=pipeline_report,
        timestamp=datetime.utcnow().isoformat(),
        git_hash=_git_hash(),
        config=vars(args),
    )
    output_path = save_benchmark_report(report)
    logger.info("Benchmark report saved to %s", output_path)


if __name__ == "__main__":
    main()
