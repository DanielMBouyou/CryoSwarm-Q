"""CLI entry point for CryoSwarm-Q ML training.

Usage::

    # Phase 1: Train surrogate from saved pipeline data
    python -m scripts.train_ml --phase surrogate --data data/candidates.npz --epochs 200

    # Phase 2: Train RL policy
    python -m scripts.train_ml --phase rl --updates 500

    # Phase 3: Full pipeline (surrogate → RL with surrogate as simulator)
    python -m scripts.train_ml --phase full --data data/candidates.npz

    # Generate training data from demo pipeline
    python -m scripts.train_ml --phase generate --runs 10

All phases are disabled by default until explicitly invoked.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.core.logging import get_logger

logger = get_logger(__name__)


def generate_training_data(n_runs: int, output_path: str) -> None:
    """Run the heuristic pipeline repeatedly to build a training dataset."""
    from packages.core.models import ExperimentGoal
    from packages.ml.dataset import CandidateDatasetBuilder
    from packages.orchestration.pipeline import CryoSwarmPipeline

    builder = CandidateDatasetBuilder()
    pipeline = CryoSwarmPipeline(repository=None, parallel=False)

    for i in range(n_runs):
        atom_count = 4 + (i % 7)  # Vary atom count 4-10
        goal = ExperimentGoal(
            title=f"Training data generation run {i+1}",
            scientific_objective="Generate training data for ML models.",
            target_observable="rydberg_density",
            desired_atom_count=atom_count,
            preferred_geometry="mixed",
        )
        logger.info("Run %d/%d — %d atoms", i + 1, n_runs, atom_count)

        try:
            summary = pipeline.run(goal)
            if summary.ranked_candidates and summary.robustness_reports:
                added = builder.add_from_pipeline(
                    registers=summary.registers,
                    sequences=summary.sequences,
                    reports=summary.robustness_reports,
                    evaluations=summary.ranked_candidates,
                )
                logger.info("  → %d samples added (total: %d)", added, builder.size)
        except Exception as exc:
            logger.error("  → Run %d failed: %s", i + 1, exc)
            continue

    builder.save(output_path)
    logger.info("Dataset saved to %s — %d total samples", output_path, builder.size)


def train_surrogate(data_path: str, epochs: int, checkpoint_dir: str) -> None:
    """Train the surrogate model from a saved .npz dataset."""
    from packages.ml.dataset import CandidateDatasetBuilder
    from packages.ml.training_runner import TrainingConfig, TrainingRunner

    builder = CandidateDatasetBuilder()
    builder.load(data_path)
    logger.info("Loaded %d samples from %s", builder.size, data_path)

    if builder.size < 10:
        logger.error("Need at least 10 samples to train. Run --phase generate first.")
        return

    dataset = builder.to_torch_dataset()

    # 80/20 split
    import torch

    n_val = max(1, builder.size // 5)
    n_train = builder.size - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    config = TrainingConfig(
        checkpoint_dir=checkpoint_dir,
        surrogate_epochs=epochs,
    )
    runner = TrainingRunner(config)
    result = runner.run_surrogate(train_ds, val_ds)

    final_train_loss = result["history"]["train_loss"][-1]
    logger.info(
        "Training complete. Final train loss: %.6f. Checkpoint: %s",
        final_train_loss,
        result["checkpoint"],
    )


def train_rl(total_updates: int, checkpoint_dir: str) -> None:
    """Train the RL policy using the heuristic pipeline as environment."""
    from packages.core.models import ExperimentGoal
    from packages.agents.problem_agent import ProblemFramingAgent
    from packages.agents.geometry_agent import GeometryAgent
    from packages.ml.training_runner import TrainingConfig, TrainingRunner

    goal = ExperimentGoal(
        title="RL training environment",
        scientific_objective="Train RL agent for pulse design.",
        target_observable="rydberg_density",
        desired_atom_count=6,
        preferred_geometry="mixed",
    )

    spec = ProblemFramingAgent().run(goal)
    registers = GeometryAgent().run(spec, "rl_training", memory_records=[])

    if not registers:
        logger.error("No register candidates generated.")
        return

    logger.info("Training RL with %d register candidates", len(registers))

    config = TrainingConfig(
        checkpoint_dir=checkpoint_dir,
        rl_total_updates=total_updates,
    )
    runner = TrainingRunner(config)
    result = runner.run_rl(spec, registers)

    n_episodes = len(result["history"]["episode_rewards"])
    avg_reward = (
        sum(result["history"]["episode_rewards"][-10:]) / min(10, n_episodes)
        if n_episodes
        else 0.0
    )
    logger.info(
        "RL training complete. %d episodes, avg last-10 reward: %.4f. Checkpoint: %s",
        n_episodes,
        avg_reward,
        result["checkpoint"],
    )


def train_full(data_path: str, epochs: int, updates: int, checkpoint_dir: str) -> None:
    """Phase 3: Train surrogate then RL using surrogate as fast evaluator."""
    from packages.core.models import ExperimentGoal
    from packages.agents.problem_agent import ProblemFramingAgent
    from packages.agents.geometry_agent import GeometryAgent
    from packages.ml.dataset import CandidateDatasetBuilder
    from packages.ml.training_runner import TrainingConfig, TrainingRunner

    builder = CandidateDatasetBuilder()
    builder.load(data_path)

    if builder.size < 10:
        logger.error("Need at least 10 samples.")
        return

    dataset = builder.to_torch_dataset()
    import torch

    n_val = max(1, builder.size // 5)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [builder.size - n_val, n_val])

    goal = ExperimentGoal(
        title="Full ML pipeline training",
        scientific_objective="Train surrogate + RL for pulse design.",
        target_observable="rydberg_density",
        desired_atom_count=6,
        preferred_geometry="mixed",
    )
    spec = ProblemFramingAgent().run(goal)
    registers = GeometryAgent().run(spec, "full_training", memory_records=[])

    config = TrainingConfig(
        checkpoint_dir=checkpoint_dir,
        surrogate_epochs=epochs,
        rl_total_updates=updates,
    )
    runner = TrainingRunner(config)
    result = runner.run_full_pipeline(train_ds, spec, registers, val_ds)
    logger.info("Full pipeline training complete: %s", list(result.keys()))


def generate_training_data_v2(
    n_samples: int,
    workers: int,
    sampling: str,
    output_dir: str,
    seed: int,
) -> None:
    from packages.ml.data_generator import DatasetGenerator, GenerationConfig

    config = GenerationConfig(
        n_samples=n_samples,
        n_workers=workers,
        sampling_method=sampling,
        output_dir=output_dir,
        seed=seed,
    )
    stats = DatasetGenerator(config).generate()
    logger.info(stats.summary())


def run_active_learning(
    data_path: str,
    checkpoint_dir: str,
    al_iterations: int,
    al_top_k: int,
) -> None:
    import numpy as np

    from packages.agents.geometry_agent import GeometryAgent
    from packages.agents.problem_agent import ProblemFramingAgent
    from packages.core.models import ExperimentGoal
    from packages.ml.dataset import CandidateDatasetBuilder
    from packages.ml.training_runner import TrainingConfig, TrainingRunner

    builder = CandidateDatasetBuilder()
    builder.load(data_path)
    features, targets = builder.to_numpy()
    if len(features) < 10:
        logger.error("Need at least 10 samples before active learning.")
        return

    goal = ExperimentGoal(
        title="Active learning training",
        scientific_objective="Iteratively refine surrogate and RL policy.",
        target_observable="rydberg_density",
        desired_atom_count=6,
        preferred_geometry="mixed",
    )
    spec = ProblemFramingAgent().run(goal)
    registers = GeometryAgent().run(spec, "active_training", memory_records=[])

    runner = TrainingRunner(
        TrainingConfig(
            checkpoint_dir=checkpoint_dir,
            seed=42,
        )
    )
    results = runner.run_active_learning(
        initial_features=features,
        initial_targets=targets,
        spec=spec,
        register_candidates=registers,
        n_iterations=al_iterations,
        top_k=al_top_k,
    )

    output_path = str(Path(data_path).with_suffix("")) + "_enriched.npz"
    np.savez(
        output_path,
        features=results["final_features"],
        targets=results["final_targets"],
    )
    logger.info(
        "Active learning complete. Iterations=%d, final dataset=%d",
        len(results["iterations"]),
        int(results["final_features"].shape[0]),
    )
    logger.info("Enriched dataset saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CryoSwarm-Q ML Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "generate_v2", "surrogate", "rl", "full", "active"],
        required=True,
        help="Training phase to execute.",
    )
    parser.add_argument("--data", default="data/candidates.npz", help="Dataset path.")
    parser.add_argument("--runs", type=int, default=10, help="Pipeline runs for data generation.")
    parser.add_argument("--n-samples", type=int, default=100000, help="Sample count for v2 dataset generation.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for v2 dataset generation.")
    parser.add_argument(
        "--sampling",
        default="lhs",
        choices=["lhs", "grid", "random", "sobol"],
        help="Sampling scheme for v2 dataset generation.",
    )
    parser.add_argument("--output-dir", default="data/generated", help="Output directory for v2 dataset generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for v2 dataset generation.")
    parser.add_argument("--epochs", type=int, default=100, help="Surrogate training epochs.")
    parser.add_argument("--updates", type=int, default=200, help="RL policy updates.")
    parser.add_argument("--al-iterations", type=int, default=5, help="Active-learning iterations.")
    parser.add_argument("--al-top-k", type=int, default=200, help="Top-K configs to re-simulate per active-learning iteration.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory.")

    args = parser.parse_args()

    if args.phase == "generate":
        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        generate_training_data(args.runs, args.data)
    elif args.phase == "generate_v2":
        generate_training_data_v2(
            n_samples=args.n_samples,
            workers=args.workers,
            sampling=args.sampling,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.phase == "surrogate":
        train_surrogate(args.data, args.epochs, args.checkpoint_dir)
    elif args.phase == "rl":
        train_rl(args.updates, args.checkpoint_dir)
    elif args.phase == "full":
        train_full(args.data, args.epochs, args.updates, args.checkpoint_dir)
    elif args.phase == "active":
        run_active_learning(
            data_path=args.data,
            checkpoint_dir=args.checkpoint_dir,
            al_iterations=args.al_iterations,
            al_top_k=args.al_top_k,
        )


if __name__ == "__main__":
    main()
