"""Ablation study runner for CryoSwarm-Q."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from packages.core.logging import get_logger
from scripts.benchmark import (
    benchmark_pipeline_run,
    compute_rl_benchmark,
    compute_surrogate_benchmark,
)

logger = get_logger(__name__)


ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline_heuristic": {
        "description": "Heuristic sequence agent only, no ML",
        "use_surrogate": False,
        "use_rl": False,
    },
    "surrogate_filter_v1": {
        "description": "Surrogate pre-filter with V1 features (10-dim)",
        "use_surrogate": True,
        "feature_version": "v1",
        "use_rl": False,
    },
    "surrogate_filter_v2": {
        "description": "Surrogate pre-filter with V2 features (18-dim)",
        "use_surrogate": True,
        "feature_version": "v2",
        "use_rl": False,
    },
    "rl_single_step": {
        "description": "RL with max_steps=1 (original)",
        "use_rl": True,
        "rl_max_steps": 1,
        "reward_shaping": False,
    },
    "rl_multi_step": {
        "description": "RL with max_steps=5 and reward shaping",
        "use_rl": True,
        "rl_max_steps": 5,
        "reward_shaping": True,
    },
    "rl_curriculum": {
        "description": "RL with curriculum learning",
        "use_rl": True,
        "rl_max_steps": 5,
        "use_curriculum": True,
    },
    "hybrid_heuristic_rl": {
        "description": "Hybrid mode: heuristic + RL candidates together",
        "use_rl": True,
        "strategy_mode": "hybrid",
    },
    "ensemble_3": {
        "description": "Ensemble of 3 surrogate models",
        "use_surrogate": True,
        "use_ensemble": True,
        "n_ensemble": 3,
    },
    "full_pipeline": {
        "description": "Everything enabled: ensemble + RL + curriculum + active learning",
        "use_surrogate": True,
        "use_ensemble": True,
        "use_rl": True,
        "use_curriculum": True,
        "strategy_mode": "adaptive",
    },
}


def _resolve_cases(name: str) -> list[str]:
    if name == "all":
        return list(ABLATION_CONFIGS)
    if name == "features":
        return ["surrogate_filter_v1", "surrogate_filter_v2"]
    if name == "architecture":
        return ["surrogate_filter_v2", "ensemble_3", "full_pipeline"]
    if name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {name}")
    return [name]


def _load_dataset(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_path)
    return np.asarray(data["features"], dtype=np.float32), np.asarray(data["targets"], dtype=np.float32)


def _split_dataset(features: np.ndarray, targets: np.ndarray) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    n_val = max(1, len(dataset) // 5)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    val_features = targets[:n_val] * 0.0  # placeholder to keep shapes aligned below
    return train_ds, val_ds, features[:n_val], targets[:n_val]


def _train_surrogate_case(
    config_name: str,
    case_config: dict[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
    checkpoint_root: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import TensorDataset

    from packages.ml.surrogate import EnsembleTrainer, SurrogateEnsemble
    from packages.ml.training_runner import TrainingConfig, TrainingRunner

    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    n_val = max(1, len(dataset) // 5)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    val_indices = getattr(val_ds, "indices", list(range(n_val)))
    x_val = features[np.asarray(val_indices)]
    y_val = targets[np.asarray(val_indices)]

    case_dir = checkpoint_root / config_name
    case_dir.mkdir(parents=True, exist_ok=True)

    if case_config.get("use_ensemble"):
        ensemble = SurrogateEnsemble(
            n_models=int(case_config.get("n_ensemble", 3)),
            input_dim=features.shape[1],
            hidden=128,
            n_blocks=3,
        )
        trainer = EnsembleTrainer(ensemble, bootstrap=True)
        trainer.fit(train_ds, val_ds, epochs=5, batch_size=32, log_dir=None)
        ensemble_dir = case_dir / "ensemble"
        ensemble.save(ensemble_dir)
        predictions, uncertainty = ensemble.predict_with_uncertainty(x_val)
        metrics = compute_surrogate_benchmark(y_val, predictions)
        return {
            "benchmark": asdict(metrics),
            "uncertainty_mean": float(np.mean(uncertainty)),
            "checkpoint": str(ensemble_dir),
        }

    runner = TrainingRunner(
        TrainingConfig(
            checkpoint_dir=str(case_dir),
            surrogate_epochs=5,
            surrogate_batch_size=32,
            use_curriculum=False,
        )
    )
    result = runner.run_surrogate(train_ds, val_ds)
    from packages.ml.surrogate import SurrogateModel, SurrogateModelV2

    model = SurrogateModelV2(input_dim=features.shape[1]) if result["model_version"] == "v2" else SurrogateModel(input_dim=features.shape[1])
    model.load(result["checkpoint"])
    predictions = model.predict_numpy(x_val)
    metrics = compute_surrogate_benchmark(y_val, predictions)
    return {
        "benchmark": asdict(metrics),
        "checkpoint": result["checkpoint"],
    }


def _train_rl_case(
    config_name: str,
    case_config: dict[str, Any],
    checkpoint_root: Path,
) -> dict[str, Any]:
    from packages.agents.geometry_agent import GeometryAgent
    from packages.agents.problem_agent import ProblemFramingAgent
    from packages.core.models import ExperimentGoal
    from packages.ml.training_runner import TrainingConfig, TrainingRunner

    goal = ExperimentGoal(
        title=f"Ablation {config_name}",
        scientific_objective="Ablation study.",
        target_observable="rydberg_density",
        desired_atom_count=6,
        preferred_geometry="mixed",
    )
    spec = ProblemFramingAgent().run(goal)
    registers = GeometryAgent().run(spec, f"ablation_{config_name}", memory_records=[])
    case_dir = checkpoint_root / config_name
    case_dir.mkdir(parents=True, exist_ok=True)

    runner = TrainingRunner(
        TrainingConfig(
            checkpoint_dir=str(case_dir),
            rl_total_updates=5,
            rl_rollout_steps=32,
            rl_max_steps=int(case_config.get("rl_max_steps", 5)),
            rl_reward_shaping=bool(case_config.get("reward_shaping", True)),
            use_curriculum=bool(case_config.get("use_curriculum", False)),
        )
    )
    result = runner.run_rl(spec, registers)
    metrics = compute_rl_benchmark(result["history"]["episode_rewards"])
    return {
        "benchmark": asdict(metrics),
        "checkpoint": result["checkpoint"],
    }


def run_ablation(
    ablation: str,
    data_path: str,
    output_dir: str = "ablations",
) -> dict[str, Any]:
    features, targets = _load_dataset(data_path)
    checkpoint_root = Path(output_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "ablation": ablation,
        "cases": {},
    }

    for name in _resolve_cases(ablation):
        case_config = ABLATION_CONFIGS[name]
        case_result: dict[str, Any] = {
            "description": case_config["description"],
            "config": case_config,
        }
        if case_config.get("use_surrogate"):
            case_result["surrogate"] = _train_surrogate_case(name, case_config, features, targets, checkpoint_root)
        if case_config.get("use_rl"):
            case_result["rl"] = _train_rl_case(name, case_config, checkpoint_root)
        if not case_config.get("use_surrogate") and not case_config.get("use_rl"):
            case_result["pipeline"] = asdict(benchmark_pipeline_run(sequence_strategy_mode="heuristic_only"))
        elif case_config.get("strategy_mode"):
            case_result["pipeline"] = asdict(
                benchmark_pipeline_run(
                    rl_checkpoint_path=case_result.get("rl", {}).get("checkpoint"),
                    sequence_strategy_mode=str(case_config["strategy_mode"]),
                )
            )
        results["cases"][name] = case_result

    output_path = checkpoint_root / f"{ablation}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Ablation results saved to %s", output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CryoSwarm-Q ablation study runner")
    parser.add_argument("--ablation", default="all")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="ablations")
    args = parser.parse_args()
    run_ablation(args.ablation, args.data, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
