"""Systematic dataset generator for CryoSwarm-Q ML training."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import itertools
import json
from math import dist
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from packages.agents.geometry_agent import GeometryAgent
from packages.core.enums import NoiseLevel, SequenceFamily
from packages.core.models import ExperimentSpec, NoiseScenario, RegisterCandidate, SequenceCandidate
from packages.core.parameter_space import PhysicsParameterSpace
from packages.ml.dataset import OUTPUT_DIM, INPUT_DIM_V2, build_feature_vector_v2
from packages.ml.normalizer import DatasetNormalizer
from packages.scoring.robustness import clamp_score, robustness_score
from packages.simulation.evaluators import evaluate_candidate_robustness
from packages.pasqal_adapters.pulser_adapter import create_simple_register, summarize_register_physics

try:
    from scipy.stats.qmc import LatinHypercube, Sobol

    SCIPY_QMC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional fallback
    LatinHypercube = None  # type: ignore[assignment]
    Sobol = None  # type: ignore[assignment]
    SCIPY_QMC_AVAILABLE = False


@dataclass
class GenerationConfig:
    """Configuration for large-scale data generation."""

    n_samples: int = 100_000
    sampling_method: str = "lhs"
    atom_counts: list[int] = field(default_factory=lambda: list(range(3, 16)))
    layouts: list[str] = field(
        default_factory=lambda: ["square", "line", "triangular", "ring", "zigzag", "honeycomb"]
    )
    families: list[str] = field(
        default_factory=lambda: [
            "adiabatic_sweep",
            "detuning_scan",
            "global_ramp",
            "constant_drive",
            "blackman_sweep",
        ]
    )
    include_noise_variation: bool = True
    noise_samples_per_config: int = 1
    n_workers: int = 4
    batch_size: int = 100
    output_dir: str = "data/generated"
    save_interval: int = 1000
    resume: bool = True
    max_atoms_for_full_sim: int = 12
    timeout_per_eval: float = 30.0
    seed: int = 42


@dataclass
class DatasetStats:
    """Statistics about a generated dataset."""

    total_samples: int
    successful_evals: int
    failed_evals: int
    eval_time_seconds: float
    atom_count_distribution: dict[int, int]
    layout_distribution: dict[str, int]
    family_distribution: dict[str, int]
    robustness_mean: float
    robustness_std: float
    robustness_min: float
    robustness_max: float
    nominal_mean: float
    worst_case_mean: float
    feature_means: NDArray[np.float32]
    feature_stds: NDArray[np.float32]
    feature_mins: NDArray[np.float32]
    feature_maxs: NDArray[np.float32]

    def summary(self) -> str:
        return (
            f"Dataset generated: {self.successful_evals}/{self.total_samples} successful evaluations, "
            f"robustness mean={self.robustness_mean:.4f}, std={self.robustness_std:.4f}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "successful_evals": self.successful_evals,
            "failed_evals": self.failed_evals,
            "eval_time_seconds": self.eval_time_seconds,
            "atom_count_distribution": self.atom_count_distribution,
            "layout_distribution": self.layout_distribution,
            "family_distribution": self.family_distribution,
            "robustness_mean": self.robustness_mean,
            "robustness_std": self.robustness_std,
            "robustness_min": self.robustness_min,
            "robustness_max": self.robustness_max,
            "nominal_mean": self.nominal_mean,
            "worst_case_mean": self.worst_case_mean,
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist(),
            "feature_mins": self.feature_mins.tolist(),
            "feature_maxs": self.feature_maxs.tolist(),
        }


def _unit_samples(method: str, n_samples: int, n_dims: int, rng: np.random.Generator) -> NDArray[np.float32]:
    if n_samples <= 0:
        return np.empty((0, n_dims), dtype=np.float32)

    if method == "lhs":
        if SCIPY_QMC_AVAILABLE:
            return LatinHypercube(d=n_dims, seed=rng).random(n=n_samples).astype(np.float32)
    elif method == "sobol":
        if SCIPY_QMC_AVAILABLE:
            return Sobol(d=n_dims, scramble=True, seed=rng).random(n=n_samples).astype(np.float32)
    elif method == "random":
        return rng.random((n_samples, n_dims), dtype=np.float32)

    if method == "grid":
        points_per_axis = max(2, int(np.ceil(n_samples ** (1.0 / max(n_dims, 1)))))
        axes = [np.linspace(0.0, 1.0, points_per_axis, dtype=np.float32) for _ in range(n_dims)]
        grid = np.array(list(itertools.product(*axes)), dtype=np.float32)
        return grid[:n_samples]

    result = np.zeros((n_samples, n_dims), dtype=np.float32)
    intervals = np.linspace(0.0, 1.0, n_samples + 1, dtype=np.float32)
    for dim in range(n_dims):
        points = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(points)
        result[:, dim] = points
    return result


def _fast_target_estimate(
    register: RegisterCandidate,
    sequence: SequenceCandidate,
    param_space: PhysicsParameterSpace,
) -> tuple[float, float, float, float]:
    spacing = float(register.metadata.get("spacing_um", param_space.geometry.spacing_um.default))
    blockade_fraction = register.blockade_pair_count / max(register.atom_count * (register.atom_count - 1) / 2, 1)
    interaction_energy = 862690.0 / max(spacing**6, 1e-6)
    omega_ratio = sequence.amplitude / max(interaction_energy, 1e-6)
    density_proxy = clamp_score(0.55 * blockade_fraction + 0.45 * min(omega_ratio, 1.0))
    nominal = density_proxy
    worst_case = clamp_score(max(0.0, nominal - 0.12))
    avg = clamp_score(max(0.0, nominal - 0.06))
    robustness = robustness_score(nominal, avg, worst_case, 0.05, param_space=param_space)
    return robustness, nominal, worst_case, nominal


def _evaluate_sample_task(
    spec: ExperimentSpec,
    register: RegisterCandidate,
    sequence: SequenceCandidate,
    param_space_dict: dict[str, Any],
    fast_mode: bool,
    noise_payload: dict[str, float] | None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]] | None:
    try:
        param_space = PhysicsParameterSpace.from_dict(param_space_dict)
        if fast_mode:
            robustness, nominal, worst_case, observable = _fast_target_estimate(register, sequence, param_space)
        else:
            scenarios = None
            if noise_payload is not None:
                scenarios = [NoiseScenario(label=NoiseLevel.MEDIUM, **noise_payload)]
            (
                nominal,
                _scenario_scores,
                _average,
                worst_case,
                _score_std,
                _penalty,
                robustness,
                nominal_observables,
                _scenario_observables,
                _hamiltonian_metrics,
            ) = evaluate_candidate_robustness(
                spec,
                register,
                sequence,
                scenarios=scenarios,
                param_space=param_space,
            )
            observable = float(nominal_observables.get("observable_score", nominal))

        features = build_feature_vector_v2(register, sequence, param_space)
        targets = np.array([robustness, nominal, worst_case, observable], dtype=np.float32)
        metadata = {
            "atom_count": register.atom_count,
            "layout": register.layout_type,
            "family": sequence.sequence_family.value,
        }
        return features, targets, metadata
    except Exception:
        return None


class DatasetGenerator:
    """Systematic large-scale dataset generator."""

    def __init__(
        self,
        config: GenerationConfig,
        param_space: PhysicsParameterSpace | None = None,
        evaluate_fn: Callable[..., tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]] | None] | None = None,
    ) -> None:
        self.config = config
        self.param_space = param_space or PhysicsParameterSpace.default()
        self.geometry_agent = GeometryAgent(param_space=self.param_space)
        self.evaluate_fn = evaluate_fn or _evaluate_sample_task
        self.rng = np.random.default_rng(config.seed)
        self.output_dir = Path(config.output_dir)
        self.parts_dir = self.output_dir / "parts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.json"
        self.dataset_path = self.output_dir / "dataset.npz"
        self.stats_path = self.output_dir / "stats.json"
        self.normalizer_path = self.output_dir / "normalizer.npz"

    def generate(self) -> DatasetStats:
        start_time = time.perf_counter()
        configurations = self._plan_configurations()
        progress = self._load_progress()
        start_idx = int(progress.get("processed", 0))
        part_idx = int(progress.get("part_idx", 0))

        feature_buffer: list[NDArray[np.float32]] = []
        target_buffer: list[NDArray[np.float32]] = []
        meta_buffer: list[dict[str, Any]] = []

        for batch_start in range(start_idx, len(configurations), self.config.batch_size):
            configs_batch = configurations[batch_start : batch_start + self.config.batch_size]
            batch_payload: list[tuple[ExperimentSpec, RegisterCandidate, SequenceCandidate, dict[str, float] | None, bool]] = []
            for local_idx, config in enumerate(configs_batch):
                campaign_id = f"dataset_{batch_start + local_idx}"
                spec = self._build_spec(config)
                register = self._build_register(config["atom_count"], config["layout"], config["spacing_um"], campaign_id)
                sequence = self._build_sequence(config, register, campaign_id)
                fast_mode = bool(config["atom_count"] > self.config.max_atoms_for_full_sim)
                noise_payload = config.get("noise_payload")
                batch_payload.append((spec, register, sequence, noise_payload, fast_mode))

            for result in self._evaluate_batch(batch_payload):
                if result is None:
                    progress["failed_evals"] = int(progress.get("failed_evals", 0)) + 1
                    continue
                features, targets, metadata = result
                feature_buffer.append(features)
                target_buffer.append(targets)
                meta_buffer.append(metadata)
                progress["successful_evals"] = int(progress.get("successful_evals", 0)) + 1

            progress["processed"] = batch_start + len(configs_batch)
            if feature_buffer and (
                len(feature_buffer) >= self.config.save_interval or progress["processed"] == len(configurations)
            ):
                self._save_checkpoint(
                    np.stack(feature_buffer),
                    np.stack(target_buffer),
                    meta_buffer,
                    part_idx,
                )
                part_idx += 1
                progress["part_idx"] = part_idx
                feature_buffer = []
                target_buffer = []
                meta_buffer = []
            self._save_progress(progress)

        stats = self._merge_checkpoints(str(self.dataset_path), total_samples=len(configurations))
        stats = DatasetStats(
            total_samples=stats.total_samples,
            successful_evals=stats.successful_evals,
            failed_evals=stats.failed_evals,
            eval_time_seconds=round(time.perf_counter() - start_time, 3),
            atom_count_distribution=stats.atom_count_distribution,
            layout_distribution=stats.layout_distribution,
            family_distribution=stats.family_distribution,
            robustness_mean=stats.robustness_mean,
            robustness_std=stats.robustness_std,
            robustness_min=stats.robustness_min,
            robustness_max=stats.robustness_max,
            nominal_mean=stats.nominal_mean,
            worst_case_mean=stats.worst_case_mean,
            feature_means=stats.feature_means,
            feature_stds=stats.feature_stds,
            feature_mins=stats.feature_mins,
            feature_maxs=stats.feature_maxs,
        )
        self.stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
        return stats

    def _load_progress(self) -> dict[str, Any]:
        if self.config.resume and self.progress_path.exists():
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        return {"processed": 0, "part_idx": 0, "successful_evals": 0, "failed_evals": 0}

    def _save_progress(self, progress: dict[str, Any]) -> None:
        self.progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    def _plan_configurations(self) -> list[dict[str, Any]]:
        combos = [
            (atom_count, layout, SequenceFamily(family))
            for atom_count in self.config.atom_counts
            for layout in self.config.layouts
            for family in self.config.families
        ]
        weights = np.array(
            [1.0 if atom_count <= self.config.max_atoms_for_full_sim else 0.35 for atom_count, _, _ in combos],
            dtype=float,
        )
        weights /= weights.sum()
        raw_counts = weights * self.config.n_samples
        counts = np.floor(raw_counts).astype(int)
        remainder = self.config.n_samples - counts.sum()
        if remainder > 0:
            order = np.argsort(raw_counts - counts)[::-1][:remainder]
            counts[order] += 1

        configs: list[dict[str, Any]] = []
        for idx, ((atom_count, layout, family), sample_count) in enumerate(zip(combos, counts, strict=False)):
            if sample_count <= 0:
                continue
            pulse_space = self.param_space.pulse[family]
            dimensions = {
                "amplitude": pulse_space.amplitude,
                "detuning": pulse_space.detuning_start,
                "duration_ns": pulse_space.duration_ns,
                "spacing_um": self.param_space.geometry.spacing_um,
            }
            if pulse_space.detuning_end is not None:
                dimensions["detuning_end"] = pulse_space.detuning_end
            if pulse_space.amplitude_start is not None:
                dimensions["amplitude_start"] = pulse_space.amplitude_start
            combo_rng = np.random.default_rng(self.config.seed + idx)
            samples = _unit_samples(self.config.sampling_method, sample_count, len(dimensions), combo_rng)
            names = list(dimensions.keys())
            for row in samples:
                config = {
                    "atom_count": atom_count,
                    "layout": layout,
                    "family": family.value,
                    "waveform_kind": family.value,
                }
                for dim_idx, name in enumerate(names):
                    value = dimensions[name].denormalize(float(row[dim_idx]))
                    if name == "duration_ns":
                        value = int(dimensions[name].clip(value))
                    config[name] = value
                if "amplitude_start" in config:
                    config["amplitude_start"] = min(config["amplitude_start"], config["amplitude"])
                if "detuning_end" in config and config["detuning_end"] <= config["detuning"]:
                    config["detuning_end"] = pulse_space.detuning_end.clip(config["detuning"] + 1.0)  # type: ignore[union-attr]
                if self.config.include_noise_variation:
                    config["noise_payload"] = {
                        "amplitude_jitter": self.param_space.noise.amplitude_jitter.sample(combo_rng),
                        "detuning_jitter": self.param_space.noise.detuning_jitter.sample(combo_rng),
                        "dephasing_rate": self.param_space.noise.dephasing_rate.sample(combo_rng),
                        "atom_loss_rate": self.param_space.noise.atom_loss_rate.sample(combo_rng),
                        "temperature_uk": self.param_space.noise.temperature_uk.sample(combo_rng),
                        "state_prep_error": self.param_space.noise.state_prep_error.sample(combo_rng),
                        "false_positive_rate": self.param_space.noise.false_positive_rate.sample(combo_rng),
                        "false_negative_rate": self.param_space.noise.false_negative_rate.sample(combo_rng),
                    }
                configs.append(config)
        return configs

    def _build_spec(self, config: dict[str, Any]) -> ExperimentSpec:
        family = SequenceFamily(config["family"])
        return ExperimentSpec(
            goal_id=f"dataset_goal_{config['atom_count']}_{config['layout']}",
            objective_class="dataset_generation",
            target_observable="rydberg_density",
            min_atoms=int(config["atom_count"]),
            max_atoms=int(config["atom_count"]),
            preferred_layouts=[str(config["layout"])],
            sequence_families=[family],
            target_density=0.5,
            reasoning_summary="Synthetic specification for dataset generation.",
            metadata={"dataset_generation": True},
        )

    def _build_register(
        self,
        atom_count: int,
        layout: str,
        spacing_um: float,
        campaign_id: str,
    ) -> RegisterCandidate:
        coordinates = self.geometry_agent._coordinates_for_layout(layout, atom_count, spacing_um)  # noqa: SLF001
        physics = summarize_register_physics(coordinates)
        if not physics.valid:
            distances = [
                dist(coordinates[i], coordinates[j])
                for i in range(len(coordinates))
                for j in range(i + 1, len(coordinates))
            ]
            min_distance = min(distances) if distances else 0.0
            blockade_radius = (862690.0 / 5.0) ** (1.0 / 6.0) if min_distance else 0.0
            blockade_pairs = sum(1 for value in distances if value <= blockade_radius)
            physics = type(
                "FallbackPhysics",
                (),
                {
                    "min_distance_um": round(min_distance, 6),
                    "blockade_radius_um": round(blockade_radius, 6),
                    "blockade_pair_count": blockade_pairs,
                    "van_der_waals_matrix": [
                        [
                            0.0 if i == j else round(862690.0 / max(dist(coordinates[i], coordinates[j]) ** 6, 1e-6), 6)
                            for j in range(len(coordinates))
                        ]
                        for i in range(len(coordinates))
                    ],
                },
            )()
        feasibility = self.geometry_agent._feasibility_score(atom_count, physics.min_distance_um, physics.blockade_pair_count)  # noqa: SLF001
        return RegisterCandidate(
            campaign_id=campaign_id,
            spec_id=f"spec_{campaign_id}",
            label=f"{layout}-{atom_count}-s{spacing_um:.2f}",
            layout_type=layout,
            atom_count=atom_count,
            coordinates=coordinates,
            device_constraints={"min_spacing_um": self.param_space.geometry.min_spacing_um.default},
            min_distance_um=float(physics.min_distance_um),
            blockade_radius_um=float(physics.blockade_radius_um),
            blockade_pair_count=int(physics.blockade_pair_count),
            van_der_waals_matrix=physics.van_der_waals_matrix,
            feasibility_score=feasibility,
            reasoning_summary="Register generated for dataset sampling.",
            pulser_register_summary=create_simple_register(coordinates),
            metadata={"spacing_um": float(spacing_um)},
        )

    def _build_sequence(
        self,
        config: dict[str, Any],
        register: RegisterCandidate,
        campaign_id: str,
    ) -> SequenceCandidate:
        family = SequenceFamily(config["family"])
        metadata = {
            "spacing_um": register.metadata.get("spacing_um"),
            "layout_type": register.layout_type,
            "atom_count": register.atom_count,
        }
        if "detuning_end" in config:
            metadata["detuning_end"] = float(config["detuning_end"])
        if "amplitude_start" in config:
            metadata["amplitude_start"] = float(config["amplitude_start"])
        return SequenceCandidate(
            campaign_id=campaign_id,
            spec_id=f"spec_{campaign_id}",
            register_candidate_id=register.id,
            label=f"{register.label}-{family.value}-dataset",
            sequence_family=family,
            channel_id="rydberg_global",
            duration_ns=int(config["duration_ns"]),
            amplitude=float(config["amplitude"]),
            detuning=float(config["detuning"]),
            phase=0.0,
            waveform_kind=str(config.get("waveform_kind", family.value)),
            predicted_cost=self.param_space.cost_for(register.atom_count, int(config["duration_ns"])),
            reasoning_summary="Sequence generated for dataset sampling.",
            metadata=metadata,
        )

    def _evaluate_batch(
        self,
        batch: list[tuple[ExperimentSpec, RegisterCandidate, SequenceCandidate, dict[str, float] | None, bool]],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]] | None]:
        tasks = [
            (spec, register, sequence, self.param_space.to_dict(), fast_mode, noise_payload)
            for spec, register, sequence, noise_payload, fast_mode in batch
        ]
        if self.config.n_workers <= 1:
            return [self.evaluate_fn(*task) for task in tasks]

        results: list[tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]] | None] = []
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = [executor.submit(self.evaluate_fn, *task) for task in tasks]
            for future in as_completed(futures):
                try:
                    results.append(future.result(timeout=self.config.timeout_per_eval))
                except Exception:
                    results.append(None)
        return results

    def _save_checkpoint(
        self,
        features: NDArray[np.float32],
        targets: NDArray[np.float32],
        metadata: list[dict[str, Any]],
        checkpoint_idx: int,
    ) -> None:
        checkpoint_path = self.parts_dir / f"part_{checkpoint_idx:05d}.npz"
        metadata_encoded = np.array([json.dumps(item) for item in metadata], dtype=object)
        np.savez(checkpoint_path, features=features, targets=targets, metadata=metadata_encoded)

    def _merge_checkpoints(self, output_path: str, total_samples: int | None = None) -> DatasetStats:
        part_paths = sorted(self.parts_dir.glob("part_*.npz"))
        feature_chunks: list[NDArray[np.float32]] = []
        target_chunks: list[NDArray[np.float32]] = []
        metadata_records: list[dict[str, Any]] = []

        for part_path in part_paths:
            data = np.load(part_path, allow_pickle=True)
            feature_chunks.append(data["features"].astype(np.float32))
            target_chunks.append(data["targets"].astype(np.float32))
            metadata_records.extend(json.loads(item) for item in data["metadata"].tolist())

        if feature_chunks:
            features = np.concatenate(feature_chunks, axis=0).astype(np.float32)
            targets = np.concatenate(target_chunks, axis=0).astype(np.float32)
        else:
            features = np.empty((0, INPUT_DIM_V2), dtype=np.float32)
            targets = np.empty((0, OUTPUT_DIM), dtype=np.float32)

        np.savez(output_path, features=features, targets=targets)
        normalizer = DatasetNormalizer().fit(features) if len(features) else DatasetNormalizer()
        if len(features):
            normalizer.save(str(self.normalizer_path))
        return self._compute_stats(features, targets, metadata_records, total_samples or len(features))

    def _compute_stats(
        self,
        features: NDArray[np.float32],
        targets: NDArray[np.float32],
        metadata_records: list[dict[str, Any]],
        total_samples: int,
    ) -> DatasetStats:
        atom_counts = Counter(int(item["atom_count"]) for item in metadata_records)
        layouts = Counter(str(item["layout"]) for item in metadata_records)
        families = Counter(str(item["family"]) for item in metadata_records)

        if len(features) == 0:
            empty = np.zeros((INPUT_DIM_V2,), dtype=np.float32)
            return DatasetStats(
                total_samples=total_samples,
                successful_evals=0,
                failed_evals=total_samples,
                eval_time_seconds=0.0,
                atom_count_distribution={},
                layout_distribution={},
                family_distribution={},
                robustness_mean=0.0,
                robustness_std=0.0,
                robustness_min=0.0,
                robustness_max=0.0,
                nominal_mean=0.0,
                worst_case_mean=0.0,
                feature_means=empty,
                feature_stds=empty,
                feature_mins=empty,
                feature_maxs=empty,
            )

        return DatasetStats(
            total_samples=total_samples,
            successful_evals=int(targets.shape[0]),
            failed_evals=max(total_samples - int(targets.shape[0]), 0),
            eval_time_seconds=0.0,
            atom_count_distribution=dict(atom_counts),
            layout_distribution=dict(layouts),
            family_distribution=dict(families),
            robustness_mean=float(targets[:, 0].mean()),
            robustness_std=float(targets[:, 0].std()),
            robustness_min=float(targets[:, 0].min()),
            robustness_max=float(targets[:, 0].max()),
            nominal_mean=float(targets[:, 1].mean()),
            worst_case_mean=float(targets[:, 2].mean()),
            feature_means=features.mean(axis=0).astype(np.float32),
            feature_stds=features.std(axis=0).astype(np.float32),
            feature_mins=features.min(axis=0).astype(np.float32),
            feature_maxs=features.max(axis=0).astype(np.float32),
        )
