from __future__ import annotations

import numpy as np

from packages.core.parameter_space import PhysicsParameterSpace
from packages.ml.data_generator import DatasetGenerator, GenerationConfig
from packages.ml.dataset import INPUT_DIM_V2, build_feature_vector_v2
from packages.ml.normalizer import DatasetNormalizer


def test_data_generator_small_dataset_and_resume(tmp_path) -> None:
    param_space = PhysicsParameterSpace.default()

    def stub_evaluate(spec, register, sequence, param_space_dict, fast_mode, noise_payload):  # type: ignore[no-untyped-def]
        features = build_feature_vector_v2(register, sequence, PhysicsParameterSpace.from_dict(param_space_dict))
        robustness = min(1.0, 0.4 + 0.02 * register.atom_count)
        targets = np.array([robustness, robustness - 0.02, robustness - 0.05, robustness - 0.01], dtype=np.float32)
        metadata = {
            "atom_count": register.atom_count,
            "layout": register.layout_type,
            "family": sequence.sequence_family.value,
        }
        return features, targets, metadata

    config = GenerationConfig(
        n_samples=100,
        atom_counts=[3, 4],
        layouts=["line", "square"],
        families=["adiabatic_sweep", "constant_drive"],
        include_noise_variation=False,
        n_workers=1,
        batch_size=20,
        save_interval=25,
        output_dir=str(tmp_path / "generated"),
        seed=123,
    )

    generator = DatasetGenerator(config, param_space=param_space, evaluate_fn=stub_evaluate)
    stats = generator.generate()

    assert stats.total_samples == 100
    assert stats.successful_evals == 100
    assert stats.failed_evals == 0
    assert (tmp_path / "generated" / "dataset.npz").exists()
    assert (tmp_path / "generated" / "normalizer.npz").exists()
    assert stats.feature_means.shape == (INPUT_DIM_V2,)

    loaded = np.load(tmp_path / "generated" / "dataset.npz")
    assert loaded["features"].shape == (100, INPUT_DIM_V2)
    assert loaded["targets"].shape == (100, 4)

    resumed_stats = generator.generate()
    loaded_resumed = np.load(tmp_path / "generated" / "dataset.npz")
    assert resumed_stats.total_samples == 100
    assert loaded_resumed["features"].shape == (100, INPUT_DIM_V2)


def test_dataset_normalizer_roundtrip() -> None:
    features = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    normalizer = DatasetNormalizer()
    normalized = normalizer.fit_transform(features)
    restored = normalizer.inverse_transform(normalized)

    np.testing.assert_allclose(restored, features, atol=1e-6)
