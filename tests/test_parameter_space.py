from __future__ import annotations

import numpy as np

from packages.core.enums import SequenceFamily
from packages.core.parameter_space import ParameterRange, PhysicsParameterSpace


def test_parameter_range_sampling_and_clipping() -> None:
    rng = np.random.default_rng(123)
    parameter = ParameterRange(
        name="duration",
        min_val=100.0,
        max_val=200.0,
        default=120.0,
        unit="ns",
        description="Test duration",
        quantize=4.0,
    )

    sample = parameter.sample(rng)
    assert 100.0 <= sample <= 200.0
    assert sample % 4.0 == 0.0
    assert parameter.clip(500.0) == 200.0
    assert parameter.clip(50.0) == 100.0


def test_parameter_range_normalize_roundtrip() -> None:
    parameter = ParameterRange(
        name="amplitude",
        min_val=1.0,
        max_val=15.0,
        default=5.0,
        unit="rad/us",
        description="Test amplitude",
    )
    value = 7.0
    normalized = parameter.normalize(value)
    restored = parameter.denormalize(normalized)
    assert 0.0 <= normalized <= 1.0
    assert restored == value


def test_default_parameter_space_contains_all_families() -> None:
    param_space = PhysicsParameterSpace.default()
    assert set(param_space.pulse) == set(SequenceFamily)
    assert param_space.geometry.min_spacing_um.default == 5.0
    assert param_space.scoring.nominal_weight.default == 0.25


def test_grid_search_configs_respects_family_space() -> None:
    param_space = PhysicsParameterSpace.default()
    configs = param_space.grid_search_configs(SequenceFamily.DETUNING_SCAN, n_amplitude=3, n_detuning=2, n_duration=2)
    assert len(configs) == 12
    for config in configs:
        assert config["family"] == SequenceFamily.DETUNING_SCAN.value
        assert 2.0 <= config["amplitude"] <= 12.0
        assert -30.0 <= config["detuning"] <= -5.0
        assert 1000.0 <= config["duration_ns"] <= 5000.0


def test_latin_hypercube_sample_returns_physically_valid_points() -> None:
    param_space = PhysicsParameterSpace.default()
    rng = np.random.default_rng(42)
    configs = param_space.latin_hypercube_sample(
        SequenceFamily.ADIABATIC_SWEEP,
        n_samples=100,
        atom_count=6,
        rng=rng,
    )
    assert len(configs) == 100
    assert len({config["amplitude"] for config in configs}) > 50
    for config in configs:
        assert config["family"] == SequenceFamily.ADIABATIC_SWEEP.value
        assert 1.0 <= config["amplitude"] <= 15.0
        assert -40.0 <= config["detuning"] <= -5.0
        assert config["detuning_end"] > config["detuning"]


def test_parameter_space_json_yaml_roundtrip(tmp_path) -> None:
    param_space = PhysicsParameterSpace.default()
    path = tmp_path / "parameter_space.yaml"
    param_space.to_yaml(str(path))
    restored = PhysicsParameterSpace.from_yaml(str(path))

    assert restored.geometry.spacing_um.default == param_space.geometry.spacing_um.default
    assert restored.pulse[SequenceFamily.BLACKMAN_SWEEP].amplitude.default == 7.0


def test_sample_noise_scenario_is_bounded() -> None:
    param_space = PhysicsParameterSpace.default()
    rng = np.random.default_rng(7)
    scenario = param_space.sample_noise_scenario(rng)

    assert param_space.noise.amplitude_jitter.min_val <= scenario.amplitude_jitter <= param_space.noise.amplitude_jitter.max_val
    assert param_space.noise.false_negative_rate.min_val <= scenario.false_negative_rate <= param_space.noise.false_negative_rate.max_val
