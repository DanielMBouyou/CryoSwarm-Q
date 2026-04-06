from __future__ import annotations

from packages.core.enums import NoiseLevel
from packages.core.models import NoiseScenario
from packages.core.parameter_space import PhysicsParameterSpace


def _with_spatial_default(
    scenario: NoiseScenario,
    level: NoiseLevel,
) -> NoiseScenario:
    defaults = {
        NoiseLevel.LOW: 0.02,
        NoiseLevel.MEDIUM: 0.05,
        NoiseLevel.STRESSED: 0.08,
    }
    metadata = dict(scenario.metadata)
    metadata.setdefault("spatial_inhomogeneity", defaults[level])
    return scenario.model_copy(update={"metadata": metadata})


def low_noise(param_space: PhysicsParameterSpace | None = None) -> NoiseScenario:
    base = (param_space or PhysicsParameterSpace.default()).noise_profile(NoiseLevel.LOW)
    return _with_spatial_default(base, NoiseLevel.LOW)


def medium_noise(param_space: PhysicsParameterSpace | None = None) -> NoiseScenario:
    base = (param_space or PhysicsParameterSpace.default()).noise_profile(NoiseLevel.MEDIUM)
    return _with_spatial_default(base, NoiseLevel.MEDIUM)


def stressed_noise(param_space: PhysicsParameterSpace | None = None) -> NoiseScenario:
    base = (param_space or PhysicsParameterSpace.default()).noise_profile(NoiseLevel.STRESSED)
    return _with_spatial_default(base, NoiseLevel.STRESSED)


def default_noise_scenarios(param_space: PhysicsParameterSpace | None = None) -> list[NoiseScenario]:
    return [
        low_noise(param_space),
        medium_noise(param_space),
        stressed_noise(param_space),
    ]
