from __future__ import annotations

from packages.core.enums import NoiseLevel
from packages.core.models import NoiseScenario


def low_noise() -> NoiseScenario:
    return NoiseScenario(
        label=NoiseLevel.LOW,
        amplitude_jitter=0.03,
        detuning_jitter=0.02,
        dephasing_rate=0.03,
        atom_loss_rate=0.01,
        temperature_uk=30.0,
        state_prep_error=0.003,
        false_positive_rate=0.008,
        false_negative_rate=0.02,
    )


def medium_noise() -> NoiseScenario:
    return NoiseScenario(
        label=NoiseLevel.MEDIUM,
        amplitude_jitter=0.06,
        detuning_jitter=0.05,
        dephasing_rate=0.07,
        atom_loss_rate=0.03,
        temperature_uk=50.0,
        state_prep_error=0.005,
        false_positive_rate=0.01,
        false_negative_rate=0.05,
    )


def stressed_noise() -> NoiseScenario:
    return NoiseScenario(
        label=NoiseLevel.STRESSED,
        amplitude_jitter=0.10,
        detuning_jitter=0.08,
        dephasing_rate=0.11,
        atom_loss_rate=0.05,
        temperature_uk=75.0,
        state_prep_error=0.01,
        false_positive_rate=0.02,
        false_negative_rate=0.08,
    )


def default_noise_scenarios() -> list[NoiseScenario]:
    return [low_noise(), medium_noise(), stressed_noise()]
