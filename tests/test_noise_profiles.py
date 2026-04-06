"""Tests for noise profiles and their physical relevance."""
from __future__ import annotations

import pytest

from packages.simulation.noise_profiles import (
    default_noise_scenarios,
    low_noise,
    medium_noise,
    stressed_noise,
)


class TestNoiseProfiles:
    def test_three_default_scenarios(self) -> None:
        scenarios = default_noise_scenarios()
        assert len(scenarios) == 3

    def test_noise_severity_ordering(self) -> None:
        lo = low_noise()
        med = medium_noise()
        hi = stressed_noise()
        assert lo.amplitude_jitter < med.amplitude_jitter < hi.amplitude_jitter
        assert lo.dephasing_rate < med.dephasing_rate < hi.dephasing_rate
        assert lo.atom_loss_rate < med.atom_loss_rate < hi.atom_loss_rate

    def test_temperature_ordering(self) -> None:
        lo = low_noise()
        med = medium_noise()
        hi = stressed_noise()
        assert lo.temperature_uk < med.temperature_uk < hi.temperature_uk

    def test_spam_errors_in_range(self) -> None:
        for scenario in default_noise_scenarios():
            assert 0.0 <= scenario.state_prep_error <= 0.1
            assert 0.0 <= scenario.false_positive_rate <= 0.1
            assert 0.0 <= scenario.false_negative_rate <= 0.2

    def test_amplitude_jitter_reasonable(self) -> None:
        """Amplitude jitter should be <20% (realistic for lab conditions)."""
        for scenario in default_noise_scenarios():
            assert scenario.amplitude_jitter <= 0.20

    def test_labels_unique(self) -> None:
        labels = [s.label for s in default_noise_scenarios()]
        assert len(set(labels)) == 3

    def test_spatial_inhomogeneity_present(self) -> None:
        scenarios = default_noise_scenarios()
        values = [float(s.metadata.get("spatial_inhomogeneity", 0.0)) for s in scenarios]
        assert values == [0.02, 0.05, 0.08]
