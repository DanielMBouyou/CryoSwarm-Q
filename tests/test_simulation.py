"""Tests for the numpy exact-diagonal simulation backend."""
from __future__ import annotations

import numpy as np
import pytest

from packages.simulation.hamiltonian import blockade_radius, ground_state
from packages.simulation.observables import rydberg_density, total_rydberg_fraction

pytest.importorskip("scipy", reason="scipy required for numpy backend")
from packages.simulation.numpy_backend import (  # noqa: E402
    PulseSchedule,
    compute_schedule_diagnostics,
    estimate_discretization_error,
    simulate_rydberg_evolution,
)


class TestNumPySimulation:
    """End-to-end tests using the scipy-based time evolution."""

    def test_ground_state_for_zero_omega(self) -> None:
        """With Omega=0 the system stays in |g...g>."""
        coords = [(0.0, 0.0), (7.0, 0.0)]
        result = simulate_rydberg_evolution(
            coords, omega_max=0.0, delta_start=0.0, delta_end=0.0,
            duration_ns=1000, n_steps=20,
        )
        np.testing.assert_allclose(
            result["total_rydberg_fraction"], 0.0, atol=1e-8
        )

    def test_rabi_oscillation_two_atoms(self) -> None:
        """Resonant drive creates a non-trivial Rydberg fraction."""
        coords = [(0.0, 0.0), (50.0, 0.0)]  # far apart → negligible interaction
        result = simulate_rydberg_evolution(
            coords, omega_max=5.0, delta_start=0.0, delta_end=0.0,
            duration_ns=1000, n_steps=100,
        )
        frac = result["total_rydberg_fraction"]
        assert 0.05 < frac < 0.95, f"Expected non-trivial Rydberg fraction, got {frac}"

    def test_blockade_suppresses_double_excitation(self) -> None:
        """Two close atoms: simultaneous excitation should be suppressed."""
        coords = [(0.0, 0.0), (5.0, 0.0)]
        result = simulate_rydberg_evolution(
            coords, omega_max=2.0, delta_start=0.0, delta_end=0.0,
            duration_ns=1500, n_steps=100,
        )
        psi = result["final_state"]
        prob_both = abs(psi[3]) ** 2  # |11>
        assert prob_both < 0.05, f"|rr> probability {prob_both:.4f} too high"

    def test_adiabatic_sweep_creates_excitations(self) -> None:
        """Sweeping delta from negative to positive should create Rydberg excitations."""
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        result = simulate_rydberg_evolution(
            coords, omega_max=5.0, delta_start=-20.0, delta_end=10.0,
            duration_ns=3000, n_steps=200,
        )
        frac = result["total_rydberg_fraction"]
        assert frac > 0.1, f"Expected Rydberg excitations after sweep, got {frac}"

    def test_blackman_envelope(self) -> None:
        """Blackman envelope produces a valid final state."""
        coords = [(0.0, 0.0), (7.0, 0.0)]
        result = simulate_rydberg_evolution(
            coords, omega_max=6.0, delta_start=-15.0, delta_end=8.0,
            duration_ns=3000, n_steps=150, omega_shape="blackman",
        )
        psi = result["final_state"]
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-8

    def test_entanglement_from_interaction(self) -> None:
        """Interacting atoms develop entanglement entropy > 0."""
        coords = [(0.0, 0.0), (7.0, 0.0)]
        result = simulate_rydberg_evolution(
            coords, omega_max=5.0, delta_start=-10.0, delta_end=5.0,
            duration_ns=2000, n_steps=150,
        )
        assert result["entanglement_entropy"] > 0.01

    def test_three_atom_chain_af_order(self) -> None:
        """Adiabatic sweep on a 3-atom chain should produce some AF order."""
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        result = simulate_rydberg_evolution(
            coords, omega_max=5.0, delta_start=-25.0, delta_end=15.0,
            duration_ns=4000, n_steps=250,
        )
        af = result["antiferromagnetic_order"]
        assert af > 0.1, f"Expected some AF order, got {af}"

    def test_output_keys(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0)]
        result = simulate_rydberg_evolution(
            coords, omega_max=3.0, delta_start=-5.0, delta_end=3.0,
            duration_ns=1000, n_steps=50,
        )
        expected_keys = {
            "final_state", "n_atoms", "rydberg_densities",
            "pair_correlations", "connected_correlations",
            "total_rydberg_fraction", "antiferromagnetic_order",
            "entanglement_entropy", "top_bitstrings", "backend", "n_steps", "dt_us",
            "integration_order", "pulse_schedule", "adiabatic_ratios",
            "adiabatic_ratio_max", "adiabatic_warning",
            "blockade_radius_min_um", "blockade_radius_max_um",
        }
        assert expected_keys <= set(result.keys())

    def test_dynamic_blockade_radius_changes_for_ramp(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0)]
        result = simulate_rydberg_evolution(
            coords,
            omega_max=6.0,
            delta_start=-15.0,
            delta_end=8.0,
            duration_ns=3000,
            n_steps=60,
            omega_shape="ramp",
        )
        assert result["blockade_radius_min_um"] is not None
        assert result["blockade_radius_max_um"] is not None
        assert result["blockade_radius_max_um"] > result["blockade_radius_min_um"]

    def test_fast_sweep_flags_adiabatic_warning(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        result = simulate_rydberg_evolution(
            coords,
            omega_max=2.0,
            delta_start=-30.0,
            delta_end=30.0,
            duration_ns=120.0,
            n_steps=80,
        )
        assert result["adiabatic_warning"] is True


class TestPulseSchedule:
    def test_custom_linear_schedule_interpolates_profiles(self) -> None:
        schedule = PulseSchedule(
            duration_us=2.0,
            omega_times_us=np.array([0.0, 1.0, 2.0]),
            omega_values=np.array([0.0, 2.0, 4.0]),
            delta_times_us=np.array([0.0, 2.0]),
            delta_values=np.array([-2.0, 2.0]),
        )

        np.testing.assert_allclose(schedule.omega_at(np.array([0.5, 1.5])), [1.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(schedule.delta_at(np.array([0.5, 1.5])), [-1.0, 1.0], atol=1e-12)

    def test_simulation_accepts_custom_schedule(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0)]
        schedule = PulseSchedule(
            duration_us=1.5,
            omega_times_us=np.array([0.0, 0.75, 1.5]),
            omega_values=np.array([0.0, 5.0, 5.0]),
            delta_times_us=np.array([0.0, 1.5]),
            delta_values=np.array([-10.0, 8.0]),
        )
        result = simulate_rydberg_evolution(
            coords,
            omega_max=0.0,
            delta_start=0.0,
            delta_end=0.0,
            duration_ns=1500,
            n_steps=50,
            schedule=schedule,
        )

        assert result["pulse_schedule"]["interpolation"] == "linear"
        assert len(result["pulse_schedule"]["omega_profile"]) == 50
        assert result["integration_order"] == 2

    def test_schedule_diagnostics_reports_gap_and_blockade(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0)]
        diagnostics = compute_schedule_diagnostics(
            coords=coords,
            n_steps=40,
            omega_max=5.0,
            delta_start=-10.0,
            delta_end=10.0,
            duration_ns=2000.0,
        )

        assert diagnostics["adiabatic_ratio_max"] is not None
        assert diagnostics["adiabatic_gap_min"] is not None
        assert diagnostics["blockade_radius_min_um"] is not None
        assert diagnostics["pulse_schedule"]["times_us"]


class TestDiscretizationError:
    """Tests for the time-discretization error estimator."""

    def test_constant_hamiltonian_zero_bound(self) -> None:
        """Constant Omega and delta give zero estimated dH/dt."""
        result = estimate_discretization_error(
            coords=[(0.0, 0.0), (7.0, 0.0)],
            omega_max=5.0,
            delta_start=-10.0,
            delta_end=-10.0,
            duration_ns=1000,
            n_steps=50,
            omega_shape="constant",
        )
        assert result["analytical_bound"] == pytest.approx(0.0, abs=1e-6)
        assert result["fidelity_half_steps"] > 0.9999

    def test_sweep_produces_nonzero_bound(self) -> None:
        """A detuning sweep should induce a non-zero analytical bound."""
        result = estimate_discretization_error(
            coords=[(0.0, 0.0), (7.0, 0.0)],
            omega_max=5.0,
            delta_start=-20.0,
            delta_end=10.0,
            duration_ns=2000,
            n_steps=100,
        )
        assert result["analytical_bound"] > 0.0

    def test_more_steps_improve_fidelity(self) -> None:
        """Increasing the number of slices should not degrade the estimate."""
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        result_50 = estimate_discretization_error(
            coords=coords,
            omega_max=5.0,
            delta_start=-20.0,
            delta_end=10.0,
            duration_ns=3000,
            n_steps=50,
        )
        result_200 = estimate_discretization_error(
            coords=coords,
            omega_max=5.0,
            delta_start=-20.0,
            delta_end=10.0,
            duration_ns=3000,
            n_steps=200,
        )
        assert result_200["fidelity_half_steps"] >= result_50["fidelity_half_steps"] - 1e-6

    def test_recommended_steps_reasonable(self) -> None:
        """Recommended n_steps should never be below the current one."""
        result = estimate_discretization_error(
            coords=[(0.0, 0.0), (7.0, 0.0)],
            omega_max=5.0,
            delta_start=-20.0,
            delta_end=10.0,
            duration_ns=2000,
            n_steps=100,
        )
        assert result["recommended_n_steps"] >= result["n_steps_used"]

    def test_result_keys(self) -> None:
        """Output dict should contain the full diagnostic payload."""
        result = estimate_discretization_error(
            coords=[(0.0, 0.0), (7.0, 0.0)],
            omega_max=5.0,
            delta_start=-10.0,
            delta_end=5.0,
            duration_ns=1000,
            n_steps=50,
        )
        assert {
            "analytical_bound",
            "fidelity_half_steps",
            "recommended_n_steps",
            "n_steps_used",
            "adiabatic_ratio_max",
            "adiabatic_warning",
        } <= set(result.keys())


class TestGroundStatePhysics:
    """Verify that the static Hamiltonian ground state captures known physics."""

    def test_large_negative_detuning_ground_state_is_all_ground(self) -> None:
        """Deep in the disordered phase (delta << 0), GS ≈ |gg...g>."""
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        gs = ground_state(coords, omega=5.0, delta=-50.0)
        prob_all_ground = abs(gs[0]) ** 2
        assert prob_all_ground > 0.9

    def test_large_positive_detuning_creates_excitations(self) -> None:
        """Deep in the ordered phase (delta >> 0), GS has Rydberg excitations."""
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        gs = ground_state(coords, omega=5.0, delta=30.0)
        dens = rydberg_density(gs, 3)
        assert total_rydberg_fraction(gs, 3) > 0.2
