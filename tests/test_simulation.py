"""Tests for the numpy exact-diagonal simulation backend."""
from __future__ import annotations

import numpy as np
import pytest

from packages.simulation.hamiltonian import blockade_radius, ground_state
from packages.simulation.observables import rydberg_density, total_rydberg_fraction

pytest.importorskip("scipy", reason="scipy required for numpy backend")
from packages.simulation.numpy_backend import simulate_rydberg_evolution  # noqa: E402


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
            "entanglement_entropy", "top_bitstrings", "backend", "n_steps",
        }
        assert expected_keys <= set(result.keys())


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
