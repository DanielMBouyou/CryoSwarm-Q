"""Tests for quantum observables computed from state vectors."""
from __future__ import annotations

import numpy as np
import pytest

from packages.simulation.observables import (
    antiferromagnetic_order,
    bitstring_probabilities,
    connected_correlation,
    entanglement_entropy,
    mis_overlap,
    pair_correlation,
    rydberg_density,
    state_fidelity,
    total_rydberg_fraction,
)


# ---- helpers ----

def _ground(n: int) -> np.ndarray:
    """All-ground state |00...0>."""
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def _excited(n: int) -> np.ndarray:
    """All-excited state |11...1>."""
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[-1] = 1.0
    return psi


def _bell_state(n: int = 2) -> np.ndarray:
    """Bell state (|00> + |11>) / sqrt(2) for 2 qubits."""
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    return psi


def _neel_state(n: int) -> np.ndarray:
    """Neel state |0101...> (perfect antiferromagnetic order)."""
    bits = "".join("0" if i % 2 == 0 else "1" for i in range(n))
    idx = int(bits, 2)
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[idx] = 1.0
    return psi


# ---- rydberg density ----


class TestRydbergDensity:
    def test_ground_state(self) -> None:
        dens = rydberg_density(_ground(3), 3)
        np.testing.assert_allclose(dens, [0.0, 0.0, 0.0])

    def test_all_excited(self) -> None:
        dens = rydberg_density(_excited(3), 3)
        np.testing.assert_allclose(dens, [1.0, 1.0, 1.0])

    def test_bell_state(self) -> None:
        dens = rydberg_density(_bell_state(), 2)
        np.testing.assert_allclose(dens, [0.5, 0.5], atol=1e-12)

    def test_neel_state(self) -> None:
        psi = _neel_state(4)
        dens = rydberg_density(psi, 4)
        np.testing.assert_allclose(dens, [0.0, 1.0, 0.0, 1.0])


# ---- pair correlation ----


class TestPairCorrelation:
    def test_ground_state_zero(self) -> None:
        corr = pair_correlation(_ground(3), 3)
        np.testing.assert_allclose(corr, np.zeros((3, 3)))

    def test_all_excited(self) -> None:
        corr = pair_correlation(_excited(3), 3)
        expected = np.ones((3, 3)) - np.eye(3)
        np.testing.assert_allclose(corr, expected)

    def test_bell_state_correlations(self) -> None:
        corr = pair_correlation(_bell_state(), 2)
        assert corr[0, 1] == pytest.approx(0.5, abs=1e-12)


# ---- connected correlation ----


class TestConnectedCorrelation:
    def test_product_state_zero(self) -> None:
        """Connected correlation vanishes for product states."""
        g = connected_correlation(_ground(3), 3)
        np.testing.assert_allclose(g, np.zeros((3, 3)), atol=1e-14)

    def test_bell_state_nonzero(self) -> None:
        """Bell state shows positive connected correlation (bunching)."""
        g = connected_correlation(_bell_state(), 2)
        assert g[0, 1] > 0  # <n0 n1> - <n0><n1> = 0.5 - 0.25 = 0.25


# ---- antiferromagnetic order ----


class TestAFOrder:
    def test_neel_state_perfect(self) -> None:
        af = antiferromagnetic_order(_neel_state(4), 4)
        assert af == pytest.approx(1.0, abs=1e-12)

    def test_ground_state_zero(self) -> None:
        """All-ground has zero Rydberg density → zero AF order."""
        af = antiferromagnetic_order(_ground(4), 4)
        assert af == pytest.approx(0.0, abs=1e-12)

    def test_bell_state_reduced(self) -> None:
        af = antiferromagnetic_order(_bell_state(), 2)
        assert af < 1.0


# ---- entanglement entropy ----


class TestEntanglementEntropy:
    def test_product_state_zero(self) -> None:
        s = entanglement_entropy(_ground(4), 4)
        assert s == pytest.approx(0.0, abs=1e-12)

    def test_bell_state_one_ebit(self) -> None:
        s = entanglement_entropy(_bell_state(), 2, partition_a=[0])
        assert s == pytest.approx(1.0, abs=1e-10)

    def test_single_atom(self) -> None:
        s = entanglement_entropy(_ground(1), 1)
        assert s == 0.0


# ---- fidelity ----


class TestFidelity:
    def test_same_state(self) -> None:
        psi = _bell_state()
        assert state_fidelity(psi, psi) == pytest.approx(1.0, abs=1e-12)

    def test_orthogonal_states(self) -> None:
        assert state_fidelity(_ground(2), _excited(2)) == pytest.approx(0.0, abs=1e-12)


# ---- bitstring probabilities ----


class TestBitstringProbabilities:
    def test_ground_state_dominant(self) -> None:
        probs = bitstring_probabilities(_ground(3), 3, top_k=3)
        assert probs[0] == ("000", pytest.approx(1.0, abs=1e-12))

    def test_bell_state_two_entries(self) -> None:
        probs = bitstring_probabilities(_bell_state(), 2, top_k=5)
        bitstrings = [bs for bs, _ in probs]
        assert "00" in bitstrings
        assert "11" in bitstrings


# ---- MIS overlap ----


class TestMISOverlap:
    def test_exact_overlap(self) -> None:
        psi = np.zeros(8, dtype=np.complex128)
        psi[5] = 1.0  # |101>
        overlap = mis_overlap(psi, 3, ["101"])
        assert overlap == pytest.approx(1.0, abs=1e-12)

    def test_no_overlap(self) -> None:
        overlap = mis_overlap(_ground(3), 3, ["101"])
        assert overlap == pytest.approx(0.0, abs=1e-12)
