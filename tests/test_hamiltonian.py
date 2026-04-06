"""Tests for Rydberg Hamiltonian construction and analysis."""
from __future__ import annotations

import numpy as np
import pytest

from packages.simulation.hamiltonian import (
    C6_RB87_70S,
    blockade_radius,
    build_hamiltonian_matrix,
    find_maximum_independent_sets,
    ground_state,
    interaction_graph,
    mis_bitstrings,
    pairwise_distances,
    spectral_gap,
    van_der_waals_matrix,
)


# ---- distance / interaction helpers ----


class TestPairwiseDistances:
    def test_three_atoms_line(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        dists = pairwise_distances(coords)
        assert dists.shape == (3, 3)
        np.testing.assert_allclose(dists[0, 1], 7.0)
        np.testing.assert_allclose(dists[0, 2], 14.0)
        np.testing.assert_allclose(dists[1, 2], 7.0)
        np.testing.assert_allclose(np.diag(dists), 0.0)

    def test_symmetry(self) -> None:
        coords = [(1.0, 2.0), (4.0, 6.0), (0.0, 0.0)]
        dists = pairwise_distances(coords)
        np.testing.assert_array_equal(dists, dists.T)


class TestVanDerWaals:
    def test_interaction_decreases_with_distance(self) -> None:
        coords = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        U = van_der_waals_matrix(coords)
        assert U[0, 1] > U[0, 2], "Closer atoms should interact more strongly"

    def test_matches_c6_formula(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0)]
        U = van_der_waals_matrix(coords)
        expected = C6_RB87_70S / 7.0**6
        np.testing.assert_allclose(U[0, 1], expected, rtol=1e-10)


# ---- blockade ----


class TestBlockadeRadius:
    def test_decreases_with_omega(self) -> None:
        r1 = blockade_radius(1.0)
        r5 = blockade_radius(5.0)
        r10 = blockade_radius(10.0)
        assert r1 > r5 > r10

    def test_known_value(self) -> None:
        omega = 5.0
        r_b = blockade_radius(omega)
        expected = (C6_RB87_70S / omega) ** (1.0 / 6.0)
        np.testing.assert_allclose(r_b, expected, rtol=1e-10)

    def test_zero_omega_returns_inf(self) -> None:
        assert blockade_radius(0.0) == float("inf")


class TestInteractionGraph:
    def test_three_atom_chain(self) -> None:
        """At spacing 7 um and Omega=5, nearest neighbors are blockaded."""
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        r_b = blockade_radius(5.0)
        graph = interaction_graph(coords, 5.0)
        assert graph[0, 1]  # nn blockaded
        assert graph[1, 2]  # nn blockaded
        assert not graph[0, 2] or 14.0 < r_b  # nnn depends on r_b


# ---- MIS ----


class TestMIS:
    def test_three_atom_chain_mis(self) -> None:
        """For a 3-atom chain with only nn edges, MIS = {0,2}."""
        adj = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ])
        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1
        assert [0, 2] in sets

    def test_bitstrings_format(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        bs = mis_bitstrings(coords, omega=5.0)
        for s in bs:
            assert len(s) == 3
            assert all(c in "01" for c in s)

    def test_disconnected_graph_all_selected(self) -> None:
        """No edges implies every vertex belongs to the MIS."""
        adj = np.zeros((4, 4), dtype=bool)
        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1
        assert [0, 1, 2, 3] in sets

    def test_complete_graph_single_vertex(self) -> None:
        """A clique has MIS size 1."""
        n = 5
        adj = np.ones((n, n), dtype=bool)
        np.fill_diagonal(adj, False)
        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1
        assert all(len(candidate) == 1 for candidate in sets)

    def test_greedy_fallback_large_system(self) -> None:
        """For n > 15, the greedy heuristic should return valid sets."""
        n = 20
        rng = np.random.default_rng(42)
        adj = rng.random((n, n)) < 0.3
        adj = np.logical_or(adj, adj.T)
        np.fill_diagonal(adj, False)

        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1
        for candidate in sets:
            for index, vertex_a in enumerate(candidate):
                for vertex_b in candidate[index + 1 :]:
                    assert not adj[vertex_a, vertex_b]

    def test_greedy_result_is_maximal(self) -> None:
        """Heuristic MIS outputs must be maximal independent sets."""
        n = 18
        adj = np.zeros((n, n), dtype=bool)
        for idx in range(n - 1):
            adj[idx, idx + 1] = True
            adj[idx + 1, idx] = True

        sets = find_maximum_independent_sets(adj)
        assert len(sets) >= 1
        for candidate in sets:
            candidate_set = set(candidate)
            for vertex in range(n):
                if vertex in candidate_set:
                    continue
                assert any(adj[vertex, neighbour] for neighbour in candidate_set)

    def test_greedy_matches_exact_for_small(self) -> None:
        """For small systems, greedy restarts should match the exact MIS size."""
        from packages.simulation.hamiltonian import _greedy_mis_multi

        n = 10
        adj = np.zeros((n, n), dtype=bool)
        for idx in range(n - 1):
            adj[idx, idx + 1] = True
            adj[idx + 1, idx] = True

        exact = find_maximum_independent_sets(adj)
        greedy = _greedy_mis_multi(adj, n_restarts=50)
        exact_size = len(exact[0]) if exact else 0
        greedy_size = len(greedy[0]) if greedy else 0
        assert greedy_size == exact_size

    def test_very_large_system_returns_empty(self) -> None:
        """Very large systems are skipped for safety."""
        adj = np.zeros((60, 60), dtype=bool)
        sets = find_maximum_independent_sets(adj)
        assert sets == []


# ---- Hamiltonian ----


class TestHamiltonian:
    def test_hermitian(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0)]
        H = build_hamiltonian_matrix(coords, omega=5.0, delta=-3.0)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_dimension(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0), (14.0, 0.0)]
        H = build_hamiltonian_matrix(coords, omega=5.0, delta=-3.0)
        assert H.shape == (8, 8)

    def test_ground_state_normalised(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0)]
        gs = ground_state(coords, omega=5.0, delta=0.0)
        np.testing.assert_allclose(np.linalg.norm(gs), 1.0, atol=1e-12)

    def test_spectral_gap_positive(self) -> None:
        coords = [(0.0, 0.0), (7.0, 0.0)]
        gap = spectral_gap(coords, omega=5.0, delta=0.0)
        assert gap > 0

    def test_strong_blockade_suppresses_double_excitation(self) -> None:
        """Two very close atoms at low Omega: ground state should have
        negligible weight on |rr> (both excited)."""
        coords = [(0.0, 0.0), (5.0, 0.0)]  # close spacing
        gs = ground_state(coords, omega=1.0, delta=0.0)  # low Omega = strong blockade
        prob_rr = abs(gs[3]) ** 2  # |11> = both excited
        assert prob_rr < 0.01, f"|rr> probability {prob_rr:.4f} too high under blockade"
