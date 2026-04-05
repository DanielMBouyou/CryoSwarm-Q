from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="scipy required for sparse Hamiltonian tests")

from packages.simulation.hamiltonian import (  # noqa: E402
    build_hamiltonian_matrix,
    build_sparse_hamiltonian,
    ground_state,
    sparse_ground_state,
    sparse_spectral_gap,
    spectral_gap,
)


def _line_coords(atom_count: int, spacing: float = 7.0) -> list[tuple[float, float]]:
    return [(float(index) * spacing, 0.0) for index in range(atom_count)]


def test_sparse_and_dense_hamiltonians_match_eigenvalues_for_four_atoms() -> None:
    coords = _line_coords(4)
    dense = build_hamiltonian_matrix(coords, omega=5.0, delta=-10.0)
    sparse = build_sparse_hamiltonian(coords, omega=5.0, delta=-10.0)

    dense_eigs = np.sort(np.linalg.eigvalsh(dense).real)
    sparse_eigs = np.sort(np.linalg.eigvalsh(sparse.toarray()).real)

    np.testing.assert_allclose(sparse_eigs, dense_eigs, atol=1e-9)


def test_sparse_ground_state_matches_dense_ground_state_for_six_atoms() -> None:
    coords = _line_coords(6)
    dense_state = ground_state(coords, omega=5.0, delta=-8.0)
    _, sparse_states = sparse_ground_state(coords, omega=5.0, delta=-8.0, k=2)
    sparse_state = sparse_states[:, 0]

    overlap = abs(np.vdot(dense_state, sparse_state))
    assert overlap > 0.999


def test_sparse_hamiltonian_builds_for_sixteen_atoms() -> None:
    coords = _line_coords(16)
    sparse = build_sparse_hamiltonian(coords, omega=4.0, delta=-6.0)

    assert sparse.shape == (2**16, 2**16)
    assert sparse.nnz > 0


def test_sparse_spectral_gap_matches_dense_for_small_system() -> None:
    coords = _line_coords(5)
    dense_gap = spectral_gap(coords, omega=5.0, delta=-10.0)
    sparse_gap = sparse_spectral_gap(coords, omega=5.0, delta=-10.0)

    assert sparse_gap == pytest.approx(dense_gap, rel=1e-7, abs=1e-9)
