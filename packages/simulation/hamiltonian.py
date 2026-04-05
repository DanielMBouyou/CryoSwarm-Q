"""Rydberg Hamiltonian construction and interaction analysis.

Provides standalone Hamiltonian building, blockade-graph analysis,
and Maximum Independent Set (MIS) enumeration for neutral-atom registers.

Units
-----
- Distances: micrometers (um)
- Frequencies / energies: rad/us  (with hbar = 1)
- C6 coefficient: rad * um^6 / us

The default C6 value corresponds to 87-Rb |70S_{1/2}>.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.sparse import csc_matrix, diags, eye, kron
    from scipy.sparse.linalg import eigsh

    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    SCIPY_SPARSE_AVAILABLE = False
    csc_matrix = None  # type: ignore[assignment]
    diags = None  # type: ignore[assignment]
    eye = None  # type: ignore[assignment]
    kron = None  # type: ignore[assignment]
    eigsh = None  # type: ignore[assignment]


C6_RB87_70S: float = 862690.0
"""C6 / (2*pi*hbar) for 87-Rb |70S_{1/2}> in rad*um^6/us."""


# ---------- distance / interaction helpers ----------


def pairwise_distances(
    coords: list[tuple[float, float]],
) -> NDArray[np.float64]:
    """Euclidean distance matrix between atom positions (um)."""
    pts = np.array(coords, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def van_der_waals_matrix(
    coords: list[tuple[float, float]],
    c6: float = C6_RB87_70S,
) -> NDArray[np.float64]:
    """Interaction matrix U_ij = C6 / r_ij^6 (rad/us)."""
    dists = pairwise_distances(coords)
    n = len(coords)
    U = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] > 0:
                U[i, j] = c6 / dists[i, j] ** 6
                U[j, i] = U[i, j]
    return U


# ---------- blockade ----------


def blockade_radius(omega: float, c6: float = C6_RB87_70S) -> float:
    """R_b = (C6 / Omega)^{1/6}  in um."""
    if omega <= 0:
        return float("inf")
    return float((c6 / omega) ** (1.0 / 6.0))


def interaction_graph(
    coords: list[tuple[float, float]],
    omega: float,
    c6: float = C6_RB87_70S,
) -> NDArray[np.bool_]:
    """Boolean adjacency matrix: True where atoms are within blockade radius."""
    r_b = blockade_radius(omega, c6)
    dists = pairwise_distances(coords)
    return (dists > 0) & (dists < r_b)


# ---------- Maximum Independent Set ----------


def find_maximum_independent_sets(
    adjacency: NDArray[np.bool_],
    max_results: int = 20,
) -> list[list[int]]:
    """Brute-force MIS enumeration (feasible for ≤20 atoms).

    An independent set contains no two adjacent (blockaded) vertices.
    Returns all sets of maximum cardinality, up to *max_results*.
    """
    n = adjacency.shape[0]
    if n > 20:
        return []
    best_size = 0
    results: list[list[int]] = []
    for size in range(n, 0, -1):
        if size < best_size:
            break
        for subset in combinations(range(n), size):
            valid = True
            for idx_a, a in enumerate(subset):
                for b in subset[idx_a + 1 :]:
                    if adjacency[a, b]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_size = size
                results.append(list(subset))
                if len(results) >= max_results:
                    return results
    return results


def mis_bitstrings(
    coords: list[tuple[float, float]],
    omega: float,
    c6: float = C6_RB87_70S,
) -> list[str]:
    """Bitstring representations of MIS solutions on the blockade graph."""
    n = len(coords)
    adj = interaction_graph(coords, omega, c6)
    mis_sets = find_maximum_independent_sets(adj)
    out: list[str] = []
    for s in mis_sets:
        bits = ["0"] * n
        for idx in s:
            bits[idx] = "1"
        out.append("".join(bits))
    return out


# ---------- Hamiltonian matrix ----------


def build_hamiltonian_matrix(
    coords: list[tuple[float, float]],
    omega: float,
    delta: float,
    c6: float = C6_RB87_70S,
) -> NDArray[np.complex128]:
    r"""Full Rydberg Hamiltonian for exact diagonalisation.

    H = (Omega/2) sum_i sigma_x^{(i)}
        - delta   sum_i n_i
        + sum_{i<j} U_{ij} n_i n_j

    Returns a dense 2^N x 2^N matrix (feasible for N <= 14).
    """
    n = len(coords)
    if n > 14:
        raise ValueError(
            "Dense Hamiltonian limited to 14 atoms (2^14 = 16384 dim). Use MPS for larger systems."
        )
    dim = 2**n
    U = van_der_waals_matrix(coords, c6)

    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    n_op = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    eye2 = np.eye(2, dtype=np.complex128)

    def _site_op(op: NDArray[np.complex128], site: int) -> NDArray[np.complex128]:
        result = np.array([[1.0]], dtype=np.complex128)
        for j in range(n):
            result = np.kron(result, op if j == site else eye2)
        return result

    H = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(n):
        H += (omega / 2.0) * _site_op(sigma_x, i)
        H -= delta * _site_op(n_op, i)

    for i in range(n):
        for j in range(i + 1, n):
            if U[i, j] < 1e-10:
                continue
            ni_nj = np.array([[1.0]], dtype=np.complex128)
            for k in range(n):
                ni_nj = np.kron(ni_nj, n_op if k in (i, j) else eye2)
            H += U[i, j] * ni_nj

    return H


def build_sparse_hamiltonian(
    coords: list[tuple[float, float]],
    omega: float,
    delta: float,
    c6: float = C6_RB87_70S,
):
    """Construct the Rydberg Hamiltonian as a sparse CSC matrix."""
    if not SCIPY_SPARSE_AVAILABLE:
        raise RuntimeError("scipy.sparse is required for sparse Hamiltonian construction.")

    n = len(coords)
    dim = 2**n
    U = van_der_waals_matrix(coords, c6)

    sigma_x = csc_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    eye2 = eye(2, format="csc", dtype=np.complex128)

    def _site_sigma_x(site: int):
        operator = csc_matrix([[1.0 + 0.0j]])
        for idx in range(n):
            operator = kron(operator, sigma_x if idx == site else eye2, format="csc")
        return operator

    H = csc_matrix((dim, dim), dtype=np.complex128)
    for site in range(n):
        H += (omega / 2.0) * _site_sigma_x(site)

    basis = np.arange(dim, dtype=np.uint64)
    diagonal = np.zeros(dim, dtype=np.float64)
    occupancies: list[NDArray[np.float64]] = []
    for site in range(n):
        occ = ((basis >> (n - 1 - site)) & 1).astype(np.float64)
        occupancies.append(occ)
        diagonal -= delta * occ

    for row_index in range(n):
        for col_index in range(row_index + 1, n):
            interaction = U[row_index, col_index]
            if interaction <= 1e-10:
                continue
            diagonal += interaction * occupancies[row_index] * occupancies[col_index]

    return (H + diags(diagonal.astype(np.complex128), format="csc")).tocsc()


# ---------- spectral analysis ----------


def spectral_gap(
    coords: list[tuple[float, float]],
    omega: float,
    delta: float,
    c6: float = C6_RB87_70S,
) -> float:
    """Energy gap E_1 - E_0 between ground state and first excited state."""
    H = build_hamiltonian_matrix(coords, omega, delta, c6)
    eigvals = np.sort(np.linalg.eigvalsh(H).real)
    if len(eigvals) < 2:
        return 0.0
    return float(eigvals[1] - eigvals[0])


def sparse_ground_state(
    coords: list[tuple[float, float]],
    omega: float,
    delta: float,
    c6: float = C6_RB87_70S,
    k: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return the lowest sparse eigenpairs of the Rydberg Hamiltonian."""
    H = build_sparse_hamiltonian(coords, omega, delta, c6)
    dim = H.shape[0]
    if dim <= 2:
        dense = build_hamiltonian_matrix(coords, omega, delta, c6)
        eigvals, eigvecs = np.linalg.eigh(dense)
        return eigvals.real[:1], eigvecs[:, :1]

    effective_k = min(max(k, 1), dim - 1)
    eigvals, eigvecs = eigsh(H, k=effective_k, which="SA")
    order = np.argsort(eigvals.real)
    return eigvals.real[order], eigvecs[:, order]


def sparse_spectral_gap(
    coords: list[tuple[float, float]],
    omega: float,
    delta: float,
    c6: float = C6_RB87_70S,
) -> float:
    """Estimate the many-body spectral gap using a sparse eigensolver."""
    eigvals, _ = sparse_ground_state(coords, omega, delta, c6, k=2)
    if len(eigvals) < 2:
        return 0.0
    return float(eigvals[1] - eigvals[0])


def ground_state(
    coords: list[tuple[float, float]],
    omega: float,
    delta: float,
    c6: float = C6_RB87_70S,
) -> NDArray[np.complex128]:
    """Ground-state eigenvector of the Rydberg Hamiltonian."""
    H = build_hamiltonian_matrix(coords, omega, delta, c6)
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs[:, np.argmin(eigvals)]
