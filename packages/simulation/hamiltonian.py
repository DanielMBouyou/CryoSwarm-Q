"""Rydberg Hamiltonian construction and interaction analysis.

Provides standalone Hamiltonian building, blockade-graph analysis,
and Maximum Independent Set (MIS) enumeration for neutral-atom registers.

Units
-----
- Distances: micrometers (um)
- Frequencies / energies: rad/us  (with hbar = 1)
- C6 coefficient: rad * um^6 / us

Bitstring convention
--------------------
All bit-level operations use the project-wide MSB convention:
atom *i* corresponds to bit position ``(n_atoms - 1 - i)`` in the
integer basis index. See ``packages/simulation/observables.py``.

The default C6 value corresponds to 87-Rb |70S_{1/2}>.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from packages.core.logging import get_logger
from packages.core.parameter_space import PhysicsParameterSpace

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


C6_RB87_70S: float = PhysicsParameterSpace.default().c6_coefficient
"""C6 / (2*pi*hbar) for 87-Rb |70S_{1/2}> in rad*um^6/us."""

logger = get_logger(__name__)


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
    """R_b = (C6 / Omega)^{1/6} in um."""
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


def _greedy_independent_set(
    adjacency: NDArray[np.bool_],
) -> list[int]:
    """Greedy MIS heuristic using repeated minimum-degree vertex selection."""
    available = set(range(adjacency.shape[0]))
    result: list[int] = []

    while available:
        best = min(
            available,
            key=lambda vertex: sum(
                1
                for neighbour in available
                if neighbour != vertex and adjacency[vertex, neighbour]
            ),
        )
        result.append(best)
        neighbours = {vertex for vertex in available if adjacency[best, vertex]}
        available -= neighbours | {best}

    return sorted(result)


def _greedy_mis_multi(
    adjacency: NDArray[np.bool_],
    n_restarts: int = 10,
    max_results: int = 20,
) -> list[list[int]]:
    """Run the greedy MIS heuristic with randomised restarts.

    Returns up to *max_results* distinct maximal independent sets having the
    largest cardinality observed across the randomised restarts.
    """
    n = adjacency.shape[0]
    rng = np.random.default_rng(42)
    best_size = 0
    results: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    for restart in range(n_restarts):
        available = set(range(n))
        result: list[int] = []
        order = list(range(n)) if restart == 0 else rng.permutation(n).tolist()

        while available:
            candidates = [vertex for vertex in order if vertex in available]
            if not candidates:
                break
            best = min(
                candidates,
                key=lambda vertex: (
                    sum(
                        1
                        for neighbour in available
                        if neighbour != vertex and adjacency[vertex, neighbour]
                    ),
                    order.index(vertex),
                ),
            )
            result.append(best)
            neighbours = {vertex for vertex in available if adjacency[best, vertex]}
            available -= neighbours | {best}

        result_sorted = sorted(result)
        key = tuple(result_sorted)
        size = len(result_sorted)

        if size > best_size:
            best_size = size
            results = [result_sorted]
            seen = {key}
        elif size == best_size and key not in seen:
            results.append(result_sorted)
            seen.add(key)
            if len(results) >= max_results:
                break

    if not results:
        fallback = _greedy_independent_set(adjacency)
        return [fallback] if fallback else []
    return results


_MIS_EXACT_THRESHOLD: int = 15
"""Maximum atom count for exact MIS enumeration before switching to heuristics."""


def find_maximum_independent_sets(
    adjacency: NDArray[np.bool_],
    max_results: int = 20,
) -> list[list[int]]:
    """Find maximum independent sets on the blockade graph.

    For n <= 15 atoms, performs exact brute-force enumeration.
    For 15 < n <= 50, uses a randomised greedy heuristic.
    For n > 50, returns an empty list.

    An independent set contains no two adjacent (blockaded) vertices.
    Returns sets of maximum cardinality found, up to *max_results*.
    """
    n = adjacency.shape[0]
    if n > 50:
        logger.warning("MIS computation skipped for n=%d atoms (limit: 50).", n)
        return []

    if n > _MIS_EXACT_THRESHOLD:
        logger.debug(
            "Using greedy MIS heuristic for n=%d atoms (exact threshold: %d).",
            n,
            _MIS_EXACT_THRESHOLD,
        )
        return _greedy_mis_multi(adjacency, n_restarts=max(20, n), max_results=max_results)

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
    for independent_set in mis_sets:
        bits = ["0"] * n
        for idx in independent_set:
            bits[idx] = "1"
        out.append("".join(bits))
    return out


# ---------- Hamiltonian matrix ----------


def build_hamiltonian_matrix(
    coords: list[tuple[float, float]],
    omega: float,
    delta: float,
    c6: float = C6_RB87_70S,
    max_atoms_dense: int | None = None,
) -> NDArray[np.complex128]:
    r"""Full Rydberg Hamiltonian for exact diagonalisation.

    H = (Omega/2) sum_i sigma_x^{(i)}
        - delta   sum_i n_i
        + sum_{i<j} U_{ij} n_i n_j

    Returns a dense 2^N x 2^N matrix with the default threshold sourced from
    ``PhysicsParameterSpace.max_atoms_evaluator_parallel``.
    """
    n = len(coords)
    dense_limit = (
        max_atoms_dense
        if max_atoms_dense is not None
        else PhysicsParameterSpace.default().max_atoms_evaluator_parallel
    )
    if n > dense_limit:
        raise ValueError(
            f"Dense Hamiltonian limited to {dense_limit} atoms. Use sparse methods for larger systems."
        )
    dim = 2**n
    U = van_der_waals_matrix(coords, c6)

    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    n_op = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    eye2 = np.eye(2, dtype=np.complex128)

    def _site_op(op: NDArray[np.complex128], site: int) -> NDArray[np.complex128]:
        result = np.array([[1.0]], dtype=np.complex128)
        for idx in range(n):
            result = np.kron(result, op if idx == site else eye2)
        return result

    H = np.zeros((dim, dim), dtype=np.complex128)

    for idx in range(n):
        H += (omega / 2.0) * _site_op(sigma_x, idx)
        H -= delta * _site_op(n_op, idx)

    for row_index in range(n):
        for col_index in range(row_index + 1, n):
            if U[row_index, col_index] < 1e-10:
                continue
            ni_nj = np.array([[1.0]], dtype=np.complex128)
            for idx in range(n):
                ni_nj = np.kron(ni_nj, n_op if idx in (row_index, col_index) else eye2)
            H += U[row_index, col_index] * ni_nj

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
