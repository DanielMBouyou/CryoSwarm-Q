"""Quantum observables for neutral-atom Rydberg systems.

Computes expectation values, correlations, and entanglement measures
from quantum state vectors in the {|g>, |r>} computational basis.

Bitstring convention (project-wide)
-----------------------------------
- |g> = |0>, |r> = |1> per atom.
- Basis ordering: |q0 q1 ... q_{N-1}> with **q0 as the most significant bit (MSB)**.
- Integer index of state |b_0 b_1 ... b_{N-1}> = sum_i b_i * 2^{N-1-i}.
- Bit extraction: atom *i* is excited iff ``(index >> (n_atoms - 1 - i)) & 1``.
- Bitstring format: ``format(index, f'0{n_atoms}b')`` — leftmost character is q0.
- n_i = |r_i><r_i| measures Rydberg occupation of atom i.

This convention is consistent with Pulser's default qubit ordering.
All modules in ``packages/simulation/`` and ``packages/ml/gpu_backend.py``
MUST follow this convention.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def atom_excited(basis_index: int, atom: int, n_atoms: int) -> bool:
    """Check if *atom* is in |r> in the given computational-basis index.

    Uses the project-wide MSB convention: atom 0 is the most significant bit.
    """
    return bool((basis_index >> (n_atoms - 1 - atom)) & 1)


def rydberg_density(
    state_vector: NDArray[np.complex128],
    n_atoms: int,
) -> NDArray[np.float64]:
    """Per-site Rydberg occupation <n_i> for each atom.

    Returns an array of length *n_atoms* with entry *i* giving
    the probability that atom *i* is in the Rydberg state |r>.
    """
    dim = 2**n_atoms
    probs = np.abs(state_vector.ravel()[:dim]) ** 2
    densities = np.zeros(n_atoms, dtype=np.float64)
    for basis in range(dim):
        p = probs[basis]
        if p < 1e-15:
            continue
        for i in range(n_atoms):
            if atom_excited(basis, i, n_atoms):
                densities[i] += p
    return densities


def pair_correlation(
    state_vector: NDArray[np.complex128],
    n_atoms: int,
) -> NDArray[np.float64]:
    """Joint Rydberg excitation <n_i n_j> for all atom pairs.

    Returns an *n_atoms x n_atoms* symmetric matrix.
    """
    dim = 2**n_atoms
    probs = np.abs(state_vector.ravel()[:dim]) ** 2
    corr = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for basis in range(dim):
        p = probs[basis]
        if p < 1e-15:
            continue
        excited = [i for i in range(n_atoms) if atom_excited(basis, i, n_atoms)]
        for idx_a, a in enumerate(excited):
            for b in excited[idx_a + 1 :]:
                corr[a, b] += p
                corr[b, a] += p
    return corr


def connected_correlation(
    state_vector: NDArray[np.complex128],
    n_atoms: int,
) -> NDArray[np.float64]:
    """Connected correlation function g_ij = <n_i n_j> - <n_i><n_j>.

    Positive values indicate bunching; negative values indicate
    anti-bunching characteristic of antiferromagnetic order.
    """
    dens = rydberg_density(state_vector, n_atoms)
    corr = pair_correlation(state_vector, n_atoms)
    g = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            g[i, j] = corr[i, j] - dens[i] * dens[j]
            g[j, i] = g[i, j]
    return g


def antiferromagnetic_order(
    state_vector: NDArray[np.complex128],
    n_atoms: int,
) -> float:
    """Staggered magnetization for 1-D chains.

    m_AF = (1/N) |sum_i (-1)^i (2<n_i> - 1)|

    Returns 0 for a disordered state and 1 for perfect AF ordering.
    """
    dens = rydberg_density(state_vector, n_atoms)
    staggered = sum((-1) ** i * (2.0 * dens[i] - 1.0) for i in range(n_atoms))
    return abs(staggered) / n_atoms


def total_rydberg_fraction(
    state_vector: NDArray[np.complex128],
    n_atoms: int,
) -> float:
    """Average Rydberg excitation fraction <N_r>/N."""
    return float(np.mean(rydberg_density(state_vector, n_atoms)))


def entanglement_entropy(
    state_vector: NDArray[np.complex128],
    n_atoms: int,
    partition_a: list[int] | None = None,
) -> float:
    """Von Neumann entanglement entropy S_A = -Tr(rho_A log2 rho_A).

    Default partition: first floor(N/2) atoms.
    Uses the Schmidt decomposition (SVD) of the reshaped state vector.
    """
    if n_atoms < 2:
        return 0.0
    partition_a = partition_a or list(range(n_atoms // 2))
    partition_b = [i for i in range(n_atoms) if i not in partition_a]
    n_a = len(partition_a)
    n_b = len(partition_b)
    dim_a = 2**n_a
    dim_b = 2**n_b
    dim = 2**n_atoms

    psi = state_vector.ravel()[:dim]
    qubit_order = partition_a + partition_b

    psi_reordered = np.zeros(dim, dtype=np.complex128)
    for old_idx in range(dim):
        bits = [(old_idx >> (n_atoms - 1 - q)) & 1 for q in range(n_atoms)]
        new_bits = [bits[q] for q in qubit_order]
        new_idx = 0
        for bit in new_bits:
            new_idx = (new_idx << 1) | bit
        psi_reordered[new_idx] = psi[old_idx]

    psi_matrix = psi_reordered.reshape(dim_a, dim_b)
    singular_values = np.linalg.svd(psi_matrix, compute_uv=False)
    schmidt_weights = singular_values**2
    schmidt_weights = schmidt_weights[schmidt_weights > 1e-15]
    return float(-np.sum(schmidt_weights * np.log2(schmidt_weights)))


def state_fidelity(
    state1: NDArray[np.complex128],
    state2: NDArray[np.complex128],
) -> float:
    """Fidelity |<psi1|psi2>|^2 between two pure states."""
    return float(np.abs(np.vdot(state1.ravel(), state2.ravel())) ** 2)


def bitstring_probabilities(
    state_vector: NDArray[np.complex128],
    n_atoms: int,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Top-*k* most probable computational-basis bitstrings."""
    dim = 2**n_atoms
    probs = np.abs(state_vector.ravel()[:dim]) ** 2
    indices = np.argsort(probs)[::-1][:top_k]
    return [
        (format(idx, f"0{n_atoms}b"), float(probs[idx]))
        for idx in indices
        if probs[idx] > 1e-10
    ]


def mis_overlap(
    state_vector: NDArray[np.complex128],
    n_atoms: int,
    mis_bitstrings: list[str],
) -> float:
    """Total probability weight on Maximum Independent Set bitstrings."""
    dim = 2**n_atoms
    probs = np.abs(state_vector.ravel()[:dim]) ** 2
    total = 0.0
    for bs in mis_bitstrings:
        idx = int(bs, 2)
        if idx < dim:
            total += probs[idx]
    return float(total)
