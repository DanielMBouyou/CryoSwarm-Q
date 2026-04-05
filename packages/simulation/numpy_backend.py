"""NumPy / SciPy exact simulation backend for small neutral-atom systems.

Solves the time-dependent Schrodinger equation using Trotter-Suzuki
decomposition when Pulser / QutipEmulator is unavailable.

Limited to N <= 12 atoms (Hilbert-space dimension 4096).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from packages.simulation.hamiltonian import (
    C6_RB87_70S,
    build_hamiltonian_matrix,
    build_sparse_hamiltonian,
)
from packages.simulation.observables import (
    antiferromagnetic_order,
    bitstring_probabilities,
    connected_correlation,
    entanglement_entropy,
    pair_correlation,
    rydberg_density,
    total_rydberg_fraction,
)

try:
    from scipy.linalg import expm
    from scipy.sparse.linalg import expm_multiply

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


MAX_ATOMS_DENSE = 12
MAX_ATOMS_SPARSE = 18


def simulate_rydberg_evolution(
    coords: list[tuple[float, float]],
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    n_steps: int = 200,
    omega_shape: str = "constant",
    c6: float = C6_RB87_70S,
) -> dict[str, Any]:
    """Time-evolve the Rydberg Hamiltonian with a detuning sweep.

    Parameters
    ----------
    coords : atom positions in um.
    omega_max : peak Rabi frequency in rad/us.
    delta_start, delta_end : detuning sweep endpoints in rad/us.
    duration_ns : total pulse duration in ns.
    n_steps : Trotter steps (higher = more accurate).
    omega_shape : ``"constant"``, ``"ramp"`` or ``"blackman"``.
    c6 : van-der-Waals coefficient.

    Returns
    -------
    dict with ``final_state``, per-site densities, correlations,
    entanglement entropy, AF order, and top bitstrings.
    """
    n_atoms = len(coords)
    if n_atoms > MAX_ATOMS_SPARSE:
        raise ValueError(
            f"NumPy backend limited to {MAX_ATOMS_SPARSE} atoms "
            f"(requested {n_atoms}, dim={2**n_atoms})."
        )
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for the numpy backend (pip install scipy).")

    dim = 2**n_atoms
    total_time_us = duration_ns / 1000.0
    dt = total_time_us / n_steps

    # |g ... g> = |0 ... 0>
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0

    backend_label = "numpy_exact" if n_atoms <= MAX_ATOMS_DENSE else "scipy_sparse"

    for step in range(n_steps):
        frac = (step + 0.5) / n_steps

        if omega_shape == "blackman":
            omega_t = omega_max * (
                0.42 - 0.5 * np.cos(2 * np.pi * frac) + 0.08 * np.cos(4 * np.pi * frac)
            )
        elif omega_shape == "ramp":
            omega_t = omega_max * min(frac * 2.0, 1.0)
        else:
            omega_t = omega_max

        delta_t = delta_start + (delta_end - delta_start) * frac
        if n_atoms <= MAX_ATOMS_DENSE:
            H = build_hamiltonian_matrix(coords, omega_t, delta_t, c6)
            psi = expm(-1j * H * dt) @ psi
        else:
            H = build_sparse_hamiltonian(coords, omega_t, delta_t, c6)
            psi = expm_multiply((-1j * H * dt), psi)

    psi /= np.linalg.norm(psi)

    dens = rydberg_density(psi, n_atoms)
    return {
        "final_state": psi,
        "n_atoms": n_atoms,
        "rydberg_densities": dens.tolist(),
        "pair_correlations": pair_correlation(psi, n_atoms).tolist(),
        "connected_correlations": connected_correlation(psi, n_atoms).tolist(),
        "total_rydberg_fraction": total_rydberg_fraction(psi, n_atoms),
        "antiferromagnetic_order": antiferromagnetic_order(psi, n_atoms),
        "entanglement_entropy": entanglement_entropy(psi, n_atoms),
        "top_bitstrings": bitstring_probabilities(psi, n_atoms, top_k=5),
        "backend": backend_label,
        "n_steps": n_steps,
    }
