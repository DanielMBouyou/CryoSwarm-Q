"""NumPy / SciPy exact simulation backend for small neutral-atom systems.

Solves the time-dependent Schrodinger equation using Trotter-Suzuki
decomposition when Pulser / QutipEmulator is unavailable. Default dense and
sparse limits are sourced from ``PhysicsParameterSpace``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from packages.core.parameter_space import PhysicsParameterSpace
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
    state_fidelity,
    total_rydberg_fraction,
)

try:
    from scipy.linalg import expm
    from scipy.sparse.linalg import expm_multiply

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


_DEFAULT_PARAM_SPACE = PhysicsParameterSpace.default()


def simulate_rydberg_evolution(
    coords: list[tuple[float, float]],
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    n_steps: int = 200,
    omega_shape: str = "constant",
    c6: float = C6_RB87_70S,
    max_atoms_dense: int | None = None,
    max_atoms_sparse: int | None = None,
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
    dense_limit = max_atoms_dense if max_atoms_dense is not None else _DEFAULT_PARAM_SPACE.max_atoms_dense
    sparse_limit = (
        max_atoms_sparse
        if max_atoms_sparse is not None
        else _DEFAULT_PARAM_SPACE.max_atoms_sparse
    )
    n_atoms = len(coords)
    if n_atoms > sparse_limit:
        raise ValueError(
            f"NumPy backend limited to {sparse_limit} atoms "
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

    backend_label = "numpy_exact" if n_atoms <= dense_limit else "scipy_sparse"

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
        if n_atoms <= dense_limit:
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
        "dt_us": dt,
    }


def estimate_discretization_error(
    coords: list[tuple[float, float]],
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    n_steps: int = 200,
    omega_shape: str = "constant",
    c6: float = C6_RB87_70S,
    max_atoms_dense: int | None = None,
    max_atoms_sparse: int | None = None,
) -> dict[str, float | int]:
    """Estimate time-discretization error for the piecewise-constant evolution.

    Returns
    -------
    dict with:
        - ``analytical_bound``: epsilon <= (T^2 / 2N) * max ||dH/dt||.
        - ``fidelity_half_steps``: |<psi_N | psi_{N/2}>|^2.
        - ``recommended_n_steps``: suggested step count for fidelity > 0.999.
        - ``n_steps_used``: the reference step count used for the estimate.
    """
    dense_limit = (
        max_atoms_dense if max_atoms_dense is not None else _DEFAULT_PARAM_SPACE.max_atoms_dense
    )
    n_atoms = len(coords)
    total_time_us = duration_ns / 1000.0

    def _omega(frac: float) -> float:
        if omega_shape == "blackman":
            return omega_max * (
                0.42 - 0.5 * np.cos(2 * np.pi * frac) + 0.08 * np.cos(4 * np.pi * frac)
            )
        if omega_shape == "ramp":
            return omega_max * min(frac * 2.0, 1.0)
        return omega_max

    n_samples = min(n_steps, 10)
    h_norms: list[float] = []
    for sample in range(n_samples):
        frac_a = sample / n_samples
        frac_b = (sample + 1) / n_samples
        omega_a = _omega(frac_a)
        omega_b = _omega(frac_b)
        delta_a = delta_start + (delta_end - delta_start) * frac_a
        delta_b = delta_start + (delta_end - delta_start) * frac_b

        if n_atoms <= dense_limit:
            hamiltonian_a = build_hamiltonian_matrix(coords, omega_a, delta_a, c6)
            hamiltonian_b = build_hamiltonian_matrix(coords, omega_b, delta_b, c6)
            diff_norm = float(np.linalg.norm(hamiltonian_b - hamiltonian_a))
        else:
            hamiltonian_a = build_sparse_hamiltonian(coords, omega_a, delta_a, c6)
            hamiltonian_b = build_sparse_hamiltonian(coords, omega_b, delta_b, c6)
            diff = hamiltonian_b - hamiltonian_a
            if SCIPY_AVAILABLE:
                from scipy.sparse.linalg import norm as sparse_norm

                diff_norm = float(sparse_norm(diff))
            else:
                diff_norm = float(np.linalg.norm(diff.toarray()))

        delta_frac = 1.0 / n_samples
        dh_dt = diff_norm / (delta_frac * total_time_us) if total_time_us > 0 else 0.0
        h_norms.append(dh_dt)

    max_dh_dt = max(h_norms) if h_norms else 0.0
    analytical_bound = (total_time_us**2 / (2.0 * n_steps)) * max_dh_dt

    result_full = simulate_rydberg_evolution(
        coords,
        omega_max,
        delta_start,
        delta_end,
        duration_ns,
        n_steps=n_steps,
        omega_shape=omega_shape,
        c6=c6,
        max_atoms_dense=max_atoms_dense,
        max_atoms_sparse=max_atoms_sparse,
    )
    half_steps = max(n_steps // 2, 1)
    result_half = simulate_rydberg_evolution(
        coords,
        omega_max,
        delta_start,
        delta_end,
        duration_ns,
        n_steps=half_steps,
        omega_shape=omega_shape,
        c6=c6,
        max_atoms_dense=max_atoms_dense,
        max_atoms_sparse=max_atoms_sparse,
    )
    fidelity = state_fidelity(result_full["final_state"], result_half["final_state"])

    recommended = n_steps
    if fidelity < 0.999:
        ratio = (1.0 - fidelity) / 0.001
        recommended = int(np.ceil(n_steps * max(ratio, 1.0)))
        recommended = min(recommended, n_steps * 8)

    return {
        "analytical_bound": round(analytical_bound, 8),
        "fidelity_half_steps": round(fidelity, 8),
        "recommended_n_steps": recommended,
        "n_steps_used": n_steps,
    }
