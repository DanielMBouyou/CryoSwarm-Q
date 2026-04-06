"""GPU-accelerated simulation backend for neutral-atom Hamiltonians.

Provides sparse Hamiltonian construction and approximate time evolution for
larger neutral-atom systems when PyTorch and accelerator hardware are
available. Falls back to CPU callers when GPU execution is unavailable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from packages.core.logging import get_logger
from packages.core.parameter_space import PhysicsParameterSpace
from packages.simulation.hamiltonian import C6_RB87_70S, van_der_waals_matrix

try:
    import torch
    import torch.sparse

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = get_logger(__name__)

# ---------- Bitstring convention ----------
# This module follows the project-wide MSB convention:
#   Atom i corresponds to bit position (n_atoms - 1 - i).
#   |q0 q1 ... q_{N-1}> with q0 = most significant bit.
# See packages/simulation/observables.py for the canonical reference.


def get_device() -> str:
    """Detect the best available execution device for PyTorch workloads."""
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore[attr-defined]
        return "rocm"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sigma_x_elements(n_atoms: int, qubit: int) -> tuple[list[list[int]], list[complex]]:
    """Return sparse COO indices and values for sigma_x acting on one qubit."""
    dim = 2**n_atoms
    bit = 1 << (n_atoms - 1 - qubit)
    rows: list[int] = []
    cols: list[int] = []
    for state in range(dim):
        flipped = state ^ bit
        rows.append(state)
        cols.append(flipped)
    return [rows, cols], [1.0 + 0j] * dim


def _n_op_diagonal(n_atoms: int, qubit: int) -> NDArray[np.float64]:
    """Return the computational-basis diagonal of the Rydberg number operator."""
    dim = 2**n_atoms
    bit = 1 << (n_atoms - 1 - qubit)
    diagonal = np.zeros(dim, dtype=np.float64)
    for state in range(dim):
        if state & bit:
            diagonal[state] = 1.0
    return diagonal


def build_gpu_hamiltonian(
    coords: list[tuple[float, float]],
    omega: float,
    delta: float,
    c6: float = C6_RB87_70S,
    device: str | None = None,
    max_atoms_gpu: int | None = None,
) -> torch.Tensor:
    """Build the Rydberg Hamiltonian as a sparse COO tensor on the target device."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for GPU backend.")

    dev = device or get_device()
    n_atoms = len(coords)
    gpu_limit = (
        max_atoms_gpu
        if max_atoms_gpu is not None
        else PhysicsParameterSpace.default().max_atoms_gpu
    )
    if n_atoms > gpu_limit:
        raise ValueError(f"GPU backend limited to {gpu_limit} atoms (requested {n_atoms}).")

    dimension = 2**n_atoms
    interactions = van_der_waals_matrix(coords, c6)

    diagonal = np.zeros(dimension, dtype=np.float64)
    occupations = [_n_op_diagonal(n_atoms, index) for index in range(n_atoms)]

    for index in range(n_atoms):
        diagonal -= delta * occupations[index]

    for row_index in range(n_atoms):
        for col_index in range(row_index + 1, n_atoms):
            if interactions[row_index, col_index] > 0:
                diagonal += interactions[row_index, col_index] * occupations[row_index] * occupations[col_index]

    all_rows: list[int] = list(range(dimension))
    all_cols: list[int] = list(range(dimension))
    all_vals: list[complex] = [complex(value) for value in diagonal]

    for qubit in range(n_atoms):
        indices, values = _sigma_x_elements(n_atoms, qubit)
        for row_index, col_index in zip(indices[0], indices[1], strict=False):
            all_rows.append(row_index)
            all_cols.append(col_index)
        all_vals.extend([value * (omega / 2.0) for value in values])

    indices_t = torch.tensor([all_rows, all_cols], dtype=torch.long)
    values_t = torch.tensor(all_vals, dtype=torch.complex128)
    hamiltonian = torch.sparse_coo_tensor(indices_t, values_t, (dimension, dimension)).coalesce()
    return hamiltonian.to(dev)


def _lanczos_expm_multiply(
    hamiltonian: torch.Tensor,
    psi: torch.Tensor,
    dt: float,
    krylov_dim: int = 20,
) -> torch.Tensor:
    """Compute e^{-i H dt} |psi> via a short Lanczos Krylov projection."""
    norm_psi = psi.norm()
    if norm_psi.item() < 1e-15 or dt == 0.0:
        return psi.clone()

    device = psi.device
    dtype = psi.dtype
    dim = psi.shape[0]
    m = max(1, min(krylov_dim, dim))

    V = torch.zeros((m, dim), dtype=dtype, device=device)
    alpha = torch.zeros(m, dtype=torch.float64, device=device)
    beta = torch.zeros(m, dtype=torch.float64, device=device)
    V[0] = psi / norm_psi

    def _matvec(vector: torch.Tensor) -> torch.Tensor:
        if hamiltonian.is_sparse:
            return torch.sparse.mm(hamiltonian, vector.unsqueeze(1)).squeeze(1)
        return torch.mv(hamiltonian, vector)

    w = _matvec(V[0])
    alpha[0] = torch.vdot(V[0], w).real.to(torch.float64)
    w = w - alpha[0].to(dtype=dtype) * V[0]

    actual_m = m
    for j in range(1, m):
        beta_j = w.norm().real.to(torch.float64)
        beta[j] = beta_j
        if beta_j.item() < 1e-14:
            actual_m = j
            break

        V[j] = w / beta_j.to(dtype=dtype)
        w = _matvec(V[j])
        alpha[j] = torch.vdot(V[j], w).real.to(torch.float64)
        w = w - alpha[j].to(dtype=dtype) * V[j] - beta_j.to(dtype=dtype) * V[j - 1]

        for k in range(j + 1):
            overlap = torch.vdot(V[k], w)
            w = w - overlap * V[k]
    else:
        actual_m = m

    T = torch.zeros((actual_m, actual_m), dtype=torch.float64, device="cpu")
    for j in range(actual_m):
        T[j, j] = alpha[j].cpu()
    for j in range(1, actual_m):
        T[j, j - 1] = beta[j].cpu()
        T[j - 1, j] = beta[j].cpu()

    eigvals, eigvecs = torch.linalg.eigh(T)
    exp_diag = torch.exp(-1j * dt * eigvals.to(torch.complex128))
    eigvecs_complex = eigvecs.to(torch.complex128)
    expT = eigvecs_complex @ torch.diag(exp_diag) @ eigvecs_complex.T

    e1 = torch.zeros(actual_m, dtype=torch.complex128, device="cpu")
    e1[0] = 1.0 + 0.0j
    coeffs = (expT @ e1).to(device)

    result = norm_psi.to(dtype=dtype) * torch.matmul(
        V[:actual_m].transpose(0, 1),
        coeffs.to(dtype=dtype),
    )
    result_norm = result.norm()
    if result_norm.item() < 1e-15:
        return result
    return result / result_norm


def gpu_time_evolution(
    coords: list[tuple[float, float]],
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    n_steps: int = 200,
    omega_shape: str = "constant",
    c6: float = C6_RB87_70S,
    device: str | None = None,
    max_atoms_gpu: int | None = None,
    krylov_dim: int = 20,
) -> dict[str, Any]:
    """Approximate neutral-atom time evolution with sparse GPU Hamiltonians."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for GPU backend.")

    dev = device or get_device()
    n_atoms = len(coords)
    gpu_limit = (
        max_atoms_gpu
        if max_atoms_gpu is not None
        else PhysicsParameterSpace.default().max_atoms_gpu
    )
    if n_atoms > gpu_limit:
        raise ValueError(f"GPU backend limited to {gpu_limit} atoms.")

    dimension = 2**n_atoms
    dt = (duration_ns / 1000.0) / n_steps

    psi = torch.zeros(dimension, dtype=torch.complex128, device=dev)
    psi[0] = 1.0

    for step in range(n_steps):
        fraction = (step + 0.5) / n_steps

        if omega_shape == "blackman":
            omega_t = omega_max * (
                0.42 - 0.5 * np.cos(2 * np.pi * fraction) + 0.08 * np.cos(4 * np.pi * fraction)
            )
        elif omega_shape == "ramp":
            omega_t = omega_max * min(fraction * 2.0, 1.0)
        else:
            omega_t = omega_max

        delta_t = delta_start + (delta_end - delta_start) * fraction
        hamiltonian = build_gpu_hamiltonian(
            coords,
            omega_t,
            delta_t,
            c6=c6,
            device=dev,
            max_atoms_gpu=gpu_limit,
        )
        psi = _lanczos_expm_multiply(hamiltonian, psi, dt, krylov_dim=krylov_dim)

    psi_np = psi.cpu().numpy()

    from packages.simulation.observables import (
        antiferromagnetic_order,
        bitstring_probabilities,
        connected_correlation,
        entanglement_entropy,
        pair_correlation,
        rydberg_density,
        total_rydberg_fraction,
    )

    densities = rydberg_density(psi_np, n_atoms)
    return {
        "final_state": psi_np,
        "n_atoms": n_atoms,
        "rydberg_densities": densities.tolist(),
        "pair_correlations": pair_correlation(psi_np, n_atoms).tolist(),
        "connected_correlations": connected_correlation(psi_np, n_atoms).tolist(),
        "total_rydberg_fraction": total_rydberg_fraction(psi_np, n_atoms),
        "antiferromagnetic_order": antiferromagnetic_order(psi_np, n_atoms),
        "entanglement_entropy": entanglement_entropy(psi_np, n_atoms),
        "top_bitstrings": bitstring_probabilities(psi_np, n_atoms, top_k=5),
        "backend": f"gpu_{dev}",
        "n_steps": n_steps,
        "device": dev,
        "krylov_dim": krylov_dim,
    }
