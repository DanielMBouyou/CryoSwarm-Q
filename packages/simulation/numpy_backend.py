"""NumPy / SciPy exact simulation backend for small neutral-atom systems.

Uses a symmetric second-order Strang splitting between the transverse drive
term and the diagonal detuning / interaction term. Pulse envelopes are carried
by a reusable ``PulseSchedule`` abstraction rather than hard-coded per-shape
branching in the time-evolution loop.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from packages.core.parameter_space import PhysicsParameterSpace
from packages.simulation.hamiltonian import C6_RB87_70S, blockade_radius, pairwise_distances
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
    from scipy.interpolate import interp1d
    from scipy.linalg import expm
    from scipy.sparse import csc_matrix, diags, eye, kron
    from scipy.sparse.linalg import eigsh, expm_multiply, norm as sparse_norm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    interp1d = None  # type: ignore[assignment]
    csc_matrix = None  # type: ignore[assignment]
    diags = None  # type: ignore[assignment]
    eye = None  # type: ignore[assignment]
    kron = None  # type: ignore[assignment]
    eigsh = None  # type: ignore[assignment]
    sparse_norm = None  # type: ignore[assignment]


_DEFAULT_PARAM_SPACE = PhysicsParameterSpace.default()
_ADIABATIC_WARNING_THRESHOLD = 0.1
_GAP_EPS = 1e-10


@dataclass(slots=True)
class PulseSchedule:
    """Reusable time-dependent pulse schedule for ``omega(t)`` and ``delta(t)``.

    The schedule accepts either explicit control points or arbitrary callables.
    Control-point interpolation is linear by default and can use higher-order
    SciPy interpolation when available.
    """

    duration_us: float
    omega_times_us: NDArray[np.float64] | None = None
    omega_values: NDArray[np.float64] | None = None
    delta_times_us: NDArray[np.float64] | None = None
    delta_values: NDArray[np.float64] | None = None
    interpolation: str = "linear"
    omega_function: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
    delta_function: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None

    def __post_init__(self) -> None:
        if self.duration_us <= 0.0:
            raise ValueError("PulseSchedule.duration_us must be strictly positive.")
        self.omega_times_us = self._as_array(self.omega_times_us)
        self.omega_values = self._as_array(self.omega_values)
        self.delta_times_us = self._as_array(self.delta_times_us)
        self.delta_values = self._as_array(self.delta_values)
        self._validate_component(self.omega_times_us, self.omega_values, "omega")
        self._validate_component(self.delta_times_us, self.delta_values, "delta")

    @classmethod
    def from_legacy(
        cls,
        omega_max: float,
        delta_start: float,
        delta_end: float,
        duration_ns: float,
        omega_shape: str = "constant",
        interpolation: str = "linear",
    ) -> "PulseSchedule":
        """Build a schedule from the legacy sweep arguments."""

        duration_us = duration_ns / 1000.0
        delta_times = np.array([0.0, duration_us], dtype=np.float64)
        delta_values = np.array([delta_start, delta_end], dtype=np.float64)

        if omega_shape == "ramp":
            omega_times = np.array([0.0, duration_us * 0.5, duration_us], dtype=np.float64)
            omega_values = np.array([0.0, omega_max, omega_max], dtype=np.float64)
        elif omega_shape == "blackman":
            fractions = np.linspace(0.0, 1.0, 33, dtype=np.float64)
            omega_times = duration_us * fractions
            omega_values = omega_max * (
                0.42
                - 0.5 * np.cos(2.0 * np.pi * fractions)
                + 0.08 * np.cos(4.0 * np.pi * fractions)
            )
        else:
            omega_times = np.array([0.0, duration_us], dtype=np.float64)
            omega_values = np.array([omega_max, omega_max], dtype=np.float64)

        return cls(
            duration_us=duration_us,
            omega_times_us=omega_times,
            omega_values=omega_values,
            delta_times_us=delta_times,
            delta_values=delta_values,
            interpolation=interpolation,
        )

    @staticmethod
    def _as_array(values: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
        if values is None:
            return None
        return np.asarray(values, dtype=np.float64)

    def _validate_component(
        self,
        times: NDArray[np.float64] | None,
        values: NDArray[np.float64] | None,
        name: str,
    ) -> None:
        if times is None and values is None:
            return
        if times is None or values is None:
            raise ValueError(f"PulseSchedule requires both {name}_times_us and {name}_values.")
        if times.ndim != 1 or values.ndim != 1 or len(times) != len(values):
            raise ValueError(f"PulseSchedule {name} control points must be matching 1-D arrays.")
        if len(times) == 0:
            raise ValueError(f"PulseSchedule {name} control point arrays cannot be empty.")
        if times[0] < 0.0 or times[-1] > self.duration_us:
            raise ValueError(f"PulseSchedule {name} control times must lie inside [0, duration_us].")
        if np.any(np.diff(times) < 0.0):
            raise ValueError(f"PulseSchedule {name} control times must be sorted ascending.")

    def _interpolate(
        self,
        times_us: NDArray[np.float64],
        control_times: NDArray[np.float64],
        control_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if len(control_times) == 1:
            return np.full_like(times_us, control_values[0], dtype=np.float64)
        if self.interpolation == "linear":
            return np.interp(times_us, control_times, control_values).astype(np.float64)
        if not SCIPY_AVAILABLE or interp1d is None:
            raise RuntimeError(
                f"Interpolation mode '{self.interpolation}' requires scipy.interpolate."
            )
        interpolator = interp1d(
            control_times,
            control_values,
            kind=self.interpolation,
            bounds_error=False,
            fill_value=(control_values[0], control_values[-1]),
            assume_sorted=True,
        )
        return np.asarray(interpolator(times_us), dtype=np.float64)

    def _evaluate_component(
        self,
        times_us: NDArray[np.float64],
        control_times: NDArray[np.float64] | None,
        control_values: NDArray[np.float64] | None,
        sampler: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
        name: str,
    ) -> NDArray[np.float64]:
        clipped_times = np.clip(np.asarray(times_us, dtype=np.float64), 0.0, self.duration_us)
        if sampler is not None:
            return np.asarray(sampler(clipped_times), dtype=np.float64)
        if control_times is None or control_values is None:
            raise ValueError(f"PulseSchedule has no sampler configured for {name}.")
        return self._interpolate(clipped_times, control_times, control_values)

    def omega_at(self, times_us: float | NDArray[np.float64]) -> NDArray[np.float64]:
        sample_times = np.atleast_1d(np.asarray(times_us, dtype=np.float64))
        return self._evaluate_component(
            sample_times,
            self.omega_times_us,
            self.omega_values,
            self.omega_function,
            "omega",
        )

    def delta_at(self, times_us: float | NDArray[np.float64]) -> NDArray[np.float64]:
        sample_times = np.atleast_1d(np.asarray(times_us, dtype=np.float64))
        return self._evaluate_component(
            sample_times,
            self.delta_times_us,
            self.delta_values,
            self.delta_function,
            "delta",
        )

    def sample_midpoints(
        self,
        n_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if n_steps <= 0:
            raise ValueError("n_steps must be strictly positive.")
        dt = self.duration_us / n_steps
        times_us = (np.arange(n_steps, dtype=np.float64) + 0.5) * dt
        return times_us, self.omega_at(times_us), self.delta_at(times_us)


@dataclass(slots=True)
class _SimulationInputs:
    times_us: NDArray[np.float64]
    omega_profile: NDArray[np.float64]
    delta_profile: NDArray[np.float64]
    dt_us: float
    dense: bool
    drive_dense: NDArray[np.complex128] | None
    drive_sparse: Any
    occupation_sum: NDArray[np.float64]
    interaction_diag: NDArray[np.float64]
    schedule: PulseSchedule


def _build_dense_drive_operator(n_atoms: int) -> NDArray[np.complex128]:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    eye2 = np.eye(2, dtype=np.complex128)
    drive = np.zeros((2**n_atoms, 2**n_atoms), dtype=np.complex128)
    for site in range(n_atoms):
        operator = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        for idx in range(n_atoms):
            operator = np.kron(operator, sigma_x if idx == site else eye2)
        drive += 0.5 * operator
    return drive


def _build_sparse_drive_operator(n_atoms: int):
    if not SCIPY_AVAILABLE or csc_matrix is None or eye is None or kron is None:
        raise RuntimeError("scipy is required for sparse Strang splitting.")
    sigma_x = csc_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128))
    eye2 = eye(2, format="csc", dtype=np.complex128)
    drive = csc_matrix((2**n_atoms, 2**n_atoms), dtype=np.complex128)
    for site in range(n_atoms):
        operator = csc_matrix([[1.0 + 0.0j]])
        for idx in range(n_atoms):
            operator = kron(operator, sigma_x if idx == site else eye2, format="csc")
        drive += 0.5 * operator
    return drive.tocsc()


def _build_diagonal_components(
    coords: list[tuple[float, float]],
    c6: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_atoms = len(coords)
    dim = 2**n_atoms
    basis = np.arange(dim, dtype=np.uint64)
    occupation_sum = np.zeros(dim, dtype=np.float64)
    interaction_diag = np.zeros(dim, dtype=np.float64)
    dists = pairwise_distances(coords)
    occupancies: list[NDArray[np.float64]] = []

    for site in range(n_atoms):
        occ = ((basis >> (n_atoms - 1 - site)) & 1).astype(np.float64)
        occupancies.append(occ)
        occupation_sum += occ

    for row_index in range(n_atoms):
        for col_index in range(row_index + 1, n_atoms):
            distance = dists[row_index, col_index]
            if distance <= 0.0:
                continue
            interaction = c6 / distance**6
            if interaction <= 1e-10:
                continue
            interaction_diag += interaction * occupancies[row_index] * occupancies[col_index]

    return occupation_sum, interaction_diag


def _diagonal_energy(
    occupation_sum: NDArray[np.float64],
    interaction_diag: NDArray[np.float64],
    delta: float,
) -> NDArray[np.float64]:
    return -delta * occupation_sum + interaction_diag


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _blockade_radius_or_none(omega: float, c6: float) -> float | None:
    if omega <= 0.0:
        return None
    return blockade_radius(omega, c6)


def _prepare_simulation_inputs(
    coords: list[tuple[float, float]],
    n_steps: int,
    c6: float,
    max_atoms_dense: int | None,
    max_atoms_sparse: int | None,
    schedule: PulseSchedule | None,
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    omega_shape: str,
) -> _SimulationInputs:
    dense_limit = max_atoms_dense if max_atoms_dense is not None else _DEFAULT_PARAM_SPACE.max_atoms_dense
    sparse_limit = max_atoms_sparse if max_atoms_sparse is not None else _DEFAULT_PARAM_SPACE.max_atoms_sparse
    n_atoms = len(coords)
    if n_atoms > sparse_limit:
        raise ValueError(
            f"NumPy backend limited to {sparse_limit} atoms "
            f"(requested {n_atoms}, dim={2**n_atoms})."
        )
    pulse_schedule = schedule or PulseSchedule.from_legacy(
        omega_max=omega_max,
        delta_start=delta_start,
        delta_end=delta_end,
        duration_ns=duration_ns,
        omega_shape=omega_shape,
    )
    times_us, omega_profile, delta_profile = pulse_schedule.sample_midpoints(n_steps)
    occupation_sum, interaction_diag = _build_diagonal_components(coords, c6)
    dense = n_atoms <= dense_limit
    drive_dense = _build_dense_drive_operator(n_atoms) if dense else None
    drive_sparse = _build_sparse_drive_operator(n_atoms) if not dense else None
    return _SimulationInputs(
        times_us=times_us,
        omega_profile=omega_profile,
        delta_profile=delta_profile,
        dt_us=pulse_schedule.duration_us / n_steps,
        dense=dense,
        drive_dense=drive_dense,
        drive_sparse=drive_sparse,
        occupation_sum=occupation_sum,
        interaction_diag=interaction_diag,
        schedule=pulse_schedule,
    )


def compute_schedule_diagnostics(
    coords: list[tuple[float, float]],
    *,
    n_steps: int,
    schedule: PulseSchedule | None = None,
    omega_max: float,
    delta_start: float,
    delta_end: float,
    duration_ns: float,
    omega_shape: str = "constant",
    c6: float = C6_RB87_70S,
    max_atoms_dense: int | None = None,
    max_atoms_sparse: int | None = None,
) -> dict[str, Any]:
    """Compute adiabaticity and dynamic-blockade diagnostics on the simulation grid."""

    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for schedule diagnostics.")

    inputs = _prepare_simulation_inputs(
        coords=coords,
        n_steps=n_steps,
        c6=c6,
        max_atoms_dense=max_atoms_dense,
        max_atoms_sparse=max_atoms_sparse,
        schedule=schedule,
        omega_max=omega_max,
        delta_start=delta_start,
        delta_end=delta_end,
        duration_ns=duration_ns,
        omega_shape=omega_shape,
    )

    if len(inputs.times_us) == 1:
        domega_dt = np.zeros(1, dtype=np.float64)
        ddelta_dt = np.zeros(1, dtype=np.float64)
    else:
        domega_dt = np.gradient(inputs.omega_profile, inputs.times_us)
        ddelta_dt = np.gradient(inputs.delta_profile, inputs.times_us)

    adiabatic_ratios: list[float | None] = []
    gaps: list[float] = []
    undefined_steps = 0
    max_hdot_norm = 0.0

    for step, (omega_t, delta_t) in enumerate(zip(inputs.omega_profile, inputs.delta_profile, strict=True)):
        diagonal_energy = _diagonal_energy(inputs.occupation_sum, inputs.interaction_diag, delta_t)
        d_diagonal = (-ddelta_dt[step] * inputs.occupation_sum).astype(np.complex128)

        if inputs.dense:
            assert inputs.drive_dense is not None
            hamiltonian = omega_t * inputs.drive_dense + np.diag(diagonal_energy.astype(np.complex128))
            eigvals = np.sort(np.linalg.eigvalsh(hamiltonian).real)
            dh_dt = domega_dt[step] * inputs.drive_dense + np.diag(d_diagonal)
            dh_norm = float(np.linalg.norm(dh_dt))
        else:
            assert inputs.drive_sparse is not None
            assert diags is not None and eigsh is not None and sparse_norm is not None
            hamiltonian = (
                omega_t * inputs.drive_sparse
                + diags(diagonal_energy.astype(np.complex128), format="csc")
            ).tocsc()
            dim = hamiltonian.shape[0]
            if dim <= 2:
                eigvals = np.sort(np.linalg.eigvalsh(hamiltonian.toarray()).real)
            else:
                eigvals = np.sort(
                    eigsh(hamiltonian, k=min(2, dim - 1), which="SA", return_eigenvectors=False).real
                )
            dh_dt = (
                domega_dt[step] * inputs.drive_sparse
                + diags(d_diagonal, format="csc")
            ).tocsc()
            dh_norm = float(sparse_norm(dh_dt))

        max_hdot_norm = max(max_hdot_norm, dh_norm)
        gap = float(eigvals[1] - eigvals[0]) if len(eigvals) >= 2 else 0.0
        gaps.append(gap)
        if gap <= _GAP_EPS:
            adiabatic_ratios.append(None)
            undefined_steps += 1
            continue
        adiabatic_ratios.append(float(dh_norm / (gap**2)))

    finite_ratios = [value for value in adiabatic_ratios if value is not None]
    finite_gaps = [gap for gap in gaps if gap > _GAP_EPS]
    blockade_profile = [
        _blockade_radius_or_none(float(omega_t), c6)
        for omega_t in inputs.omega_profile
    ]
    finite_blockade = [radius for radius in blockade_profile if radius is not None]

    return {
        "pulse_schedule": {
            "times_us": [round(float(value), 8) for value in inputs.times_us],
            "omega_profile": [round(float(value), 8) for value in inputs.omega_profile],
            "delta_profile": [round(float(value), 8) for value in inputs.delta_profile],
            "interpolation": inputs.schedule.interpolation,
        },
        "adiabatic_ratios": [_round_or_none(value, 8) for value in adiabatic_ratios],
        "adiabatic_ratio_max": _round_or_none(max(finite_ratios) if finite_ratios else None, 8),
        "adiabatic_gap_min": _round_or_none(min(finite_gaps) if finite_gaps else None, 8),
        "adiabatic_warning": bool(
            undefined_steps > 0
            or any(value is not None and value > _ADIABATIC_WARNING_THRESHOLD for value in adiabatic_ratios)
        ),
        "adiabatic_undefined_steps": undefined_steps,
        "blockade_radius_profile_um": [_round_or_none(value, 6) for value in blockade_profile],
        "blockade_radius_min_um": _round_or_none(min(finite_blockade) if finite_blockade else None, 6),
        "blockade_radius_max_um": _round_or_none(max(finite_blockade) if finite_blockade else None, 6),
        "blockade_radius_undefined_steps": len(blockade_profile) - len(finite_blockade),
        "max_hdot_norm": round(max_hdot_norm, 8),
        "integration_order": 2,
    }


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
    schedule: PulseSchedule | None = None,
    compute_diagnostics: bool = True,
) -> dict[str, Any]:
    """Time-evolve the Rydberg Hamiltonian with a detuning sweep.

    The evolution uses a second-order symmetric Strang splitting:
    ``exp(-i H dt) ~= exp(-i H_drive dt/2) exp(-i H_diag dt) exp(-i H_drive dt/2)``,
    with ``H_drive = (Omega/2) sum sigma_x`` and
    ``H_diag = -delta sum n_i + sum U_ij n_i n_j``.
    """

    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for the numpy backend (pip install scipy).")

    inputs = _prepare_simulation_inputs(
        coords=coords,
        n_steps=n_steps,
        c6=c6,
        max_atoms_dense=max_atoms_dense,
        max_atoms_sparse=max_atoms_sparse,
        schedule=schedule,
        omega_max=omega_max,
        delta_start=delta_start,
        delta_end=delta_end,
        duration_ns=duration_ns,
        omega_shape=omega_shape,
    )

    n_atoms = len(coords)
    dim = 2**n_atoms
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    backend_label = "numpy_exact" if inputs.dense else "scipy_sparse"
    half_dt = inputs.dt_us / 2.0

    for omega_t, delta_t in zip(inputs.omega_profile, inputs.delta_profile, strict=True):
        diagonal_energy = _diagonal_energy(inputs.occupation_sum, inputs.interaction_diag, float(delta_t))
        diagonal_phase = np.exp(-1j * diagonal_energy * inputs.dt_us)

        if inputs.dense:
            assert inputs.drive_dense is not None
            if abs(omega_t) > 0.0:
                drive_half = expm(-1j * (omega_t * inputs.drive_dense) * half_dt)
                psi = drive_half @ psi
            psi *= diagonal_phase
            if abs(omega_t) > 0.0:
                psi = drive_half @ psi
        else:
            assert inputs.drive_sparse is not None
            if abs(omega_t) > 0.0:
                psi = expm_multiply((-1j * omega_t * inputs.drive_sparse * half_dt), psi)
            psi *= diagonal_phase
            if abs(omega_t) > 0.0:
                psi = expm_multiply((-1j * omega_t * inputs.drive_sparse * half_dt), psi)

    psi /= np.linalg.norm(psi)
    dens = rydberg_density(psi, n_atoms)

    result: dict[str, Any] = {
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
        "dt_us": inputs.dt_us,
        "integration_order": 2,
    }

    if compute_diagnostics:
        result.update(
            compute_schedule_diagnostics(
                coords=coords,
                n_steps=n_steps,
                schedule=inputs.schedule,
                omega_max=omega_max,
                delta_start=delta_start,
                delta_end=delta_end,
                duration_ns=duration_ns,
                omega_shape=omega_shape,
                c6=c6,
                max_atoms_dense=max_atoms_dense,
                max_atoms_sparse=max_atoms_sparse,
            )
        )

    return result


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
    schedule: PulseSchedule | None = None,
) -> dict[str, float | int | bool | None]:
    """Estimate time-discretization error for the piecewise-constant evolution."""

    pulse_schedule = schedule or PulseSchedule.from_legacy(
        omega_max=omega_max,
        delta_start=delta_start,
        delta_end=delta_end,
        duration_ns=duration_ns,
        omega_shape=omega_shape,
    )
    diagnostics = compute_schedule_diagnostics(
        coords=coords,
        n_steps=n_steps,
        schedule=pulse_schedule,
        omega_max=omega_max,
        delta_start=delta_start,
        delta_end=delta_end,
        duration_ns=duration_ns,
        omega_shape=omega_shape,
        c6=c6,
        max_atoms_dense=max_atoms_dense,
        max_atoms_sparse=max_atoms_sparse,
    )
    analytical_bound = (pulse_schedule.duration_us**2 / (2.0 * n_steps)) * float(diagnostics["max_hdot_norm"])

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
        schedule=pulse_schedule,
        compute_diagnostics=False,
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
        schedule=pulse_schedule,
        compute_diagnostics=False,
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
        "adiabatic_ratio_max": diagnostics["adiabatic_ratio_max"],
        "adiabatic_warning": bool(diagnostics["adiabatic_warning"]),
    }
