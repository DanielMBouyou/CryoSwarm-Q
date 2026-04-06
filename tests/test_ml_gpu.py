"""Tests for Phase 3 — GPU backend and training runner."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from packages.core.parameter_space import PhysicsParameterSpace
from packages.ml.gpu_backend import _n_op_diagonal, _sigma_x_elements, get_device

pytestmark = pytest.mark.gpu


# ---- device detection (always works) ----


class TestDeviceDetection:
    def test_returns_string(self):
        dev = get_device()
        assert isinstance(dev, str)
        assert dev in {"cpu", "cuda", "rocm", "mps"}


# ---- operator construction (CPU, needs PyTorch) ----


class TestOperatorConstruction:
    def test_sigma_x_symmetry(self):
        """σ_x elements should be symmetric (flip bit both ways)."""
        n = 3
        indices, values = _sigma_x_elements(n, 0)
        rows, cols = indices
        assert len(rows) == 2**n
        # Each row->col is a bit flip, so col->row is the reverse flip
        for r, c in zip(rows, cols):
            assert r ^ (1 << (n - 1)) == c

    def test_n_op_diagonal(self):
        """n_i projector should count excited states."""
        n = 2
        diag = _n_op_diagonal(n, 0)  # q0 = MSB
        # States: 00=0, 01=1, 10=2, 11=3
        # q0 excited: states where bit (1<<1) is set = 2, 3
        assert diag[0] == 0.0
        assert diag[1] == 0.0
        assert diag[2] == 1.0
        assert diag[3] == 1.0

    def test_n_op_qubit1(self):
        n = 2
        diag = _n_op_diagonal(n, 1)  # q1 = LSB
        assert diag[0] == 0.0
        assert diag[1] == 1.0
        assert diag[2] == 0.0
        assert diag[3] == 1.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestGPUHamiltonian:
    def test_build_2atoms(self):
        from packages.ml.gpu_backend import build_gpu_hamiltonian

        coords = [(0.0, 0.0), (7.0, 0.0)]
        H = build_gpu_hamiltonian(coords, omega=5.0, delta=-10.0, device="cpu")
        assert H.shape == (4, 4)
        # Hamiltonian should be Hermitian
        H_dense = H.to_dense()
        torch.testing.assert_close(H_dense, H_dense.T.conj(), atol=1e-10, rtol=1e-10)

    def test_build_3atoms(self):
        from packages.ml.gpu_backend import build_gpu_hamiltonian

        coords = [(0.0, 0.0), (7.0, 0.0), (3.5, 6.06)]
        H = build_gpu_hamiltonian(coords, omega=5.0, delta=-10.0, device="cpu")
        assert H.shape == (8, 8)

    def test_too_many_atoms(self):
        from packages.ml.gpu_backend import build_gpu_hamiltonian

        max_atoms_gpu = PhysicsParameterSpace.default().max_atoms_gpu
        coords = [(float(i), 0.0) for i in range(max_atoms_gpu + 1)]
        with pytest.raises(ValueError, match="limited"):
            build_gpu_hamiltonian(coords, omega=5.0, delta=-10.0, device="cpu")

    def test_matches_numpy_backend(self):
        """GPU Hamiltonian diagonal should match CPU for a simple case."""
        from packages.ml.gpu_backend import build_gpu_hamiltonian
        from packages.simulation.hamiltonian import build_hamiltonian_matrix

        coords = [(0.0, 0.0), (7.0, 0.0)]
        omega, delta = 5.0, -10.0

        H_gpu = build_gpu_hamiltonian(coords, omega, delta, device="cpu").to_dense()
        H_cpu = build_hamiltonian_matrix(coords, omega, delta)

        H_gpu_np = H_gpu.numpy().real
        np.testing.assert_array_almost_equal(
            np.diag(H_gpu_np), np.diag(H_cpu), decimal=4,
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestLanczos:
    """Tests for the Lanczos time-evolution kernel."""

    def test_identity_for_zero_dt(self) -> None:
        """Lanczos with dt=0 should return the original state."""
        from packages.ml.gpu_backend import _lanczos_expm_multiply, build_gpu_hamiltonian

        coords = [(0.0, 0.0), (7.0, 0.0)]
        hamiltonian = build_gpu_hamiltonian(coords, omega=5.0, delta=-10.0, device="cpu")
        dim = hamiltonian.shape[0]
        psi = torch.zeros(dim, dtype=torch.complex128)
        psi[0] = 1.0

        psi_out = _lanczos_expm_multiply(hamiltonian, psi, dt=0.0, krylov_dim=10)
        np.testing.assert_allclose(psi_out.numpy(), psi.numpy(), atol=1e-12)

    def test_unitarity_preserved(self) -> None:
        """The evolved state should remain unit-normalised."""
        from packages.ml.gpu_backend import _lanczos_expm_multiply, build_gpu_hamiltonian

        coords = [(0.0, 0.0), (7.0, 0.0), (3.5, 6.06)]
        hamiltonian = build_gpu_hamiltonian(coords, omega=5.0, delta=-10.0, device="cpu")
        dim = hamiltonian.shape[0]
        psi = torch.zeros(dim, dtype=torch.complex128)
        psi[0] = 1.0

        for _ in range(20):
            psi = _lanczos_expm_multiply(hamiltonian, psi, dt=0.01, krylov_dim=15)

        assert abs(psi.norm().item() - 1.0) < 1e-10

    def test_matches_scipy_expm(self) -> None:
        """Lanczos evolution should match dense scipy expm on small systems."""
        pytest.importorskip("scipy")
        from scipy.linalg import expm as scipy_expm

        from packages.ml.gpu_backend import _lanczos_expm_multiply, build_gpu_hamiltonian
        from packages.simulation.hamiltonian import build_hamiltonian_matrix

        coords = [(0.0, 0.0), (7.0, 0.0)]
        omega = 5.0
        delta = -10.0
        dt = 0.005

        H_cpu = build_hamiltonian_matrix(coords, omega, delta)
        dim = H_cpu.shape[0]
        psi_np = np.zeros(dim, dtype=np.complex128)
        psi_np[0] = 1.0
        psi_exact = scipy_expm(-1j * H_cpu * dt) @ psi_np
        psi_exact /= np.linalg.norm(psi_exact)

        H_torch = build_gpu_hamiltonian(coords, omega, delta, device="cpu")
        psi_torch = torch.zeros(dim, dtype=torch.complex128)
        psi_torch[0] = 1.0
        psi_lanczos = _lanczos_expm_multiply(H_torch, psi_torch, dt, krylov_dim=15)

        np.testing.assert_allclose(psi_lanczos.numpy(), psi_exact, atol=1e-8)

    def test_convergence_with_krylov_dim(self) -> None:
        """Higher krylov_dim should not perform worse than a tiny Krylov basis."""
        pytest.importorskip("scipy")
        from scipy.linalg import expm as scipy_expm

        from packages.ml.gpu_backend import _lanczos_expm_multiply, build_gpu_hamiltonian
        from packages.simulation.hamiltonian import build_hamiltonian_matrix

        coords = [(0.0, 0.0), (7.0, 0.0), (3.5, 6.06)]
        omega = 5.0
        delta = -10.0
        dt = 0.01

        H_cpu = build_hamiltonian_matrix(coords, omega, delta)
        dim = H_cpu.shape[0]
        psi_np = np.zeros(dim, dtype=np.complex128)
        psi_np[0] = 1.0
        psi_exact = scipy_expm(-1j * H_cpu * dt) @ psi_np
        psi_exact /= np.linalg.norm(psi_exact)

        H_torch = build_gpu_hamiltonian(coords, omega, delta, device="cpu")
        psi_torch = torch.zeros(dim, dtype=torch.complex128)
        psi_torch[0] = 1.0

        errors: list[float] = []
        for krylov_dim in [5, 8, 15]:
            psi_m = _lanczos_expm_multiply(H_torch, psi_torch, dt, krylov_dim=krylov_dim)
            errors.append(float(np.linalg.norm(psi_m.numpy() - psi_exact)))

        assert errors[-1] <= errors[0] + 1e-12


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestGPUTimeEvolution:
    def test_small_system(self):
        from packages.ml.gpu_backend import gpu_time_evolution

        coords = [(0.0, 0.0), (7.0, 0.0)]
        result = gpu_time_evolution(
            coords,
            omega_max=5.0,
            delta_start=-10.0,
            delta_end=5.0,
            duration_ns=500,
            n_steps=50,
            device="cpu",
        )
        assert result["n_atoms"] == 2
        assert len(result["rydberg_densities"]) == 2
        psi = result["final_state"]
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-6
        assert result["krylov_dim"] == 20

    def test_blackman_shape(self):
        from packages.ml.gpu_backend import gpu_time_evolution

        coords = [(0.0, 0.0), (7.0, 0.0)]
        result = gpu_time_evolution(
            coords,
            omega_max=5.0,
            delta_start=-10.0,
            delta_end=5.0,
            duration_ns=500,
            n_steps=50,
            omega_shape="blackman",
            device="cpu",
        )
        assert abs(np.linalg.norm(result["final_state"]) - 1.0) < 1e-6


# ---- training runner tests ----


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTrainingRunner:
    def test_surrogate_run(self, tmp_path):
        from packages.ml.dataset import INPUT_DIM, OUTPUT_DIM
        from packages.ml.training_runner import TrainingConfig, TrainingRunner

        X = torch.randn(30, INPUT_DIM)
        Y = torch.sigmoid(torch.randn(30, OUTPUT_DIM))
        dataset = torch.utils.data.TensorDataset(X, Y)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [24, 6])

        config = TrainingConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            surrogate_epochs=3,
        )
        runner = TrainingRunner(config)
        result = runner.run_surrogate(train_ds, val_ds)

        assert "history" in result
        assert (tmp_path / "ckpt" / "surrogate_latest.pt").exists()
