"""Tests for surrogate V2, ensembles, and normalizer integration."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    from torch.utils.data import TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def test_surrogate_save_load_with_normalizer(tmp_path):
    from packages.ml.surrogate import SurrogateModel

    model = SurrogateModel(input_dim=10)
    means = np.random.randn(10).astype(np.float32)
    stds = np.abs(np.random.randn(10).astype(np.float32)) + 0.1
    model.set_normalizer(means, stds)
    path = tmp_path / "model.pt"
    model.save(path)

    model2 = SurrogateModel(input_dim=10)
    model2.load(path)
    assert model2._normalizer_means is not None
    np.testing.assert_array_almost_equal(model2._normalizer_means.numpy(), means)
    np.testing.assert_array_almost_equal(model2._normalizer_stds.numpy(), stds)


def test_surrogate_backward_compat_load(tmp_path):
    from packages.ml.surrogate import SurrogateModel

    model = SurrogateModel(input_dim=10)
    torch.save(model.state_dict(), tmp_path / "old.pt")

    model2 = SurrogateModel(input_dim=10)
    model2.load(tmp_path / "old.pt")
    assert model2._normalizer_means is None


def test_trainer_computes_normalizer():
    from packages.ml.surrogate import SurrogateModel, SurrogateTrainer

    x = torch.randn(100, 10)
    y = torch.rand(100, 4)
    dataset = TensorDataset(x, y)
    model = SurrogateModel(input_dim=10)
    trainer = SurrogateTrainer(model)
    trainer.fit(dataset, epochs=2)
    assert model._normalizer_means is not None
    assert model._normalizer_stds is not None


def test_residual_block_preserves_dim():
    from packages.ml.surrogate import ResidualBlock

    block = ResidualBlock(dim=128, dropout=0.1)
    x = torch.randn(16, 128)
    assert block(x).shape == x.shape


def test_surrogate_v2_forward():
    from packages.ml.surrogate import SurrogateModelV2

    model = SurrogateModelV2(input_dim=18, hidden=128, n_blocks=3)
    x = torch.randn(32, 18)
    y = model(x)
    assert y.shape == (32, 4)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_ensemble_uncertainty():
    from packages.ml.surrogate import SurrogateEnsemble

    ensemble = SurrogateEnsemble(n_models=3, input_dim=18, hidden=64, n_blocks=2)
    features = np.random.randn(10, 18).astype(np.float32)
    mean_pred, uncertainty = ensemble.predict_with_uncertainty(features)
    assert mean_pred.shape == (10, 4)
    assert uncertainty.shape == (10, 4)
    assert np.all(uncertainty >= 0.0)


def test_ensemble_save_load(tmp_path):
    from packages.ml.surrogate import SurrogateEnsemble

    ensemble = SurrogateEnsemble(n_models=3, input_dim=10, hidden=64, n_blocks=2)
    ensemble.save(tmp_path / "ensemble")

    ensemble2 = SurrogateEnsemble(n_models=3, input_dim=10, hidden=64, n_blocks=2)
    ensemble2.load(tmp_path / "ensemble")

    x = np.random.randn(5, 10).astype(np.float32)
    m1, _ = ensemble.predict_with_uncertainty(x)
    m2, _ = ensemble2.predict_with_uncertainty(x)
    np.testing.assert_array_almost_equal(m1, m2)


def test_ensemble_trainer_reports_stratified_kfold_calibration():
    from packages.ml.surrogate import EnsembleTrainer, SurrogateEnsemble

    x = torch.randn(24, 10)
    robustness = torch.linspace(0.1, 0.9, steps=24).unsqueeze(1)
    other_targets = torch.sigmoid(torch.randn(24, 3))
    y = torch.cat([robustness, other_targets], dim=1)
    dataset = TensorDataset(x, y)

    ensemble = SurrogateEnsemble(n_models=2, input_dim=10, hidden=32, n_blocks=1)
    trainer = EnsembleTrainer(
        ensemble,
        lr=5e-3,
        bootstrap=True,
        k_folds=3,
        stratify_bins=4,
        cv_seed=7,
    )
    histories = trainer.fit(dataset, epochs=1, batch_size=8)

    assert len(histories) == 2
    assert trainer.last_cv_report["enabled"] is True
    assert trainer.last_cv_report["k_folds"] == 3
    assert len(trainer.last_cv_report["folds"]) == 3
    assert "uncertainty_error_rank_corr" in trainer.last_cv_report
    assert "calibration_error" in trainer.last_cv_report
