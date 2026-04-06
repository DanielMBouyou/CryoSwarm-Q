"""Tests for Phase 1 — Surrogate model and trainer."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    from torch.utils.data import TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from packages.ml.dataset import INPUT_DIM, OUTPUT_DIM

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.fixture
def dummy_dataset():
    """Small random dataset for testing."""
    n = 50
    X = torch.randn(n, INPUT_DIM)
    Y = torch.sigmoid(torch.randn(n, OUTPUT_DIM))
    return TensorDataset(X, Y)


@pytest.fixture
def train_val_split(dummy_dataset):
    n = len(dummy_dataset)
    n_val = 10
    train, val = torch.utils.data.random_split(dummy_dataset, [n - n_val, n_val])
    return train, val


class TestSurrogateModel:
    def test_forward_shape(self):
        from packages.ml.surrogate import SurrogateModel

        model = SurrogateModel()
        x = torch.randn(8, INPUT_DIM)
        y = model(x)
        assert y.shape == (8, OUTPUT_DIM)

    def test_output_range(self):
        from packages.ml.surrogate import SurrogateModel

        model = SurrogateModel()
        x = torch.randn(16, INPUT_DIM)
        y = model(x)
        assert (y >= 0.0).all()
        assert (y <= 1.0).all()

    def test_predict_numpy(self):
        from packages.ml.surrogate import SurrogateModel

        model = SurrogateModel()
        features = np.random.randn(5, INPUT_DIM).astype(np.float32)
        preds = model.predict_numpy(features)
        assert preds.shape == (5, OUTPUT_DIM)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_predict_single(self):
        from packages.ml.surrogate import SurrogateModel

        model = SurrogateModel()
        features = np.random.randn(INPUT_DIM).astype(np.float32)
        score = model.predict_robustness(features)
        assert 0.0 <= score <= 1.0

    def test_save_load(self, tmp_path):
        from packages.ml.surrogate import SurrogateModel

        model = SurrogateModel()
        path = tmp_path / "model.pt"
        model.save(path)

        model2 = SurrogateModel()
        model2.load(path)

        x = torch.randn(4, INPUT_DIM)
        torch.testing.assert_close(model(x), model2(x))


class TestSurrogateTrainer:
    def test_fit_runs(self, train_val_split):
        from packages.ml.surrogate import SurrogateModel, SurrogateTrainer

        model = SurrogateModel()
        trainer = SurrogateTrainer(model)
        train_ds, val_ds = train_val_split
        history = trainer.fit(train_ds, val_ds, epochs=5, batch_size=16)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 5
        assert all(loss > 0 for loss in history["train_loss"])

    def test_loss_decreases(self, train_val_split):
        from packages.ml.surrogate import SurrogateModel, SurrogateTrainer

        model = SurrogateModel()
        trainer = SurrogateTrainer(model, lr=5e-3)
        train_ds, _ = train_val_split
        history = trainer.fit(train_ds, epochs=30, batch_size=16)

        # Loss should generally decrease (allow some noise)
        first_5 = np.mean(history["train_loss"][:5])
        last_5 = np.mean(history["train_loss"][-5:])
        assert last_5 < first_5
