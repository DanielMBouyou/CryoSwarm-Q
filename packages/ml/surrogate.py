"""Phase 1 - Surrogate models, trainers, and uncertainty-aware ensembles."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from packages.ml.dataset import INPUT_DIM, INPUT_DIM_V2, OUTPUT_DIM

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore[assignment]


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required: pip install 'cryoswarm-q[ml]'")


def _as_numpy_row(features: Any) -> NDArray[np.float32]:
    if TORCH_AVAILABLE and isinstance(features, torch.Tensor):
        return features.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(features, dtype=np.float32)


class _NormalizerMixin:
    _normalizer_means: torch.Tensor | None
    _normalizer_stds: torch.Tensor | None

    def set_normalizer(self, means: NDArray[np.float32], stds: NDArray[np.float32]) -> None:
        self._normalizer_means = torch.from_numpy(np.asarray(means, dtype=np.float32)).float()
        self._normalizer_stds = torch.from_numpy(np.asarray(stds, dtype=np.float32)).float()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._normalizer_means is None or self._normalizer_stds is None:
            return x
        means = self._normalizer_means.to(x.device)
        stds = self._normalizer_stds.to(x.device)
        return (x - means) / stds


class SurrogateModel(_NormalizerMixin, nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Baseline MLP surrogate for robustness score prediction."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden: int = 64,
    ) -> None:
        _check_torch()
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim),
            nn.Sigmoid(),
        )
        self._normalizer_means = None
        self._normalizer_stds = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._normalize(x.float()))

    def predict_numpy(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(features, dtype=np.float32))
            if x.ndim == 1:
                x = x.unsqueeze(0)
            preds = self.forward(x)
            return preds.cpu().numpy()

    def predict_robustness(self, features: NDArray[np.float32]) -> float:
        preds = self.predict_numpy(features)
        return float(preds[0, 0])

    def save(self, path: str | Path) -> None:
        state = {
            "model": self.state_dict(),
            "normalizer_means": self._normalizer_means,
            "normalizer_stds": self._normalizer_stds,
            "version": "v1",
            "config": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden": self.hidden,
            },
        }
        torch.save(state, str(path))

    def load(self, path: str | Path, strict: bool = True) -> None:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.load_state_dict(checkpoint["model"], strict=strict)
            self._normalizer_means = checkpoint.get("normalizer_means")
            self._normalizer_stds = checkpoint.get("normalizer_stds")
            return
        self.load_state_dict(checkpoint, strict=strict)
        self._normalizer_means = None
        self._normalizer_stds = None


class ResidualBlock(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Residual block with pre-activation LayerNorm."""

    def __init__(self, dim: int, dropout: float = 0.1):
        _check_torch()
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SurrogateModelV2(_NormalizerMixin, nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Residual surrogate model tuned for physics-informed feature sets."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM_V2,
        output_dim: int = OUTPUT_DIM,
        hidden: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        _check_torch()
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim),
            nn.Sigmoid(),
        )
        self._normalizer_means = None
        self._normalizer_stds = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x.float())
        x = self.input_proj(x)
        x = self.residual_blocks(x)
        return self.output_head(x)

    def predict_numpy(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(features, dtype=np.float32))
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return self.forward(x).cpu().numpy()

    def predict_robustness(self, features: NDArray[np.float32]) -> float:
        preds = self.predict_numpy(features)
        return float(preds[0, 0])

    def save(self, path: str | Path) -> None:
        state = {
            "model": self.state_dict(),
            "normalizer_means": self._normalizer_means,
            "normalizer_stds": self._normalizer_stds,
            "version": "v2",
            "config": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden": self.hidden,
                "n_blocks": self.n_blocks,
                "dropout": self.dropout,
            },
        }
        torch.save(state, str(path))

    def load(self, path: str | Path, strict: bool = True) -> None:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.load_state_dict(checkpoint["model"], strict=strict)
            self._normalizer_means = checkpoint.get("normalizer_means")
            self._normalizer_stds = checkpoint.get("normalizer_stds")
            return
        self.load_state_dict(checkpoint, strict=strict)
        self._normalizer_means = None
        self._normalizer_stds = None


class SurrogateEnsemble:
    """Deep-ensemble wrapper for uncertainty-aware surrogate prediction."""

    def __init__(
        self,
        n_models: int = 3,
        model_class: type = SurrogateModelV2,
        **model_kwargs: Any,
    ) -> None:
        self.n_models = n_models
        self.model_class = model_class
        self.model_kwargs = dict(model_kwargs)
        self.models = [model_class(**model_kwargs) for _ in range(n_models)]

    def predict_with_uncertainty(
        self,
        features: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        predictions = [model.predict_numpy(features) for model in self.models]
        stacked = np.stack(predictions, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def predict_robustness_with_uncertainty(
        self,
        features: NDArray[np.float32],
    ) -> tuple[float, float]:
        mean_preds, uncertainty = self.predict_with_uncertainty(features)
        return float(mean_preds[0, 0]), float(uncertainty[0, 0])

    def set_normalizer(self, means: NDArray[np.float32], stds: NDArray[np.float32]) -> None:
        for model in self.models:
            model.set_normalizer(means, stds)

    def save(self, directory: str | Path) -> None:
        import json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for index, model in enumerate(self.models):
            model.save(directory / f"member_{index}.pt")

        meta = {
            "n_models": self.n_models,
            "version": "ensemble_v1",
            "model_class": getattr(self.model_class, "__name__", str(self.model_class)),
            "model_kwargs": self.model_kwargs,
        }
        (directory / "ensemble_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        for index, model in enumerate(self.models):
            model.load(directory / f"member_{index}.pt")

    def eval(self) -> None:
        for model in self.models:
            model.eval()

    def train(self) -> None:
        for model in self.models:
            model.train()


class SurrogateTrainer:
    """Training loop for single surrogate models."""

    def __init__(
        self,
        model: Any,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        target_weights: list[float] | None = None,
    ) -> None:
        _check_torch()
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )
        weights = target_weights or [2.0, 1.0, 1.5, 1.0]
        self.target_weights = torch.tensor(weights, dtype=torch.float32)

    def _weighted_mse(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_per_target = ((preds - targets) ** 2).mean(dim=0)
        return (mse_per_target * self.target_weights.to(preds.device)).mean()

    def _attach_normalizer(self, train_dataset: Any) -> None:
        if not hasattr(self.model, "set_normalizer"):
            return
        if len(train_dataset) == 0:
            return

        features = [_as_numpy_row(train_dataset[index][0]) for index in range(len(train_dataset))]
        stacked = np.stack(features).astype(np.float32, copy=False)
        means = stacked.mean(axis=0).astype(np.float32, copy=False)
        stds = stacked.std(axis=0).astype(np.float32, copy=False)
        stds[stds < 1e-8] = 1.0
        self.model.set_normalizer(means, stds)

    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset | None = None,
        epochs: int = 100,
        batch_size: int = 64,
        log_dir: str | None = None,
    ) -> dict[str, list[float]]:
        writer = None
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(log_dir)
            except ImportError:
                pass

        self._attach_normalizer(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = None
        if val_dataset and len(val_dataset) > 0:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for features, targets in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(features.float())
                loss = self._weighted_mse(preds, targets.float())
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)

            avg_val = 0.0
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for features, targets in val_loader:
                        preds = self.model(features.float())
                        val_loss += float(self._weighted_mse(preds, targets.float()).item())
                        n_val += 1
                avg_val = val_loss / max(n_val, 1)
                history["val_loss"].append(avg_val)

            self.scheduler.step(avg_val if val_loader else avg_train)

            if writer:
                writer.add_scalar("loss/train", avg_train, epoch)
                if val_loader:
                    writer.add_scalar("loss/val", avg_val, epoch)

        if writer:
            writer.close()

        return history


class EnsembleTrainer:
    """Train all members of a surrogate ensemble independently."""

    def __init__(
        self,
        ensemble: SurrogateEnsemble,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        target_weights: list[float] | None = None,
        bootstrap: bool = False,
    ) -> None:
        _check_torch()
        self.ensemble = ensemble
        self.lr = lr
        self.weight_decay = weight_decay
        self.target_weights = target_weights
        self.bootstrap = bootstrap

    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset | None = None,
        epochs: int = 200,
        batch_size: int = 64,
        log_dir: str | None = None,
    ) -> list[dict[str, list[float]]]:
        histories: list[dict[str, list[float]]] = []
        for index, model in enumerate(self.ensemble.models):
            member_dataset = train_dataset
            if self.bootstrap:
                sample_count = len(train_dataset)
                indices = torch.randint(0, sample_count, (sample_count,))
                member_dataset = torch.utils.data.Subset(train_dataset, indices.tolist())

            trainer = SurrogateTrainer(
                model,
                lr=self.lr,
                weight_decay=self.weight_decay,
                target_weights=self.target_weights,
            )
            member_log_dir = f"{log_dir}/member_{index}" if log_dir else None
            history = trainer.fit(
                member_dataset,
                val_dataset,
                epochs=epochs,
                batch_size=batch_size,
                log_dir=member_log_dir,
            )
            histories.append(history)
        return histories
