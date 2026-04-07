from __future__ import annotations

import inspect

import numpy as np
import pytest

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from packages.ml.dataset import INPUT_DIM_V2
from packages.ml.surrogate import (
    DEFAULT_SURROGATE_DROPOUT,
    DEFAULT_SURROGATE_ENSEMBLE_MODELS,
    DEFAULT_SURROGATE_HIDDEN,
    DEFAULT_SURROGATE_LR,
    DEFAULT_SURROGATE_N_BLOCKS,
    DEFAULT_SURROGATE_SCHEDULER_FACTOR,
    DEFAULT_SURROGATE_SCHEDULER_PATIENCE,
    DEFAULT_SURROGATE_TARGET_WEIGHTS,
    DEFAULT_SURROGATE_TARGET_WEIGHT_SUM,
    DEFAULT_SURROGATE_WEIGHT_DECAY,
)

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def _make_multitask_dataset(
    *,
    seed: int = 0,
    n_samples: int = 160,
    noisy_aux_targets: bool = False,
) -> tuple[TensorDataset, TensorDataset]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    features = torch.rand(n_samples, INPUT_DIM_V2)
    robustness = (
        0.55 * features[:, 0]
        + 0.25 * features[:, 2]
        + 0.20 * (1.0 - features[:, 3])
    ).clamp(0.0, 1.0)

    if noisy_aux_targets:
        nominal = (1.0 - robustness).clamp(0.0, 1.0)
        worst_case = torch.rand(n_samples)
        observable = torch.rand(n_samples)
    else:
        nominal = (
            0.45 * features[:, 4]
            + 0.35 * features[:, 5]
            + 0.20 * features[:, 6]
        ).clamp(0.0, 1.0)
        worst_case = (
            0.40 * features[:, 1]
            + 0.30 * features[:, 7]
            + 0.30 * (1.0 - features[:, 8])
        ).clamp(0.0, 1.0)
        observable = (
            0.50 * features[:, 9]
            + 0.30 * features[:, 10]
            + 0.20 * features[:, 11]
        ).clamp(0.0, 1.0)

    targets = torch.stack([robustness, nominal, worst_case, observable], dim=1)
    dataset = TensorDataset(features, targets)
    train_dataset, val_dataset = random_split(
        dataset,
        [int(n_samples * 0.8), n_samples - int(n_samples * 0.8)],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_dataset, val_dataset


def _train_surrogate(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    *,
    seed: int,
    epochs: int,
    batch_size: int = 32,
    lr: float = DEFAULT_SURROGATE_LR,
    weight_decay: float = DEFAULT_SURROGATE_WEIGHT_DECAY,
    dropout: float = DEFAULT_SURROGATE_DROPOUT,
    target_weights: list[float] | None = None,
    hidden: int = DEFAULT_SURROGATE_HIDDEN,
    n_blocks: int = DEFAULT_SURROGATE_N_BLOCKS,
):
    from packages.ml.surrogate import SurrogateModelV2, SurrogateTrainer

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SurrogateModelV2(
        input_dim=INPUT_DIM_V2,
        hidden=hidden,
        n_blocks=n_blocks,
        dropout=dropout,
    )
    trainer = SurrogateTrainer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        target_weights=target_weights,
    )
    history = trainer.fit(train_dataset, val_dataset, epochs=epochs, batch_size=batch_size)
    return model, history


def _robustness_val_mse(model: torch.nn.Module, val_dataset: TensorDataset) -> float:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for features, batch_targets in DataLoader(val_dataset, batch_size=32):
            predictions.append(model(features.float())[:, 0])
            targets.append(batch_targets[:, 0])
    pred_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    return float(torch.mean((pred_tensor - target_tensor) ** 2).item())


def test_surrogate_defaults_are_explicitly_named() -> None:
    from packages.ml.surrogate import SurrogateEnsemble, SurrogateModelV2, SurrogateTrainer

    model_signature = inspect.signature(SurrogateModelV2.__init__)
    ensemble_signature = inspect.signature(SurrogateEnsemble.__init__)
    trainer_signature = inspect.signature(SurrogateTrainer.__init__)

    assert model_signature.parameters["hidden"].default == DEFAULT_SURROGATE_HIDDEN
    assert model_signature.parameters["n_blocks"].default == DEFAULT_SURROGATE_N_BLOCKS
    assert model_signature.parameters["dropout"].default == DEFAULT_SURROGATE_DROPOUT
    assert ensemble_signature.parameters["n_models"].default == DEFAULT_SURROGATE_ENSEMBLE_MODELS
    assert trainer_signature.parameters["lr"].default == DEFAULT_SURROGATE_LR
    assert trainer_signature.parameters["weight_decay"].default == DEFAULT_SURROGATE_WEIGHT_DECAY
    assert DEFAULT_SURROGATE_SCHEDULER_PATIENCE == 10
    assert DEFAULT_SURROGATE_SCHEDULER_FACTOR == pytest.approx(0.5)
    assert DEFAULT_SURROGATE_TARGET_WEIGHTS == pytest.approx((2.0, 1.0, 1.5, 1.0))
    assert DEFAULT_SURROGATE_TARGET_WEIGHT_SUM == pytest.approx(5.5)


def test_default_surrogate_hyperparameters_reduce_validation_loss() -> None:
    train_dataset, val_dataset = _make_multitask_dataset(seed=3)
    _, history = _train_surrogate(
        train_dataset,
        val_dataset,
        seed=3,
        epochs=15,
    )

    first_window = float(np.mean(history["val_loss"][:3]))
    last_window = float(np.mean(history["val_loss"][-3:]))

    assert history["val_loss"]
    assert last_window < first_window


@pytest.mark.parametrize(
    ("parameter_name", "values"),
    [
        ("lr", [3e-4, 1e-3, 3e-3]),
        ("weight_decay", [0.0, 1e-5, 1e-3]),
        ("dropout", [0.0, 0.1, 0.3]),
    ],
)
def test_surrogate_hyperparameter_sensitivity_stays_finite(
    parameter_name: str,
    values: list[float],
) -> None:
    train_dataset, val_dataset = _make_multitask_dataset(seed=5)

    for value in values:
        kwargs: dict[str, float] = {parameter_name: value}
        _, history = _train_surrogate(
            train_dataset,
            val_dataset,
            seed=5,
            epochs=6,
            **kwargs,
        )
        final_val_loss = history["val_loss"][-1]

        assert np.isfinite(final_val_loss)
        assert final_val_loss < 1.0


def test_default_target_weights_improve_robustness_prediction_over_uniform_weights() -> None:
    train_dataset, val_dataset = _make_multitask_dataset(
        seed=9,
        n_samples=128,
        noisy_aux_targets=True,
    )
    weighted_model, _ = _train_surrogate(
        train_dataset,
        val_dataset,
        seed=9,
        epochs=8,
        hidden=32,
        n_blocks=1,
        target_weights=list(DEFAULT_SURROGATE_TARGET_WEIGHTS),
    )
    uniform_model, _ = _train_surrogate(
        train_dataset,
        val_dataset,
        seed=9,
        epochs=8,
        hidden=32,
        n_blocks=1,
        target_weights=[1.0, 1.0, 1.0, 1.0],
    )

    weighted_mse = _robustness_val_mse(weighted_model, val_dataset)
    uniform_mse = _robustness_val_mse(uniform_model, val_dataset)

    assert weighted_mse < uniform_mse
