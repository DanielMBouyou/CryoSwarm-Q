from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class DatasetNormalizer:
    """Feature normalization with saved statistics for inference."""

    def __init__(self) -> None:
        self.means: NDArray[np.float32] | None = None
        self.stds: NDArray[np.float32] | None = None
        self.fitted = False

    def fit(self, features: NDArray[np.float32]) -> "DatasetNormalizer":
        self.means = features.mean(axis=0).astype(np.float32)
        self.stds = features.std(axis=0).astype(np.float32)
        self.stds[self.stds < 1e-8] = 1.0
        self.fitted = True
        return self

    def transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self.fitted or self.means is None or self.stds is None:
            raise RuntimeError("Call fit() before transform().")
        return ((features - self.means) / self.stds).astype(np.float32)

    def fit_transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.fit(features).transform(features)

    def inverse_transform(self, normalized: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self.fitted or self.means is None or self.stds is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return (normalized * self.stds + self.means).astype(np.float32)

    def save(self, path: str) -> None:
        if not self.fitted or self.means is None or self.stds is None:
            raise RuntimeError("Call fit() before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, means=self.means, stds=self.stds)

    def load(self, path: str) -> "DatasetNormalizer":
        data = np.load(path)
        self.means = data["means"].astype(np.float32)
        self.stds = data["stds"].astype(np.float32)
        self.fitted = True
        return self
