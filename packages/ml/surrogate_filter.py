"""Phase 1 - Surrogate-based pre-filtering for sequence candidates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from packages.core.logging import get_logger
from packages.core.models import RegisterCandidate, SequenceCandidate
from packages.ml.dataset import (
    INPUT_DIM_V2,
    build_feature_vector,
    build_feature_vector_v2,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger(__name__)


class SurrogateFilter:
    """Pre-filter sequence candidates using a single model or an ensemble."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        top_k: int = 20,
        min_score: float = 0.1,
        max_uncertainty: float = 0.15,
        enabled: bool = False,
        use_ensemble: bool = False,
    ) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.max_uncertainty = max_uncertainty
        self.enabled = enabled
        self.use_ensemble = use_ensemble
        self._model = None
        self._ensemble = None
        self._feature_builder: Callable[[RegisterCandidate, SequenceCandidate], np.ndarray] = build_feature_vector

        if enabled and model_path and TORCH_AVAILABLE:
            path = Path(model_path)
            if use_ensemble:
                self._load_ensemble(path)
            else:
                self._load_model(path)
        elif enabled and not TORCH_AVAILABLE:
            logger.warning("SurrogateFilter enabled but PyTorch not installed - passthrough mode.")
            self.enabled = False

    def _configure_feature_builder(self, input_dim: int) -> None:
        self._feature_builder = build_feature_vector_v2 if input_dim >= INPUT_DIM_V2 else build_feature_vector

    def _load_model(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Surrogate checkpoint not found at %s - passthrough mode.", path)
            self.enabled = False
            return

        from packages.ml.surrogate import SurrogateModel, SurrogateModelV2

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        version = checkpoint.get("version") if isinstance(checkpoint, dict) else None
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

        if version == "v2":
            input_dim = int(config.get("input_dim", INPUT_DIM_V2))
            hidden = int(config.get("hidden", 128))
            n_blocks = int(config.get("n_blocks", 3))
            dropout = float(config.get("dropout", 0.1))
            output_dim = int(config.get("output_dim", 4))
            self._model = SurrogateModelV2(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden=hidden,
                n_blocks=n_blocks,
                dropout=dropout,
            )
        else:
            if isinstance(checkpoint, dict) and "model" not in checkpoint:
                input_dim = int(checkpoint["net.0.weight"].shape[1])
                hidden = int(checkpoint["net.0.weight"].shape[0])
            else:
                input_dim = int(config.get("input_dim", 10))
                hidden = int(config.get("hidden", 64))
            output_dim = int(config.get("output_dim", 4))
            self._model = SurrogateModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden=hidden,
            )

        self._model.load(path)
        self._model.eval()
        self._configure_feature_builder(input_dim)
        logger.info("Surrogate model loaded from %s (input_dim=%d)", path, input_dim)

    def _load_ensemble(self, directory: Path) -> None:
        if not directory.is_dir():
            logger.warning("Ensemble dir not found: %s - passthrough mode.", directory)
            self.enabled = False
            return

        from packages.ml.surrogate import SurrogateEnsemble, SurrogateModel, SurrogateModelV2

        meta_path = directory / "ensemble_meta.json"
        if not meta_path.exists():
            logger.warning("Ensemble metadata not found at %s - passthrough mode.", meta_path)
            self.enabled = False
            return

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_kwargs = dict(meta.get("model_kwargs", {}))
        model_class_name = meta.get("model_class", "SurrogateModelV2")
        model_class = SurrogateModelV2 if model_class_name == "SurrogateModelV2" else SurrogateModel
        self._ensemble = SurrogateEnsemble(
            n_models=int(meta.get("n_models", 3)),
            model_class=model_class,
            **model_kwargs,
        )
        self._ensemble.load(directory)
        self._ensemble.eval()
        input_dim = int(model_kwargs.get("input_dim", INPUT_DIM_V2 if model_class is SurrogateModelV2 else 10))
        self._configure_feature_builder(input_dim)
        logger.info(
            "Surrogate ensemble loaded from %s (%d members, input_dim=%d)",
            directory,
            self._ensemble.n_models,
            input_dim,
        )

    def _extract_batch_features(
        self,
        sequences: list[SequenceCandidate],
        register_lookup: dict[str, RegisterCandidate],
    ) -> tuple[np.ndarray | None, list[int]]:
        features_list: list[np.ndarray] = []
        valid_indices: list[int] = []
        for index, sequence in enumerate(sequences):
            register = register_lookup.get(sequence.register_candidate_id)
            if register is None:
                continue
            features_list.append(self._feature_builder(register, sequence))
            valid_indices.append(index)
        if not features_list:
            return None, valid_indices
        return np.stack(features_list), valid_indices

    def _filter_with_ensemble(
        self,
        sequences: list[SequenceCandidate],
        register_lookup: dict[str, RegisterCandidate],
    ) -> list[SequenceCandidate]:
        if self._ensemble is None:
            return sequences

        features, valid_indices = self._extract_batch_features(sequences, register_lookup)
        if features is None:
            return sequences

        mean_preds, uncertainty = self._ensemble.predict_with_uncertainty(features)
        robustness_scores = mean_preds[:, 0]
        robustness_uncertainty = uncertainty[:, 0]

        scored = [
            (idx, score, unc)
            for idx, score, unc in zip(valid_indices, robustness_scores, robustness_uncertainty)
            if unc < self.max_uncertainty
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        kept = [
            sequences[idx]
            for idx, score, _ in scored[:self.top_k]
            if score >= self.min_score
        ]
        logger.info(
            "EnsembleFilter: %d/%d kept (rejected %d uncertain, top_k=%d)",
            len(kept),
            len(sequences),
            len(valid_indices) - len(scored),
            self.top_k,
        )
        return kept

    def filter(
        self,
        sequences: list[SequenceCandidate],
        register_lookup: dict[str, RegisterCandidate],
    ) -> list[SequenceCandidate]:
        if not self.enabled:
            return sequences
        if not sequences:
            return []
        if self._ensemble is not None:
            return self._filter_with_ensemble(sequences, register_lookup)
        if self._model is None:
            return sequences

        features, valid_indices = self._extract_batch_features(sequences, register_lookup)
        if features is None:
            return sequences

        predictions = self._model.predict_numpy(features)
        robustness_scores = predictions[:, 0]

        scored = list(zip(valid_indices, robustness_scores))
        scored.sort(key=lambda item: item[1], reverse=True)
        kept_indices = [
            idx
            for idx, score in scored[:self.top_k]
            if score >= self.min_score
        ]
        filtered = [sequences[idx] for idx in kept_indices]

        logger.info(
            "SurrogateFilter: %d/%d candidates kept (top_k=%d, min_score=%.2f)",
            len(filtered),
            len(sequences),
            self.top_k,
            self.min_score,
        )
        return filtered
