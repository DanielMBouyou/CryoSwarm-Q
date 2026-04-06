# P1 — AMÉLIORATION DU TRAINING + P2 — CRÉDIBILITÉ RECHERCHE

## Prérequis

Ce prompt suppose que **P0 est implémenté** :
- `PhysicsParameterSpace` existe dans `packages/core/parameter_space.py`
- `SequenceStrategy` existe dans `packages/agents/sequence_strategy.py`
- `DatasetGenerator` existe dans `packages/ml/data_generator.py`
- `DatasetNormalizer` existe dans `packages/ml/normalizer.py`
- Les agents consomment le `param_space`
- Le RL est connecté au pipeline
- Un dataset de 100k+ points peut être généré

---

## Contexte technique — État actuel du code

### Surrogate actuel (`packages/ml/surrogate.py`)

```python
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=4, hidden=64):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),     # 10 → 64
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),         # 64 → 64
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2),    # 64 → 32
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim), # 32 → 4
            nn.Sigmoid(),
        )
```

Problèmes :
- Pas de skip connections → vanishing gradients sur des modèles plus profonds
- Pas de dropout → overfitting garanti sur 100k+ points avec augmentation
- BatchNorm1d mal placé (avant activation = ok, mais pas de LayerNorm alternative)
- Un seul modèle → pas d'estimation d'incertitude
- Pas d'ensemble → pas de variance de prédiction pour filtrer les candidats incertains
- Architecture trop petite pour 18 features (P0 feature_v2)
- Sigmoid en sortie force [0,1] mais ne permet pas d'extrapolation

### RL actuel (`packages/ml/rl_env.py` + `packages/ml/ppo.py`)

```python
class PulseDesignEnv:
    def __init__(self, ..., max_steps: int = 1, ...):  # SINGLE-SHOT
        self.max_steps = max_steps  # = 1 → une seule action par épisode
```

Problèmes :
- `max_steps=1` : l'agent fait UNE action puis c'est fini. Pas de trajectoire, pas de raffinement
- Le reward est la robustness_score brute, pas de shaping progressif
- Observation ne contient pas l'historique de l'épisode
- Pas de curriculum : l'agent voit 3-15 atomes dès le début
- L'environnement ne change jamais de difficulté

### Pipeline de training (`packages/ml/training_runner.py`)

```python
class TrainingRunner:
    def run_full_pipeline(self, ...):
        # Phase 1: Train surrogate
        surrogate_result = self.run_surrogate(train_dataset, val_dataset)
        # Phase 2: Train RL with surrogate
        rl_result = self.run_rl(spec, registers, simulate_fn=surrogate_simulate)
        return {"surrogate": surrogate_result, "rl": rl_result}
```

Problèmes :
- One-shot : le surrogate est entraîné une fois, le RL une fois, terminé
- Pas de boucle itérative surrogate ↔ RL
- Le RL découvre potentiellement des régions mal modélisées par le surrogate
- Les meilleures actions RL ne sont pas re-simulées avec le vrai simulateur
- Pas de raffinement progressif du surrogate dans les régions intéressantes

### Dataset actuel (`packages/ml/dataset.py`)

```python
INPUT_DIM = 10
OUTPUT_DIM = 4

def build_feature_vector(register, sequence):
    return np.array([
        atom_count, spacing, amplitude, detuning, duration_ns,
        layout_id, family_id, blockade_radius, feasibility, cost
    ])
```

Problèmes :
- Features non normalisées → amplitudes de gradient déséquilibrées
- Pas de StandardScaler sauvé avec le modèle
- Le training et l'inférence utilisent des échelles différentes

---

---

## P1.4 — NORMALISATION DES FEATURES

### Objectif

Intégrer un `StandardScaler` dans le pipeline de training du surrogate pour que :
1. Les features soient normalisées (mean=0, std=1) avant d'entrer dans le modèle
2. Les statistiques de normalisation soient sauvées avec le checkpoint
3. L'inférence (SurrogateFilter, RLSequenceAgent) utilise les mêmes statistiques
4. Le passage de v1 (10-dim) à v2 (18-dim, P0) soit transparent

### Ce qui doit être modifié

#### `packages/ml/surrogate.py` — Intégration du normalizer

Le `SurrogateModel` doit porter un normalizer optionnel :

```python
class SurrogateModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden=64):
        ...
        self._normalizer_means: torch.Tensor | None = None
        self._normalizer_stds: torch.Tensor | None = None
    
    def set_normalizer(self, means: NDArray, stds: NDArray) -> None:
        """Attach normalization statistics (computed from training data)."""
        self._normalizer_means = torch.from_numpy(means).float()
        self._normalizer_stds = torch.from_numpy(stds).float()
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stored normalization. Pass-through if no stats attached."""
        if self._normalizer_means is None:
            return x
        means = self._normalizer_means.to(x.device)
        stds = self._normalizer_stds.to(x.device)
        return (x - means) / stds
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._normalize(x))
    
    def save(self, path: str | Path) -> None:
        """Save model weights AND normalization statistics."""
        state = {
            "model": self.state_dict(),
            "normalizer_means": self._normalizer_means,
            "normalizer_stds": self._normalizer_stds,
            "input_dim": self.net[0].in_features,  # Save architecture info
        }
        torch.save(state, str(path))
    
    def load(self, path: str | Path, strict: bool = True) -> None:
        """Load model weights AND normalization statistics."""
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.load_state_dict(checkpoint["model"], strict=strict)
            self._normalizer_means = checkpoint.get("normalizer_means")
            self._normalizer_stds = checkpoint.get("normalizer_stds")
        else:
            # Backward compat: old checkpoints are raw state_dicts
            self.load_state_dict(checkpoint, strict=strict)
```

**Note sur la rétrocompatibilité du save/load** : L'ancien format sauvait directement `state_dict()`. Le nouveau `load()` doit détecter l'ancien format (dict sans clé `"model"`) et le charger en mode legacy. Tous les tests existants doivent continuer à passer.

#### `packages/ml/surrogate.py` — SurrogateTrainer

Le `SurrogateTrainer.fit()` doit calculer les statistiques de normalisation sur le training set et les attacher au modèle :

```python
class SurrogateTrainer:
    def fit(self, train_dataset, val_dataset=None, epochs=100, batch_size=64, log_dir=None):
        # --- NEW: compute normalizer from training data ---
        all_features = []
        for i in range(len(train_dataset)):
            features, _ = train_dataset[i]
            all_features.append(features.numpy() if isinstance(features, torch.Tensor) else features)
        all_features = np.stack(all_features)
        means = all_features.mean(axis=0)
        stds = all_features.std(axis=0)
        stds[stds < 1e-8] = 1.0  # Prevent division by zero
        self.model.set_normalizer(means, stds)
        
        # ... rest of training loop unchanged ...
```

#### `packages/ml/surrogate_filter.py`

Le `SurrogateFilter._load_model()` charge le modèle avec ses statistiques de normalisation intégrées. Aucune modification nécessaire car le `SurrogateModel.load()` gère déjà le normalizer.

#### `packages/ml/rl_sequence_agent.py`

Le `RLSequenceAgent._load_policy()` n'est pas affecté (c'est l'ActorCritic, pas le surrogate). Mais quand `training_runner.run_full_pipeline()` utilise le surrogate comme simulator pour le RL, le normalizer est automatiquement appliqué via `model.forward()`.

### Tests requis

```python
# test_normalizer_integration.py

def test_surrogate_save_load_with_normalizer():
    """Normalizer statistics survive save/load cycle."""
    model = SurrogateModel(input_dim=10)
    means = np.random.randn(10).astype(np.float32)
    stds = np.abs(np.random.randn(10).astype(np.float32)) + 0.1
    model.set_normalizer(means, stds)
    model.save(tmp_path / "model.pt")
    
    model2 = SurrogateModel(input_dim=10)
    model2.load(tmp_path / "model.pt")
    assert model2._normalizer_means is not None
    np.testing.assert_array_almost_equal(
        model2._normalizer_means.numpy(), means
    )

def test_surrogate_backward_compat_load():
    """Old-format checkpoints (raw state_dict) still load correctly."""
    model = SurrogateModel(input_dim=10)
    # Save in old format
    torch.save(model.state_dict(), tmp_path / "old.pt")
    
    model2 = SurrogateModel(input_dim=10)
    model2.load(tmp_path / "old.pt")
    assert model2._normalizer_means is None  # No normalizer in old format

def test_trainer_computes_normalizer():
    """SurrogateTrainer.fit() automatically sets normalizer on model."""
    X = torch.randn(100, 10)
    Y = torch.rand(100, 4)
    dataset = TensorDataset(X, Y)
    model = SurrogateModel(input_dim=10)
    trainer = SurrogateTrainer(model)
    trainer.fit(dataset, epochs=2)
    assert model._normalizer_means is not None
    assert model._normalizer_stds is not None
```

---

---

## P1.5 — ARCHITECTURE SURROGATE ENRICHIE

### Objectif

Remplacer le MLP naïf actuel par une architecture research-grade avec :
1. **Residual connections** (skip connections) pour un meilleur gradient flow
2. **Dropout** pour la régularisation
3. **LayerNorm** (plus stable que BatchNorm pour les petits batches)
4. **Ensemble de modèles** pour estimation d'incertitude
5. Architecture compatible avec 18 features (P0 feature_v2) ET 10 features (legacy)

### Ce qui doit être créé/modifié

#### `packages/ml/surrogate.py` — Nouveau ResidualBlock + SurrogateModelV2

Ajouter les classes suivantes **en plus** du `SurrogateModel` existant (ne pas supprimer l'ancien) :

```python
class ResidualBlock(nn.Module):
    """Residual block with pre-activation LayerNorm.
    
    Architecture: LayerNorm → Linear → GELU → Dropout → Linear → + skip
    
    GELU is preferred over ReLU for scientific ML applications:
    - Smoother gradient landscape
    - Better convergence on regression tasks
    - Standard in modern transformer-based architectures
    
    References:
        He et al., "Deep Residual Learning" (2015)
        Hendrycks & Gimpel, "Gaussian Error Linear Units" (2016)
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
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


class SurrogateModelV2(nn.Module):
    """Research-grade surrogate with residual connections and uncertainty.
    
    Architecture:
        Input(D) → Linear(D, hidden) → GELU
        → ResidualBlock(hidden) × n_blocks
        → LayerNorm(hidden)
        → Linear(hidden, hidden//2) → GELU → Dropout
        → Linear(hidden//2, output_dim) → Sigmoid
    
    For an ensemble of K models, use SurrogateEnsemble which wraps
    K independent SurrogateModelV2 instances.
    
    Default configuration for 18-dim input (P0 feature_v2):
        hidden=128, n_blocks=3, dropout=0.1
        → ~25k parameters (vs ~6k for V1)
        → Still fast enough for pre-filtering (< 1ms for 1000 candidates)
    
    Parameters
    ----------
    input_dim : Feature vector dimensionality (10 for v1, 18 for v2).
    output_dim : Target vector dimensionality (4).
    hidden : Hidden layer width.
    n_blocks : Number of residual blocks.
    dropout : Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int = 18,   # Default to v2 features
        output_dim: int = 4,
        hidden: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        _check_torch()
        super().__init__()
        
        # Projection from input to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
        )
        
        # Residual tower
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim),
            nn.Sigmoid(),
        )
        
        # Normalizer (same interface as V1)
        self._normalizer_means: torch.Tensor | None = None
        self._normalizer_stds: torch.Tensor | None = None
    
    def set_normalizer(self, means: NDArray, stds: NDArray) -> None:
        self._normalizer_means = torch.from_numpy(means).float()
        self._normalizer_stds = torch.from_numpy(stds).float()
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._normalizer_means is None:
            return x
        means = self._normalizer_means.to(x.device)
        stds = self._normalizer_stds.to(x.device)
        return (x - means) / stds
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        x = self.input_proj(x)
        x = self.residual_blocks(x)
        return self.output_head(x)
    
    def predict_numpy(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features)
            if features.ndim == 1:
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
                "input_dim": self.input_proj[0].in_features,
                "hidden": self.input_proj[0].out_features,
                "n_blocks": len(self.residual_blocks),
                "output_dim": self.output_head[-2].out_features,
            },
        }
        torch.save(state, str(path))
    
    def load(self, path: str | Path, strict: bool = True) -> None:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.load_state_dict(checkpoint["model"], strict=strict)
            self._normalizer_means = checkpoint.get("normalizer_means")
            self._normalizer_stds = checkpoint.get("normalizer_stds")
        else:
            self.load_state_dict(checkpoint, strict=strict)
```

#### `packages/ml/surrogate.py` — SurrogateEnsemble

Ajouter un wrapper d'ensemble pour estimation d'incertitude :

```python
class SurrogateEnsemble:
    """Ensemble of K surrogate models for uncertainty-aware prediction.
    
    Each model in the ensemble is trained independently on the same data
    (with different random initialization and shuffling). The ensemble
    provides:
    
    1. Mean prediction: average of K model outputs
    2. Epistemic uncertainty: standard deviation across K predictions
    3. Calibrated confidence: uncertainty-based filtering threshold
    
    The uncertainty estimate is crucial for:
    - Active learning (P1.8): identify high-uncertainty regions to re-simulate
    - Surrogate filter: reject candidates where the surrogate is unsure
    - RL training: discount surrogate reward when uncertainty is high
    
    Usage::
    
        ensemble = SurrogateEnsemble(n_models=3, input_dim=18, hidden=128)
        trainer = EnsembleTrainer(ensemble)
        trainer.fit(train_dataset, val_dataset, epochs=200)
        
        # Prediction with uncertainty
        mean_pred, uncertainty = ensemble.predict_with_uncertainty(features)
        # uncertainty > 0.1 → low confidence, consider re-simulation
    
    References:
        Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
        Estimation using Deep Ensembles" (NeurIPS 2017)
    """
    
    def __init__(
        self,
        n_models: int = 3,
        model_class: type = SurrogateModelV2,
        **model_kwargs,
    ):
        self.n_models = n_models
        self.models = [model_class(**model_kwargs) for _ in range(n_models)]
    
    def predict_with_uncertainty(
        self, features: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return (mean_prediction, uncertainty) across ensemble members.
        
        Parameters
        ----------
        features : Input features, shape (N, D) or (D,).
        
        Returns
        -------
        mean_preds : Mean prediction, shape (N, output_dim).
        uncertainty : Per-output std across ensemble, shape (N, output_dim).
        """
        predictions = []
        for model in self.models:
            pred = model.predict_numpy(features)
            predictions.append(pred)
        
        stacked = np.stack(predictions, axis=0)  # (K, N, output_dim)
        mean_preds = stacked.mean(axis=0)
        uncertainty = stacked.std(axis=0)
        return mean_preds, uncertainty
    
    def predict_robustness_with_uncertainty(
        self, features: NDArray[np.float32],
    ) -> tuple[float, float]:
        """Single-sample robustness prediction with uncertainty.
        
        Returns (mean_robustness, uncertainty).
        """
        mean_preds, uncertainty = self.predict_with_uncertainty(features)
        return float(mean_preds[0, 0]), float(uncertainty[0, 0])
    
    def set_normalizer(self, means: NDArray, stds: NDArray) -> None:
        """Set the same normalizer on all ensemble members."""
        for model in self.models:
            model.set_normalizer(means, stds)
    
    def save(self, directory: str | Path) -> None:
        """Save all ensemble members to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            model.save(directory / f"member_{i}.pt")
        # Save ensemble metadata
        import json
        meta = {"n_models": self.n_models, "version": "ensemble_v1"}
        (directory / "ensemble_meta.json").write_text(json.dumps(meta))
    
    def load(self, directory: str | Path) -> None:
        """Load all ensemble members from a directory."""
        directory = Path(directory)
        for i, model in enumerate(self.models):
            model.load(directory / f"member_{i}.pt")
    
    def eval(self) -> None:
        for model in self.models:
            model.eval()
    
    def train(self) -> None:
        for model in self.models:
            model.train()


class EnsembleTrainer:
    """Train all members of a SurrogateEnsemble independently.
    
    Each member sees the same data but with:
    - Different random initialization (from model construction)
    - Different batch shuffling (different DataLoader seed per member)
    - Optionally different bootstrap samples (bagging)
    
    This diversity ensures meaningful uncertainty estimates.
    """
    
    def __init__(
        self,
        ensemble: SurrogateEnsemble,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        target_weights: list[float] | None = None,
        bootstrap: bool = False,  # If True, each member trains on a bootstrap sample
    ):
        self.ensemble = ensemble
        self.lr = lr
        self.weight_decay = weight_decay
        self.target_weights = target_weights
        self.bootstrap = bootstrap
    
    def fit(
        self,
        train_dataset,
        val_dataset=None,
        epochs: int = 200,
        batch_size: int = 64,
        log_dir: str | None = None,
    ) -> list[dict[str, list[float]]]:
        """Train each ensemble member sequentially.
        
        Returns a list of history dicts, one per member.
        """
        histories = []
        for i, model in enumerate(self.ensemble.models):
            logger.info("Training ensemble member %d/%d", i + 1, self.ensemble.n_models)
            
            # Optionally create a bootstrap sample
            if self.bootstrap:
                n = len(train_dataset)
                indices = torch.randint(0, n, (n,))
                member_dataset = torch.utils.data.Subset(train_dataset, indices.tolist())
            else:
                member_dataset = train_dataset
            
            trainer = SurrogateTrainer(
                model, lr=self.lr, weight_decay=self.weight_decay,
                target_weights=self.target_weights,
            )
            member_log_dir = f"{log_dir}/member_{i}" if log_dir else None
            history = trainer.fit(
                member_dataset, val_dataset, epochs=epochs,
                batch_size=batch_size, log_dir=member_log_dir,
            )
            histories.append(history)
        
        return histories
```

#### `packages/ml/surrogate_filter.py` — Support de l'ensemble

Le `SurrogateFilter` doit pouvoir utiliser soit un modèle unique soit un ensemble :

```python
class SurrogateFilter:
    def __init__(
        self,
        model_path: str | Path | None = None,
        top_k: int = 20,
        min_score: float = 0.1,
        max_uncertainty: float = 0.15,  # NEW: reject high-uncertainty candidates
        enabled: bool = False,
        use_ensemble: bool = False,     # NEW: use ensemble directory
    ):
        ...
        self._ensemble = None
        self.max_uncertainty = max_uncertainty
        self.use_ensemble = use_ensemble
        
        if enabled and use_ensemble and model_path:
            self._load_ensemble(Path(model_path))
        elif enabled and model_path and TORCH_AVAILABLE:
            self._load_model(Path(model_path))
    
    def _load_ensemble(self, directory: Path) -> None:
        if not directory.is_dir():
            logger.warning("Ensemble dir not found: %s — passthrough.", directory)
            self.enabled = False
            return
        from packages.ml.surrogate import SurrogateEnsemble, SurrogateModelV2
        import json
        meta = json.loads((directory / "ensemble_meta.json").read_text())
        self._ensemble = SurrogateEnsemble(
            n_models=meta["n_models"], model_class=SurrogateModelV2,
        )
        self._ensemble.load(directory)
        self._ensemble.eval()
        logger.info("Surrogate ensemble loaded from %s (%d members)", directory, meta["n_models"])
    
    def filter(self, sequences, register_lookup):
        if not self.enabled:
            return sequences
        
        if self._ensemble is not None:
            return self._filter_with_ensemble(sequences, register_lookup)
        
        # ... existing single-model filter logic ...
    
    def _filter_with_ensemble(self, sequences, register_lookup):
        """Filter using ensemble predictions with uncertainty rejection."""
        features_list, valid_indices = [], []
        for idx, seq in enumerate(sequences):
            reg = register_lookup.get(seq.register_candidate_id)
            if reg is None:
                continue
            features_list.append(build_feature_vector(reg, seq))
            valid_indices.append(idx)
        
        if not features_list:
            return sequences
        
        features = np.stack(features_list)
        mean_preds, uncertainty = self._ensemble.predict_with_uncertainty(features)
        robustness_scores = mean_preds[:, 0]
        robustness_uncertainty = uncertainty[:, 0]
        
        # Rank by score, but reject high-uncertainty candidates
        scored = [
            (idx, score, unc)
            for idx, score, unc in zip(valid_indices, robustness_scores, robustness_uncertainty)
            if unc < self.max_uncertainty  # Reject if too uncertain
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        kept = [sequences[idx] for idx, score, unc in scored[:self.top_k] if score >= self.min_score]
        
        logger.info(
            "EnsembleFilter: %d/%d kept (rejected %d uncertain, top_k=%d)",
            len(kept), len(sequences),
            len(valid_indices) - len(scored), self.top_k,
        )
        return kept
```

### Architecture specs

| Property | SurrogateModel (V1) | SurrogateModelV2 |
|----------|-------------------|-----------------|
| Input dim | 10 | 18 (P0 feature_v2) |
| Hidden dim | 64 | 128 |
| Depth | 3 linéaires | 1 projection + 3 residual blocks + output head |
| Normalization | BatchNorm1d | LayerNorm (per-block) |
| Activation | ReLU | GELU |
| Dropout | 0 | 0.1 |
| Skip connections | Non | Oui (x + block(x)) |
| Parameters | ~6k | ~25k |
| Uncertainty | Non | Via ensemble (K=3) |
| Latency (1k samples) | < 1ms | < 2ms |

### Tests requis

```python
def test_surrogate_v2_forward():
    model = SurrogateModelV2(input_dim=18, hidden=128, n_blocks=3)
    x = torch.randn(32, 18)
    y = model(x)
    assert y.shape == (32, 4)
    assert (y >= 0).all() and (y <= 1).all()  # Sigmoid output

def test_residual_block_preserves_dim():
    block = ResidualBlock(dim=128, dropout=0.1)
    x = torch.randn(16, 128)
    assert block(x).shape == x.shape

def test_ensemble_uncertainty():
    ensemble = SurrogateEnsemble(n_models=3, input_dim=18, hidden=64, n_blocks=2)
    features = np.random.randn(10, 18).astype(np.float32)
    mean_pred, uncertainty = ensemble.predict_with_uncertainty(features)
    assert mean_pred.shape == (10, 4)
    assert uncertainty.shape == (10, 4)
    assert (uncertainty >= 0).all()

def test_ensemble_save_load(tmp_path):
    ensemble = SurrogateEnsemble(n_models=3, input_dim=10, hidden=64, n_blocks=2)
    ensemble.save(tmp_path / "ensemble")
    ensemble2 = SurrogateEnsemble(n_models=3, input_dim=10, hidden=64, n_blocks=2)
    ensemble2.load(tmp_path / "ensemble")
    # Verify same predictions
    x = np.random.randn(5, 10).astype(np.float32)
    m1, _ = ensemble.predict_with_uncertainty(x)
    m2, _ = ensemble2.predict_with_uncertainty(x)
    np.testing.assert_array_almost_equal(m1, m2)
```

---

---

## P1.6 — RL MULTI-STEP AVEC REWARD SHAPING

### Objectif

Transformer le RL d'un problème **single-shot** (1 action → 1 reward → terminé) en un vrai problème séquentiel où l'agent **raffine progressivement** ses paramètres de pulse sur plusieurs steps.

### Problème fondamental actuel

Avec `max_steps=1`, l'agent PPO :
- Ne peut PAS apprendre de trajectoires
- Ne bénéficie PAS du GAE (Generalized Advantage Estimation) car il n'y a qu'un seul step
- Ne raffine PAS son action en fonction du feedback
- Chaque épisode est indépendant → c'est un bandit, pas du RL

### Ce qui doit être modifié

#### `packages/ml/rl_env.py` — Épisodes multi-step

Le `PulseDesignEnv` doit supporter des épisodes de 5-10 steps avec reward shaping :

```python
class PulseDesignEnv:
    """Gymnasium-compatible environment for pulse-sequence parameter selection.
    
    Multi-step episode structure (NEW):
    
    1. reset() → sample register, observe initial state
    2. step(action_0) → simulate → get reward_0, observe episode-best feedback
    3. step(action_1) → simulate → get reward_1 (shaped: includes improvement bonus)
    4. ...
    5. step(action_{T-1}) → final step, terminated=True
    
    The agent sees its previous best in the observation, incentivizing
    iterative refinement. Each step evaluates a DIFFERENT pulse configuration
    on the SAME register.
    
    Reward shaping:
        raw_reward = robustness_score(params)
        improvement = max(0, raw_reward - best_so_far)
        shaped_reward = 0.5 * raw_reward + 0.5 * improvement * 5.0
    
    This gives credit both for absolute quality AND for improving over
    the best previous attempt in the episode. The improvement bonus
    (×5.0) heavily incentivizes exploration of better configurations.
    
    Episode features:
    - max_steps: number of refinement attempts (default 5)
    - Observation includes episode-best parameters as context
    - Step budget encoding: remaining_steps / max_steps
    - Register stays fixed within episode, changes across episodes
    """
    
    def __init__(
        self,
        spec: ExperimentSpec,
        register_candidates: list[RegisterCandidate],
        max_steps: int = 5,        # CHANGED from 1 to 5
        simulate_fn: Any | None = None,
        reward_shaping: bool = True,  # NEW
        improvement_weight: float = 0.5,  # NEW: weight on improvement bonus
        improvement_scale: float = 5.0,   # NEW: scaling for improvement bonus
        param_space: PhysicsParameterSpace | None = None,  # From P0
    ) -> None:
        ...
        self._episode_history: list[dict[str, Any]] = []  # NEW: track all steps
        self._reward_shaping = reward_shaping
        self._improvement_weight = improvement_weight
        self._improvement_scale = improvement_scale
```

**Observation space étendu** (14 → 16 dim) :

```python
OBS_DIM = 16  # Was 14

def _build_observation(self) -> NDArray[np.float32]:
    reg = self._current_register
    if reg is None:
        return np.zeros(OBS_DIM, dtype=np.float32)
    
    spacing = float(reg.metadata.get("spacing_um", 7.0))
    from packages.ml.dataset import _encode_layout
    
    return np.array([
        # Register context (same as before)
        float(reg.atom_count),
        spacing,
        float(reg.blockade_radius_um),
        float(reg.feasibility_score),
        float(self.spec.target_density),
        float(self.spec.min_atoms),
        float(self.spec.max_atoms),
        float(_encode_layout(reg.layout_type)),
        
        # Episode-best feedback (updated after each step)
        self._best_robustness,
        self._best_params.get("amplitude", 0.0) / 15.0,   # Normalized
        self._best_params.get("detuning", 0.0) / 30.0,     # Normalized
        self._best_params.get("duration_ns", 0.0) / 5500.0, # Normalized
        
        # NEW: step progress encoding
        self._step_count / max(self.max_steps, 1),  # How far into episode
        max(0, self.max_steps - self._step_count) / max(self.max_steps, 1),  # Steps remaining
        
        # NEW: best nominal and observable from previous steps
        self._best_nominal,  # NEW field
        self._best_observable,  # NEW field
    ], dtype=np.float32)
```

**Reward shaping** :

```python
def step(self, action):
    self._step_count += 1
    params = rescale_action(action)
    reg = self._current_register
    
    raw_reward = 0.0
    info: dict[str, Any] = {"params": params, "step": self._step_count}
    
    if reg is not None:
        raw_reward = self._evaluate(reg, params)
        info["raw_robustness"] = raw_reward
        
        # Reward shaping: credit improvement over episode-best
        improvement = max(0.0, raw_reward - self._best_robustness)
        
        if self._reward_shaping:
            base_weight = 1.0 - self._improvement_weight
            shaped_reward = (
                base_weight * raw_reward 
                + self._improvement_weight * improvement * self._improvement_scale
            )
        else:
            shaped_reward = raw_reward
        
        reward = shaped_reward
        info["shaped_reward"] = reward
        info["improvement"] = improvement
        
        # Update episode-best
        if raw_reward > self._best_robustness:
            self._best_robustness = raw_reward
            self._best_params = params
        
        # Track episode history
        self._episode_history.append({
            "step": self._step_count,
            "params": params,
            "raw_reward": raw_reward,
            "shaped_reward": reward,
            "best_so_far": self._best_robustness,
        })
    else:
        reward = 0.0
    
    terminated = self._step_count >= self.max_steps
    
    # Terminal bonus: reward the episode-best score at end
    if terminated and self._reward_shaping:
        reward += self._best_robustness * 0.5  # Bonus for best found
    
    info["robustness_score"] = raw_reward
    info["episode_best"] = self._best_robustness
    info["episode_history"] = self._episode_history if terminated else []
    
    obs = self._build_observation()
    return obs, reward, terminated, False, info

def reset(self, seed=None):
    ...
    self._episode_history = []  # NEW: clear history
    self._best_nominal = 0.0   # NEW
    self._best_observable = 0.0 # NEW
    ...
```

#### `packages/ml/ppo.py` — Adapter ActorCritic au nouveau obs_dim

Le `ActorCritic` doit accepter le nouveau `obs_dim=16` :

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 128):
        # OBS_DIM est maintenant 16 (importé de rl_env)
        ...
```

Le `PPOConfig` doit avoir des rollout_steps plus grands pour les épisodes multi-step :

```python
@dataclass
class PPOConfig:
    ...
    rollout_steps: int = 256    # CHANGED from 128 : more steps per rollout for multi-step episodes
    total_updates: int = 500    # CHANGED from 200 : more updates needed for multi-step
```

#### `packages/ml/training_runner.py` — Passer max_steps

```python
def run_rl(self, spec, register_candidates, simulate_fn=None):
    env = PulseDesignEnv(
        spec=spec,
        register_candidates=register_candidates,
        max_steps=self.config.rl_max_steps,  # NEW config field
        simulate_fn=simulate_fn,
        reward_shaping=True,
    )
    ...
```

Ajouter au `TrainingConfig` :

```python
@dataclass
class TrainingConfig:
    ...
    rl_max_steps: int = 5         # NEW: steps per episode
    rl_total_updates: int = 500   # CHANGED from 200
    rl_rollout_steps: int = 256   # CHANGED from 128
```

### Rétrocompatibilité

- `PulseDesignEnv(max_steps=1, reward_shaping=False)` reproduit le comportement actuel
- L'ancien `OBS_DIM=14` doit rester comme constante `OBS_DIM_V1 = 14` pour les checkpoints existants
- Le nouveau `OBS_DIM = 16` est le défaut
- Le `ActorCritic` accepte `obs_dim` en paramètre donc les anciens checkpoints (obs_dim=14) fonctionnent à condition de passer `obs_dim=14`

### Tests requis

```python
def test_multi_step_episode():
    """Agent can take 5 steps in one episode."""
    env = PulseDesignEnv(spec, registers, max_steps=5, simulate_fn=mock_sim)
    obs, info = env.reset(seed=42)
    rewards = []
    for step in range(5):
        action = np.random.uniform(-1, 1, 4).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if step < 4:
            assert not terminated
    assert terminated  # Step 5 terminates

def test_reward_shaping_improvement():
    """Improvement bonus rewards better actions."""
    env = PulseDesignEnv(spec, registers, max_steps=3, reward_shaping=True,
                         simulate_fn=lambda r, p: p["amplitude"] / 15.0)
    obs, _ = env.reset(seed=42)
    
    # Step 1: low amplitude → low reward
    action1 = np.array([-0.8, 0, 0, 0], dtype=np.float32)  # low amplitude
    _, r1, _, _, info1 = env.step(action1)
    
    # Step 2: high amplitude → improvement bonus
    action2 = np.array([0.8, 0, 0, 0], dtype=np.float32)  # high amplitude
    _, r2, _, _, info2 = env.step(action2)
    
    assert info2["improvement"] > 0  # There was improvement
    assert r2 > r1  # Shaped reward should be higher

def test_obs_dim_16():
    """Observation space is 16-dimensional."""
    env = PulseDesignEnv(spec, registers, max_steps=5, simulate_fn=mock)
    obs, _ = env.reset()
    assert obs.shape == (16,)

def test_backward_compat_single_step():
    """max_steps=1 with reward_shaping=False matches old behavior."""
    env = PulseDesignEnv(spec, registers, max_steps=1, reward_shaping=False, 
                         simulate_fn=mock)
    obs, _ = env.reset()
    assert obs.shape == (16,)  # New obs size but same behavior
    _, reward, terminated, _, _ = env.step(np.zeros(4, dtype=np.float32))
    assert terminated  # Terminates after 1 step
```

---

---

## P1.7 — CURRICULUM LEARNING

### Objectif

Implémenter un curriculum qui augmente progressivement la complexité du problème pendant le training RL :

1. **Phase 1** (0-30% du training) : 3-5 atomes seulement → problèmes simples, convergence rapide
2. **Phase 2** (30-60% du training) : 5-8 atomes → complexité intermédiaire
3. **Phase 3** (60-100% du training) : 3-15 atomes → full range

L'idée est que l'agent apprenne d'abord les bases de la physique Rydberg sur des systèmes petits (où la simulation est rapide et le landscape simple), puis généralise aux systèmes plus grands.

### Ce qui doit être créé

#### `packages/ml/curriculum.py`

```python
"""Curriculum learning for progressive RL training.

Implements a curriculum that gradually increases problem complexity
during PPO training. The agent starts with simple problems (few atoms,
simple geometries) and progresses to harder ones (more atoms, diverse
geometries).

The curriculum is managed by a CurriculumScheduler that:
1. Controls which register candidates are available to the environment
2. Tracks agent performance to decide when to advance
3. Supports multiple progression strategies (linear, adaptive, threshold)

Curriculum stages:
    Stage 1 (warm-up):    3-5 atoms,  square/line only
    Stage 2 (expansion):  5-8 atoms,  + triangular/ring
    Stage 3 (full):       3-15 atoms, all geometries

References:
    Bengio et al., "Curriculum Learning" (ICML 2009)
    Narvekar et al., "Curriculum Learning for RL: A Survey" (JMLR 2020)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from packages.core.logging import get_logger
from packages.core.models import RegisterCandidate

logger = get_logger(__name__)


@dataclass
class CurriculumStage:
    """Definition of a single curriculum stage."""
    name: str
    min_atoms: int
    max_atoms: int
    allowed_layouts: list[str]
    min_performance: float      # Avg reward to advance to next stage
    min_episodes: int           # Minimum episodes before considering advancement
    
    def accepts(self, register: RegisterCandidate) -> bool:
        """Check if a register candidate is valid for this stage."""
        if register.atom_count < self.min_atoms or register.atom_count > self.max_atoms:
            return False
        if register.layout_type not in self.allowed_layouts:
            return False
        return True


class CurriculumScheduler:
    """Manages progression through curriculum stages.
    
    Progression logic:
    1. LINEAR: advance after fixed fraction of total training
    2. ADAPTIVE: advance when agent reaches performance threshold
    3. CYCLING: cycle through stages to prevent forgetting
    
    Default: ADAPTIVE with fallback to LINEAR
    (advance after threshold OR after max_fraction of training, whichever first)
    """
    
    def __init__(
        self,
        stages: list[CurriculumStage] | None = None,
        mode: str = "adaptive",   # "linear", "adaptive", "cycling"
        total_updates: int = 500,
    ):
        self.stages = stages or self._default_stages()
        self.mode = mode
        self.total_updates = total_updates
        self.current_stage_idx = 0
        self._episode_rewards: list[float] = []
        self._stage_episode_count = 0
        self._update_count = 0
    
    @staticmethod
    def _default_stages() -> list[CurriculumStage]:
        return [
            CurriculumStage(
                name="warm-up",
                min_atoms=3, max_atoms=5,
                allowed_layouts=["square", "line"],
                min_performance=0.3,
                min_episodes=50,
            ),
            CurriculumStage(
                name="expansion",
                min_atoms=4, max_atoms=8,
                allowed_layouts=["square", "line", "triangular", "ring"],
                min_performance=0.35,
                min_episodes=100,
            ),
            CurriculumStage(
                name="full",
                min_atoms=3, max_atoms=15,
                allowed_layouts=["square", "line", "triangular", "ring", "zigzag", "honeycomb"],
                min_performance=0.0,  # No threshold — final stage
                min_episodes=0,
            ),
        ]
    
    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[min(self.current_stage_idx, len(self.stages) - 1)]
    
    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1
    
    def filter_registers(
        self, candidates: list[RegisterCandidate],
    ) -> list[RegisterCandidate]:
        """Filter register candidates to those valid for the current stage."""
        stage = self.current_stage
        valid = [r for r in candidates if stage.accepts(r)]
        if not valid:
            # Fallback: if no candidates match, use the closest atom count
            logger.warning(
                "No registers match curriculum stage '%s' (atoms %d-%d). "
                "Using all candidates as fallback.",
                stage.name, stage.min_atoms, stage.max_atoms,
            )
            return candidates
        return valid
    
    def record_episode(self, reward: float) -> None:
        """Record an episode reward and check for stage advancement."""
        self._episode_rewards.append(reward)
        self._stage_episode_count += 1
    
    def step_update(self) -> bool:
        """Called after each PPO update. Returns True if stage changed."""
        self._update_count += 1
        
        if self.is_final_stage:
            return False
        
        stage = self.current_stage
        
        if self.mode == "linear":
            # Advance at fixed fractions of training
            fraction = self._update_count / max(self.total_updates, 1)
            stage_fraction = (self.current_stage_idx + 1) / len(self.stages)
            if fraction >= stage_fraction:
                return self._advance()
        
        elif self.mode == "adaptive":
            # Advance when performance threshold is met
            if self._stage_episode_count < stage.min_episodes:
                return False
            
            recent_rewards = self._episode_rewards[-stage.min_episodes:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            if avg_reward >= stage.min_performance:
                logger.info(
                    "Curriculum: advancing from '%s' (avg_reward=%.3f >= %.3f)",
                    stage.name, avg_reward, stage.min_performance,
                )
                return self._advance()
            
            # Fallback: advance after 40% of remaining training even if threshold not met
            remaining_fraction = 1.0 - self._update_count / max(self.total_updates, 1)
            max_stage_fraction = 0.4 / (len(self.stages) - self.current_stage_idx)
            if self._update_count > 0:
                stage_time = self._stage_episode_count / (self._update_count * 5)  # rough estimate
                if stage_time > max_stage_fraction:
                    logger.info("Curriculum: advancing from '%s' (time limit)", stage.name)
                    return self._advance()
        
        return False
    
    def _advance(self) -> bool:
        if self.is_final_stage:
            return False
        self.current_stage_idx += 1
        self._stage_episode_count = 0
        logger.info("Curriculum: now at stage '%s'", self.current_stage.name)
        return True
    
    def get_report(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage.name,
            "stage_index": self.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_episodes": self._stage_episode_count,
            "total_episodes": len(self._episode_rewards),
            "update_count": self._update_count,
            "avg_recent_reward": (
                sum(self._episode_rewards[-20:]) / min(20, len(self._episode_rewards))
                if self._episode_rewards else 0.0
            ),
        }
```

### Intégration dans le PPO Trainer

#### `packages/ml/ppo.py` — PPOTrainer avec curriculum

```python
class PPOTrainer:
    def __init__(self, config=None, curriculum: CurriculumScheduler | None = None):
        ...
        self.curriculum = curriculum
    
    def train(self, env, seed=42):
        obs, info = env.reset(seed=seed)
        buffer = RolloutBuffer()
        
        for update in range(self.config.total_updates):
            episode_reward = 0.0
            
            for step in range(self.config.rollout_steps):
                action, log_prob, value = self.policy.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                buffer.add(obs, action, log_prob, reward, value, done)
                episode_reward += reward
                obs = next_obs
                
                if done:
                    history["episode_rewards"].append(episode_reward)
                    
                    # --- Curriculum: record episode and filter registers ---
                    if self.curriculum is not None:
                        self.curriculum.record_episode(episode_reward)
                        # Update env's register pool based on curriculum stage
                        filtered_registers = self.curriculum.filter_registers(
                            env._all_register_candidates  # Full pool
                        )
                        env.register_candidates = filtered_registers
                    
                    episode_reward = 0.0
                    obs, info = env.reset()
            
            # ... PPO update (unchanged) ...
            
            # --- Curriculum: check for advancement ---
            if self.curriculum is not None:
                self.curriculum.step_update()
```

#### `packages/ml/rl_env.py` — Store full register pool

```python
class PulseDesignEnv:
    def __init__(self, spec, register_candidates, ...):
        ...
        self._all_register_candidates = list(register_candidates)  # Full pool
        self.register_candidates = list(register_candidates)       # Active pool (filtered by curriculum)
```

### Intégration dans TrainingRunner

```python
class TrainingRunner:
    def run_rl(self, spec, register_candidates, simulate_fn=None):
        from packages.ml.curriculum import CurriculumScheduler
        
        curriculum = CurriculumScheduler(
            mode="adaptive",
            total_updates=self.config.rl_total_updates,
        ) if self.config.use_curriculum else None  # NEW config field
        
        env = PulseDesignEnv(
            spec=spec,
            register_candidates=register_candidates,
            max_steps=self.config.rl_max_steps,
            simulate_fn=simulate_fn,
        )
        
        ppo_config = PPOConfig(
            total_updates=self.config.rl_total_updates,
            rollout_steps=self.config.rl_rollout_steps,
        )
        
        trainer = PPOTrainer(ppo_config, curriculum=curriculum)
        history = trainer.train(env, seed=self.config.seed)
        
        # Include curriculum report in history
        if curriculum:
            history["curriculum"] = curriculum.get_report()
        
        ...
```

Ajouter au `TrainingConfig` :
```python
@dataclass
class TrainingConfig:
    ...
    use_curriculum: bool = True   # NEW
```

### Tests requis

```python
def test_curriculum_stage_progression():
    """Scheduler advances through stages."""
    scheduler = CurriculumScheduler(mode="adaptive", total_updates=100)
    assert scheduler.current_stage.name == "warm-up"
    
    # Simulate 50 episodes with good reward
    for _ in range(60):
        scheduler.record_episode(0.4)
    scheduler.step_update()
    
    assert scheduler.current_stage.name == "expansion"

def test_curriculum_filters_registers():
    """Stage 1 only allows small registers."""
    scheduler = CurriculumScheduler()
    registers = [
        make_register(atom_count=4, layout="square"),   # OK for stage 1
        make_register(atom_count=10, layout="ring"),     # Too big for stage 1
        make_register(atom_count=3, layout="line"),      # OK for stage 1
    ]
    filtered = scheduler.filter_registers(registers)
    assert len(filtered) == 2
    assert all(r.atom_count <= 5 for r in filtered)

def test_curriculum_fallback_on_no_match():
    """If no registers match, uses all candidates."""
    scheduler = CurriculumScheduler()
    registers = [make_register(atom_count=12, layout="honeycomb")]
    filtered = scheduler.filter_registers(registers)
    assert len(filtered) == 1  # Fallback to all
```

---

---

## P1.8 — BOUCLE ACTIVE LEARNING

### Objectif

Implémenter une boucle itérative surrogate ↔ RL ↔ simulation qui :

1. Entraîne le surrogate sur le dataset initial (100k+ points)
2. Entraîne le RL avec le surrogate comme simulateur rapide
3. **Re-simule** les meilleures actions découvertes par le RL avec le VRAI simulateur
4. **Ajoute** ces nouveaux points au dataset du surrogate
5. **Re-entraîne** le surrogate avec le dataset enrichi
6. **Répète** pour N itérations

C'est de l'**active learning** : le RL explore l'espace, le surrogate s'améliore dans les régions que le RL trouve intéressantes, le RL bénéficie d'un surrogate plus précis dans ces régions.

### Formulation mathématique

À chaque itération $k$ :

$$\mathcal{D}_{k+1} = \mathcal{D}_k \cup \{(x_i, y_i) \mid x_i \in \text{TopK}(\pi_k), \; y_i = f_{\text{sim}}(x_i)\}$$

Où :
- $\mathcal{D}_k$ est le dataset de training à l'itération $k$
- $\pi_k$ est la politique RL entraînée à l'itération $k$
- $\text{TopK}(\pi_k)$ sont les K meilleures configurations trouvées par $\pi_k$
- $f_{\text{sim}}$ est la simulation exacte (Hamiltonian diagonalization)

Le surrogate $\hat{f}_k$ est entraîné sur $\mathcal{D}_k$ et sert de reward pour le RL :

$$r_k(s, a) = \hat{f}_k(s, a)$$

### Ce qui doit être créé

#### `packages/ml/active_learning.py`

```python
"""Active learning loop for iterative surrogate-RL co-training.

Implements the research-grade active learning pipeline:

    1. Train surrogate S_0 on dataset D_0 (from data_generator)
    2. Train RL policy π_0 using S_0 as fast simulator
    3. Extract top-K diverse configurations from π_0 rollouts
    4. Re-simulate top-K with REAL simulator → new training points
    5. D_1 = D_0 ∪ new_points
    6. Train S_1 on D_1
    7. Train π_1 using S_1
    8. Repeat for N_iterations

Diversity selection:
    We don't just take the top-K highest-reward actions. We want
    DIVERSE configurations that cover different regions of parameter space.
    Use greedy farthest-point sampling in feature space after ranking by
    predicted reward.

Uncertainty-guided acquisition (if ensemble available):
    Prioritize configurations where the surrogate ensemble has HIGH
    uncertainty. These are regions where more simulation data would
    be most valuable.

    acquisition_score = α * predicted_reward + (1-α) * uncertainty

References:
    Settles, "Active Learning Literature Survey" (2009)
    Cohn et al., "Active Learning with Statistical Models" (JAIR 1996)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from packages.core.logging import get_logger
from packages.core.models import (
    ExperimentSpec,
    RegisterCandidate,
    SequenceCandidate,
)

logger = get_logger(__name__)


@dataclass
class ActiveLearningConfig:
    """Configuration for the active learning loop."""
    n_iterations: int = 5               # Number of surrogate-RL cycles
    top_k_per_iteration: int = 200      # Configs to re-simulate per iteration
    diversity_fraction: float = 0.5     # Fraction selected by diversity vs. reward
    uncertainty_weight: float = 0.3     # Weight on uncertainty for acquisition
    
    # Surrogate training per iteration
    surrogate_epochs: int = 50          # Fewer epochs per iteration (fine-tuning)
    surrogate_lr: float = 5e-4          # Lower LR for fine-tuning
    
    # RL training per iteration
    rl_updates: int = 100               # Fewer updates per iteration
    rl_max_steps: int = 5
    rl_rollout_episodes: int = 200      # Episodes to collect for acquisition
    
    # Simulation
    max_atoms_for_resim: int = 12       # Safety limit for re-simulation
    n_workers: int = 4                  # Parallel workers for re-simulation
    
    # Persistence
    checkpoint_dir: str = "checkpoints/active_learning"
    seed: int = 42


class ActiveLearningLoop:
    """Iterative surrogate-RL co-training with simulation-based data augmentation.
    
    Usage::
    
        config = ActiveLearningConfig(n_iterations=5, top_k_per_iteration=200)
        loop = ActiveLearningLoop(
            config=config,
            initial_dataset=dataset,       # From DatasetGenerator (100k+ pts)
            spec=spec,
            register_candidates=registers,
        )
        results = loop.run()
        # results contains per-iteration metrics, final model checkpoints
    """
    
    def __init__(
        self,
        config: ActiveLearningConfig,
        initial_features: NDArray[np.float32],   # (N, D)
        initial_targets: NDArray[np.float32],     # (N, 4)
        spec: ExperimentSpec,
        register_candidates: list[RegisterCandidate],
        param_space: Any | None = None,
        simulate_fn: Callable | None = None,      # Custom simulator (for testing)
    ):
        self.config = config
        self.features = initial_features.copy()
        self.targets = initial_targets.copy()
        self.spec = spec
        self.register_candidates = register_candidates
        self.param_space = param_space
        self.simulate_fn = simulate_fn
        self._rng = np.random.default_rng(config.seed)
    
    def run(self) -> dict[str, Any]:
        """Execute the full active learning loop.
        
        Returns:
            dict with keys:
            - iterations: list of per-iteration metrics
            - final_surrogate_path: path to final surrogate checkpoint
            - final_rl_path: path to final RL checkpoint
            - dataset_growth: list of dataset sizes per iteration
            - surrogate_improvement: list of val_loss per iteration
        """
        results: dict[str, Any] = {
            "iterations": [],
            "dataset_growth": [len(self.features)],
        }
        
        for iteration in range(self.config.n_iterations):
            logger.info(
                "=== Active Learning iteration %d/%d (dataset: %d samples) ===",
                iteration + 1, self.config.n_iterations, len(self.features),
            )
            
            iter_result = self._run_iteration(iteration)
            results["iterations"].append(iter_result)
            results["dataset_growth"].append(len(self.features))
            
            logger.info(
                "Iteration %d complete: +%d new samples, surrogate val_loss=%.6f",
                iteration + 1,
                iter_result["new_samples"],
                iter_result.get("surrogate_val_loss", float("nan")),
            )
        
        results["final_surrogate_path"] = str(
            Path(self.config.checkpoint_dir) / "surrogate_final.pt"
        )
        results["final_rl_path"] = str(
            Path(self.config.checkpoint_dir) / "ppo_final.pt"
        )
        
        return results
    
    def _run_iteration(self, iteration: int) -> dict[str, Any]:
        """Single active learning iteration.
        
        Steps:
        1. Train/fine-tune surrogate on current dataset
        2. Train RL policy using surrogate as simulator
        3. Collect diverse high-reward configurations from RL rollouts
        4. Re-simulate selected configurations with real simulator
        5. Add new (feature, target) pairs to dataset
        """
        import torch
        from torch.utils.data import TensorDataset
        from packages.ml.surrogate import SurrogateModelV2, SurrogateTrainer
        from packages.ml.training_runner import TrainingConfig, TrainingRunner
        
        checkpoint_dir = Path(self.config.checkpoint_dir) / f"iter_{iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Step 1: Train surrogate ---
        X_tensor = torch.from_numpy(self.features)
        Y_tensor = torch.from_numpy(self.targets)
        n_val = max(1, len(self.features) // 10)
        indices = torch.randperm(len(self.features))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        train_ds = TensorDataset(X_tensor[train_idx], Y_tensor[train_idx])
        val_ds = TensorDataset(X_tensor[val_idx], Y_tensor[val_idx])
        
        input_dim = self.features.shape[1]
        surrogate = SurrogateModelV2(input_dim=input_dim)
        
        # Load previous iteration's weights for warm-start (if not first iteration)
        prev_checkpoint = checkpoint_dir.parent / f"iter_{iteration - 1}" / "surrogate.pt"
        if iteration > 0 and prev_checkpoint.exists():
            surrogate.load(str(prev_checkpoint))
            logger.info("Warm-starting surrogate from iteration %d", iteration - 1)
        
        trainer = SurrogateTrainer(
            surrogate, lr=self.config.surrogate_lr,
        )
        surr_history = trainer.fit(
            train_ds, val_ds,
            epochs=self.config.surrogate_epochs,
            batch_size=64,
        )
        surrogate.save(str(checkpoint_dir / "surrogate.pt"))
        
        # Copy to final location
        surrogate.save(str(Path(self.config.checkpoint_dir) / "surrogate_final.pt"))
        
        # --- Step 2: Train RL with surrogate ---
        surrogate.eval()
        from packages.ml.dataset import build_feature_vector
        
        def surrogate_sim(register, params):
            from packages.core.enums import SequenceFamily
            family_enum = params.get("family_enum", SequenceFamily.ADIABATIC_SWEEP)
            seq = SequenceCandidate(
                campaign_id="active_learning",
                spec_id=self.spec.id,
                register_candidate_id=register.id,
                label="al-surrogate",
                sequence_family=family_enum,
                duration_ns=params["duration_ns"],
                amplitude=params["amplitude"],
                detuning=params["detuning"],
                phase=0.0,
                waveform_kind="constant",
                predicted_cost=0.0,
                reasoning_summary="Active learning surrogate eval.",
            )
            features = build_feature_vector(register, seq)
            return float(surrogate.predict_robustness(features))
        
        config = TrainingConfig(
            checkpoint_dir=str(checkpoint_dir),
            rl_total_updates=self.config.rl_updates,
            rl_max_steps=self.config.rl_max_steps,
        )
        runner = TrainingRunner(config)
        rl_result = runner.run_rl(
            self.spec, self.register_candidates,
            simulate_fn=surrogate_sim,
        )
        
        # Copy RL checkpoint to final location
        import shutil
        rl_src = checkpoint_dir / "ppo_latest.pt"
        if rl_src.exists():
            shutil.copy2(rl_src, Path(self.config.checkpoint_dir) / "ppo_final.pt")
        
        # --- Step 3: Collect diverse configurations from RL ---
        from packages.ml.ppo import ActorCritic
        from packages.ml.rl_env import PulseDesignEnv, rescale_action
        
        policy = ActorCritic()
        policy.load(str(rl_src))
        policy.eval()
        
        collected_configs = self._collect_rl_configurations(policy, surrogate)
        
        # --- Step 4: Select diverse top-K for re-simulation ---
        selected = self._select_diverse_configs(collected_configs)
        
        # --- Step 5: Re-simulate and add to dataset ---
        new_samples = self._resimulate_configs(selected)
        
        # Update dataset
        if new_samples:
            new_features = np.stack([s[0] for s in new_samples])
            new_targets = np.stack([s[1] for s in new_samples])
            self.features = np.concatenate([self.features, new_features])
            self.targets = np.concatenate([self.targets, new_targets])
        
        return {
            "iteration": iteration,
            "surrogate_val_loss": surr_history["val_loss"][-1] if surr_history.get("val_loss") else None,
            "surrogate_train_loss": surr_history["train_loss"][-1],
            "rl_episodes": len(rl_result["history"]["episode_rewards"]),
            "rl_avg_reward": (
                np.mean(rl_result["history"]["episode_rewards"][-20:])
                if rl_result["history"]["episode_rewards"] else 0.0
            ),
            "configs_collected": len(collected_configs),
            "configs_selected": len(selected),
            "new_samples": len(new_samples),
            "dataset_size": len(self.features),
        }
    
    def _collect_rl_configurations(
        self, policy, surrogate,
    ) -> list[dict[str, Any]]:
        """Run RL policy and collect (config, predicted_reward) pairs."""
        from packages.ml.rl_env import PulseDesignEnv, rescale_action
        
        env = PulseDesignEnv(
            spec=self.spec,
            register_candidates=self.register_candidates,
            max_steps=self.config.rl_max_steps,
            simulate_fn=lambda r, p: float(surrogate.predict_robustness(
                build_feature_vector(r, self._params_to_seq(r, p))
            )),
        )
        
        configs = []
        for episode in range(self.config.rl_rollout_episodes):
            obs, info = env.reset(seed=self.config.seed + episode)
            episode_best = None
            
            for step in range(self.config.rl_max_steps):
                action, _, _ = policy.get_action(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if episode_best is None or reward > episode_best["reward"]:
                    episode_best = {
                        "params": info["params"],
                        "reward": reward,
                        "register_id": env._current_register.id if env._current_register else None,
                        "register": env._current_register,
                    }
                
                if terminated:
                    break
            
            if episode_best and episode_best["register"] is not None:
                configs.append(episode_best)
        
        return configs
    
    def _select_diverse_configs(
        self, configs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Select diverse top-K configurations using farthest-point sampling.
        
        Strategy:
        1. Sort by predicted reward (descending)
        2. Take top-50% by reward
        3. From these, use greedy farthest-point sampling in feature space
           to select the most DIVERSE subset
        
        This ensures we get both high-quality AND diverse re-simulation targets.
        """
        if not configs:
            return []
        
        k = min(self.config.top_k_per_iteration, len(configs))
        
        # Sort by reward
        sorted_configs = sorted(configs, key=lambda c: c["reward"], reverse=True)
        
        # Take top candidates by reward
        reward_pool_size = max(k, len(sorted_configs) // 2)
        reward_pool = sorted_configs[:reward_pool_size]
        
        if len(reward_pool) <= k:
            return reward_pool[:k]
        
        # Greedy farthest-point sampling for diversity
        from packages.ml.dataset import build_feature_vector
        
        feature_vecs = []
        for config in reward_pool:
            reg = config["register"]
            seq = self._params_to_seq(reg, config["params"])
            feature_vecs.append(build_feature_vector(reg, seq))
        
        feature_matrix = np.stack(feature_vecs)
        
        # Normalize features for distance computation
        f_std = feature_matrix.std(axis=0)
        f_std[f_std < 1e-8] = 1.0
        normalized = (feature_matrix - feature_matrix.mean(axis=0)) / f_std
        
        # Greedy farthest-point selection
        selected_indices = [0]  # Start with the best-reward config
        for _ in range(k - 1):
            min_dists = np.full(len(normalized), np.inf)
            for sel_idx in selected_indices:
                dists = np.linalg.norm(normalized - normalized[sel_idx], axis=1)
                min_dists = np.minimum(min_dists, dists)
            
            # Mask already selected
            min_dists[selected_indices] = -np.inf
            
            # Select the farthest point
            next_idx = int(np.argmax(min_dists))
            selected_indices.append(next_idx)
        
        return [reward_pool[i] for i in selected_indices]
    
    def _params_to_seq(
        self, register: RegisterCandidate, params: dict[str, Any],
    ) -> SequenceCandidate:
        """Convert raw params dict to SequenceCandidate for feature extraction."""
        from packages.core.enums import SequenceFamily
        family_enum = params.get("family_enum", SequenceFamily.ADIABATIC_SWEEP)
        return SequenceCandidate(
            campaign_id="active_learning",
            spec_id=self.spec.id,
            register_candidate_id=register.id,
            label="al-resim",
            sequence_family=family_enum,
            duration_ns=params["duration_ns"],
            amplitude=params["amplitude"],
            detuning=params["detuning"],
            phase=0.0,
            waveform_kind="constant",
            predicted_cost=0.0,
            reasoning_summary="Active learning re-simulation candidate.",
        )
    
    def _resimulate_configs(
        self, configs: list[dict[str, Any]],
    ) -> list[tuple[NDArray, NDArray]]:
        """Re-simulate selected configs with the REAL simulator.
        
        Returns list of (feature_vector, target_vector) pairs.
        """
        from packages.ml.dataset import build_feature_vector
        from packages.simulation.evaluators import evaluate_candidate_robustness
        
        results = []
        
        for config in configs:
            register = config["register"]
            params = config["params"]
            
            # Skip large systems if configured
            if register.atom_count > self.config.max_atoms_for_resim:
                continue
            
            seq = self._params_to_seq(register, params)
            
            try:
                if self.simulate_fn is not None:
                    # Custom simulate_fn for testing
                    reward = float(self.simulate_fn(register, params))
                    feature = build_feature_vector(register, seq)
                    target = np.array([reward, reward, reward * 0.8, reward * 0.9], dtype=np.float32)
                else:
                    (
                        nominal, scenario_scores, avg, wc, std, penalty,
                        robustness, nom_obs, scen_obs, ham_metrics,
                    ) = evaluate_candidate_robustness(self.spec, register, seq)
                    
                    feature = build_feature_vector(register, seq)
                    observable_score = float(nom_obs.get("observable_score", nominal))
                    target = np.array([
                        robustness, nominal, wc, observable_score,
                    ], dtype=np.float32)
                
                results.append((feature, target))
                
            except Exception as exc:
                logger.warning("Re-simulation failed for config: %s", exc)
                continue
        
        logger.info(
            "Re-simulated %d/%d configurations successfully",
            len(results), len(configs),
        )
        return results
```

### Intégration dans `scripts/train_ml.py`

Ajouter une phase `active` :

```python
parser.add_argument("--phase", choices=[
    "generate", "generate_v2", "surrogate", "rl", "full", "active",
], required=True)

# New args for active learning
parser.add_argument("--al-iterations", type=int, default=5)
parser.add_argument("--al-top-k", type=int, default=200)

elif args.phase == "active":
    from packages.ml.active_learning import ActiveLearningLoop, ActiveLearningConfig
    from packages.ml.dataset import CandidateDatasetBuilder
    
    builder = CandidateDatasetBuilder()
    builder.load(args.data)
    X, Y = builder.to_numpy()
    
    # Setup spec and registers
    goal = ExperimentGoal(...)
    spec = ProblemFramingAgent().run(goal)
    registers = GeometryAgent().run(spec, "active_training", memory_records=[])
    
    config = ActiveLearningConfig(
        n_iterations=args.al_iterations,
        top_k_per_iteration=args.al_top_k,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    loop = ActiveLearningLoop(
        config=config,
        initial_features=X,
        initial_targets=Y,
        spec=spec,
        register_candidates=registers,
    )
    results = loop.run()
    
    # Save the enriched dataset
    np.savez(
        args.data.replace(".npz", "_enriched.npz"),
        features=loop.features,
        targets=loop.targets,
    )
    
    for iter_info in results["iterations"]:
        logger.info("Iteration %d: +%d samples, val_loss=%.6f",
            iter_info["iteration"],
            iter_info["new_samples"],
            iter_info.get("surrogate_val_loss", float("nan")),
        )
```

### Intégration dans `packages/ml/training_runner.py`

Ajouter une méthode `run_active_learning` :

```python
class TrainingRunner:
    def run_active_learning(
        self,
        initial_features: NDArray,
        initial_targets: NDArray,
        spec: Any,
        register_candidates: list[Any],
        n_iterations: int = 5,
        top_k: int = 200,
    ) -> dict[str, Any]:
        """Run the full active learning loop.
        
        Phase 1: Initial surrogate training
        Phase 2: Iterative surrogate→RL→re-simulate cycles
        """
        from packages.ml.active_learning import ActiveLearningLoop, ActiveLearningConfig
        
        config = ActiveLearningConfig(
            n_iterations=n_iterations,
            top_k_per_iteration=top_k,
            checkpoint_dir=self.config.checkpoint_dir,
            seed=self.config.seed,
        )
        
        loop = ActiveLearningLoop(
            config=config,
            initial_features=initial_features,
            initial_targets=initial_targets,
            spec=spec,
            register_candidates=register_candidates,
        )
        
        return loop.run()
```

### Tests requis

```python
def test_active_learning_basic():
    """Active learning loop runs and grows the dataset."""
    # Create small initial dataset
    X = np.random.randn(100, 10).astype(np.float32)
    Y = np.random.rand(100, 4).astype(np.float32)
    
    config = ActiveLearningConfig(
        n_iterations=2,
        top_k_per_iteration=10,
        rl_updates=5,
        rl_rollout_episodes=10,
        surrogate_epochs=3,
    )
    
    # Mock simulator
    def mock_sim(register, params):
        return min(1.0, params["amplitude"] / 15.0)
    
    loop = ActiveLearningLoop(
        config=config,
        initial_features=X,
        initial_targets=Y,
        spec=spec,
        register_candidates=registers,
        simulate_fn=mock_sim,
    )
    
    results = loop.run()
    assert len(loop.features) > 100  # Dataset grew
    assert len(results["iterations"]) == 2

def test_diversity_selection():
    """Farthest-point sampling selects diverse configs."""
    loop = ActiveLearningLoop(...)
    configs = [
        {"params": {"amplitude": 5.0, ...}, "reward": 0.8, "register": reg},
        {"params": {"amplitude": 5.1, ...}, "reward": 0.79, "register": reg},  # Similar
        {"params": {"amplitude": 12.0, ...}, "reward": 0.75, "register": reg}, # Different
    ]
    selected = loop._select_diverse_configs(configs)
    # Should prefer the diverse config over the similar one
    assert len(selected) <= loop.config.top_k_per_iteration
```

---

---

## P2 — CRÉDIBILITÉ RECHERCHE

### P2.9 — Benchmarks reproductibles

#### Objectif

Créer un script de benchmark standardisé qui mesure des métriques fixes sur des datasets fixes, permettant de comparer objectivement les améliorations.

#### Créer `scripts/benchmark.py`

```python
"""Standardized benchmark suite for CryoSwarm-Q ML models.

Measures and records:
1. Surrogate: MSE, MAE, R², per-target metrics on a FIXED test set
2. RL: avg episode reward, best episode reward, convergence speed
3. Pipeline: top candidate robustness, mean ranked robustness
4. Active learning: surrogate improvement rate, dataset efficiency

Usage:
    python -m scripts.benchmark --model checkpoints/surrogate_latest.pt --test-data data/test_set.npz
    python -m scripts.benchmark --rl-checkpoint checkpoints/ppo_latest.pt --env-atoms 6
    python -m scripts.benchmark --full --checkpoint-dir checkpoints/

Outputs JSON report to benchmarks/results_{timestamp}.json
"""
```

Métriques à calculer :

```python
@dataclass
class SurrogateBenchmark:
    mse_total: float              # Overall MSE
    mae_total: float              # Overall MAE
    r2_total: float               # R² score
    mse_per_target: dict[str, float]  # Per-target MSE {robustness, nominal, worst_case, observable}
    mae_per_target: dict[str, float]
    r2_per_target: dict[str, float]
    calibration_error: float      # |mean(pred) - mean(true)|
    worst_prediction_error: float # Max absolute error
    n_test_samples: int

@dataclass
class RLBenchmark:
    avg_reward_last_50: float
    best_episode_reward: float
    episodes_to_threshold: int | None  # Episodes to reach reward > 0.3
    reward_std: float
    n_episodes: int

@dataclass
class PipelineBenchmark:
    top_candidate_robustness: float
    mean_ranked_robustness: float
    n_candidates_evaluated: int
    heuristic_vs_rl_comparison: dict[str, float] | None  # If hybrid mode

@dataclass
class FullBenchmark:
    surrogate: SurrogateBenchmark
    rl: RLBenchmark
    pipeline: PipelineBenchmark
    timestamp: str
    git_hash: str | None
    config: dict[str, Any]
```

### P2.10 — Ablation studies

#### Créer `scripts/ablation.py`

```python
"""Ablation study runner for CryoSwarm-Q.

Compares system performance with and without each component:

1. Heuristic only vs. surrogate-filtered vs. RL vs. hybrid
2. With vs. without reward shaping
3. With vs. without curriculum learning 
4. With vs. without active learning
5. V1 features (10-dim) vs. V2 features (18-dim)
6. Single model vs. ensemble (3 models)

Each ablation:
- Trains the specified configuration
- Runs the benchmark suite
- Saves results to ablations/

Usage:
    python -m scripts.ablation --ablation all --data data/candidates.npz
    python -m scripts.ablation --ablation features --data data/candidates.npz
    python -m scripts.ablation --ablation architecture --data data/candidates.npz
"""
```

Ablation configurations :

```python
ABLATION_CONFIGS = {
    "baseline_heuristic": {
        "description": "Heuristic sequence agent only, no ML",
        "use_surrogate": False,
        "use_rl": False,
    },
    "surrogate_filter_v1": {
        "description": "Surrogate pre-filter with V1 features (10-dim)",
        "use_surrogate": True,
        "feature_version": "v1",
        "use_rl": False,
    },
    "surrogate_filter_v2": {
        "description": "Surrogate pre-filter with V2 features (18-dim)",
        "use_surrogate": True,
        "feature_version": "v2",
        "use_rl": False,
    },
    "rl_single_step": {
        "description": "RL with max_steps=1 (original)",
        "use_rl": True,
        "rl_max_steps": 1,
        "reward_shaping": False,
    },
    "rl_multi_step": {
        "description": "RL with max_steps=5 and reward shaping",
        "use_rl": True,
        "rl_max_steps": 5,
        "reward_shaping": True,
    },
    "rl_curriculum": {
        "description": "RL with curriculum learning",
        "use_rl": True,
        "rl_max_steps": 5,
        "use_curriculum": True,
    },
    "hybrid_heuristic_rl": {
        "description": "Hybrid mode: heuristic + RL candidates together",
        "use_rl": True,
        "strategy_mode": "hybrid",
    },
    "ensemble_3": {
        "description": "Ensemble of 3 surrogate models",
        "use_surrogate": True,
        "use_ensemble": True,
        "n_ensemble": 3,
    },
    "full_pipeline": {
        "description": "Everything enabled: ensemble + RL + curriculum + active learning",
        "use_surrogate": True,
        "use_ensemble": True,
        "use_rl": True,
        "use_curriculum": True,
        "strategy_mode": "adaptive",
    },
}
```

### P2.11 — Physics-informed features (déjà dans P0)

Les features v2 (18-dim) avec ratios physiques ($\Omega/U$, $\delta/\Omega$, adiabaticité, blockade fraction) sont spécifiées dans P0.3. Ici on spécifie leur **validation** :

#### Créer `tests/test_physics_features.py`

```python
"""Validate that physics-informed features are physically meaningful."""

def test_omega_over_interaction_scales_correctly():
    """Ω/U ratio increases when Ω increases or spacing increases."""
    # High Ω, close spacing → Ω/U ~ 1 (Rydberg regime)
    # Low Ω, close spacing → Ω/U << 1 (blockade regime)
    reg_close = make_register(spacing_um=5.0, atom_count=4)
    seq_high_omega = make_sequence(amplitude=10.0)
    seq_low_omega = make_sequence(amplitude=1.0)
    
    f_high = build_feature_vector_v2(reg_close, seq_high_omega)
    f_low = build_feature_vector_v2(reg_close, seq_low_omega)
    
    # Ω/U ratio is feature index 8
    assert f_high[8] > f_low[8]

def test_adiabaticity_scales_with_duration():
    """Longer durations give higher adiabaticity parameter."""
    reg = make_register(spacing_um=7.0, atom_count=6)
    seq_long = make_sequence(duration_ns=5000, amplitude=5.0)
    seq_short = make_sequence(duration_ns=1000, amplitude=5.0)
    
    f_long = build_feature_vector_v2(reg, seq_long)
    f_short = build_feature_vector_v2(reg, seq_short)
    
    # Adiabaticity T×Ω is feature index 10
    assert f_long[10] > f_short[10]

def test_blockade_fraction_ring_vs_line():
    """Ring geometry has higher blockade fraction than line for same atoms."""
    reg_ring = make_register(layout="ring", spacing_um=7.0, atom_count=6)
    reg_line = make_register(layout="line", spacing_um=7.0, atom_count=6)
    seq = make_sequence()
    
    f_ring = build_feature_vector_v2(reg_ring, seq)
    f_line = build_feature_vector_v2(reg_line, seq)
    
    # Blockade fraction is feature index 7
    # Ring should have more pairs within blockade radius
    assert f_ring[7] >= f_line[7]

def test_feature_v2_dimension():
    """Feature vector v2 has exactly 18 dimensions."""
    reg = make_register()
    seq = make_sequence()
    f = build_feature_vector_v2(reg, seq)
    assert f.shape == (18,)
    assert f.dtype == np.float32

def test_feature_v2_all_finite():
    """No NaN or Inf in feature vectors."""
    for layout in ["square", "line", "triangular", "ring", "zigzag", "honeycomb"]:
        for atoms in [3, 6, 10, 15]:
            reg = make_register(layout=layout, atom_count=atoms)
            seq = make_sequence()
            f = build_feature_vector_v2(reg, seq)
            assert np.all(np.isfinite(f)), f"Non-finite in {layout}/{atoms}"
```

### P2.12 — Noise model spatial

#### Objectif

Ajouter l'inhomogénéité spatiale du drive laser (realistic pour les tweezer arrays) aux profils de bruit.

#### Modifier `packages/simulation/noise_profiles.py`

Ajouter un champ d'inhomogénéité spatiale :

```python
def _make_scenario(label, ..., spatial_inhomogeneity=0.0):
    """Factory with optional spatial drive inhomogeneity."""
    scenario = NoiseScenario(
        label=label,
        amplitude_jitter=amplitude_jitter,
        ...,
        metadata={"spatial_inhomogeneity": spatial_inhomogeneity},
    )
    return scenario

def default_noise_scenarios():
    return [
        _make_scenario(NoiseLevel.LOW, ..., spatial_inhomogeneity=0.02),
        _make_scenario(NoiseLevel.MEDIUM, ..., spatial_inhomogeneity=0.05),
        _make_scenario(NoiseLevel.STRESSED, ..., spatial_inhomogeneity=0.08),
    ]
```

#### Modifier `packages/simulation/evaluators.py` — Appliquer l'inhomogénéité

Dans `_simulate_with_numpy_fallback()`, appliquer une variation spatiale du drive :

```python
def _simulate_with_numpy_fallback(spec, register, sequence, noise_scenario=None):
    ...
    if noise_scenario is not None:
        rng = np.random.default_rng()
        # Existing jitter (global)
        omega_max *= 1.0 + rng.normal(0.0, noise_scenario.amplitude_jitter)
        
        # NEW: spatial inhomogeneity — per-atom Ω variation
        spatial_inhom = noise_scenario.metadata.get("spatial_inhomogeneity", 0.0)
        if spatial_inhom > 0:
            n_atoms = register.atom_count
            # Each atom sees a slightly different Rabi frequency
            # Modeled as Gaussian with std = spatial_inhom * omega_max
            spatial_factors = 1.0 + rng.normal(0.0, spatial_inhom, size=n_atoms)
            # Store in metadata for the simulation backend to use
            # (Currently numpy_backend uses uniform Ω, this prepares for future per-site Ω)
    ...
```

---

---

## RÉSUMÉ DES FICHIERS

### Fichiers à créer

| Fichier | Description |
|---------|-------------|
| `packages/ml/curriculum.py` | CurriculumScheduler — progression progressive de difficulté |
| `packages/ml/active_learning.py` | ActiveLearningLoop — boucle itérative surrogate↔RL |
| `scripts/benchmark.py` | Suite de benchmarks reproductibles |
| `scripts/ablation.py` | Runner d'études d'ablation |
| `tests/test_physics_features.py` | Validation des features physiquement informées |

### Fichiers à modifier

| Fichier | Modification |
|---------|-------------|
| `packages/ml/surrogate.py` | Normalizer intégré au save/load, ResidualBlock, SurrogateModelV2, SurrogateEnsemble, EnsembleTrainer |
| `packages/ml/surrogate_filter.py` | Support ensemble, uncertainty-based rejection |
| `packages/ml/rl_env.py` | OBS_DIM=16, reward shaping, multi-step, episode history |
| `packages/ml/ppo.py` | PPOTrainer avec curriculum, PPOConfig defaults mis à jour |
| `packages/ml/training_runner.py` | TrainingConfig (rl_max_steps, use_curriculum), run_active_learning() |
| `packages/simulation/noise_profiles.py` | Spatial inhomogeneity dans les scénarios |
| `packages/simulation/evaluators.py` | Spatial noise dans numpy fallback |
| `scripts/train_ml.py` | Phase `active`, args pour active learning |

### Tests à créer

| Test | Vérifie |
|------|---------|
| `tests/test_surrogate_v2.py` | ResidualBlock, SurrogateModelV2 forward, ensemble, save/load, normalizer |
| `tests/test_multi_step_rl.py` | Épisodes multi-step, reward shaping, OBS_DIM=16, backward compat |
| `tests/test_curriculum.py` | Stage progression, register filtering, adaptive/linear modes |
| `tests/test_active_learning.py` | Loop execution, diversity selection, dataset growth |
| `tests/test_physics_features.py` | Feature v2 correctness, physical scaling laws |
| `tests/test_benchmark.py` | Benchmark metrics computation |

---

## ORDRE D'EXÉCUTION

1. **P1.4** : Modifier `surrogate.py` pour normalizer intégré (save/load + SurrogateTrainer)
2. **P1.5** : Ajouter ResidualBlock, SurrogateModelV2, SurrogateEnsemble à `surrogate.py` ; modifier `surrogate_filter.py`
3. Tester P1.4 + P1.5
4. **P1.6** : Modifier `rl_env.py` (OBS_DIM=16, multi-step, reward shaping) ; adapter `ppo.py`
5. Tester P1.6
6. **P1.7** : Créer `curriculum.py` ; intégrer dans `ppo.py` et `training_runner.py`
7. Tester P1.7
8. **P1.8** : Créer `active_learning.py` ; intégrer dans `training_runner.py` et `train_ml.py`
9. Tester P1.8
10. **P2.9** : Créer `scripts/benchmark.py`
11. **P2.10** : Créer `scripts/ablation.py`
12. **P2.11** : Créer `tests/test_physics_features.py`
13. **P2.12** : Modifier `noise_profiles.py` et `evaluators.py`
14. Lancer `py -m pytest tests/ -v` pour vérifier TOUS les tests passent

---

## CONTRAINTES

1. **Rétrocompatibilité** : Le `SurrogateModel` V1 reste fonctionnel. Les anciens checkpoints (raw state_dict) chargent correctement. `PulseDesignEnv(max_steps=1, reward_shaping=False)` reproduit l'ancien comportement. `PPOTrainer(curriculum=None)` ne change rien.

2. **Pas de nouvelles dépendances** : NumPy, SciPy, PyTorch (optionnel) seulement. Pas de gym, stable-baselines, or autre framework RL.

3. **Performance** : Le `SurrogateModelV2` avec 25k paramètres doit inférer en < 2ms pour 1000 samples. L'ensemble de 3 modèles doit rester < 5ms.

4. **Reproductibilité** : Tous les RNG seedés. Curriculum progression déterministe. Active learning reproductible avec même seed.

5. **Physique correcte** : Le reward shaping ne doit pas biaiser l'optimum trouvé (le raw_reward reste la robustness_score). Le curriculum ne doit pas empêcher l'agent de voir de grands systèmes en stage final. L'inhomogénéité spatiale doit être physiquement réaliste (2-8% variation pour tweezer arrays).

---

## CRITÈRES DE SUCCÈS

P1 est atteint quand :

1. Le surrogate V2 (128-dim, 3 residual blocks, LayerNorm, dropout 0.1) converge sur 100k points avec val_loss < V1
2. L'ensemble de 3 modèles produit des estimations d'incertitude non-triviales ($\sigma > 0$)
3. Le RL multi-step (5 steps) montre une amélioration intra-épisode (le reward au step 5 est en moyenne meilleur qu'au step 1)
4. Le curriculum accélère la convergence RL : l'agent atteint reward > 0.3 plus vite qu'en mode full-range
5. L'active learning enrichit le dataset ET améliore la val_loss du surrogate à chaque itération

P2 est atteint quand :

6. `python -m scripts.benchmark --full` produit un rapport JSON avec MSE, MAE, R² par cible
7. `python -m scripts.ablation --ablation all` montre que chaque composant ajouté apporte un gain mesurable
8. Les features v2 passent tous les tests de sanity physique (scaling laws de $\Omega/U$, adiabaticité, blockade fraction)
9. Le bruit spatial est intégré dans les profils MEDIUM et STRESSED
10. Tous les tests passent, anciens + nouveaux
