# P0 — PROMPT D'IMPLÉMENTATION CRITIQUE

## Contexte du projet

CryoSwarm-Q est un système multi-agent hardware-aware pour l'expérimentation quantique sur atomes neutres (neutral-atom quantum computing). Le backend physique est basé sur 87-Rb avec interaction van der Waals ($C_6 = 862690 \text{ rad·µm}^6/\text{µs}$), programmation pulse-level via Pulser/Pasqal.

Le projet a actuellement :
- 8 agents spécialisés (ProblemFraming, Geometry, Sequence, Noise, Routing, Campaign, Results, Memory)
- Un pipeline d'orchestration séquentiel (`packages/orchestration/pipeline.py`)
- Un hamiltonien Rydberg exact (`packages/simulation/hamiltonian.py`)
- Un évaluateur multi-scénarios de robustesse (`packages/simulation/evaluators.py`)
- Un surrogate MLP basique (`packages/ml/surrogate.py`) — Phase 1
- Un PPO basique (`packages/ml/ppo.py`) — Phase 2
- Un runner de training (`packages/ml/training_runner.py`) — Phase 3

**Problème critique** : Le training ML est au stade proof-of-concept. Les paramètres physiques sont hardcodés dans les agents au lieu d'être appris. Le RL est déconnecté du pipeline. Les datasets sont minuscules et auto-référents. Rien n'est au niveau recherche.

---

## OBJECTIF DE CE PROMPT

Implémenter les 3 actions P0 suivantes **dans cet ordre exact** :

1. **Rendre les paramètres physiques configurables** — Extraire TOUS les paramètres hardcodés dans un `PhysicsParameterSpace` configurable
2. **Connecter le RL au pipeline avec un vrai switch** — Le pipeline doit choisir intelligemment entre heuristique et RL
3. **Construire un vrai dataset generator capable de produire 100k+ points**

Chaque section ci-dessous détaille exactement ce qui doit être créé, modifié, et pourquoi.

---

---

## P0.1 — PHYSICS PARAMETER SPACE CONFIGURABLE

### Problème actuel

Le `SequenceAgent` dans `packages/agents/sequence_agent.py` contient **30+ paramètres physiques hardcodés** directement dans le code source :

```python
# ADIABATIC — lignes actuelles dans _adiabatic_variants()
(3000 + offset, 5.0, -20.0, 0.0, "adiabatic_conservative", {"amplitude_start": 0.5, "detuning_end": 10.0})
(4000 + offset, 7.5, -25.0, 0.0, "adiabatic_extended", {"amplitude_start": 0.8, "detuning_end": 15.0})

# DETUNING SCAN — dans _detuning_scan_variants()
(2000 + offset, 4.0, -15.0, 0.0, "detuning_scan_fast", {"detuning_end": 5.0})
(3000 + offset, 6.0, -20.0, 0.0, "detuning_scan_wide", {"detuning_end": 10.0})

# GLOBAL RAMP — dans _global_ramp_variants()
(2000 + offset, 6.0, -12.0, 0.0, "global_ramp_compact", {"amplitude_start": 0.5})
(3000 + offset, 8.0, -18.0, 0.0, "global_ramp_extended", {"amplitude_start": 1.0})

# CONSTANT DRIVE — dans _constant_drive_variants()
(1200 + offset, 5.0, 0.0, 0.0, "constant_resonant", {})
(1500 + offset, 5.0, -5.0, 0.0, "constant_detuned", {})

# BLACKMAN SWEEP — dans _blackman_sweep_variants()
(3000 + offset, 7.0, -20.0, 0.0, "blackman_sweep_standard", {"detuning_end": 10.0})
(4000 + offset, 8.5, -30.0, 0.0, "blackman_sweep_wide", {"detuning_end": 15.0})
```

Autres paramètres hardcodés :
- `duration_offset = atom_count * 150` (formule linéaire arbitraire)
- `predicted_cost = min(1.0, 2**atom_count * duration_ns / 500000.0)`
- Dans `rl_env.py` : `AMP_RANGE = (1.0, 15.0)`, `DET_RANGE = (-30.0, 15.0)`, `DUR_RANGE = (500.0, 5500.0)`
- Dans `GeometryAgent` : `min_spacing_um = 5.0`, les poids de feasibility `0.55 + 0.20 * margin + 0.20 * blockade - 0.04 * (atoms - 6)`
- Dans `noise_profiles.py` : les 3 scénarios LOW/MEDIUM/STRESSED avec des valeurs en dur
- Dans `evaluators.py` : les poids `0.7 * density_score + 0.3 * blockade_score`
- Dans `robustness.py` : les poids `0.25 * nominal + 0.35 * avg + 0.30 * worst + 0.10 * stability`

### Ce qui doit être créé

#### Fichier : `packages/core/parameter_space.py`

Créer un module `PhysicsParameterSpace` qui centralise **tous** les paramètres physiques configurables. Ce module doit :

1. Définir des **ranges physiquement motivés** (min, max, default, unit, description) pour chaque paramètre
2. Permettre le **sampling** aléatoire pour la génération de données
3. Supporter le **grid search** systématique
4. Supporter le **Latin Hypercube Sampling** (LHS) pour une couverture efficace de l'espace
5. Être sérialisable en JSON/YAML pour reproductibilité
6. Valider les contraintes physiques croisées entre paramètres

Voici la structure exacte attendue :

```python
"""Physics parameter space for neutral-atom experiment design.

Centralizes all tunable physical parameters across CryoSwarm-Q agents.
Replaces hardcoded constants with configurable, samplable ranges.

Parameter bounds are derived from:
- Pasqal AnalogDevice hardware specifications
- 87-Rb atomic physics (C6 = 862690 rad·µm^6/µs for |70S_{1/2}>)
- Realistic experimental constraints (tweezer array capabilities)

References:
    - Pasqal Pulser documentation: channel constraints
    - Browaeys & Lahaye, Nat. Phys. 16, 132 (2020)
    - Scholl et al., Nature 595, 233 (2021)
"""
```

Les classes et structures attendues :

```python
@dataclass
class ParameterRange:
    """A single tunable physical parameter with bounds and metadata."""
    name: str
    min_val: float
    max_val: float
    default: float
    unit: str
    description: str
    log_scale: bool = False          # Sample in log-space (for rates, etc.)
    quantize: float | None = None    # Quantization step (e.g., 4 ns for Pulser clock)
    
    def sample_uniform(self, rng: np.random.Generator) -> float: ...
    def sample_log_uniform(self, rng: np.random.Generator) -> float: ...
    def sample(self, rng: np.random.Generator) -> float: ...
    def clip(self, value: float) -> float: ...
    def normalize(self, value: float) -> float: ...  # → [0, 1]
    def denormalize(self, normalized: float) -> float: ...  # [0, 1] →


class PulseParameterSpace:
    """Parameter space for a single pulse-sequence family.
    
    Each family (adiabatic_sweep, detuning_scan, global_ramp, 
    constant_drive, blackman_sweep) has its own physically motivated ranges.
    """
    family: SequenceFamily
    amplitude: ParameterRange      # Rabi frequency Ω (rad/µs)
    detuning_start: ParameterRange # Initial detuning δ_i (rad/µs)
    detuning_end: ParameterRange   # Final detuning δ_f (rad/µs)  
    duration_ns: ParameterRange    # Total pulse duration (ns)
    amplitude_start: ParameterRange | None  # Ramp start (for sweeps)
    phase: ParameterRange          # Phase (rad)


class GeometryParameterSpace:
    """Parameter space for register geometry generation."""
    spacing_um: ParameterRange       # Inter-atom spacing (µm)
    min_spacing_um: ParameterRange   # Minimum safe distance (µm)
    atom_count: ParameterRange       # Number of atoms
    # Feasibility scoring weights (learnable!)
    feasibility_base: ParameterRange
    feasibility_spacing_weight: ParameterRange
    feasibility_blockade_weight: ParameterRange
    feasibility_atom_penalty: ParameterRange


class NoiseParameterSpace:
    """Parameter space for noise model configuration."""
    amplitude_jitter: ParameterRange     # Fractional jitter on Ω
    detuning_jitter: ParameterRange      # Fractional jitter on δ
    dephasing_rate: ParameterRange       # T2^{-1} (µs^{-1})
    atom_loss_rate: ParameterRange       # Loss rate (µs^{-1})
    temperature_uk: ParameterRange       # Atom temperature (µK)
    state_prep_error: ParameterRange     # Prep fidelity error
    false_positive_rate: ParameterRange  # Detection FP
    false_negative_rate: ParameterRange  # Detection FN


class ScoringParameterSpace:
    """Parameter space for scoring and robustness weights."""
    # Robustness aggregation weights (sum = 1)
    nominal_weight: ParameterRange
    average_weight: ParameterRange
    worst_case_weight: ParameterRange
    stability_weight: ParameterRange
    # Observable scoring weights
    density_score_weight: ParameterRange
    blockade_score_weight: ParameterRange
    # Stability scaling threshold
    stability_std_threshold: ParameterRange


class PhysicsParameterSpace:
    """Master parameter space aggregating all sub-spaces.
    
    This is the single source of truth for ALL tunable physical parameters
    in CryoSwarm-Q. Agents, environments, and datasets reference this
    instead of hardcoding values.
    """
    pulse: dict[SequenceFamily, PulseParameterSpace]
    geometry: GeometryParameterSpace
    noise: NoiseParameterSpace
    scoring: ScoringParameterSpace
    
    # Duration scaling model (replaces hardcoded `atom_count * 150`)
    duration_offset_per_atom: ParameterRange  
    
    # Cost model parameters (replaces hardcoded `2**N * T / 500000`)
    cost_base_exponent: ParameterRange
    cost_normalization: ParameterRange
    
    @classmethod 
    def default(cls) -> "PhysicsParameterSpace":
        """Create the default parameter space with research-calibrated ranges."""
        ...
    
    @classmethod
    def from_yaml(cls, path: str) -> "PhysicsParameterSpace": ...
    def to_yaml(self, path: str) -> None: ...
    
    def sample_pulse_config(
        self, 
        family: SequenceFamily, 
        atom_count: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Sample a complete pulse configuration for a given family and atom count."""
        ...
    
    def sample_noise_scenario(self, rng: np.random.Generator) -> NoiseScenario: ...
    
    def grid_search_configs(
        self,
        family: SequenceFamily,
        n_amplitude: int = 10,
        n_detuning: int = 10, 
        n_duration: int = 5,
    ) -> list[dict[str, float]]:
        """Generate a systematic grid of pulse configurations."""
        ...
    
    def latin_hypercube_sample(
        self,
        family: SequenceFamily,
        n_samples: int,
        atom_count: int,
        rng: np.random.Generator,
    ) -> list[dict[str, float]]:
        """Latin Hypercube Sampling for space-filling coverage."""
        ...
```

### Ranges physiques calibrés attendus

Voici les ranges que le `PhysicsParameterSpace.default()` doit utiliser. Ces ranges sont basés sur les specs hardware Pasqal AnalogDevice et la physique de 87-Rb :

**Pulse parameters par famille :**

| Parameter | Adiabatic Sweep | Detuning Scan | Global Ramp | Constant Drive | Blackman Sweep | Unit |
|-----------|----------------|---------------|-------------|----------------|----------------|------|
| amplitude (Ω) | [1.0, 15.0] default 5.0 | [2.0, 12.0] default 4.0 | [3.0, 15.0] default 6.0 | [1.0, 10.0] default 5.0 | [3.0, 15.0] default 7.0 | rad/µs |
| detuning_start (δ_i) | [-40.0, -5.0] default -20.0 | [-30.0, -5.0] default -15.0 | [-30.0, -5.0] default -12.0 | [-10.0, 0.0] default 0.0 | [-40.0, -10.0] default -20.0 | rad/µs |
| detuning_end (δ_f) | [5.0, 25.0] default 10.0 | [0.0, 20.0] default 5.0 | N/A | N/A | [5.0, 25.0] default 10.0 | rad/µs |
| duration | [1500, 6000] default 3000 | [1000, 5000] default 2000 | [1000, 5000] default 2000 | [500, 3000] default 1200 | [1500, 6000] default 3000 | ns, quantize=4 |
| amplitude_start | [0.1, 2.0] default 0.5 | N/A | [0.1, 3.0] default 0.5 | N/A | N/A | rad/µs |

**Geometry parameters :**

| Parameter | Min | Max | Default | Unit |
|-----------|-----|-----|---------|------|
| spacing_um | 4.0 | 15.0 | 7.0 | µm |
| min_spacing_um | 3.5 | 6.0 | 5.0 | µm |
| atom_count | 2 | 25 | 6 | atoms |
| feasibility_base | 0.3 | 0.7 | 0.55 | — |
| feasibility_spacing_w | 0.05 | 0.40 | 0.20 | — |
| feasibility_blockade_w | 0.05 | 0.40 | 0.20 | — |
| feasibility_atom_penalty | 0.01 | 0.10 | 0.04 | per atom |

**Noise parameters :**

| Parameter | Min | Max | Default | Unit |
|-----------|-----|-----|---------|------|
| amplitude_jitter | 0.01 | 0.15 | 0.06 | fractional |
| detuning_jitter | 0.01 | 0.12 | 0.05 | fractional |
| dephasing_rate | 0.01 | 0.15 | 0.07 | µs⁻¹ |
| atom_loss_rate | 0.005 | 0.08 | 0.03 | µs⁻¹ |
| temperature_uk | 10.0 | 100.0 | 50.0 | µK |
| state_prep_error | 0.001 | 0.02 | 0.005 | — |
| false_positive_rate | 0.003 | 0.05 | 0.01 | — |
| false_negative_rate | 0.01 | 0.10 | 0.05 | — |

**Scoring parameters :**

| Parameter | Min | Max | Default | Unit |
|-----------|-----|-----|---------|------|
| nominal_weight | 0.10 | 0.40 | 0.25 | — |
| average_weight | 0.15 | 0.50 | 0.35 | — |
| worst_case_weight | 0.10 | 0.50 | 0.30 | — |
| stability_weight | 0.05 | 0.20 | 0.10 | — |
| density_score_weight | 0.4 | 0.9 | 0.7 | — |
| blockade_score_weight | 0.1 | 0.6 | 0.3 | — |
| stability_std_threshold | 0.05 | 0.50 | 0.20 | — |

**Duration offset model :**
- `duration_offset_per_atom` : [50, 300] default 150 ns/atom

**Cost model :**
- `cost_base_exponent` : Actuellement `2^N`, utiliser `cost = min(1.0, base^N * T / normalization)`
- `cost_normalization` : [100000, 1000000] default 500000

### Modifications requises dans les agents

Une fois le `PhysicsParameterSpace` créé, les agents suivants doivent être modifiés pour le consommer :

#### `packages/agents/sequence_agent.py`

Le `SequenceAgent` doit accepter un `PhysicsParameterSpace` en paramètre d'initialisation et l'utiliser pour :
- Remplacer toutes les constantes hardcodées dans `_adiabatic_variants()`, `_detuning_scan_variants()`, `_global_ramp_variants()`, `_constant_drive_variants()`, `_blackman_sweep_variants()`
- Chaque variante doit utiliser `param_space.pulse[family].amplitude.default` au lieu de `5.0`
- Le `duration_offset` doit venir de `param_space.duration_offset_per_atom.default * atom_count`
- Le `predicted_cost` doit utiliser les paramètres du cost model

Signature modifiée :
```python
class SequenceAgent(BaseAgent):
    def __init__(self, param_space: PhysicsParameterSpace | None = None):
        self.param_space = param_space or PhysicsParameterSpace.default()
```

Les variantes d'exploit/memory continuent à fonctionner mais utilisent les ranges du param_space pour les bornes de clipping.

#### `packages/agents/geometry_agent.py`

Le `GeometryAgent` remplace ses constantes hardcodées :
- `min_spacing_um` vient du param_space
- Les poids de feasibility scoring viennent du param_space
- Le spacing de base provient du param_space

#### `packages/ml/rl_env.py`

Le `PulseDesignEnv` doit accepter un `PhysicsParameterSpace` :
- `AMP_RANGE`, `DET_RANGE`, `DUR_RANGE` ne sont plus des constantes globales mais viennent du param_space
- `rescale_action()` et `inverse_rescale()` utilisent les ranges du param_space
- L'observation space inclut les paramètres du param_space comme contexte

Signature modifiée :
```python
class PulseDesignEnv:
    def __init__(self, spec, register_candidates, param_space=None, max_steps=1, simulate_fn=None):
        self.param_space = param_space or PhysicsParameterSpace.default()
```

#### `packages/ml/rl_sequence_agent.py`

Le `RLSequenceAgent` passe le param_space au RL env :
```python
class RLSequenceAgent(BaseAgent):
    def __init__(self, param_space=None, checkpoint_path=None, ...):
        self.param_space = param_space or PhysicsParameterSpace.default()
```

#### `packages/scoring/robustness.py`

La fonction `robustness_score()` doit accepter les poids depuis le param_space au lieu de `0.25, 0.35, 0.30, 0.10` hardcodés.

#### `packages/simulation/evaluators.py`

Les poids `0.7 * density + 0.3 * blockade` doivent venir du param_space.

### Contrainte de rétrocompatibilité

TOUTES les modifications doivent être rétrocompatibles. Si `param_space=None`, le comportement par défaut doit être identique au code actuel. Les tests existants doivent continuer à passer sans modification.

---

---

## P0.2 — CONNECTER LE RL AU PIPELINE AVEC UN VRAI SWITCH

### Problème actuel

Le `RLSequenceAgent` dans `packages/ml/rl_sequence_agent.py` est **déconnecté** du pipeline. Actuellement :

1. Le pipeline dans `packages/orchestration/pipeline.py` instancie directement `SequenceAgent()` :
   ```python
   self.sequence_agent = SequenceAgent()
   ```
2. Le `RLSequenceAgent` a un flag `enabled=False` par défaut
3. Quand `enabled=True` mais pas de checkpoint → fallback silencieux vers heuristique
4. Pas de logique de choix intelligent entre heuristique et RL
5. Pas de métriques pour comparer les deux
6. Pas de mode hybride (RL + heuristique ensemble)

### Ce qui doit être créé

#### Fichier : `packages/agents/sequence_strategy.py`

Créer un **strategy pattern** digne de la recherche pour le choix entre heuristique et RL :

```python
"""Intelligent sequence generation strategy.

Implements a research-grade switch between heuristic and RL-based
pulse-sequence generation. The strategy considers:

1. Checkpoint availability and quality (training metrics)
2. Problem complexity (atom count, geometry)
3. Historical performance (memory records)
4. Confidence calibration (surrogate uncertainty)

The switch is NOT a simple boolean flag. It implements:
- Adaptive selection based on problem characteristics
- Hybrid mode: RL candidates + heuristic candidates in same batch
- Bandit-style exploration between strategies
- Performance tracking per strategy per problem class

References:
    Auer et al., "Finite-time Analysis of the Multiarmed Bandit Problem" (2002)
"""
```

Classes attendues :

```python
class SequenceStrategyMode(StrEnum):
    HEURISTIC_ONLY = "heuristic_only"
    RL_ONLY = "rl_only"
    HYBRID = "hybrid"              # RL + heuristic candidates mixed
    ADAPTIVE = "adaptive"          # Choix automatique par le système
    BANDIT = "bandit"              # UCB1 multi-armed bandit selection


@dataclass
class StrategyMetrics:
    """Performance tracking for a strategy on a problem class."""
    strategy: str
    problem_class: str          # e.g., "adiabatic_6atoms_square"
    n_trials: int = 0
    total_reward: float = 0.0   # Sum of robustness scores obtenues
    best_reward: float = 0.0
    avg_reward: float = 0.0
    
    def update(self, reward: float) -> None: ...
    
    @property
    def ucb1_score(self) -> float:
        """Upper Confidence Bound score for bandit selection."""
        if self.n_trials == 0:
            return float('inf')
        return self.avg_reward + math.sqrt(2 * math.log(total_trials) / self.n_trials)


class SequenceStrategy:
    """Research-grade strategy for sequence generation method selection.
    
    Manages the intelligent switch between:
    - HeuristicSequenceAgent (rule-based, hardcoded parameters)
    - RLSequenceAgent (learned policy, trained checkpoint)
    - Hybrid (both, ranked together)
    - Adaptive (automatic selection based on problem + history)
    """
    
    def __init__(
        self,
        mode: SequenceStrategyMode = SequenceStrategyMode.ADAPTIVE,
        param_space: PhysicsParameterSpace | None = None,
        rl_checkpoint_path: str | None = None,
        rl_temperature: float = 0.3,
        rl_n_candidates: int = 5,
        heuristic_enabled: bool = True,
        min_rl_confidence: float = 0.4,      # Minimum confidence to use RL
        hybrid_rl_fraction: float = 0.5,       # Fraction of candidates from RL in hybrid mode
        surrogate_model: SurrogateModel | None = None,  # For uncertainty estimation
    ): ...
    
    def select_strategy(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
        memory_records: list[MemoryRecord],
    ) -> SequenceStrategyMode:
        """Select the best strategy for this specific problem instance.
        
        Decision logic:
        1. If no RL checkpoint → HEURISTIC_ONLY
        2. If mode == HEURISTIC_ONLY or RL_ONLY → use that
        3. If mode == HYBRID → always both
        4. If mode == ADAPTIVE:
           a. Compute problem complexity score
           b. Check RL checkpoint quality metrics
           c. Check memory for this problem class
           d. If RL has history of strong performance on this class → RL_ONLY
           e. If problem is novel (no memory) → HYBRID (explore both)
           f. If RL consistently underperforms heuristic → HEURISTIC_ONLY
        5. If mode == BANDIT:
           a. UCB1 selection between strategies
           b. Exploration bonus for under-tried strategies
        """
        ...
    
    def generate_candidates(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord],
    ) -> tuple[list[SequenceCandidate], dict[str, Any]]:
        """Generate candidates using the selected strategy.
        
        Returns (candidates, metadata) where metadata includes:
        - strategy_used: which strategy was selected
        - strategy_reason: why
        - rl_candidates_count: how many from RL
        - heuristic_candidates_count: how many from heuristic
        - confidence: estimated quality
        """
        ...
    
    def update_performance(
        self,
        problem_class: str,
        strategy_used: str,
        robustness_scores: list[float],
    ) -> None:
        """Update bandit/adaptive metrics after evaluation results are known."""
        ...
    
    def _compute_problem_class(
        self,
        spec: ExperimentSpec,
        register: RegisterCandidate,
    ) -> str:
        """Compute a problem class identifier for bandit tracking.
        
        Format: "{objective_class}_{atom_count}atoms_{layout}"
        Example: "balanced_campaign_search_6atoms_square"
        """
        ...
    
    def _assess_rl_checkpoint_quality(self) -> float:
        """Assess the quality of the loaded RL checkpoint.
        
        Returns a confidence score [0, 1] based on:
        - Training reward history (if available in checkpoint metadata)
        - Number of training steps
        - Validation performance
        """
        ...
    
    def get_strategy_report(self) -> dict[str, Any]:
        """Return a full report of strategy performance for dashboard/logging."""
        ...
```

### Modifications requises dans le pipeline

#### `packages/orchestration/pipeline.py`

Le pipeline doit utiliser le `SequenceStrategy` au lieu d'instancier directement un `SequenceAgent` :

```python
class CryoSwarmPipeline:
    def __init__(
        self,
        repository=None,
        parallel=False,
        sequence_strategy_mode: str = "adaptive",
        rl_checkpoint_path: str | None = None,
        param_space: PhysicsParameterSpace | None = None,
    ):
        self.param_space = param_space or PhysicsParameterSpace.default()
        self.sequence_strategy = SequenceStrategy(
            mode=SequenceStrategyMode(sequence_strategy_mode),
            param_space=self.param_space,
            rl_checkpoint_path=rl_checkpoint_path,
        )
        # Les autres agents reçoivent aussi le param_space
        self.geometry_agent = GeometryAgent(param_space=self.param_space)
        # etc.
```

Dans la boucle de génération de séquences, remplacer :
```python
# AVANT :
sequences = self.sequence_agent.run(spec, register, campaign.id, memory_context)

# APRÈS :
sequences, strategy_meta = self.sequence_strategy.generate_candidates(
    spec, register, campaign.id, memory_context,
)
```

Après l'évaluation de robustesse, le pipeline doit appeler :
```python
# Feedback au strategy pour mise à jour des métriques bandit/adaptive
self.sequence_strategy.update_performance(
    problem_class=strategy_meta["problem_class"],
    strategy_used=strategy_meta["strategy_used"],
    robustness_scores=[r.robustness_score for r in reports_for_this_register],
)
```

Le `strategy_meta` doit être inclus dans les `AgentDecision` pour traçabilité :
```python
decisions.append(AgentDecision(
    ...
    structured_output={
        "strategy_used": strategy_meta["strategy_used"],
        "strategy_reason": strategy_meta["strategy_reason"],
        "rl_candidates": strategy_meta.get("rl_candidates_count", 0),
        "heuristic_candidates": strategy_meta.get("heuristic_candidates_count", 0),
    },
))
```

### Le mode HYBRID en détail

En mode HYBRID, le `SequenceStrategy` fait :

1. Appel au `SequenceAgent.run()` → N heuristic candidates
2. Appel au `RLSequenceAgent.run()` → M RL candidates
3. Merge des deux listes
4. Chaque candidat a un tag `metadata["source"]` = `"heuristic"` ou `"rl_policy"`
5. Les deux sont évalués par le NoiseRobustnessAgent
6. Le ranking final compare RL vs heuristique
7. Les résultats alimentent le bandit pour les prochaines décisions

C'est le mode par défaut quand un checkpoint RL existe mais qu'on n'a pas encore assez de données pour savoir lequel est meilleur. **C'est le mode recherche par excellence** : on compare les deux stratégies sur les mêmes problèmes.

### Mode ADAPTIVE en détail

L'ADAPTIVE utilise cette logique :

```
IF checkpoint manquant OR checkpoint ancien (> 1000 campagnes depuis training):
    → HEURISTIC_ONLY
ELIF memory montre RL > heuristique sur ce problem_class (>10 comparaisons, p-value < 0.05):
    → RL_ONLY  
ELIF memory montre heuristique > RL sur ce problem_class:
    → HEURISTIC_ONLY
ELSE:
    → HYBRID (pas assez de données, on explore)
```

### Mode BANDIT en détail  

Le BANDIT utilise UCB1 (Upper Confidence Bound) :

$$\text{UCB1}(a) = \bar{X}_a + \sqrt{\frac{2 \ln N}{n_a}}$$

Où $\bar{X}_a$ est le reward moyen de la stratégie $a$, $N$ le total de sélections, $n_a$ le nombre de fois que $a$ a été choisie.

Le bandit maintient des compteurs séparés **par problem_class** pour que la sélection soit contextuelle.

---

---

## P0.3 — DATASET GENERATOR 100K+ POINTS

### Problème actuel

Le générateur actuel dans `scripts/train_ml.py` :

```python
def generate_training_data(n_runs: int, output_path: str) -> None:
    for i in range(n_runs):
        atom_count = 4 + (i % 7)  # 4-10 atomes
        goal = ExperimentGoal(...)
        summary = pipeline.run(goal)
        builder.add_from_pipeline(registers=[], sequences=[], ...)  # BUG: empty lists!
```

Problèmes critiques :
1. **Les listes `registers=[]` et `sequences=[]` sont vides** → le `add_from_pipeline` ne peut rien matcher → le dataset est probablement VIDE
2. Seulement 4-10 atomes, pas de variation de géométrie contrôlée
3. Pas de couverture systématique de l'espace des paramètres
4. Pas de balayage des paramètres de bruit
5. Pour 100k points il faut une approche radicalement différente

### Ce qui doit être créé

#### Fichier : `packages/ml/data_generator.py`

Un générateur de données massif et systématique qui :

1. **N'utilise PAS le pipeline** comme source unique de données (auto-référent)
2. **Évalue directement** des configurations (register, sequence) via l'évaluateur de robustesse
3. **Couvre systématiquement** l'espace des paramètres via Latin Hypercube Sampling
4. **Supporte la parallélisation** (multi-process pour 100k évaluations)
5. **Gère les échecs gracieusement** (certaines configs ne simulent pas)
6. **Sauve en streaming** (pas tout en RAM)
7. **Supporte la reprise** (checkpoint de progression)
8. **Produit des statistiques** sur le dataset généré

```python
"""Systematic dataset generator for CryoSwarm-Q ML training.

Generates large-scale training datasets by systematically sampling
the physics parameter space and evaluating candidate configurations
through the simulation pipeline.

Architecture:
    1. Sample configurations from PhysicsParameterSpace (LHS, grid, or random)
    2. Build (register, sequence) pairs for each configuration
    3. Evaluate robustness via simulation/evaluators.py
    4. Extract (feature_vector, target_vector) pairs
    5. Save to disk in streaming fashion

Supports:
    - Latin Hypercube Sampling for efficient space coverage
    - Grid search for systematic parameter sweeps
    - Multi-process parallel evaluation
    - Streaming save to avoid OOM on large datasets
    - Checkpoint/resume for long generation runs
    - Progress tracking and statistics

Target: 100k+ data points covering the full parameter space.
"""
```

Classes attendues :

```python
@dataclass
class GenerationConfig:
    """Configuration for large-scale data generation."""
    n_samples: int = 100_000
    sampling_method: str = "lhs"  # "lhs", "grid", "random", "sobol"
    
    # Parameter space coverage
    atom_counts: list[int] = field(default_factory=lambda: list(range(3, 16)))
    layouts: list[str] = field(default_factory=lambda: [
        "square", "line", "triangular", "ring", "zigzag", "honeycomb",
    ])
    families: list[str] = field(default_factory=lambda: [
        "adiabatic_sweep", "detuning_scan", "global_ramp", 
        "constant_drive", "blackman_sweep",
    ])
    
    # Noise variation
    include_noise_variation: bool = True
    noise_samples_per_config: int = 1  # How many noise scenarios to sample per config
    
    # Parallelization
    n_workers: int = 4
    batch_size: int = 100          # Configs per batch for parallel eval
    
    # Persistence
    output_dir: str = "data/generated"
    save_interval: int = 1000      # Save checkpoint every N samples
    resume: bool = True            # Resume from last checkpoint
    
    # Simulation limits
    max_atoms_for_full_sim: int = 12  # Above this, use fast approximation
    timeout_per_eval: float = 30.0    # Seconds per single evaluation
    
    # Reproducibility
    seed: int = 42


class DatasetGenerator:
    """Systematic large-scale dataset generator.
    
    Generates training data by sampling the PhysicsParameterSpace
    and evaluating configurations through the simulation pipeline.
    """
    
    def __init__(
        self,
        config: GenerationConfig,
        param_space: PhysicsParameterSpace | None = None,
    ): ...
    
    def generate(self) -> DatasetStats:
        """Run the full generation pipeline.
        
        Steps:
        1. Plan configurations (sample parameter space)
        2. Build register candidates for each (atom_count, layout, spacing) combo
        3. Build sequence candidates for each register × pulse_config
        4. Evaluate robustness in parallel
        5. Collect features and targets
        6. Save to disk
        
        Returns statistics about the generated dataset.
        """
        ...
    
    def _plan_configurations(self) -> list[dict[str, Any]]:
        """Plan all configurations to evaluate.
        
        Uses the sampling method to cover the parameter space:
        - LHS: Latin Hypercube for uniform coverage with minimal samples
        - Grid: Cartesian product for systematic sweeps
        - Random: Uniform random for baseline
        - Sobol: Quasi-random for low-discrepancy coverage
        
        Each configuration specifies:
        {
            "atom_count": int,
            "layout": str,
            "spacing_um": float,
            "family": str,
            "amplitude": float,
            "detuning": float,
            "duration_ns": int,
            "detuning_end": float | None,
            "amplitude_start": float | None,
            "waveform_kind": str,
            "noise_jitter_amplitude": float,  # If noise variation enabled
            "noise_jitter_detuning": float,
        }
        """
        ...
    
    def _build_register(
        self, 
        atom_count: int, 
        layout: str, 
        spacing_um: float,
        campaign_id: str,
    ) -> RegisterCandidate:
        """Build a register candidate from configuration.
        
        Uses GeometryAgent internals to compute coordinates for the layout,
        then computes physics metadata (blockade_radius, vdW matrix, etc.).
        """
        ...
    
    def _build_sequence(
        self,
        config: dict[str, Any],
        register: RegisterCandidate,
        campaign_id: str,
    ) -> SequenceCandidate:
        """Build a sequence candidate from sampled parameters."""
        ...
    
    def _evaluate_batch(
        self,
        batch: list[tuple[ExperimentSpec, RegisterCandidate, SequenceCandidate]],
    ) -> list[tuple[NDArray, NDArray] | None]:
        """Evaluate a batch of configurations in parallel.
        
        Uses ProcessPoolExecutor for CPU-parallel Hamiltonian diagonalization.
        Returns (feature_vector, target_vector) for each successful evaluation,
        None for failed evaluations.
        """
        ...
    
    def _save_checkpoint(
        self,
        features: NDArray,
        targets: NDArray,
        metadata: dict[str, Any],
        checkpoint_idx: int,
    ) -> None:
        """Save intermediate results to disk."""
        ...
    
    def _merge_checkpoints(self, output_path: str) -> DatasetStats:
        """Merge all checkpoint files into final dataset."""
        ...
    
    def _compute_stats(self, features: NDArray, targets: NDArray) -> DatasetStats:
        """Compute statistics about the generated dataset."""
        ...


@dataclass
class DatasetStats:
    """Statistics about a generated dataset."""
    total_samples: int
    successful_evals: int
    failed_evals: int
    eval_time_seconds: float
    
    # Coverage statistics
    atom_count_distribution: dict[int, int]
    layout_distribution: dict[str, int]
    family_distribution: dict[str, int]
    
    # Target statistics
    robustness_mean: float
    robustness_std: float
    robustness_min: float
    robustness_max: float
    nominal_mean: float
    worst_case_mean: float
    
    # Feature statistics (for normalization)
    feature_means: NDArray
    feature_stds: NDArray
    feature_mins: NDArray
    feature_maxs: NDArray
    
    def summary(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
```

### Stratégie de sampling pour 100k+ points

Pour atteindre 100k+ points de manière **physiquement significative**, voici la stratégie de sampling :

#### Décomposition du budget de 100k :

```
Total = 100,000+ samples

Par atom_count (13 valeurs : 3-15) :
  100,000 / 13 ≈ 7,700 par atom_count

Par layout (6 layouts) :
  7,700 / 6 ≈ 1,283 par (atom_count, layout)

Par family (5 families) :
  1,283 / 5 ≈ 257 par (atom_count, layout, family)

Par (atom_count, layout, family) → 257 samples LHS sur (amplitude, detuning, duration)
  = 257 points dans un espace 3D → excellent couverture

Total réel : 13 × 6 × 5 × 257 = 100,230 configurations
```

#### Pour les grands systèmes (>12 atomes)

Les simulations sont exponentiellement plus lentes pour les grands systèmes. Pour `atom_count > 12` :

1. Utiliser le backend NumPy sparse (jusqu'à 18 atomes)
2. Pour 13-15 atomes : évaluer nominalement seulement (skip les scénarios de bruit complets), utiliser un estimateur rapide de robustesse basé sur le spectral gap et la densité Rydberg
3. Réduire le nombre de configurations à ~100/combo au lieu de 257

Budget révisé :
```
Atoms 3-12 : 10 × 6 × 5 × 300 = 90,000 (full simulation)
Atoms 13-15 : 3 × 6 × 5 × 100 = 9,000 (fast approximate)
Total : 99,000+
```

#### Latin Hypercube Sampling

Pour chaque combinaison (atom_count, layout, family), le LHS doit couvrir :

```python
# Dimensions du LHS pour chaque combo
dimensions = {
    "amplitude": param_space.pulse[family].amplitude,        # [min, max]
    "detuning_start": param_space.pulse[family].detuning_start,
    "duration_ns": param_space.pulse[family].duration_ns,
}

# Si la famille a un detuning_end (sweeps)
if family in (ADIABATIC_SWEEP, DETUNING_SCAN, BLACKMAN_SWEEP):
    dimensions["detuning_end"] = param_space.pulse[family].detuning_end

# Si la famille a un amplitude_start (ramps)  
if family in (ADIABATIC_SWEEP, GLOBAL_RAMP):
    dimensions["amplitude_start"] = param_space.pulse[family].amplitude_start

# Spacing variation (même layout, différents spacings)
dimensions["spacing_um"] = param_space.geometry.spacing_um
```

Le LHS produit N échantillons uniformément répartis dans cet espace à D dimensions.

Implémentation avec `scipy.stats.qmc.LatinHypercube` si disponible, sinon fallback NumPy :

```python
def latin_hypercube(n_samples: int, n_dims: int, rng) -> NDArray:
    """Generate LHS samples in [0, 1]^n_dims."""
    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=n_dims, seed=rng)
        return sampler.random(n=n_samples).astype(np.float32)
    except ImportError:
        # Fallback: stratified random sampling
        result = np.zeros((n_samples, n_dims), dtype=np.float32)
        for d in range(n_dims):
            intervals = np.linspace(0, 1, n_samples + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            result[:, d] = points
        return result
```

### Feature engineering amélioré

Le dataset actuel a 10 features naïves. Le nouveau dataset doit inclure des **features physiquement informées** :

```python
INPUT_DIM_V2 = 18  # Upgraded from 10

def build_feature_vector_v2(
    register: RegisterCandidate,
    sequence: SequenceCandidate,
    param_space: PhysicsParameterSpace,
) -> NDArray[np.float32]:
    """Enhanced feature vector with physics-informed features."""
    spacing = float(register.metadata.get("spacing_um", 7.0))
    
    # === Existing features (normalized) ===
    atom_count_norm = register.atom_count / 25.0       # Normalized by max
    spacing_norm = spacing / 15.0                       # Normalized by max spacing
    amplitude_norm = sequence.amplitude / 15.8           # Normalized by AnalogDevice max
    detuning_norm = (sequence.detuning + 126.0) / 252.0 # Normalized to [0, 1]
    duration_norm = sequence.duration_ns / 6000.0        # Normalized by max duration
    layout_id = _encode_layout(register.layout_type) / 5.0
    family_id = _encode_family(sequence.sequence_family.value) / 4.0
    
    # === Physics-informed features (NEW) ===
    # Blockade fraction: what fraction of pairs are within blockade radius
    blockade_fraction = register.blockade_pair_count / max(
        register.atom_count * (register.atom_count - 1) / 2, 1
    )
    
    # Rabi frequency / interaction energy ratio (dimensionless)
    # Ω / (C6 / R^6) — determines which phase the system is in
    if spacing > 0:
        interaction_energy = 862690.0 / (spacing ** 6)  # C6 / r^6
        omega_over_interaction = sequence.amplitude / max(interaction_energy, 1e-10)
    else:
        omega_over_interaction = 0.0
    omega_over_interaction_norm = min(omega_over_interaction, 10.0) / 10.0
    
    # Detuning / Ω ratio — determines detuning dominance
    detuning_over_omega = abs(sequence.detuning) / max(sequence.amplitude, 0.01)
    detuning_over_omega_norm = min(detuning_over_omega, 20.0) / 20.0
    
    # Adiabaticity parameter: T × Ω — how adiabatic is the sweep
    duration_us = sequence.duration_ns / 1000.0
    adiabaticity = duration_us * sequence.amplitude
    adiabaticity_norm = min(adiabaticity, 100.0) / 100.0
    
    # Hilbert space dimension (log-scaled)
    hilbert_dim_log = register.atom_count * np.log(2) / np.log(2**25)
    
    # Feasibility score (already computed)
    feasibility = register.feasibility_score
    
    # Predicted computational cost (log-scaled)
    cost_log = np.log1p(sequence.predicted_cost)
    
    # Blockade radius / spacing ratio — key physics ratio
    blockade_over_spacing = register.blockade_radius_um / max(spacing, 0.1)
    blockade_over_spacing_norm = min(blockade_over_spacing, 3.0) / 3.0
    
    return np.array([
        # Original features (normalized)
        atom_count_norm,
        spacing_norm,
        amplitude_norm,
        detuning_norm,
        duration_norm,
        layout_id,
        family_id,
        
        # Physics-informed features
        blockade_fraction,
        omega_over_interaction_norm,
        detuning_over_omega_norm,
        adiabaticity_norm,
        hilbert_dim_log,
        feasibility,
        cost_log,
        blockade_over_spacing_norm,
        
        # Blockade radius (normalized)
        register.blockade_radius_um / 15.0,
        
        # Min distance (normalized)
        register.min_distance_um / 15.0,
    ], dtype=np.float32)
```

### Normalisation du dataset

Le module `dataset.py` doit être enrichi avec un `DatasetNormalizer` :

```python
class DatasetNormalizer:
    """Feature normalization with saved statistics for inference.
    
    Computes mean and std from training data, applies StandardScaler,
    saves statistics alongside the model checkpoint so inference uses
    the same normalization.
    """
    
    def __init__(self):
        self.means: NDArray | None = None
        self.stds: NDArray | None = None
        self.fitted: bool = False
    
    def fit(self, features: NDArray) -> "DatasetNormalizer":
        """Compute normalization statistics from training features."""
        self.means = features.mean(axis=0)
        self.stds = features.std(axis=0)
        # Prevent division by zero for constant features
        self.stds[self.stds < 1e-8] = 1.0
        self.fitted = True
        return self
    
    def transform(self, features: NDArray) -> NDArray:
        """Apply normalization."""
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")
        return (features - self.means) / self.stds
    
    def fit_transform(self, features: NDArray) -> NDArray:
        return self.fit(features).transform(features)
    
    def inverse_transform(self, normalized: NDArray) -> NDArray:
        return normalized * self.stds + self.means
    
    def save(self, path: str) -> None:
        np.savez(path, means=self.means, stds=self.stds)
    
    def load(self, path: str) -> "DatasetNormalizer":
        data = np.load(path)
        self.means = data["means"]
        self.stds = data["stds"]
        self.fitted = True
        return self
```

### Modification du script `train_ml.py`

Ajouter une nouvelle phase `generate_v2` :

```python
parser.add_argument("--phase", choices=[
    "generate", "generate_v2", "surrogate", "rl", "full",
], required=True)

# ...

elif args.phase == "generate_v2":
    from packages.ml.data_generator import DatasetGenerator, GenerationConfig
    config = GenerationConfig(
        n_samples=args.n_samples,     # CLI arg, default 100000
        n_workers=args.workers,        # CLI arg, default 4
        sampling_method=args.sampling, # CLI arg, default "lhs"
        output_dir=args.output_dir,    # CLI arg
        seed=args.seed,
    )
    generator = DatasetGenerator(config)
    stats = generator.generate()
    print(stats.summary())
```

Nouveaux arguments CLI :
```
--n-samples     (default: 100000)
--workers       (default: 4)
--sampling      (default: "lhs", choices: ["lhs", "grid", "random", "sobol"])
--output-dir    (default: "data/generated")
--seed          (default: 42)
```

### Fix du bug dans `generate_training_data()`

Le code actuel dans `scripts/train_ml.py` passe des listes vides :
```python
builder.add_from_pipeline(
    registers=[],  # ← BUG : empty
    sequences=[],  # ← BUG : empty
    reports=summary.robustness_reports,
    evaluations=summary.ranked_candidates,
)
```

Le `PipelineSummary` ne contient pas directement les registers et sequences. Il faut soit :
1. Ajouter `registers` et `sequences` au `PipelineSummary`
2. Ou extraire les données autrement

La solution propre est d'ajouter les champs au `PipelineSummary` :
```python
class PipelineSummary(CryoSwarmModel):
    # ... existing fields ...
    registers: list[RegisterCandidate] = Field(default_factory=list)
    sequences: list[SequenceCandidate] = Field(default_factory=list)
```

Et de les remplir dans `pipeline.run()` avant de construire le summary.

---

---

## FICHIERS À CRÉER

| Fichier | Description |
|---------|-------------|
| `packages/core/parameter_space.py` | PhysicsParameterSpace — source unique de vérité pour tous les paramètres physiques |
| `packages/agents/sequence_strategy.py` | SequenceStrategy — switch intelligent heuristique/RL |
| `packages/ml/data_generator.py` | DatasetGenerator — génération systématique de 100k+ points |
| `packages/ml/normalizer.py` | DatasetNormalizer — normalisation avec statistiques sauvées |

## FICHIERS À MODIFIER

| Fichier | Modification |
|---------|-------------|
| `packages/agents/sequence_agent.py` | Accepter `param_space` en init, remplacer constantes hardcodées |
| `packages/agents/geometry_agent.py` | Accepter `param_space` en init, remplacer constantes hardcodées |
| `packages/ml/rl_env.py` | Accepter `param_space`, remplacer `AMP_RANGE`/`DET_RANGE`/`DUR_RANGE` |
| `packages/ml/rl_sequence_agent.py` | Accepter `param_space`, le passer au env |
| `packages/ml/dataset.py` | Ajouter `build_feature_vector_v2()`, `INPUT_DIM_V2=18` |
| `packages/ml/surrogate.py` | Accepter `input_dim` dynamique (déjà paramétrable) |
| `packages/scoring/robustness.py` | Accepter poids depuis param_space |
| `packages/simulation/evaluators.py` | Accepter poids depuis param_space |
| `packages/orchestration/pipeline.py` | Utiliser `SequenceStrategy`, passer `param_space` aux agents |
| `packages/core/models.py` | Ajouter `registers` et `sequences` au `PipelineSummary` |
| `scripts/train_ml.py` | Ajouter phase `generate_v2`, fixer bug des listes vides, nouveaux CLI args |

## TESTS À CRÉER

| Test | Vérifie |
|------|---------|
| `tests/test_parameter_space.py` | ParameterRange sampling, clipping, normalization, LHS, grid, serialization YAML |
| `tests/test_sequence_strategy.py` | Tous les modes (HEURISTIC_ONLY, RL_ONLY, HYBRID, ADAPTIVE, BANDIT), UCB1, fallback |
| `tests/test_data_generator.py` | Génération petit dataset (100 samples), couverture, normalisation, checkpoint/resume |
| `tests/test_feature_v2.py` | build_feature_vector_v2 correct, dimensions, valeurs physiques |

## CONTRAINTES

1. **Rétrocompatibilité** : Tous les tests existants doivent passer sans modification. Quand `param_space=None`, le comportement est identique au code actuel.

2. **Pas de nouvelles dépendances** : Utiliser numpy et scipy (déjà dans requirements). PyTorch reste optionnel. Pas de nouvelles libs.

3. **Performance** : Le dataset generator doit pouvoir produire 100k points en un temps raisonnable avec parallélisation. Pour 12 atomes max (full sim), estimer ~1-5 secondes par évaluation → ~100k points à 4 workers ≈ 7-35 heures. Prévoir un mode "fast" avec simulation simplifiée (~0.1s par point → ~3 heures à 4 workers).

4. **Reproductibilité** : Tous les RNG doivent être seedés. Le même seed + même config = même dataset.

5. **Physique correcte** : Les ranges de paramètres doivent respecter les contraintes hardware Pasqal AnalogDevice et la physique de 87-Rb.

---

## ORDRE D'EXÉCUTION

1. Créer `packages/core/parameter_space.py` (P0.1)
2. Modifier `packages/agents/sequence_agent.py` pour consommer le param_space (P0.1)
3. Modifier `packages/agents/geometry_agent.py` pour consommer le param_space (P0.1)
4. Modifier `packages/ml/rl_env.py` pour consommer le param_space (P0.1)
5. Modifier `packages/scoring/robustness.py` et `packages/simulation/evaluators.py` (P0.1)
6. Créer `tests/test_parameter_space.py` et vérifier (P0.1)
7. Créer `packages/agents/sequence_strategy.py` (P0.2)
8. Modifier `packages/orchestration/pipeline.py` pour utiliser le strategy (P0.2)
9. Modifier `packages/ml/rl_sequence_agent.py` (P0.2)
10. Créer `tests/test_sequence_strategy.py` et vérifier (P0.2)
11. Créer `packages/ml/normalizer.py` (P0.3)
12. Modifier `packages/ml/dataset.py` pour feature_v2 (P0.3)
13. Créer `packages/ml/data_generator.py` (P0.3)
14. Modifier `packages/core/models.py` — ajouter champs au PipelineSummary (P0.3)
15. Modifier `scripts/train_ml.py` — fixer bug + nouvelle phase (P0.3)
16. Créer tests pour P0.3 et vérifier
17. Lancer `py -m pytest tests/ -v` pour vérifier TOUS les tests passent

---

## CRITÈRE DE SUCCÈS

Le P0 est atteint quand :

1. `PhysicsParameterSpace.default()` retourne un espace de paramètres complet et correct
2. `param_space.latin_hypercube_sample("adiabatic_sweep", n_samples=100, atom_count=6)` retourne 100 configurations physiquement valides
3. Le pipeline peut tourner en mode `HYBRID` avec un checkpoint RL et produit des candidats des deux sources
4. Le `SequenceStrategy` en mode `BANDIT` track les performances et converge vers la meilleure stratégie
5. `python -m scripts.train_ml --phase generate_v2 --n-samples 1000 --workers 2` produit un dataset de 1000 points avec couverture correcte
6. Les features v2 (18-dim) incluent les ratios physiques (Ω/U, δ/Ω, adiabaticité)
7. Tous les tests existants + nouveaux passent
8. Le code est production-quality : typé, documenté, modulaire
