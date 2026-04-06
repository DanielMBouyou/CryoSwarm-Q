# P1 — ROBUSTESSE : Prompt d'implémentation exhaustif

> **Destinataire** : Agent IA de code.
> **Mode** : Implémentation directe — écris chaque ligne, chaque import, chaque test.
> **Philosophie** : Rigueur ingénierie logicielle de niveau recherche. Chaque patron de code doit être state-of-the-art Python 3.11+. Chaque refactoring doit être réfléchi, traceable, et soutenu par des tests. C'est un projet de quantum computing en recherche — le code doit être digne de review technique sérieuse. Fais preuve d'innovation dans les patterns d'architecture. Des milliers de lignes précises et millimétrées.

---

## Contexte du projet

CryoSwarm-Q est un système multi-agent hardware-aware pour la conception autonome d'expériences en informatique quantique à atomes neutres. Le codebase Python cible Pulser / Pasqal et orchestre 8 agents spécialisés dans un pipeline séquentiel.

**Fichier manifeste** : `CLAUDE.md` à la racine contient la vision complète du projet.

**Hypothèse** : Les tâches P0 (P0_FONDATIONS_PROMPT.md) ont été implémentées. Le pipeline utilise désormais `PipelineContext` et des phases composables. Les modèles de données critiques sont frozen via `FrozenCryoSwarmModel`. Un `conftest.py` avec factories partagées existe. Les constantes physiques sont dans `PhysicsParameterSpace`.

Si tu constates qu'une tâche P0 n'est pas encore appliquée, adapte-toi en conséquence — ne casse rien.

**Règles impératives** :
- Python 3.11+, types partout, `from __future__ import annotations`
- Pydantic v2.8+
- Pas de `# type: ignore` sans justification
- Imports absolus uniquement (`from packages.core.models import ...`)
- Chaque import groupé : stdlib, third-party, project
- `pytest tests/ -x --tb=short` après CHAQUE tâche
- Zéro `# TODO` laissé dans le code produit

---

## TÂCHE 1 — Définir un protocole `AgentProtocol` avec signatures formalisées

### Problème actuel

La classe `BaseAgent` (dans `packages/agents/base.py`) est une ABC qui ne définit **aucune méthode abstraite** `run()`. Chaque agent a une signature `run()` complètement différente :

| Agent | Signature de `run()` |
|-------|---------------------|
| `ProblemFramingAgent` | `(goal, memory_records=None) → ExperimentSpec` |
| `GeometryAgent` | `(spec, campaign_id, memory_records=None) → list[RegisterCandidate]` |
| `SequenceAgent` | `(spec, register, campaign_id, memory_records=None) → list[SequenceCandidate]` |
| `NoiseRobustnessAgent` | `(spec, register, sequence) → RobustnessReport` |
| `BackendRoutingAgent` | `(spec, sequence, report) → BackendChoice` |
| `CampaignAgent` | `(campaign, evaluations) → tuple[CampaignState, list[EvaluationResult]]` |
| `ResultsAgent` | `(goal, spec, campaign, ranked) → dict[str, object]` |
| `MemoryAgent` | `(campaign_id, ranked, seq_lookup, reg_lookup) → list[MemoryRecord]` |

Le pipeline dans `pipeline.py` connaît chaque signature exacte — couplage fort. Il n'y a pas de contrat commun vérifiable.

### Objectif

Créer un système de protocoles typés qui :
1. Formalise ce que chaque catégorie d'agent DOIT fournir
2. Permet la vérification statique de conformité
3. Maintient la rétrocompatibilité (les agents gardent leurs signatures)
4. Documente les contrats d'interface

### Architecture cible

On ne peut PAS unifier les signatures `run()` en une seule — les agents ont des entrées/sorties fondamentalement différentes. L'approche correcte est un protocole à deux niveaux :

#### Niveau 1 — `AgentProtocol` minimal (tous les agents)

```python
# packages/agents/protocols.py
"""Agent interface protocols for CryoSwarm-Q orchestration layer.

Defines the minimal contract that all agents must satisfy,
plus category-specific protocols for type-safe pipeline composition.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from packages.core.enums import AgentName, DecisionType
from packages.core.models import AgentDecision


@runtime_checkable
class AgentProtocol(Protocol):
    """Minimal contract for all CryoSwarm-Q agents.

    Every agent must:
    - expose its identity via agent_name
    - be able to produce structured decisions for audit trail
    """

    agent_name: AgentName

    def build_decision(
        self,
        campaign_id: str,
        subject_id: str,
        decision_type: DecisionType,
        status: str,
        reasoning_summary: str,
        structured_output: dict[str, Any],
    ) -> AgentDecision: ...
```

#### Niveau 2 — Protocoles par catégorie

```python
from packages.core.models import (
    BackendChoice,
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    ExperimentSpec,
    MemoryRecord,
    RegisterCandidate,
    RobustnessReport,
    SequenceCandidate,
)


@runtime_checkable
class ProblemFramingProtocol(AgentProtocol, Protocol):
    """Agent that frames an experiment goal into a structured specification."""

    def run(
        self,
        goal: ExperimentGoal,
        memory_records: list[MemoryRecord] | None = None,
    ) -> ExperimentSpec: ...


@runtime_checkable
class GeometryProtocol(AgentProtocol, Protocol):
    """Agent that generates hardware-valid atom register candidates."""

    def run(
        self,
        spec: ExperimentSpec,
        campaign_id: str,
        memory_records: list[MemoryRecord] | None = None,
    ) -> list[RegisterCandidate]: ...


@runtime_checkable
class SequenceProtocol(AgentProtocol, Protocol):
    """Agent that generates pulse-sequence candidates per register."""

    def run(
        self,
        spec: ExperimentSpec,
        register_candidate: RegisterCandidate,
        campaign_id: str,
        memory_records: list[MemoryRecord] | None = None,
    ) -> list[SequenceCandidate]: ...


@runtime_checkable
class NoiseEvaluationProtocol(AgentProtocol, Protocol):
    """Agent that evaluates robustness under noise perturbations."""

    def run(
        self,
        spec: ExperimentSpec,
        register_candidate: RegisterCandidate,
        sequence_candidate: SequenceCandidate,
    ) -> RobustnessReport: ...


@runtime_checkable
class RoutingProtocol(AgentProtocol, Protocol):
    """Agent that recommends a backend for a given candidate."""

    def run(
        self,
        spec: ExperimentSpec,
        sequence_candidate: SequenceCandidate,
        report: RobustnessReport,
    ) -> BackendChoice: ...


@runtime_checkable
class CampaignProtocol(AgentProtocol, Protocol):
    """Agent that ranks candidates and produces a campaign summary."""

    def run(
        self,
        campaign: CampaignState,
        evaluations: list[EvaluationResult],
    ) -> tuple[CampaignState, list[EvaluationResult]]: ...


@runtime_checkable
class ResultsProtocol(AgentProtocol, Protocol):
    """Agent that produces a human-readable campaign summary."""

    def run(
        self,
        goal: ExperimentGoal,
        spec: ExperimentSpec,
        campaign: CampaignState,
        ranked_candidates: list[EvaluationResult],
    ) -> dict[str, object]: ...


@runtime_checkable
class MemoryCaptureProtocol(AgentProtocol, Protocol):
    """Agent that extracts reusable lessons from campaign results."""

    def run(
        self,
        campaign_id: str,
        ranked_candidates: list[EvaluationResult],
        sequence_lookup: dict[str, SequenceCandidate],
        register_lookup: dict[str, RegisterCandidate],
    ) -> list[MemoryRecord]: ...
```

### Modifications dans `BaseAgent`

`BaseAgent` reste tel quel dans `packages/agents/base.py` — c'est une implémentation concrète. Les protocoles sont dans un fichier séparé et servent de contrat. On ajoute juste une vérification de conformité :

```python
# packages/agents/base.py
# Ajouter en fin de fichier :
from packages.agents.protocols import AgentProtocol

# Vérification statique que BaseAgent satisfait le protocole minimal
assert isinstance(BaseAgent.__new__(BaseAgent), AgentProtocol) is False  # ABC, pas instanciable directement
# La vérification effective se fait sur les sous-classes concrètes
```

En fait, la vraie vérification se fait dans le pipeline et les tests.

### Modifications dans `pipeline.py`

Typer les attributs du pipeline en utilisant les protocoles au lieu des classes concrètes :

```python
from packages.agents.protocols import (
    CampaignProtocol,
    GeometryProtocol,
    MemoryCaptureProtocol,
    NoiseEvaluationProtocol,
    ProblemFramingProtocol,
    ResultsProtocol,
    RoutingProtocol,
)

class CryoSwarmPipeline:
    def __init__(
        self,
        repository: CryoSwarmRepository | None = None,
        parallel: bool = False,
        sequence_strategy_mode: str = "adaptive",
        rl_checkpoint_path: str | None = None,
        param_space: PhysicsParameterSpace | None = None,
        # Agent injection (for testing and extensibility)
        problem_agent: ProblemFramingProtocol | None = None,
        geometry_agent: GeometryProtocol | None = None,
        noise_agent: NoiseEvaluationProtocol | None = None,
        routing_agent: RoutingProtocol | None = None,
        campaign_agent: CampaignProtocol | None = None,
        results_agent: ResultsProtocol | None = None,
        memory_agent: MemoryCaptureProtocol | None = None,
    ) -> None:
        self.repository = repository
        self.parallel = parallel
        self.param_space = param_space or PhysicsParameterSpace.default()
        self.logger = get_logger(__name__)
        self.problem_agent: ProblemFramingProtocol = problem_agent or ProblemFramingAgent()
        self.geometry_agent: GeometryProtocol = geometry_agent or GeometryAgent(param_space=self.param_space)
        # ... etc pour chaque agent
```

Cela permet :
- L'injection de mocks/stubs typés dans les tests
- Le remplacement d'un agent par un autre conforme au protocole
- La vérification mypy que les protocoles sont respectés

### Tests

Créer `tests/test_agent_protocols.py` :

```python
"""Verify that all concrete agents satisfy their respective protocols."""
from __future__ import annotations

import pytest

from packages.agents.campaign_agent import CampaignAgent
from packages.agents.geometry_agent import GeometryAgent
from packages.agents.memory_agent import MemoryAgent
from packages.agents.noise_agent import NoiseRobustnessAgent
from packages.agents.problem_agent import ProblemFramingAgent
from packages.agents.results_agent import ResultsAgent
from packages.agents.routing_agent import BackendRoutingAgent
from packages.agents.sequence_agent import SequenceAgent
from packages.agents.protocols import (
    AgentProtocol,
    CampaignProtocol,
    GeometryProtocol,
    MemoryCaptureProtocol,
    NoiseEvaluationProtocol,
    ProblemFramingProtocol,
    ResultsProtocol,
    RoutingProtocol,
    SequenceProtocol,
)


_PROTOCOL_MAP: list[tuple[type, type]] = [
    (ProblemFramingAgent, ProblemFramingProtocol),
    (GeometryAgent, GeometryProtocol),
    (SequenceAgent, SequenceProtocol),
    (NoiseRobustnessAgent, NoiseEvaluationProtocol),
    (BackendRoutingAgent, RoutingProtocol),
    (CampaignAgent, CampaignProtocol),
    (ResultsAgent, ResultsProtocol),
    (MemoryAgent, MemoryCaptureProtocol),
]


@pytest.mark.parametrize(
    "agent_class,protocol",
    _PROTOCOL_MAP,
    ids=[pair[0].__name__ for pair in _PROTOCOL_MAP],
)
def test_agent_satisfies_protocol(agent_class: type, protocol: type) -> None:
    agent = agent_class()
    assert isinstance(agent, protocol), (
        f"{agent_class.__name__} does not satisfy {protocol.__name__}"
    )


@pytest.mark.parametrize(
    "agent_class",
    [pair[0] for pair in _PROTOCOL_MAP],
    ids=[pair[0].__name__ for pair in _PROTOCOL_MAP],
)
def test_all_agents_satisfy_base_protocol(agent_class: type) -> None:
    agent = agent_class()
    assert isinstance(agent, AgentProtocol)
```

### Ce qui NE CHANGE PAS

- Les signatures `run()` de chaque agent restent identiques
- `BaseAgent` reste une classe abstraite avec `build_decision()`
- Aucun agent existant n'est modifié (sauf potentiellement les type annotations)

---

## TÂCHE 2 — Schématiser les metadata avec des TypedDict nommés

### Problème actuel

`JsonDict = dict[str, Any]` est utilisé partout pour les champs `metadata`, `signals`, `nominal_observables`, `hamiltonian_metrics`, `scenario_observables`, `summary_report`, `structured_output`. Aucun contrat sur les clés attendues. Les agents lisent des clés avec `.get("key", default)` sans aucune validation.

Exemples de clés lues sans schéma :

**`RegisterCandidate.metadata`** :
- `spacing_um` (lu par `memory_agent`, `sequence_agent`, `rl_env`, `dataset`, `data_generator`, `emulator_router`)

**`SequenceCandidate.metadata`** :
- `atom_count` (lu par `emulator_router`)
- `spacing_um` (lu par `sequence_agent`)
- `detuning_end` (lu par `evaluators.py`, `pulser_adapter`)
- `amplitude_start` (lu par `pulser_adapter`)
- `source` (lu/écrit par `sequence_strategy`)
- `problem_class` (lu/écrit par `pipeline`)
- `strategy_used` (lu/écrit par `pipeline`)

**`EvaluationResult.metadata`** :
- `score_std` (lu par `memory_agent`)
- `nominal_observables` (lu par `memory_agent`)
- `hamiltonian_metrics` (lu par `memory_agent`)
- `register_label` (écrit par `pipeline`)
- `backend_rationale` (écrit par `pipeline`)

**`MemoryRecord.signals`** :
- `objective_score`, `robustness_score`, `worst_case_score`, `backend_choice`, `sequence_family`, `layout_type`, `atom_count`, `spacing_um`, `amplitude`, `detuning`, `duration_ns`, `confidence`, `spectral_gap`, `noise_degradation`

**`NoiseScenario.metadata`** :
- `spatial_inhomogeneity` (lu par `evaluators.py`)
- `sampled` (écrit par `parameter_space`)

### Objectif

Créer des `TypedDict` nommés pour chaque usage distinct de metadata, puis les utiliser comme type annotations (sans casser les modèles Pydantic existants, qui gardent `dict[str, Any]` en runtime).

### Architecture

Créer `packages/core/metadata_schemas.py` :

```python
"""Typed metadata schemas for CryoSwarm-Q domain objects.

These TypedDict definitions document the expected structure of metadata
dictionaries used across agents, pipeline, and scoring modules.

They are used as type annotations for documentation and static analysis.
At runtime, Pydantic fields remain dict[str, Any] for backward compatibility.
"""
from __future__ import annotations

from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# RegisterCandidate.metadata
# ---------------------------------------------------------------------------

class RegisterMetadata(TypedDict, total=False):
    """Metadata attached to a RegisterCandidate by GeometryAgent."""
    spacing_um: float


# ---------------------------------------------------------------------------
# SequenceCandidate.metadata
# ---------------------------------------------------------------------------

class SequenceMetadata(TypedDict, total=False):
    """Metadata attached to a SequenceCandidate by SequenceAgent and pipeline."""
    atom_count: int
    layout_type: str
    spacing_um: float | None
    source: str  # "heuristic" | "rl_policy"
    problem_class: str
    strategy_used: str
    # Pulse parameters (family-dependent)
    detuning_end: float
    amplitude_start: float


# ---------------------------------------------------------------------------
# EvaluationResult.metadata
# ---------------------------------------------------------------------------

class EvaluationMetadata(TypedDict, total=False):
    """Metadata attached to an EvaluationResult by the pipeline."""
    register_label: str
    backend_rationale: str
    nominal_observables: dict[str, Any]
    hamiltonian_metrics: dict[str, Any]
    score_std: float


# ---------------------------------------------------------------------------
# MemoryRecord.signals
# ---------------------------------------------------------------------------

class MemorySignals(TypedDict, total=False):
    """Structured signals stored in MemoryRecord for cross-campaign reuse."""
    objective_score: float
    robustness_score: float
    worst_case_score: float
    backend_choice: str
    sequence_family: str
    layout_type: str
    atom_count: int
    spacing_um: float | None
    amplitude: float
    detuning: float
    duration_ns: int
    confidence: float
    spectral_gap: float
    noise_degradation: float


# ---------------------------------------------------------------------------
# NoiseScenario.metadata
# ---------------------------------------------------------------------------

class NoiseScenarioMetadata(TypedDict, total=False):
    """Metadata for noise scenario configuration."""
    spatial_inhomogeneity: float
    sampled: bool


# ---------------------------------------------------------------------------
# CampaignState.summary_report
# ---------------------------------------------------------------------------

class CampaignSummaryReport(TypedDict, total=False):
    """Summary report structure for CampaignState."""
    reason: str
    error: str
    backend_mix: dict[str, int]
    results_summary: dict[str, object]
    top_candidate_id: str | None
    top_objective_score: float | None
    ranked_candidate_count: int


# ---------------------------------------------------------------------------
# BackendChoice.metadata
# ---------------------------------------------------------------------------

class BackendChoiceMetadata(TypedDict, total=False):
    """Metadata attached to a BackendChoice."""
    objective_class: str
    atom_count: int
    hamiltonian_dimension: int


# ---------------------------------------------------------------------------
# ExperimentSpec.metadata
# ---------------------------------------------------------------------------

class SpecMetadata(TypedDict, total=False):
    """Metadata attached to ExperimentSpec by ProblemFramingAgent."""
    priority: str
    goal_constraints: dict[str, Any]
    memory_record_count: int
    remembered_backends: list[str | None]
    max_register_candidates: int


# ---------------------------------------------------------------------------
# Observable dictionaries (simulation output)
# ---------------------------------------------------------------------------

class NominalObservables(TypedDict, total=False):
    """Observables extracted from simulation results."""
    rydberg_density: float
    target_density: float
    density_score: float
    blockade_violation: float
    blockade_score: float
    interaction_energy: float
    interaction_energy_norm: float
    final_populations: list[float]
    top_bitstrings: dict[str, int]
    entanglement_entropy: float
    antiferromagnetic_order: float
    connected_correlations: list[float]
    observable_score: float
    hamiltonian_metrics: dict[str, float | int]
    noise_label: str
    # numpy fallback extras
    spatial_inhomogeneity: float
    spatial_drive_factors: list[float]
    backend: str


class HamiltonianMetrics(TypedDict, total=False):
    """Hamiltonian characterization metrics."""
    dimension: int
    frobenius_norm: float
    spectral_radius: float
```

### Intégration

La beauté de cette approche : **AUCUN modèle Pydantic ne change en runtime.** Les TypedDicts sont utilisés uniquement pour :

1. **Type annotations dans les agents** :
```python
# packages/agents/memory_agent.py
from packages.core.metadata_schemas import EvaluationMetadata, MemorySignals

class MemoryAgent(BaseAgent):
    def run(self, ...) -> list[MemoryRecord]:
        for result in ranked_candidates[:3]:
            # Indiquer au lecteur la structure attendue :
            meta: EvaluationMetadata = result.metadata  # type: ignore[assignment]
            score_std = float(meta.get("score_std", 0.0))
```

2. **Documentation vivante** : Au lieu de `JsonDict`, le reader sait exactement quelles clés existent.

3. **Validation optionnelle en mode debug** (ajouter si souhaité) :
```python
# packages/core/metadata_schemas.py
def validate_metadata(data: dict[str, Any], schema: type[TypedDict]) -> list[str]:
    """Check metadata dict against TypedDict schema. Returns list of warnings."""
    annotations = schema.__annotations__
    warnings = []
    for key in data:
        if key not in annotations:
            warnings.append(f"Unexpected key '{key}' not in {schema.__name__}")
    return warnings
```

### Annotations à ajouter dans les fichiers existants

Pour chaque module qui fait `.metadata.get(...)` ou `.signals.get(...)`, ajoute l'import et une annotation locale. **NE change PAS le type du champ Pydantic** — garde `JsonDict`. Ajoute seulement des annotations locales dans les fonctions :

| Fichier | Import à ajouter | Annotation |
|---------|------------------|------------|
| `packages/agents/memory_agent.py` | `EvaluationMetadata, MemorySignals` | `meta: EvaluationMetadata = result.metadata` |
| `packages/agents/sequence_agent.py` | `RegisterMetadata` | (pour `register_candidate.metadata.get("spacing_um")`) |
| `packages/agents/geometry_agent.py` | `SpecMetadata` | (pour `spec.metadata.get("max_register_candidates")`) |
| `packages/pasqal_adapters/emulator_router.py` | `SequenceMetadata` | (pour `sequence.metadata.get("atom_count")`) |
| `packages/pasqal_adapters/pulser_adapter.py` | `SequenceMetadata` | (pour `metadata.get("detuning_end")`, `metadata.get("amplitude_start")`) |
| `packages/simulation/evaluators.py` | `SequenceMetadata, NoiseScenarioMetadata` | (pour `metadata.get("detuning_end")`, `metadata.get("spatial_inhomogeneity")`) |
| `packages/orchestration/pipeline.py` | `SequenceMetadata` | (pour `sequence.metadata.setdefault(...)`) |

### Tests

Créer `tests/test_metadata_schemas.py` :

```python
"""Verify metadata schema coverage and typing."""
from __future__ import annotations

from packages.core.metadata_schemas import (
    BackendChoiceMetadata,
    CampaignSummaryReport,
    EvaluationMetadata,
    HamiltonianMetrics,
    MemorySignals,
    NoiseScenarioMetadata,
    NominalObservables,
    RegisterMetadata,
    SequenceMetadata,
    SpecMetadata,
)


def test_register_metadata_keys() -> None:
    meta: RegisterMetadata = {"spacing_um": 7.0}
    assert meta["spacing_um"] == 7.0


def test_sequence_metadata_keys() -> None:
    meta: SequenceMetadata = {
        "atom_count": 6,
        "source": "heuristic",
        "detuning_end": 10.0,
    }
    assert meta["source"] == "heuristic"


def test_evaluation_metadata_keys() -> None:
    meta: EvaluationMetadata = {
        "score_std": 0.05,
        "register_label": "square-6",
        "nominal_observables": {"rydberg_density": 0.4},
        "hamiltonian_metrics": {"spectral_radius": 3.2},
    }
    assert meta["score_std"] == 0.05


def test_memory_signals_keys() -> None:
    signals: MemorySignals = {
        "objective_score": 0.72,
        "confidence": 0.85,
        "sequence_family": "adiabatic_sweep",
        "amplitude": 5.0,
        "detuning": -10.0,
    }
    assert signals["confidence"] == 0.85


def test_all_schemas_are_total_false() -> None:
    """All schemas use total=False so partial dicts are valid."""
    schemas = [
        RegisterMetadata,
        SequenceMetadata,
        EvaluationMetadata,
        MemorySignals,
        NoiseScenarioMetadata,
        CampaignSummaryReport,
        BackendChoiceMetadata,
        SpecMetadata,
        NominalObservables,
        HamiltonianMetrics,
    ]
    for schema in schemas:
        assert schema.__total__ is False, f"{schema.__name__} should use total=False"
```

---

## TÂCHE 3 — Remplacer `except Exception` par des exceptions spécifiques

### Problème actuel

23 occurrences de `except Exception` dans `packages/`. Elles masquent des bugs (TypeError, KeyError, AttributeError) en les traitant comme des erreurs attendues. Certaines avalent l'exception sans log (`except Exception: pass`).

### Inventaire complet des `except Exception` à traiter

#### Catégorie A — Exceptions silencieuses (CRITIQUE — doivent être loguées)

| Fichier | Ligne | Contexte | Action |
|---------|-------|----------|--------|
| `evaluators.py` | 174 | `except Exception: pass` dans `_extract_observables()` | Ajouter `logger.debug("Could not extract extended observables from state vector: %s", exc)` |
| `rl_env.py` | 279 | `except Exception:` retourne `0.0, {...}` | Ajouter `logger.warning("RL environment evaluation failed: %s", exc)` |
| `data_generator.py` | 207 | `except Exception: return None` | Ajouter `logger.debug("Sample evaluation failed: %s", exc)` |
| `data_generator.py` | 500 | `except Exception: results.append(None)` | Ajouter `logger.debug("Worker evaluation failed: %s", exc)` |
| `sequence_strategy.py` | 214 | `except Exception: return 0.75` dans JSON sidecar parsing | Ajouter `logger.debug("Could not parse checkpoint sidecar: %s", exc)` |
| `pulser_adapter.py` | 265 | `except Exception: pass` dans abstract_repr | Ajouter `logger.debug("Could not serialize abstract sequence: %s", exc)` |

#### Catégorie B — `except Exception` trop large dans le pipeline (gardés mais affinés)

Les `except Exception` dans `pipeline.py` (lignes 89, 321, 418, 471, 550, 684, 723, 759, 801) sont des **fire-walls intentionnels** — le pipeline DOIT absorber toutes les exceptions d'agents pour ne pas crasher. Ces catches restent `except Exception`, MAIS :

1. Chaque catch doit loguer la **classe exacte** de l'exception :
```python
except Exception as exc:
    self.logger.error(
        "Agent %s failed with %s: %s",
        agent.agent_name.value,
        type(exc).__name__,
        exc,
    )
```

2. Le catch global de `run()` (ligne 801) doit aussi loguer le traceback :
```python
except Exception as exc:
    self.logger.error("Unhandled pipeline failure: %s", exc, exc_info=True)
```

#### Catégorie C — Imports conditionnels (acceptables)

Les `except Exception` dans les imports (lignes `pulser_adapter.py:19`, `pasqal_cloud_adapter.py:13`, `qoolqit_adapter.py:11`) sont des patterns d'import facultatif. Ils sont acceptables mais doivent être affinés :

```python
# Avant
except Exception as exc:

# Après
except (ImportError, ModuleNotFoundError) as exc:
```

#### Catégorie D — Adapters externes (à affiner)

| Fichier | Ligne | Contexte | Remplacement |
|---------|-------|----------|-------------- |
| `pulser_adapter.py` | 110 | `summarize_register_physics()` device validation | `except (ValueError, TypeError) as exc:` |
| `pulser_adapter.py` | 161 | `create_simple_register()` Pulser register creation | `except (ValueError, RuntimeError) as exc:` |

### Créer un module d'exceptions spécifiques

Créer `packages/core/exceptions.py` :

```python
"""Domain-specific exceptions for CryoSwarm-Q.

Provides a structured exception hierarchy for agent failures,
simulation errors, and infrastructure failures. Replaces broad
`except Exception` patterns with catchable domain exceptions.
"""
from __future__ import annotations


class CryoSwarmError(Exception):
    """Base exception for all CryoSwarm-Q domain errors."""


# ----- Agent errors -----

class AgentError(CryoSwarmError):
    """An agent failed during execution."""

    def __init__(self, agent_name: str, message: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"[{agent_name}] {message}")


class ProblemFramingError(AgentError):
    """Problem framing agent could not produce a specification."""

    def __init__(self, message: str) -> None:
        super().__init__("problem_framing_agent", message)


class GeometryError(AgentError):
    """Geometry agent produced invalid candidates."""

    def __init__(self, message: str) -> None:
        super().__init__("geometry_agent", message)


class SequenceError(AgentError):
    """Sequence generation failed."""

    def __init__(self, message: str) -> None:
        super().__init__("sequence_agent", message)


class EvaluationError(AgentError):
    """Robustness evaluation or scoring failed."""

    def __init__(self, message: str) -> None:
        super().__init__("noise_robustness_agent", message)


# ----- Simulation errors -----

class SimulationError(CryoSwarmError):
    """Raised when a physics simulation fails."""


class HamiltonianError(SimulationError):
    """Raised when Hamiltonian construction or diagonalization fails."""


class EmulatorError(SimulationError):
    """Raised when a backend emulator fails."""


# ----- Infrastructure errors -----

class RepositoryError(CryoSwarmError):
    """Raised when database operations fail."""


class ConfigurationError(CryoSwarmError):
    """Raised when a configuration value is invalid or missing."""


class AdapterError(CryoSwarmError):
    """Raised when a Pasqal/Pulser adapter fails."""
```

### Tests

Créer `tests/test_exceptions.py` :

```python
"""Verify exception hierarchy and picklability."""
from __future__ import annotations

import pickle

import pytest

from packages.core.exceptions import (
    AdapterError,
    AgentError,
    ConfigurationError,
    CryoSwarmError,
    EmulatorError,
    EvaluationError,
    GeometryError,
    HamiltonianError,
    ProblemFramingError,
    RepositoryError,
    SequenceError,
    SimulationError,
)


def test_hierarchy() -> None:
    assert issubclass(AgentError, CryoSwarmError)
    assert issubclass(SimulationError, CryoSwarmError)
    assert issubclass(RepositoryError, CryoSwarmError)
    assert issubclass(HamiltonianError, SimulationError)
    assert issubclass(ProblemFramingError, AgentError)


def test_agent_error_message_includes_name() -> None:
    err = ProblemFramingError("bad goal")
    assert "problem_framing_agent" in str(err)
    assert "bad goal" in str(err)


@pytest.mark.parametrize(
    "exc_class",
    [CryoSwarmError, AgentError, SimulationError, RepositoryError, ConfigurationError, AdapterError],
)
def test_exceptions_are_picklable(exc_class: type) -> None:
    if exc_class is AgentError:
        exc = exc_class("test_agent", "test message")
    else:
        exc = exc_class("test message")
    restored = pickle.loads(pickle.dumps(exc))
    assert str(restored) == str(exc)
```

---

## TÂCHE 4 — Ajouter des logs pour tous les échecs silencieux

### Problème actuel

Nombreux endroits où des erreurs ou des pertes de données passent sans aucun log. L'opérateur ne sait pas pourquoi un candidat disparaît ou un score est à 0.

### Liste exhaustive des sites à instrumenter

| Fichier | Ligne | Situation | Log à ajouter |
|---------|-------|-----------|---------------|
| `evaluators.py` | ~174 | `except Exception: pass` — observables étendus | `logger.debug("Extended observable extraction failed for %d atoms: %s", n, exc)` |
| `rl_env.py` | ~279 | Évaluation RL retourne 0 | `logger.warning("RL candidate evaluation yielded fallback score: %s", exc)` |
| `rl_env.py` | ~100 | Pool de candidats vide | `logger.warning("Empty register candidate pool; using fallback register")` — ce path existe-t-il ? Vérifie. |
| `data_generator.py` | ~207 | Sample retourne None | `logger.debug("Sample generation failed: %s", exc)` |
| `data_generator.py` | ~500 | Worker retourne None | `logger.debug("Parallel evaluation worker failed: %s", exc)` |
| `dataset.py` | ~191 | `add_from_pipeline()` perd des samples | Ajouter un compteur de samples perdus et loguer en fin : `logger.info("Pipeline ingestion: %d accepted, %d dropped (missing pairs)", accepted, dropped)` |
| `surrogate_filter.py` | ~187 | Candidat rejeté haute incertitude | `logger.debug("Candidate %s rejected: uncertainty %.4f exceeds threshold", candidate.id, uncertainty)` |
| `surrogate_filter.py` | ~200 | Candidat rejeté score trop bas | `logger.debug("Candidate %s rejected: predicted score %.4f below threshold", candidate.id, score)` |
| `sequence_strategy.py` | ~214 | JSON sidecar non parsable | `logger.debug("Checkpoint sidecar %s could not be parsed: %s", sidecar, exc)` |
| `sequence_agent.py` | ~110 | Dernier variant supprimé via `_family_has_weak_failure` | `logger.debug("Dropped trailing %s variant due to weak failure memory for family %s", family.value)` — ATTENTION : ce log doit être dans `_family_variants()`, pas dans le if/else |
| `pipeline.py` | ~89 | `_safe_repository_call()` avale l'exception | Déjà loggé (`self.logger.error`), mais le level doit être `warning` pas `error` pour les opérations non critiques. Ajouter le nom de la classe d'exception : `self.logger.warning("Repository %s failed with %s: %s", description, type(exc).__name__, exc)` |
| `pulser_adapter.py` | ~265 | abstract_repr sérialisation | `logger.debug("Could not generate abstract sequence representation: %s", exc)` |
| `memory_agent.py` | ~25-29 | Double fallback `.get()` pour `spectral_gap` | Pas besoin de log ici, c'est du code défensif légitime. |

### Règles

- **Ne JAMAIS loguer au niveau ERROR pour une situation attendue** (ex: import optionnel manquant, candidat rejeté). Utiliser `DEBUG` ou `WARNING`.
- **Loguer au niveau ERROR uniquement pour les défaillances inattendues** (ex: base de données inaccessible, agent qui crash).
- **Inclure le contexte** : nom de l'agent, ID du candidat, nombre d'atomes, famille de séquence.
- **Ne JAMAIS loguer de credentials ou tokens.**
- **Chaque log doit être une phrase complète et parsable** — pas de `log("fail")`.

### Pattern de log standard

```python
# Pour les agents :
self.logger.warning(
    "Agent %s failed during %s for %s: %s",
    self.agent_name.value,
    "robustness_evaluation",
    sequence_candidate.id,
    exc,
)

# Pour les modules standalone :
logger.debug(
    "Candidate evaluation skipped for %d-atom %s register: %s",
    register.atom_count,
    register.layout_type,
    exc,
)
```

---

## TÂCHE 5 — Unifier les limites d'atomes en une seule constante partagée

### Problème actuel

Quatre limites d'atomes différentes à travers le codebase, sans lien entre elles :

| Fichier | Constante | Valeur | Usage |
|---------|-----------|--------|-------|
| `numpy_backend.py` | `MAX_ATOMS_DENSE` | 12 | Seuil pour matrice dense |
| `numpy_backend.py` | `MAX_ATOMS_SPARSE` | 18 | Seuil pour méthodes sparse |
| `evaluators.py` | hardcoded | 14 | Seuil dense → sparse dans evaluator |
| `gpu_backend.py` | `MAX_ATOMS_GPU` | 24 | Limite GPU |

### Solution

**Si P0 Tâche 3 a été faite** : Ces constantes sont dans `PhysicsParameterSpace` (`max_atoms_dense`, `max_atoms_sparse`, `max_atoms_gpu`, `max_atoms_evaluator_parallel`). Il suffit de propager l'utilisation.

**Si P0 Tâche 3 n'a pas été faite** : Créer un unique point de vérité dans `PhysicsParameterSpace` et faire les mêmes modifications que décrites dans P0 Tâche 3. Consulte `P0_FONDATIONS_PROMPT.md` pour les spécifications.

### Modifications restantes (après P0)

Les modules doivent utiliser les constantes du param_space au lieu de littéraux locaux.

1. **`packages/simulation/numpy_backend.py`** :
```python
# Supprimer les constantes locales
# MAX_ATOMS_DENSE = 12  ← supprimer
# MAX_ATOMS_SPARSE = 18  ← supprimer

def simulate_rydberg_evolution(
    ...,
    max_atoms_dense: int | None = None,
    max_atoms_sparse: int | None = None,
) -> dict[str, Any]:
    # Utiliser les defaults depuis PhysicsParameterSpace si non fourni
    ps = PhysicsParameterSpace.default()
    dense_limit = max_atoms_dense if max_atoms_dense is not None else ps.max_atoms_dense
    sparse_limit = max_atoms_sparse if max_atoms_sparse is not None else ps.max_atoms_sparse
    ...
```

2. **`packages/simulation/evaluators.py`** — remplacer le littéral `14` :
```python
def simulate_sequence_candidate(
    ...,
    param_space: PhysicsParameterSpace | None = None,
) -> ...:
    ps = param_space or PhysicsParameterSpace.default()
    if register_candidate.atom_count > ps.max_atoms_evaluator_parallel:
        logger.warning(...)
        return _simulate_with_numpy_fallback(...)
```

3. **`packages/ml/gpu_backend.py`** — remplacer `MAX_ATOMS_GPU = 24` :
```python
# Le param_space doit être passé ou lu depuis le default
class GPUSimulator:
    def __init__(self, ..., param_space: PhysicsParameterSpace | None = None):
        ps = param_space or PhysicsParameterSpace.default()
        self.max_atoms = ps.max_atoms_gpu
```

### Tests

```python
def test_atom_limits_are_consistent() -> None:
    ps = PhysicsParameterSpace.default()
    assert ps.max_atoms_dense < ps.max_atoms_sparse
    assert ps.max_atoms_sparse <= ps.max_atoms_gpu
    assert ps.max_atoms_dense <= ps.max_atoms_evaluator_parallel <= ps.max_atoms_sparse
```

---

## TÂCHE 6 — Ajouter marqueurs pytest et configuration de couverture

### Problème actuel

- Aucun marqueur pytest défini (pas de `@pytest.mark.gpu`, `@pytest.mark.slow`, etc.)
- Pas de configuration de couverture
- Impossible de lancer un sous-ensemble de tests ciblé

### 6a — Marqueurs dans `pyproject.toml`

Ajouter la section markers dans `[tool.pytest.ini_options]` :

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = [
    "gpu: tests requiring CUDA or ROCm GPU backend",
    "slow: tests taking more than 10 seconds",
    "integration: end-to-end pipeline integration tests",
    "mongo: tests requiring a live MongoDB connection",
    "pulser: tests requiring the pulser package",
]
addopts = "--strict-markers"
```

**Note** : Si `conftest.py` a déjà été créé avec `pytest_configure()` qui enregistre les marqueurs (P0 Tâche 2), il faut s'assurer que les deux sources sont cohérentes. L'idéal est de garder les marqueurs UNIQUEMENT dans `pyproject.toml` et supprimer le `pytest_configure()` hook de conftest — `pyproject.toml` est canonique, `conftest.py` est redondant.

### 6b — Annoter les tests existants

Faire un scan systématique des fichiers de tests et ajouter les marqueurs :

| Marqueur | Fichiers à annoter |
|----------|--------------------|
| `@pytest.mark.gpu` | `test_ml_gpu.py` |
| `@pytest.mark.slow` | `test_pipeline_integration.py`, `test_parallel_pipeline.py`, `test_benchmark.py` |
| `@pytest.mark.integration` | `test_pipeline_integration.py`, `test_pipeline_failures.py` |
| `@pytest.mark.mongo` | `test_db_repository.py` |
| `@pytest.mark.pulser` | Tout test qui fait `pytest.importorskip("pulser")` ou vérifie `PULSER_AVAILABLE` |

Pour chaque fichier, ajoute le marqueur au niveau module (pour tous les tests du fichier) ou au niveau des tests individuels :

```python
# Pour appliquer à tout le module :
import pytest
pytestmark = pytest.mark.integration

# Pour un test individuel :
@pytest.mark.slow
def test_full_pipeline_runs():
    ...
```

### 6c — Configuration de couverture

Ajouter dans `pyproject.toml` :

```toml
[tool.coverage.run]
source = ["packages", "apps"]
omit = [
    "packages/ml/gpu_backend.py",
    "apps/dashboard/*",
]

[tool.coverage.report]
show_missing = true
skip_covered = true
fail_under = 60
exclude_lines = [
    "pragma: no cover",
    "if __name__",
    "if TYPE_CHECKING",
    "raise NotImplementedError",
]
```

Et dans les dev dependencies de `pyproject.toml` :

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "httpx>=0.27",
    "pytest-cov>=5.0",
]
```

### 6d — Commandes de CI canoniques

Documenter les commandes dans le prompt :

```bash
# Tous les tests
pytest tests/ -v --tb=short

# Rapide (excluant slow, gpu, mongo)
pytest tests/ -v --tb=short -m "not slow and not gpu and not mongo"

# Avec couverture
pytest tests/ --cov=packages --cov=apps --cov-report=term-missing

# Uniquement intégration
pytest tests/ -v -m integration
```

---

## TÂCHE 7 — Ajouter tests API routes et dashboard

### Problème actuel

- **API** : Seuls les cas d'erreur sont testés (`test_api_error_handling.py`) + un test de health. Aucun test pour les parcours nominaux.
- **Dashboard** : 0 test. `apps/dashboard/app.py` est directement couplé à MongoDB.

### 7a — Tests API complets

Créer `tests/test_api_routes.py` :

```python
"""Comprehensive API route tests for CryoSwarm-Q.

Tests all routes through FastAPI TestClient with mocked repository.
Does NOT require MongoDB — all persistence is faked.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.api.dependencies import get_repository
from apps.api.main import app
from packages.core.enums import BackendType, CampaignStatus, CandidateStatus, GoalStatus
from packages.core.models import (
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    ExperimentSpec,
    PipelineSummary,
)

# Import from conftest factories (P0 should have created these)
# If conftest is not yet available, define minimal factories inline
try:
    from conftest import make_campaign, make_evaluation_result, make_goal, make_spec
except ImportError:
    # Inline fallbacks if conftest doesn't exist yet
    # (Define minimal factories here)
    pass


class FakeRepository:
    """In-memory repository stub for API testing."""

    def __init__(self) -> None:
        self.goals: dict[str, ExperimentGoal] = {}
        self.campaigns: dict[str, CampaignState] = {}
        self.evaluation_results: dict[str, list[EvaluationResult]] = {}

    def create_goal(self, goal: ExperimentGoal) -> ExperimentGoal:
        self.goals[goal.id] = goal
        return goal

    def get_goal(self, goal_id: str) -> ExperimentGoal | None:
        return self.goals.get(goal_id)

    def get_campaign(self, campaign_id: str) -> CampaignState | None:
        return self.campaigns.get(campaign_id)

    def list_candidates_for_campaign(self, campaign_id: str) -> list[EvaluationResult]:
        return self.evaluation_results.get(campaign_id, [])

    def list_latest_campaigns(self, limit: int = 10) -> list[CampaignState]:
        return list(self.campaigns.values())[:limit]

    def list_agent_decisions(self, campaign_id: str) -> list:
        return []

    def list_robustness_reports(self, campaign_id: str) -> list:
        return []


@pytest.fixture()
def fake_repo() -> FakeRepository:
    return FakeRepository()


@pytest.fixture()
def client(fake_repo: FakeRepository) -> TestClient:
    app.dependency_overrides[get_repository] = lambda: fake_repo
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestHealthRoute:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestGoalRoutes:
    def test_create_goal_returns_201_or_200(self, client: TestClient) -> None:
        payload = {
            "title": "Test neutral-atom benchmark",
            "scientific_objective": "Evaluate robustness of small configurations.",
            "target_observable": "rydberg_density",
            "desired_atom_count": 6,
        }
        response = client.post("/goals", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test neutral-atom benchmark"
        assert data["status"] == "stored"
        assert data["id"].startswith("goal_")

    def test_create_goal_with_invalid_atom_count(self, client: TestClient) -> None:
        payload = {
            "title": "Bad",
            "scientific_objective": "Test",
            "desired_atom_count": 1,
        }
        response = client.post("/goals", json=payload)
        assert response.status_code == 422

    def test_create_goal_with_short_title(self, client: TestClient) -> None:
        payload = {
            "title": "AB",
            "scientific_objective": "Test",
        }
        response = client.post("/goals", json=payload)
        assert response.status_code == 422

    def test_get_existing_goal(self, client: TestClient, fake_repo: FakeRepository) -> None:
        goal = ExperimentGoal(
            title="Test goal retrieval",
            scientific_objective="Retrieve goal by ID.",
            status=GoalStatus.STORED,
        )
        fake_repo.goals[goal.id] = goal

        response = client.get(f"/goals/{goal.id}")
        assert response.status_code == 200
        assert response.json()["id"] == goal.id

    def test_get_missing_goal_returns_404(self, client: TestClient) -> None:
        response = client.get("/goals/nonexistent_id")
        assert response.status_code == 404


class TestCampaignRoutes:
    def test_get_missing_campaign_returns_404(self, client: TestClient) -> None:
        response = client.get("/campaigns/nonexistent_id")
        assert response.status_code == 404

    def test_list_campaign_candidates_404_for_missing(self, client: TestClient) -> None:
        response = client.get("/campaigns/nonexistent/candidates")
        assert response.status_code == 404

    def test_list_campaign_candidates_returns_empty(
        self, client: TestClient, fake_repo: FakeRepository,
    ) -> None:
        campaign = CampaignState(goal_id="goal_test", status=CampaignStatus.COMPLETED)
        fake_repo.campaigns[campaign.id] = campaign

        response = client.get(f"/campaigns/{campaign.id}/candidates")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_campaign_candidates_returns_results(
        self, client: TestClient, fake_repo: FakeRepository,
    ) -> None:
        campaign = CampaignState(goal_id="goal_test", status=CampaignStatus.COMPLETED)
        fake_repo.campaigns[campaign.id] = campaign
        result = EvaluationResult(
            campaign_id=campaign.id,
            sequence_candidate_id="seq_1",
            register_candidate_id="reg_1",
            nominal_score=0.8,
            robustness_score=0.75,
            worst_case_score=0.6,
            observable_score=0.78,
            objective_score=0.74,
            backend_choice=BackendType.LOCAL_PULSER_SIMULATION,
            estimated_cost=0.12,
            estimated_latency=0.15,
            final_rank=1,
            status=CandidateStatus.RANKED,
            reasoning_summary="Test result.",
        )
        fake_repo.evaluation_results[campaign.id] = [result]

        response = client.get(f"/campaigns/{campaign.id}/candidates")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["objective_score"] == 0.74


class TestRunDemo:
    def test_run_demo_default_payload(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that /campaigns/run-demo accepts the default payload."""
        import apps.api.routes.campaigns as campaigns_mod

        # Mock run_demo_campaign to avoid full pipeline execution
        def fake_run(request, repository):
            goal = ExperimentGoal(
                title=request.title,
                scientific_objective=request.scientific_objective,
                status=GoalStatus.COMPLETED,
            )
            campaign = CampaignState(goal_id=goal.id, status=CampaignStatus.COMPLETED)
            return PipelineSummary(campaign=campaign, goal=goal, status="COMPLETED")

        monkeypatch.setattr(campaigns_mod, "run_demo_campaign", fake_run)

        response = client.post("/campaigns/run-demo", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "COMPLETED"

    def test_run_demo_custom_payload(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import apps.api.routes.campaigns as campaigns_mod

        def fake_run(request, repository):
            goal = ExperimentGoal(
                title=request.title,
                scientific_objective=request.scientific_objective,
                status=GoalStatus.COMPLETED,
            )
            campaign = CampaignState(goal_id=goal.id, status=CampaignStatus.COMPLETED)
            return PipelineSummary(campaign=campaign, goal=goal, status="COMPLETED")

        monkeypatch.setattr(campaigns_mod, "run_demo_campaign", fake_run)

        response = client.post(
            "/campaigns/run-demo",
            json={
                "title": "Custom test campaign",
                "scientific_objective": "Custom objective",
                "desired_atom_count": 8,
                "preferred_geometry": "square",
            },
        )
        assert response.status_code == 200

    def test_run_demo_pipeline_crash_returns_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import apps.api.routes.campaigns as campaigns_mod

        monkeypatch.setattr(
            campaigns_mod,
            "run_demo_campaign",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated crash")),
        )

        response = client.post("/campaigns/run-demo", json={})
        assert response.status_code == 500


class TestGlobalExceptionHandler:
    def test_unhandled_error_does_not_leak_details(self, client: TestClient) -> None:
        """Verify that the global handler masks internal error details.
        
        NOTE: This test assumes P0 Tâche 7 fixed the handler to return
        a generic message. If not yet done, the test documents the expected
        behavior.
        """
        # Force an unhandled error by requesting a route with a bad dependency
        # The global handler should return a generic message
        response = client.get("/health")
        # health should work fine
        assert response.status_code == 200
```

### 7b — Tests Dashboard (logic extraction)

Le dashboard Streamlit (`apps/dashboard/app.py`) est directement couplé à MongoDB et Streamlit — impossible à tester unitairement tel quel.

**Stratégie** : Extraire la logique métier du dashboard dans des fonctions testables, puis tester ces fonctions.

Créer `apps/dashboard/logic.py` :

```python
"""Business logic for the CryoSwarm-Q dashboard.

Extracted from app.py for testability. All functions are pure
or take a repository as explicit dependency.
"""
from __future__ import annotations

from typing import Any

from packages.core.models import (
    CampaignState,
    EvaluationResult,
    RegisterCandidate,
    RobustnessReport,
)


def build_campaign_table(campaigns: list[CampaignState]) -> list[dict[str, Any]]:
    """Transform campaign models into table-ready dictionaries."""
    return [
        {
            "campaign_id": campaign.id,
            "goal_id": campaign.goal_id,
            "status": campaign.status.value,
            "candidate_count": campaign.candidate_count,
            "top_candidate_id": campaign.top_candidate_id,
        }
        for campaign in campaigns
    ]


def build_ranked_table(candidates: list[EvaluationResult]) -> list[dict[str, Any]]:
    """Transform evaluation results into table-ready dictionaries."""
    return [
        {
            "rank": candidate.final_rank,
            "sequence_candidate_id": candidate.sequence_candidate_id,
            "objective_score": candidate.objective_score,
            "robustness_score": candidate.robustness_score,
            "backend_choice": candidate.backend_choice.value,
        }
        for candidate in candidates
    ]


def build_decision_table(decisions: list[Any]) -> list[dict[str, str]]:
    """Transform agent decisions into table-ready dictionaries."""
    return [
        {
            "agent": decision.agent_name.value,
            "subject_id": decision.subject_id,
            "decision_type": decision.decision_type.value,
            "status": decision.status,
            "reasoning_summary": decision.reasoning_summary,
        }
        for decision in decisions
    ]


def select_robustness_chart_data(
    reports: list[RobustnessReport],
    limit: int = 5,
) -> tuple[list[str], list[float], list[float], list[float]]:
    """Extract chart-ready data from robustness reports.

    Returns (labels, nominal_scores, average_scores, worst_scores).
    """
    if not reports:
        return [], [], [], []
    top = reports[:limit]
    labels = [r.sequence_candidate_id[-6:] for r in top]
    nominal = [r.nominal_score for r in top]
    average = [r.perturbation_average for r in top]
    worst = [r.worst_case_score for r in top]
    return labels, nominal, average, worst


def select_noise_sensitivity_data(
    report: RobustnessReport,
) -> tuple[list[str], list[float]]:
    """Extract noise sensitivity plot data from a robustness report.

    Returns (scenario_labels, scenario_scores).
    """
    scenario_order = ["low_noise", "medium_noise", "stressed_noise"]
    labels = [label for label in scenario_order if label in report.scenario_scores]
    values = [report.scenario_scores[label] for label in labels]
    return labels, values


def build_register_lookup_from_documents(
    documents: list[dict[str, Any]],
) -> dict[str, RegisterCandidate]:
    """Parse raw MongoDB documents into a RegisterCandidate lookup."""
    return {
        doc["_id"]: RegisterCandidate.model_validate(
            {k: v for k, v in doc.items() if k != "_id"}
        )
        for doc in documents
    }
```

Créer `tests/test_dashboard_logic.py` :

```python
"""Tests for dashboard business logic (extracted from app.py).

Does NOT require Streamlit or MongoDB.
"""
from __future__ import annotations

import pytest

from apps.dashboard.logic import (
    build_campaign_table,
    build_ranked_table,
    select_noise_sensitivity_data,
    select_robustness_chart_data,
)
from packages.core.enums import BackendType, CampaignStatus, CandidateStatus
from packages.core.models import CampaignState, EvaluationResult, RobustnessReport


def test_build_campaign_table_empty() -> None:
    assert build_campaign_table([]) == []


def test_build_campaign_table() -> None:
    campaign = CampaignState(
        goal_id="goal_1",
        status=CampaignStatus.COMPLETED,
        candidate_count=5,
        top_candidate_id="seq_abc",
    )
    rows = build_campaign_table([campaign])
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["candidate_count"] == 5


def test_build_ranked_table() -> None:
    result = EvaluationResult(
        campaign_id="c1",
        sequence_candidate_id="seq_1",
        register_candidate_id="reg_1",
        nominal_score=0.8,
        robustness_score=0.7,
        worst_case_score=0.55,
        observable_score=0.78,
        objective_score=0.72,
        backend_choice=BackendType.LOCAL_PULSER_SIMULATION,
        estimated_cost=0.12,
        estimated_latency=0.15,
        final_rank=1,
        status=CandidateStatus.RANKED,
        reasoning_summary="test",
    )
    rows = build_ranked_table([result])
    assert rows[0]["rank"] == 1
    assert rows[0]["objective_score"] == 0.72


def test_robustness_chart_data_empty() -> None:
    labels, nom, avg, worst = select_robustness_chart_data([])
    assert labels == []


def test_robustness_chart_data() -> None:
    report = RobustnessReport(
        campaign_id="c1",
        sequence_candidate_id="seq_123456",
        nominal_score=0.8,
        perturbation_average=0.72,
        robustness_penalty=0.08,
        robustness_score=0.75,
        worst_case_score=0.6,
        score_std=0.05,
        target_observable="rydberg_density",
        scenario_scores={"low_noise": 0.78, "medium_noise": 0.72},
        reasoning_summary="test",
    )
    labels, nominal, average, worst = select_robustness_chart_data([report])
    assert labels == ["123456"]
    assert nominal == [0.8]
    assert average == [0.72]


def test_noise_sensitivity_data() -> None:
    report = RobustnessReport(
        campaign_id="c1",
        sequence_candidate_id="seq_1",
        nominal_score=0.8,
        perturbation_average=0.72,
        robustness_penalty=0.08,
        robustness_score=0.75,
        worst_case_score=0.6,
        score_std=0.05,
        target_observable="rydberg_density",
        scenario_scores={
            "low_noise": 0.78,
            "medium_noise": 0.72,
            "stressed_noise": 0.6,
        },
        reasoning_summary="test",
    )
    labels, values = select_noise_sensitivity_data(report)
    assert labels == ["low_noise", "medium_noise", "stressed_noise"]
    assert values == [0.78, 0.72, 0.6]


def test_noise_sensitivity_data_missing_scenarios() -> None:
    report = RobustnessReport(
        campaign_id="c1",
        sequence_candidate_id="seq_1",
        nominal_score=0.8,
        perturbation_average=0.72,
        robustness_penalty=0.08,
        robustness_score=0.75,
        worst_case_score=0.6,
        score_std=0.05,
        target_observable="rydberg_density",
        scenario_scores={"medium_noise": 0.72},
        reasoning_summary="test",
    )
    labels, values = select_noise_sensitivity_data(report)
    assert labels == ["medium_noise"]
    assert len(values) == 1
```

Puis modifier `apps/dashboard/app.py` pour importer les fonctions depuis `logic.py` au lieu de les avoir inline :

```python
# Dans app.py, remplacer la construction inline des tables par :
from apps.dashboard.logic import (
    build_campaign_table,
    build_ranked_table,
    build_decision_table,
    select_robustness_chart_data,
    select_noise_sensitivity_data,
)

# ... et appeler ces fonctions au lieu du code inline
campaign_rows = build_campaign_table(latest_campaigns)
```

Les parties Streamlit pur (rendering matplotlib, `st.form`, etc.) restent dans `app.py` — elles ne sont pas testables unitairement et n'ont pas besoin de l'être.

---

## ORDRE D'EXÉCUTION

1. **TÂCHE 3** — Exceptions spécifiques (crée le module, aucune dépendance)
2. **TÂCHE 4** — Logs pour échecs silencieux (utilise les nouvelles exceptions)
3. **TÂCHE 2** — TypedDict metadata schemas (crée le module, aucune dépendance)
4. **TÂCHE 1** — Agent protocols (crée le module, modifie pipeline pour injection)
5. **TÂCHE 5** — Unification limites d'atomes (dépend de P0 Tâche 3)
6. **TÂCHE 6** — Marqueurs pytest + couverture (infrastructure, aucune dépendance code)
7. **TÂCHE 7** — Tests API + dashboard (dépend de toutes les tâches précédentes)

Après chaque tâche : `pytest tests/ -x --tb=short`

---

## CONTRAINTES FINALES

- **Zéro `# TODO` laissé dans le code produit.**
- **Chaque nouveau fichier a une docstring de module** qui explique son rôle en 1-3 phrases.
- **Les imports sont groupés** : stdlib, third-party, project.
- **Pas de lignes > 120 caractères.**
- **`from __future__ import annotations` en première ligne de chaque nouveau fichier.**
- **Les TypedDicts utilisent `total=False`** pour accepter les dicts partiels.
- **Les protocoles utilisent `@runtime_checkable`** pour les assertions isinstance.
- **Chaque exception custom est picklable** (important pour multiprocessing).
- **Les logs utilisent le format `%s` (lazy formatting)**, jamais f-strings dans les appels logger.
- **Si un test existant casse, corrige-le immédiatement** en préservant le comportement testé.

## VÉRIFICATION FINALE

```bash
# Tous les tests
pytest tests/ -v --tb=short

# Vérification stricte des marqueurs
pytest tests/ --strict-markers -v --tb=short 2>&1 | tail -50

# Couverture
pytest tests/ --cov=packages --cov=apps --cov-report=term-missing --tb=short
```

Tous les tests doivent passer. Aucune régression tolérée.
