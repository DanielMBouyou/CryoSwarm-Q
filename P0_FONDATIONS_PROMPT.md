# P0 — FONDATIONS : Prompt d'implémentation exhaustif

> **Destinataire** : Agent IA de code.
> **Mode** : Implémentation directe — écris chaque ligne, chaque import, chaque test.
> **Philosophie** : Innovation, rigueur, state-of-the-art. Chaque modification doit être millimétrée, chaque refactoring doit être traceable. Pas de demi-mesures, pas de raccourcis, pas de TODOs laissés en suspens. Tu es un ingénieur recherche senior, pas un stagiaire. Chaque ligne que tu produis sera lue par des reviewers exigeants en quantum computing.

---

## Contexte du projet

CryoSwarm-Q est un système multi-agent hardware-aware pour la conception autonome d'expériences en informatique quantique à atomes neutres (neutral-atom). Le codebase Python cible Pulser / Pasqal et orchestre 8 agents spécialisés dans un pipeline séquentiel.

**Structure du workspace** :

```
packages/
  agents/         → 8 agents spécialisés (problem, geometry, sequence, noise, routing, campaign, results, memory)
  core/           → models.py, enums.py, config.py, logging.py, parameter_space.py
  orchestration/  → pipeline.py (CryoSwarmPipeline), runner.py
  scoring/        → objective.py, ranking.py, robustness.py
  simulation/     → evaluators.py, hamiltonian.py, noise_profiles.py, numpy_backend.py, observables.py
  pasqal_adapters/→ pulser_adapter.py, emulator_router.py, pasqal_cloud_adapter.py, qoolqit_adapter.py
  db/             → mongodb.py, repositories.py, init_db.py
  ml/             → surrogate.py, ppo.py, rl_env.py, rl_sequence_agent.py, training_runner.py, dataset.py, surrogate_filter.py, gpu_backend.py, data_generator.py
apps/
  api/            → main.py, dependencies.py, routes/ (health.py, goals.py, campaigns.py, candidates.py)
  dashboard/      → app.py (Streamlit)
tests/            → 42 fichiers de tests, AUCUN conftest.py
```

**Règles impératives** :
- Python 3.11+, types partout
- Pydantic v2 (`model_config = ConfigDict(...)`)
- Pas de commentaires vagues — chaque docstring doit être technique et précise
- Chaque modification doit conserver la rétrocompatibilité des tests existants (lance `pytest tests/` après chaque changement)
- Imports absolus depuis la racine du projet (`from packages.core.models import ...`)
- Pas de `# type: ignore` sauf si absolument nécessaire avec justification

---

## TÂCHE 1 — Extraire `run()` de `pipeline.py` en phases composables

### Problème actuel

Le fichier `packages/orchestration/pipeline.py` contient une classe `CryoSwarmPipeline` dont la méthode `run()` fait **239 lignes** (lignes 339–835). C'est un monolithe avec :
- 6 appels à `_build_summary()` avec des kwargs quasi-identiques
- 5 blocs `_persist_state()` + `return _build_summary()` copiés-collés
- 5 points de sortie anticipée (early return) pour les cas d'échec (problem framing, no registers, no sequences, no evaluations, general exception)
- Chaque bloc try-except suit le même pattern : appeler un agent, capturer la décision, gérer l'échec
- Ajouter un nouvel agent nécessite de toucher 6+ endroits

Le fichier complet fait ~835 lignes. La méthode `run()` représente presque 30% du fichier.

### Objectif

Refactorer `run()` en une architecture composable par phases. Chaque phase est une méthode privée autonome qui :
1. Reçoit un objet d'état mutable (`PipelineContext`)
2. Exécute sa logique
3. Retourne un signal de continuation ou d'arrêt

### Architecture cible

```python
@dataclass
class PipelineContext:
    """Mutable state threaded through all pipeline phases."""
    goal: ExperimentGoal
    campaign: CampaignState
    spec: ExperimentSpec | None = None
    registers: list[RegisterCandidate] = field(default_factory=list)
    sequences: list[SequenceCandidate] = field(default_factory=list)
    reports: list[RobustnessReport] = field(default_factory=list)
    ranked_candidates: list[EvaluationResult] = field(default_factory=list)
    decisions: list[AgentDecision] = field(default_factory=list)
    memory_records: list[MemoryRecord] = field(default_factory=list)
    memory_context: list[MemoryRecord] = field(default_factory=list)
    backend_counter: Counter = field(default_factory=Counter)
    error: str | None = None
    status: str = "COMPLETED"

    @property
    def failed(self) -> bool:
        return self.status == "FAILED"

    @property
    def has_no_candidates(self) -> bool:
        return self.status == "NO_CANDIDATES"

    @property
    def should_stop(self) -> bool:
        return self.failed or self.has_no_candidates

    def fail(self, reason: str) -> None:
        self.status = "FAILED"
        self.error = reason

    def no_candidates(self, reason: str) -> None:
        self.status = "NO_CANDIDATES"
        self.error = reason
```

La méthode `run()` devient :

```python
def run(self, goal: ExperimentGoal) -> PipelineSummary:
    ctx = self._init_context(goal)

    self._phase_problem_framing(ctx)
    if ctx.should_stop:
        return self._finalize(ctx)

    self._phase_geometry_generation(ctx)
    if ctx.should_stop:
        return self._finalize(ctx)

    self._phase_sequence_generation(ctx)
    if ctx.should_stop:
        return self._finalize(ctx)

    self._phase_evaluation(ctx)
    if ctx.should_stop:
        return self._finalize(ctx)

    self._phase_ranking(ctx)
    self._phase_results_summary(ctx)
    self._phase_memory_capture(ctx)

    return self._finalize(ctx)
```

### Spécification détaillée de chaque méthode

#### `_init_context(self, goal: ExperimentGoal) -> PipelineContext`

```python
def _init_context(self, goal: ExperimentGoal) -> PipelineContext:
    stored_goal = goal.model_copy(update={"status": GoalStatus.RUNNING})
    self._safe_repository_call(lambda: self.repository.create_goal(stored_goal), "create_goal")

    memory_context = self._safe_repository_call(
        lambda: self.repository.list_recent_memory(limit=12),
        "list_recent_memory",
        default=[],
    ) or []

    campaign = CampaignState(goal_id=stored_goal.id, status=CampaignStatus.RUNNING)
    self._safe_repository_call(lambda: self.repository.create_campaign(campaign), "create_campaign")

    return PipelineContext(
        goal=stored_goal,
        campaign=campaign,
        memory_context=memory_context,
    )
```

#### `_phase_problem_framing(self, ctx: PipelineContext) -> None`

Reproduit exactement le bloc try-except actuel lignes 347–400 mais écrit dans ctx :
- Sur succès : `ctx.spec = spec`, met à jour `ctx.campaign` avec `spec_id`
- Sur échec : `ctx.fail("Problem framing failed.")`, met à jour `ctx.goal.status` et `ctx.campaign.status`
- Dans TOUS les cas : enregistre la décision dans `ctx.decisions`

#### `_phase_geometry_generation(self, ctx: PipelineContext) -> None`

Reproduit le bloc lignes 401–475 :
- Appelle `self.geometry_agent.run(ctx.spec, ctx.campaign.id, ctx.memory_context)`
- Si registers vides : `ctx.no_candidates("No feasible register candidates generated.")`
- Enregistre la décision

#### `_phase_sequence_generation(self, ctx: PipelineContext) -> None`

Reproduit le bloc lignes 476–575 :
- Itère sur `ctx.registers`, appelle `self.sequence_strategy.generate_candidates()`
- Accumule dans `ctx.sequences`
- Si aucune séquence : `ctx.no_candidates("No sequence candidates generated.")`
- Enregistre les décisions (par register + summary)

#### `_phase_evaluation(self, ctx: PipelineContext) -> None`

Reproduit le bloc lignes 610–670 :
- Construit `register_lookup`
- Appelle `self._evaluate_sequences()`
- Met à jour `ctx.reports`, `ctx.ranked_candidates`, `ctx.backend_counter`
- Met à jour le strategy performance tracking
- Si aucun candidat évalué : `ctx.no_candidates("No evaluation results were produced.")`

#### `_phase_ranking(self, ctx: PipelineContext) -> None`

Reproduit le bloc lignes 671–710 :
- Appelle `self.campaign_agent.run()`
- Sur échec : met campaign en FAILED mais ne stoppe PAS le pipeline (comme actuellement)

#### `_phase_results_summary(self, ctx: PipelineContext) -> None`

Reproduit le bloc lignes 711–755 :
- Appelle `self.results_agent.run()`
- Sur échec : absorbe l'erreur, continue

#### `_phase_memory_capture(self, ctx: PipelineContext) -> None`

Reproduit le bloc lignes 756–790 :
- Appelle `self.memory_agent.run()`
- Sur échec : `ctx.memory_records = []`, continue

#### `_finalize(self, ctx: PipelineContext) -> PipelineSummary`

```python
def _finalize(self, ctx: PipelineContext) -> PipelineSummary:
    # Update final goal status
    final_goal_status = (
        GoalStatus.FAILED if ctx.failed else GoalStatus.COMPLETED
    )
    ctx.goal = ctx.goal.model_copy(update={"status": final_goal_status})

    # Update campaign status for early exits
    if ctx.has_no_candidates and ctx.campaign.status == CampaignStatus.RUNNING:
        ctx.campaign = ctx.campaign.model_copy(
            update={
                "status": CampaignStatus.NO_CANDIDATES,
                "candidate_count": 0,
                "summary": ctx.error,
                "summary_report": {"reason": ctx.error},
            }
        )
    elif ctx.failed and ctx.campaign.status == CampaignStatus.RUNNING:
        ctx.campaign = ctx.campaign.model_copy(
            update={
                "status": CampaignStatus.FAILED,
                "summary": ctx.error,
            }
        )

    self._persist_state(
        ctx.goal, ctx.campaign, ctx.registers, ctx.sequences,
        ctx.reports, ctx.ranked_candidates, ctx.decisions, ctx.memory_records,
    )

    return self._build_summary(
        status=ctx.status,
        campaign=ctx.campaign,
        goal=ctx.goal,
        spec=ctx.spec,
        ranked_candidates=ctx.ranked_candidates,
        decisions=ctx.decisions,
        reports=ctx.reports,
        memory_records=ctx.memory_records,
        backend_mix=dict(ctx.backend_counter),
        registers=ctx.registers,
        sequences=ctx.sequences,
        error=ctx.error,
    )
```

### Gestion du try-except global

Le `run()` actuel a un `except Exception` global (lignes 808–835) qui capture les erreurs non gérées. Ce catch doit rester au niveau de `run()` :

```python
def run(self, goal: ExperimentGoal) -> PipelineSummary:
    ctx = self._init_context(goal)
    try:
        self._phase_problem_framing(ctx)
        if ctx.should_stop:
            return self._finalize(ctx)
        # ... suite des phases ...
        return self._finalize(ctx)
    except Exception as exc:
        self.logger.error("Unhandled pipeline failure: %s", exc)
        ctx.fail(f"Unhandled pipeline failure: {exc}")
        return self._finalize(ctx)
```

### Tests à adapter

Les tests existants dans `tests/test_pipeline_integration.py` et `tests/test_pipeline_failures.py` ne testent que le comportement observable (input → PipelineSummary output). Ils ne doivent PAS casser car la signature `run(goal) → PipelineSummary` ne change pas.

Vérifie que les tests passent : `pytest tests/test_pipeline_integration.py tests/test_pipeline_failures.py tests/test_parallel_pipeline.py -v`

### Note d'innovation

Place `PipelineContext` dans `packages/orchestration/pipeline.py` au-dessus de la classe `CryoSwarmPipeline`. N'en fais PAS un fichier séparé — le contexte est interne au pipeline. Utilise `@dataclass` avec `field(default_factory=...)` et NON une Pydantic model (le contexte est mutable par design, Pydantic serait du over-engineering ici).

### Contrainte stricte

Le comportement observable du pipeline DOIT rester identique. Chaque `PipelineSummary` retourné doit contenir les mêmes données qu'avant le refactoring. C'est un refactoring interne pur, pas un changement de comportement.

---

## TÂCHE 2 — Créer `conftest.py` avec factories partagées

### Problème actuel

Il y a **42 fichiers de tests** dans `tests/` et **aucun `conftest.py`**. Les helpers de construction d'objets sont dupliqués dans 9+ fichiers avec des signatures incompatibles. 14 fichiers construisent les objets inline à chaque test.

Exemples de duplication (toutes ces fonctions font essentiellement la même chose) :

- `tests/test_curriculum.py:8` → `_make_register(atom_count, layout)`
- `tests/test_ml_dataset.py:28` → `_make_register(layout, atom_count)`
- `tests/test_ml_filter.py:19` → `_make_register(rid, atom_count)`
- `tests/test_sequence_agent.py:30` → `_make_register()` (sans paramètres)
- `tests/test_physics_features.py:47` → `_make_register(layout, spacing_um, atom_count)`
- `tests/test_sequence_strategy.py:25` → `_make_register()` (sans paramètres)

### Objectif

Créer `tests/conftest.py` avec des factories pytest centralisées, configurables, et des marqueurs pytest standards.

### Spécification complète du fichier `tests/conftest.py`

```python
"""Shared test fixtures and factories for CryoSwarm-Q test suite.

Provides reusable, parameterized factories for all core domain objects.
Tests should use these factories instead of constructing objects inline.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from packages.core.enums import (
    BackendType,
    CampaignStatus,
    CandidateStatus,
    NoiseLevel,
    SequenceFamily,
)
from packages.core.models import (
    AgentDecision,
    BackendChoice,
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    ExperimentSpec,
    MemoryRecord,
    RegisterCandidate,
    RobustnessReport,
    ScoringWeights,
    SequenceCandidate,
)
from packages.core.parameter_space import PhysicsParameterSpace


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: tests requiring a GPU (CUDA/ROCm)")
    config.addinivalue_line("markers", "slow: tests taking more than 5 seconds")
    config.addinivalue_line("markers", "integration: end-to-end integration tests")
    config.addinivalue_line("markers", "mongo: tests requiring a live MongoDB connection")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def make_coordinates(
    layout: str = "square",
    atom_count: int = 4,
    spacing_um: float = 7.0,
) -> list[tuple[float, float]]:
    """Generate physically realistic atom coordinates for a given layout.

    Supports: line, square, triangular, ring, zigzag, honeycomb.
    Falls back to line layout for unknown layout types.
    """
    if layout == "line":
        return [(i * spacing_um, 0.0) for i in range(atom_count)]

    if layout == "square":
        side = math.isqrt(atom_count)
        if side * side < atom_count:
            side += 1
        coords: list[tuple[float, float]] = []
        for idx in range(atom_count):
            row, col = divmod(idx, side)
            coords.append((col * spacing_um, row * spacing_um))
        return coords

    if layout == "triangular":
        coords = []
        side = math.isqrt(atom_count) + 1
        placed = 0
        for row in range(side):
            x_offset = spacing_um * 0.5 if row % 2 else 0.0
            for col in range(side):
                if placed >= atom_count:
                    break
                coords.append((col * spacing_um + x_offset, row * spacing_um * math.sqrt(3) / 2))
                placed += 1
        return coords

    if layout == "ring":
        return [
            (
                spacing_um * math.cos(2 * math.pi * i / atom_count),
                spacing_um * math.sin(2 * math.pi * i / atom_count),
            )
            for i in range(atom_count)
        ]

    if layout == "zigzag":
        return [
            (i * spacing_um, spacing_um * 0.5 * (i % 2))
            for i in range(atom_count)
        ]

    if layout == "honeycomb":
        coords = []
        placed = 0
        col = 0
        while placed < atom_count:
            for row_offset in (0.0, spacing_um * math.sqrt(3) / 3):
                if placed >= atom_count:
                    break
                coords.append((col * spacing_um * 1.5, row_offset + (col % 2) * spacing_um * math.sqrt(3) / 6))
                placed += 1
            col += 1
        return coords

    # Fallback: line
    return [(i * spacing_um, 0.0) for i in range(atom_count)]


def compute_blockade_pairs(
    coordinates: list[tuple[float, float]],
    blockade_radius_um: float = 9.5,
) -> int:
    """Count atom pairs within blockade radius."""
    count = 0
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            if math.sqrt(dx * dx + dy * dy) <= blockade_radius_um:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Domain object factories
# ---------------------------------------------------------------------------

def make_goal(
    *,
    title: str = "Test neutral-atom benchmark",
    scientific_objective: str = "Evaluate robustness of small neutral-atom configurations.",
    target_observable: str = "rydberg_density",
    desired_atom_count: int = 6,
    preferred_geometry: str = "mixed",
    priority: str = "balanced",
    constraints: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> ExperimentGoal:
    """Build an ExperimentGoal with sensible defaults."""
    return ExperimentGoal(
        title=title,
        scientific_objective=scientific_objective,
        target_observable=target_observable,
        desired_atom_count=desired_atom_count,
        preferred_geometry=preferred_geometry,
        priority=priority,
        constraints=constraints or {},
        metadata=metadata or {},
        **overrides,
    )


def make_spec(
    *,
    goal_id: str = "goal_test",
    objective_class: str = "balanced_campaign_search",
    target_observable: str = "rydberg_density",
    min_atoms: int = 4,
    max_atoms: int = 8,
    preferred_layouts: list[str] | None = None,
    sequence_families: list[SequenceFamily] | None = None,
    target_density: float = 0.5,
    perturbation_budget: int = 3,
    latency_budget: float = 0.30,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> ExperimentSpec:
    """Build an ExperimentSpec with sensible defaults."""
    return ExperimentSpec(
        goal_id=goal_id,
        objective_class=objective_class,
        target_observable=target_observable,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        preferred_layouts=preferred_layouts or ["square", "line", "triangular"],
        sequence_families=sequence_families or [
            SequenceFamily.GLOBAL_RAMP,
            SequenceFamily.DETUNING_SCAN,
            SequenceFamily.ADIABATIC_SWEEP,
            SequenceFamily.CONSTANT_DRIVE,
            SequenceFamily.BLACKMAN_SWEEP,
        ],
        target_density=target_density,
        perturbation_budget=perturbation_budget,
        latency_budget=latency_budget,
        reasoning_summary="Test experiment specification.",
        metadata=metadata or {},
        **overrides,
    )


def make_register(
    *,
    campaign_id: str = "campaign_test",
    spec_id: str = "spec_test",
    label: str | None = None,
    layout_type: str = "square",
    atom_count: int = 4,
    spacing_um: float = 7.0,
    coordinates: list[tuple[float, float]] | None = None,
    blockade_radius_um: float = 9.5,
    feasibility_score: float = 0.85,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> RegisterCandidate:
    """Build a RegisterCandidate with physically consistent defaults.

    Coordinates are auto-generated from layout_type, atom_count, and spacing_um
    unless explicitly provided.
    """
    if coordinates is None:
        coordinates = make_coordinates(layout_type, atom_count, spacing_um)
    if label is None:
        label = f"{layout_type}-{atom_count}"

    min_dist = float("inf")
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d
    if min_dist == float("inf"):
        min_dist = spacing_um

    blockade_pair_count = compute_blockade_pairs(coordinates, blockade_radius_um)

    return RegisterCandidate(
        campaign_id=campaign_id,
        spec_id=spec_id,
        label=label,
        layout_type=layout_type,
        atom_count=atom_count,
        coordinates=coordinates,
        min_distance_um=round(min_dist, 4),
        blockade_radius_um=blockade_radius_um,
        blockade_pair_count=blockade_pair_count,
        feasibility_score=feasibility_score,
        reasoning_summary=f"Test {layout_type} register with {atom_count} atoms.",
        metadata={"spacing_um": spacing_um, **(metadata or {})},
        **overrides,
    )


def make_sequence(
    *,
    campaign_id: str = "campaign_test",
    spec_id: str = "spec_test",
    register_candidate_id: str = "reg_test",
    label: str | None = None,
    sequence_family: SequenceFamily = SequenceFamily.ADIABATIC_SWEEP,
    channel_id: str = "rydberg_global",
    duration_ns: int = 3000,
    amplitude: float = 5.0,
    detuning: float = -10.0,
    phase: float = 0.0,
    waveform_kind: str = "adiabatic_conservative",
    predicted_cost: float = 0.15,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> SequenceCandidate:
    """Build a SequenceCandidate with sensible pulse defaults."""
    if label is None:
        label = f"test-{sequence_family.value}-{waveform_kind}"
    return SequenceCandidate(
        campaign_id=campaign_id,
        spec_id=spec_id,
        register_candidate_id=register_candidate_id,
        label=label,
        sequence_family=sequence_family,
        channel_id=channel_id,
        duration_ns=duration_ns,
        amplitude=amplitude,
        detuning=detuning,
        phase=phase,
        waveform_kind=waveform_kind,
        predicted_cost=predicted_cost,
        reasoning_summary=f"Test {sequence_family.value} pulse sequence.",
        metadata=metadata or {},
        **overrides,
    )


def make_robustness_report(
    *,
    campaign_id: str = "campaign_test",
    sequence_candidate_id: str = "seq_test",
    nominal_score: float = 0.75,
    robustness_score: float = 0.70,
    worst_case_score: float = 0.55,
    perturbation_average: float = 0.68,
    robustness_penalty: float = 0.07,
    score_std: float = 0.05,
    target_observable: str = "rydberg_density",
    scenario_scores: dict[str, float] | None = None,
    nominal_observables: dict[str, Any] | None = None,
    hamiltonian_metrics: dict[str, Any] | None = None,
    **overrides: Any,
) -> RobustnessReport:
    """Build a RobustnessReport with physically plausible defaults."""
    return RobustnessReport(
        campaign_id=campaign_id,
        sequence_candidate_id=sequence_candidate_id,
        nominal_score=nominal_score,
        perturbation_average=perturbation_average,
        robustness_penalty=robustness_penalty,
        robustness_score=robustness_score,
        worst_case_score=worst_case_score,
        score_std=score_std,
        target_observable=target_observable,
        scenario_scores=scenario_scores or {"low": 0.72, "medium": 0.68, "high": 0.55},
        nominal_observables=nominal_observables or {"rydberg_density": 0.75, "observable_score": 0.75},
        hamiltonian_metrics=hamiltonian_metrics or {"spectral_gap": 1.2, "energy_spread": 3.5},
        reasoning_summary="Test robustness evaluation.",
        **overrides,
    )


def make_evaluation_result(
    *,
    campaign_id: str = "campaign_test",
    sequence_candidate_id: str = "seq_test",
    register_candidate_id: str = "reg_test",
    nominal_score: float = 0.75,
    robustness_score: float = 0.70,
    worst_case_score: float = 0.55,
    observable_score: float = 0.75,
    objective_score: float = 0.72,
    backend_choice: BackendType = BackendType.LOCAL_PULSER_SIMULATION,
    estimated_cost: float = 0.12,
    estimated_latency: float = 0.15,
    **overrides: Any,
) -> EvaluationResult:
    """Build an EvaluationResult with balanced defaults."""
    return EvaluationResult(
        campaign_id=campaign_id,
        sequence_candidate_id=sequence_candidate_id,
        register_candidate_id=register_candidate_id,
        nominal_score=nominal_score,
        robustness_score=robustness_score,
        worst_case_score=worst_case_score,
        observable_score=observable_score,
        objective_score=objective_score,
        backend_choice=backend_choice,
        estimated_cost=estimated_cost,
        estimated_latency=estimated_latency,
        reasoning_summary="Test evaluation result.",
        **overrides,
    )


def make_memory_record(
    *,
    campaign_id: str = "campaign_test",
    source_candidate_id: str = "seq_test",
    lesson_type: str = "candidate_pattern",
    summary: str = "Test memory record.",
    signals: dict[str, Any] | None = None,
    reusable_tags: list[str] | None = None,
    **overrides: Any,
) -> MemoryRecord:
    """Build a MemoryRecord with sensible defaults."""
    return MemoryRecord(
        campaign_id=campaign_id,
        source_candidate_id=source_candidate_id,
        lesson_type=lesson_type,
        summary=summary,
        signals=signals or {"confidence": 0.8, "sequence_family": "adiabatic_sweep"},
        reusable_tags=reusable_tags or ["strong"],
        **overrides,
    )


def make_campaign(
    *,
    goal_id: str = "goal_test",
    status: CampaignStatus = CampaignStatus.CREATED,
    **overrides: Any,
) -> CampaignState:
    """Build a CampaignState."""
    return CampaignState(goal_id=goal_id, status=status, **overrides)


# ---------------------------------------------------------------------------
# Pytest fixtures — use when you need DI-style injection
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_param_space() -> PhysicsParameterSpace:
    """Provide the default PhysicsParameterSpace for tests."""
    return PhysicsParameterSpace.default()


@pytest.fixture()
def sample_goal() -> ExperimentGoal:
    """Provide a ready-made ExperimentGoal."""
    return make_goal()


@pytest.fixture()
def sample_spec() -> ExperimentSpec:
    """Provide a ready-made ExperimentSpec."""
    return make_spec()


@pytest.fixture()
def sample_register() -> RegisterCandidate:
    """Provide a ready-made RegisterCandidate."""
    return make_register()


@pytest.fixture()
def sample_sequence(sample_register: RegisterCandidate) -> SequenceCandidate:
    """Provide a SequenceCandidate linked to sample_register."""
    return make_sequence(register_candidate_id=sample_register.id)


@pytest.fixture()
def sample_report(sample_sequence: SequenceCandidate) -> RobustnessReport:
    """Provide a RobustnessReport linked to sample_sequence."""
    return make_robustness_report(sequence_candidate_id=sample_sequence.id)
```

### Règles d'implémentation

1. **Les factories sont des fonctions libres** (pas des fixtures) pour être appelables depuis n'importe quel test sans injection. Les fixtures sont des wrappers de convenance.

2. **make_coordinates() doit produire des coordonnées géométriquement correctes** — pas de coordonnées bidon `[(0,0), (1,0)]`. Les layouts doivent correspondre à ce que fait `GeometryAgent`.

3. **compute_blockade_pairs() calcule le vrai nombre de paires** en blockade — pas de hardcode. Utilise le rayon de blockade par défaut de 9.5 µm.

4. **Chaque factory accepte `**overrides`** pour permettre l'override de n'importe quel champ Pydantic. Cela rend les factories extensibles sans modifications futures.

5. **Les imports sont groupés** par stdlib / third-party / project.

### NE PAS migrer les tests existants dans cette tâche

La migration des tests existants vers conftest est une tâche distincte (P1). Ici, on crée juste le fichier conftest.py. Les tests existants fonctionnent toujours avec leurs helpers locaux — conftest ne casse rien.

---

## TÂCHE 3 — Centraliser les constantes physiques dans `PhysicsParameterSpace`

### Problème actuel

Des constantes physiques sont hardcodées à travers tout le codebase sans lien avec `PhysicsParameterSpace` :

| Constante | Valeur | Fichier | Ligne |
|-----------|--------|---------|-------|
| C₆ coefficient | `862690.0` | `packages/ml/dataset.py` | ~91 |
| C₆ coefficient | `862690` | `packages/simulation/hamiltonian.py` | variable |
| C₆ coefficient | `862690` | `packages/simulation/evaluators.py` | variable |
| Max atoms dense | `12` | `packages/simulation/numpy_backend.py` | 33 |
| Max atoms sparse | `18` | `packages/simulation/numpy_backend.py` | 34 |
| Max atoms evaluator | `14` | `packages/simulation/evaluators.py` | variable |
| Max atoms GPU | `24` | `packages/ml/gpu_backend.py` | variable |
| Blockade radius | `9.5` | divers fichiers de tests | multiple |
| Default spacing | `7.0` | `packages/ml/dataset.py` | fallback |
| Amplitude clipping | `0.85` | `packages/pasqal_adapters/pulser_adapter.py` | ~50 |
| OBS_DIM | `16` | `packages/ml/rl_env.py` | ~25 |
| Action normalizers | `15.0, 30.0, 5500.0` | `packages/ml/rl_env.py` | ~166 |

### Objectif

Ajouter un bloc de constantes physiques et de limites de calcul dans `PhysicsParameterSpace`. Chaque module doit désormais lire ces valeurs depuis le parameter space au lieu d'utiliser des littéraux.

### Spécification

Ajouter les attributs suivants à la dataclass `PhysicsParameterSpace` :

```python
@dataclass
class PhysicsParameterSpace:
    # ... champs existants ...

    # Fundamental constants
    c6_coefficient: float = 862690.0
    """C₆ van der Waals coefficient for 87-Rb |70S₁/₂⟩ state, in rad·µm⁶/µs."""

    default_blockade_radius_um: float = 9.5
    """Default Rydberg blockade radius in µm for typical inter-atom spacings."""

    # Simulation backend limits
    max_atoms_dense: int = 12
    """Maximum atom count for dense (full) Hamiltonian diagonalization."""

    max_atoms_sparse: int = 18
    """Maximum atom count for sparse Hamiltonian methods (Lanczos/ARPACK)."""

    max_atoms_gpu: int = 24
    """Maximum atom count for GPU-accelerated simulation."""

    max_atoms_evaluator_parallel: int = 14
    """Atom count threshold for switching from dense to sparse in evaluator."""

    # Pasqal hardware margins
    amplitude_safety_margin: float = 0.85
    """Fraction of device maximum amplitude/detuning used for safety margin."""
```

### Modifications dans les modules consommateurs

Pour chaque fichier qui utilise un littéral physique :

1. **`packages/simulation/hamiltonian.py`** : Ajoute un paramètre `c6: float = 862690.0` aux fonctions qui utilisent le C₆. Les appels existants continuent de fonctionner. Les nouveaux appels depuis le pipeline peuvent passer `param_space.c6_coefficient`.

2. **`packages/simulation/numpy_backend.py`** : Remplace `MAX_ATOMS_DENSE = 12` et `MAX_ATOMS_SPARSE = 18` par des paramètres qui prennent des valeurs du param_space. Comme ce module est souvent appelé en standalone, garde les constantes actuelles comme fallback :

```python
def simulate(
    ...,
    max_atoms_dense: int = 12,
    max_atoms_sparse: int = 18,
) -> ...:
```

3. **`packages/simulation/evaluators.py`** : Le seuil `atom_count > 14` pour basculer vers sparse doit devenir un paramètre. L'evaluator reçoit déjà un `PhysicsParameterSpace` (ou pourrait le recevoir). Remplace le littéral par `param_space.max_atoms_evaluator_parallel` là où c'est applicable.

4. **`packages/ml/dataset.py`** : La constante `C6 = 862690.0` (ligne ~91) doit être remplacée par un paramètre passé depuis l'appelant. En fallback, la valeur par défaut de `PhysicsParameterSpace.default().c6_coefficient` est utilisée.

5. **`packages/ml/gpu_backend.py`** : `MAX_ATOMS_GPU = 24` → paramètre ou lecture depuis param_space.

6. **`packages/pasqal_adapters/pulser_adapter.py`** : Le facteur `0.85` doit être lu depuis `param_space.amplitude_safety_margin`. Si l'adapter n'a pas accès au param_space, ajoute-le en paramètre optionnel de `build_simple_sequence_summary()`.

7. **`packages/ml/rl_env.py`** : Les normalizers `15.0, 30.0, 5500.0` doivent être calculés depuis `param_space.rl_action_ranges()` au lieu d'être hardcodés.

### Règles

- **Ne casse JAMAIS la signature d'une fonction publique.** Ajoute des paramètres avec des valeurs par défaut.
- **Chaque constante remplacée doit conserver exactement la même valeur numérique par défaut.** Le refactoring est transparent.
- **Commente chaque constante dans PhysicsParameterSpace avec sa signification physique et ses unités.**
- Les tests existants doivent passer sans modification.
- Lance `pytest tests/ -x --tb=short` pour vérifier.

### Ce qui ne doit PAS changer

- La méthode `PhysicsParameterSpace.default()` retourne les mêmes valeurs qu'avant pour tous les champs existants.
- La sérialisation/désérialisation (`to_dict()` / `from_dict()`) doit intégrer les nouveaux champs avec des valeurs par défaut pour la rétrocompatibilité des fichiers YAML existants.

---

## TÂCHE 4 — Ajouter `model_config = ConfigDict(frozen=True)` aux modèles Pydantic critiques

### Problème actuel

Tous les modèles Pydantic héritent de `CryoSwarmModel` qui a `ConfigDict(extra="forbid", populate_by_name=True)`. Aucun modèle n'est frozen. Cela signifie que les invariants validés à la construction (comme `ScoringWeights` dont les poids doivent sommer à 1.0) peuvent être violés par mutation post-construction :

```python
w = ScoringWeights()  # alpha=0.45, beta=0.35, gamma=0.10, delta=0.10 → total=1.0 ✓
w.alpha = 0.99  # aucune erreur ! total=1.59 ← invariant violé
```

### Objectif

Rendre immutables les modèles qui représentent des **données validées** (pas des conteneurs d'état). Le pipeline utilise déjà `model_copy(update={...})` pour tous les changements d'état, ce qui est compatible avec frozen.

### Modèles à rendre frozen

Ces modèles sont des **value objects** — une fois construits, ils ne doivent jamais être mutés :

1. **`ScoringWeights`** — les poids doivent toujours sommer à 1.0
2. **`ExperimentSpec`** — spécification d'expérience figée après framing
3. **`RegisterCandidate`** — candidat de registre figé après géométrie
4. **`SequenceCandidate`** — candidat de séquence figé après génération
5. **`RobustnessReport`** — rapport figé après évaluation
6. **`BackendChoice`** — choix de backend figé après routing
7. **`AgentDecision`** — décision figée après enregistrement
8. **`EvaluationResult`** — résultat figé après scoring (SAUF `final_rank` qui est assigné par le ranking)
9. **`MemoryRecord`** — enregistrement mémoire figé après capture

### Modèles à garder mutables

Ces modèles sont des **conteneurs d'état** qui évoluent pendant le pipeline :

1. **`ExperimentGoal`** — le `status` change (DRAFT → RUNNING → COMPLETED)
2. **`CampaignState`** — le `status`, `candidate_count`, `summary`, etc. changent
3. **`PipelineSummary`** — assemblé progressivement
4. **`DemoGoalRequest`** — DTO d'entrée, pas de validation complexe
5. **`ExperimentGoalCreate`** — DTO d'entrée

### Implémentation

#### Option 1 — Classe de base intermédiaire (recommandée)

```python
class CryoSwarmModel(BaseModel):
    """Base model — mutable, for state containers."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FrozenCryoSwarmModel(CryoSwarmModel):
    """Immutable base — for validated value objects."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)
```

Ensuite faire hériter les modèles figés de `FrozenCryoSwarmModel` au lieu de `CryoSwarmModel`.

#### Cas spécial : `EvaluationResult`

`EvaluationResult` est un cas intéressant :
- Il est construit dans `_route_and_score_sequence()` avec `final_rank=None`
- Le `final_rank` est assigné plus tard par `CampaignAgent.run()` via `ranking.py`

Il faut vérifier comment `final_rank` est assigné. Si c'est via mutation directe (`result.final_rank = 3`), il faut soit :
- a) Laisser `EvaluationResult` mutable
- b) Faire passer par `model_copy(update={"final_rank": 3})`

Lis `packages/scoring/ranking.py` pour voir quelle approche est utilisée, et adapte en conséquence.

#### Cas spécial : `SequenceCandidate.metadata` et `.serialized_pulser_sequence`

Dans `pipeline.py` lignes 537–538 (phase de génération des séquences), le code fait :
```python
sequence.metadata.setdefault("problem_class", strategy_meta["problem_class"])
sequence.metadata.setdefault("strategy_used", strategy_meta["strategy_used"])
```

Et dans `sequence_agent.py`, le code fait :
```python
seq.serialized_pulser_sequence = build_simple_sequence_summary(register_candidate, seq)
```

Ces deux mutations DOIVENT être déplacées AVANT la construction (ou via `model_copy()`) si le modèle devient frozen. Le code qui mute `metadata` post-construction doit être refactoré pour passer les valeurs au constructeur.

Dans `sequence_agent.py`, la mutation de `serialized_pulser_sequence` après construction doit devenir :
```python
# Construire d'abord sans pulser_sequence, puis recréer avec
seq_data = { ...tous les champs... }
seq_data["serialized_pulser_sequence"] = build_simple_sequence_summary(register_candidate, SequenceCandidate(**seq_data))
seq = SequenceCandidate(**seq_data)
```

Ou bien, construire le summary AVANT et le passer directement au constructeur. C'est la solution la plus propre.

### Tests

Après avoir rendu les modèles frozen :

```python
def test_scoring_weights_frozen():
    w = ScoringWeights()
    with pytest.raises(ValidationError):
        w.alpha = 0.99

def test_experiment_spec_frozen():
    spec = make_spec()
    with pytest.raises(ValidationError):
        spec.min_atoms = 999
    # model_copy still works
    updated = spec.model_copy(update={"min_atoms": 3})
    assert updated.min_atoms == 3
```

Ajoute ces tests dans un nouveau fichier `tests/test_frozen_models.py` (8–12 tests vérifiant l'immutabilité des modèles critiques ET que `model_copy()` fonctionne toujours).

### Impact sur les tests existants

Fais un `grep -rn "\.status\s*=" tests/` et `grep -rn "\.final_rank\s*=" tests/` pour trouver les tests qui mutent directement les modèles. Ces tests devront utiliser `model_copy()` si le modèle muté est maintenant frozen.

Lance tous les tests après modification : `pytest tests/ -x --tb=short`

---

## TÂCHE 5 — Corriger la logique inversée de `target_density` dans `problem_agent.py`

### Problème actuel

Dans `packages/agents/problem_agent.py`, ligne 32 :

```python
target_density = 0.5
if "single excitation" in objective_text:
    target_density = 1.0 / max(desired_atoms, 1)
```

Quand l'objectif mentionne "single excitation" et qu'on veut une excitation unique sur N atomes, `target_density = 1/N` est correct **physiquement** (une excitation parmi N atomes → densité d'excitation de 1/N).

**CORRECTION : À la relecture, cette logique n'est PAS inversée.** Pour une "single excitation" (un seul atome excité dans le registre), la densité Rydberg attendue est bien `1/N`. Par exemple, 6 atomes avec 1 excité → densité 1/6 ≈ 0.167.

Cependant, le vrai problème est plus subtil :

1. `target_density` est un champ de `ExperimentSpec` mais **il n'est jamais utilisé en aval** par aucun agent ni scoring. C'est un champ orphelin.
2. La valeur par défaut `0.5` est arbitraire et ne correspond à aucune physique particulière.
3. Le champ devrait être utilisé par le scoring pour comparer la densité Rydberg observée à la densité attendue.

### Action requise

1. **Vérifie que `target_density` est bien utilisé quelque part**. Fais un grep :
   ```
   grep -rn "target_density" packages/ apps/
   ```
   
2. **Si `target_density` n'est utilisé nulle part en aval** (ce qui est le cas d'après l'audit) :
   - Documente le champ dans `ExperimentSpec` avec un commentaire expliquant son rôle prévu
   - Ajoute une utilisation dans `packages/scoring/robustness.py` ou `packages/simulation/evaluators.py` pour comparer la densité Rydberg observée à `target_density`

3. **Implémentation d'une utilisation minimale dans le scoring** :

Dans `packages/scoring/robustness.py`, ajoute une fonction :

```python
def density_alignment_bonus(
    observed_density: float,
    target_density: float,
    tolerance: float = 0.15,
) -> float:
    """Compute a bonus [0, 1] based on how close observed density is to target.
    
    Uses a Gaussian-shaped reward centered on target_density with width
    controlled by tolerance. Returns 1.0 for perfect match, decays smoothly.
    """
    if target_density <= 0:
        return 0.5  # no target → neutral
    delta = abs(observed_density - target_density)
    return float(math.exp(-(delta ** 2) / (2 * tolerance ** 2)))
```

Et intègre ce bonus dans le calcul existant si approprié (vois comment les scores sont agrégés).

### Ce qui ne doit PAS changer

- La signature de `ProblemFramingAgent.run()` ne change pas.
- La valeur de `target_density` pour le cas "single excitation" reste `1/N` — c'est physiquement correct.

---

## TÂCHE 6 — Supprimer le code mort

### 6a — `_family_defaults()` dans `sequence_agent.py`

Le fichier `packages/agents/sequence_agent.py` contient cette méthode (lignes 227–229) :

```python
def _family_defaults(self, family: SequenceFamily) -> tuple[ParameterRange, ParameterRange, ParameterRange]:
    pulse_space = self.param_space.pulse[family]
    return pulse_space.amplitude, pulse_space.detuning_start, pulse_space.duration_ns
```

**Cette méthode n'est appelée NULLE PART** dans tout le codebase. Les méthodes `_adiabatic_variants()`, `_detuning_scan_variants()`, etc. accèdent directement à `self.param_space.pulse[family]`.

### Action

1. Supprime la méthode `_family_defaults()` de `sequence_agent.py`.
2. Fais un grep pour confirmer qu'aucun test ne l'appelle : `grep -rn "_family_defaults" tests/ packages/`
3. Lance les tests : `pytest tests/test_sequence_agent.py -v`

### 6b — `NoiseScenario` : PAS du code mort !

**ATTENTION : Contrairement à ce que l'audit précédent suggérait, `NoiseScenario` est utilisé dans :**
- `packages/simulation/evaluators.py` (lignes 11, 87, 206, 242, 301)
- `packages/core/parameter_space.py` (lignes 17, 496, 532, 534, 541, 561)
- `packages/simulation/noise_profiles.py` (lignes 4, 9, 11, 22, 27)
- `packages/ml/data_generator.py` (lignes 19, 178)

**NE PAS SUPPRIMER `NoiseScenario`.** C'est un modèle actif et utilisé.

---

## TÂCHE 7 — Fixer le bug de routing API double préfixe

### Problème actuel

Fichier `apps/api/routes/campaigns.py` :
```python
router = APIRouter(prefix="/campaigns", tags=["campaigns"])
```

Fichier `apps/api/routes/candidates.py` :
```python
router = APIRouter(prefix="/campaigns", tags=["candidates"])

@router.get("/{campaign_id}/candidates", ...)
```

Fichier `apps/api/main.py` :
```python
app.include_router(campaigns_router)
app.include_router(candidates_router)
```

Le router `candidates` a son propre prefix `/campaigns`, et le endpoint est `/{campaign_id}/candidates`. Cela fonctionne techniquement (le chemin résultant est `/campaigns/{campaign_id}/candidates`) mais c'est pour deux raisons problématique :

1. **Deux routers avec le même prefix `/campaigns`** — confusant et source potentielle de conflits
2. **Le router `candidates` n'est pas nestable** — si on veut ajouter d'autres endpoints candidates, on doit toujours préfixer avec `/campaigns`

### Solution recommandée

**Option A — Router candidates inclus comme sous-router de campaigns** (préférable) :

```python
# apps/api/routes/candidates.py
router = APIRouter(tags=["candidates"])

@router.get("/{campaign_id}/candidates", response_model=list[EvaluationResult])
def list_campaign_candidates(...):
    ...
```

```python
# apps/api/main.py
from apps.api.routes.candidates import router as candidates_router

# Include candidates under campaigns prefix
app.include_router(candidates_router, prefix="/campaigns")
```

**Option B — Nesting dans campaigns.py** (plus simple, 1 seul fichier modifié) :

```python
# apps/api/routes/campaigns.py
# Ajoute les endpoints candidates directement ici
from packages.core.models import EvaluationResult

@router.get("/{campaign_id}/candidates", response_model=list[EvaluationResult])
def list_campaign_candidates(
    campaign_id: str,
    repository: CryoSwarmRepository = Depends(get_repository),
) -> list[EvaluationResult]:
    campaign = repository.get_campaign(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found.")
    return repository.list_candidates_for_campaign(campaign_id)
```

Et dans `main.py`, retirer l'import de `candidates_router`.

### Choix : Implémente l'Option A

C'est plus propre architecturalement et permet d'isoler les endpoints candidates dans leur fichier dédié.

### Modifications concrètes

1. **`apps/api/routes/candidates.py`** — Retirer le `prefix="/campaigns"` du router :
   ```python
   router = APIRouter(tags=["candidates"])
   ```

2. **`apps/api/main.py`** — Ajouter le prefix au include :
   ```python
   app.include_router(candidates_router, prefix="/campaigns")
   ```

3. **Tests** — Vérifie que le endpoint `/campaigns/{id}/candidates` fonctionne toujours. Si `tests/test_health.py` ou `tests/test_api_error_handling.py` testent ce endpoint, assure-toi qu'ils passent.

### Bonus : Corriger la fuite d'information dans le global exception handler

Tant que tu es dans `apps/api/main.py`, corrige aussi le handler global qui leak des détails d'erreur :

**Avant :**
```python
@app.exception_handler(Exception)
async def global_handler(request, exc):
    logger.error("Unhandled: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc)})
```

**Après :**
```python
@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Check server logs for details."},
    )
```

Ajoute l'import `from fastapi import Request` en haut du fichier.

---

## ORDRE D'EXÉCUTION

Exécute les tâches dans cet ordre pour minimiser les conflits :

1. **TÂCHE 6** — Supprimer le code mort (le plus simple, zéro risque de régression)
2. **TÂCHE 5** — Corriger target_density (changement isolé dans un seul agent)
3. **TÂCHE 7** — Fixer le routing API (changement isolé dans 2 fichiers)
4. **TÂCHE 2** — Créer conftest.py (ajout pur, aucune modification de fichier existant)
5. **TÂCHE 3** — Centraliser les constantes physiques (modifications transversales mais rétrocompatibles)
6. **TÂCHE 4** — Frozen models (potentiellement le plus de tests à adapter)
7. **TÂCHE 1** — Refactoring pipeline (le plus gros morceau, à faire en dernier pour éviter les merge conflicts)

Après chaque tâche, lance : `pytest tests/ -x --tb=short`

---

## CONTRAINTES FINALES

- **Zéro `# TODO` laissé dans le code.** Si tu n'implémentes pas quelque chose, ne laisse pas un placeholder.
- **Chaque nouveau fichier doit avoir une docstring de module** expliquant son rôle.
- **Chaque nouvelle classe/fonction publique doit avoir une docstring** technique et concise.
- **Les imports doivent être triés** : stdlib, third-party, project.
- **Pas de lignes de plus de 120 caractères** (cohérent avec le style existant).
- **Chaque modification doit être testable.** Si tu ajoutes une fonction, écris un test.
- **Lance les tests après CHAQUE tâche, pas seulement à la fin.**
- **Si un test existant casse, corrige-le immédiatement** en préservant le comportement testé.

## VÉRIFICATION FINALE

À la fin de toutes les tâches, lance :

```bash
pytest tests/ -v --tb=short 2>&1 | tail -50
```

Tous les tests doivent passer. Aucune régression tolérée.
