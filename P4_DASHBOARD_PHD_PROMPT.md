# P4 — DASHBOARD PHD-GRADE : Prompt d'implémentation exhaustif

> **Destinataire** : Agent IA de code.
> **Mode** : Implémentation directe — écris chaque ligne, chaque import, chaque test.
> **Philosophie** : Ce dashboard sera utilisé par des chercheurs PhD en physique quantique et en optimisation. Chaque page doit combiner visualisation interactive de données, formules LaTeX rigoureuses, et navigation fluide à travers le workflow CryoSwarm-Q. Pas de graphes statiques matplotlib — tout en Plotly interactif. Pas de formules approximatives — chaque équation doit être exacte et référençable. Le dashboard est la vitrine scientifique du projet.

---

## Contexte du projet

CryoSwarm-Q est un système multi-agent hardware-aware pour la conception autonome d'expériences en informatique quantique à atomes neutres. Le pipeline orchestre 8 agents spécialisés dans des phases séquentielles : problem framing → geometry → sequence → evaluation → routing → ranking → results → memory.

**Structure actuelle du workspace** :

```
packages/
  agents/         → 8 agents : problem, geometry, sequence, noise, routing, campaign, results, memory
                    Protocoles dans protocols.py, base commune dans base.py
  core/           → models.py (Pydantic v2), enums.py (StrEnum), config.py, parameter_space.py, exceptions.py, metadata_schemas.py
  orchestration/  → pipeline.py (CryoSwarmPipeline, PipelineContext), runner.py
  scoring/        → objective.py (compute_objective_score), ranking.py (rank_evaluations), robustness.py
  simulation/     → evaluators.py, hamiltonian.py, numpy_backend.py, observables.py, noise_profiles.py
  pasqal_adapters/→ pulser_adapter.py, emulator_router.py, pasqal_cloud_adapter.py, qoolqit_adapter.py
  db/             → mongodb.py, repositories.py (CryoSwarmRepository), init_db.py
  ml/             → surrogate.py (V1/V2, Ensemble), ppo.py (ActorCritic, PPOTrainer), rl_env.py, rl_sequence_agent.py,
                    dataset.py, data_generator.py, surrogate_filter.py, gpu_backend.py, curriculum.py, active_learning.py
apps/
  api/            → main.py (FastAPI), auth.py, dependencies.py, routes/ (health, goals, campaigns, candidates)
  dashboard/      → app.py (Streamlit single-page actuel), logic.py (helpers)
tests/            → 50+ fichiers de tests, conftest.py
```

**Dashboard actuel** (`apps/dashboard/app.py`, 213 lignes) :
- Page unique Streamlit avec formulaire de lancement + tableaux + graphes matplotlib statiques
- Fonctions helper dans `logic.py` (103 lignes) : `build_campaign_table`, `build_ranked_table`, `build_decision_table`, `select_robustness_chart_data`, `select_noise_sensitivity_data`, `build_register_lookup_from_documents`
- Graphes : register scatter plot (matplotlib), robustness bar chart (matplotlib), noise sensitivity line (matplotlib)
- Aucune formule LaTeX, aucun graphe interactif, aucune navigation multi-page

**Fichier manifeste** : `CLAUDE.md` à la racine contient la vision complète du projet.

**Hypothèse** : Les tâches P0, P1, P2, P3 ont été implémentées. Le pipeline fonctionne end-to-end avec MongoDB Atlas, les observables sont complets (rydberg_density, entanglement_entropy, antiferromagnetic_order, connected_correlation, bitstring_probabilities, MIS overlap), le Hamiltonian supporte dense et sparse, le ML stack (surrogate ensemble + PPO + RL sequence agent) est en place.

---

## Règles impératives

- Python 3.11+, types partout, `from __future__ import annotations`
- Imports absolus uniquement (`from packages.core.models import ...`)
- Chaque import groupé : stdlib, third-party, project
- `pytest tests/ -x --tb=short` après CHAQUE tâche
- Zéro `# TODO` laissé dans le code produit
- **Plotly** pour TOUS les graphes interactifs (pas matplotlib, sauf register 2D si nécessaire)
- **LaTeX** via `st.latex()` et `st.markdown()` avec `$...$` / `$$...$$`
- **Streamlit multi-page** : utiliser la convention native `pages/` de Streamlit
- Les modifications doivent être rétro-compatibles : l'ancien `app.py` single-page doit rester fonctionnel comme fallback
- Ne pas dupliquer la logique métier — importer depuis `packages/`
- Chaque page doit fonctionner indépendamment (pas de dépendance d'état inter-pages sauf via `st.session_state` explicite)

---

## Dépendances à ajouter

Dans `pyproject.toml`, ajouter aux `dependencies` :

```toml
"plotly>=5.22",
"networkx>=3.2",
```

Vérifier que `streamlit>=1.38` est déjà présent (oui). Streamlit supporte nativement Plotly via `st.plotly_chart()` et LaTeX via `st.latex()`.

---

## Architecture cible

```
apps/dashboard/
  app.py                      # Entry point : navigation sidebar, session state init, MongoDB connect
  logic.py                    # Existant — enrichir avec nouvelles fonctions helper
  components/
    __init__.py
    plotly_charts.py           # Toutes les fonctions Plotly réutilisables
    latex_panels.py            # Blocs LaTeX pré-formatés par thème physique
    data_loaders.py            # Fonctions de chargement et cache depuis MongoDB
  pages/
    1_Campaign_Control.py      # Lancement, monitoring, pipeline progress
    2_Register_Physics.py      # Géométrie atomique, interactions, blockade graph
    3_Hamiltonian_Lab.py       # Spectre énergétique, gap spectral, MIS
    4_Pulse_Studio.py          # Profils temporels, waveforms, comparaison
    5_Robustness_Arena.py      # Analyse de bruit, sensibilité, dégradation
    6_ML_Observatory.py        # Surrogate, PPO, RL, bandit strategy
    7_Campaign_Analytics.py    # Cross-campaign, tendances, mémoire
    8_Theory_Reference.py      # Référence mathématique complète
tests/
  test_dashboard_components.py # Tests des fonctions Plotly et LaTeX (sans Streamlit runtime)
  test_data_loaders.py         # Tests des loaders avec mocks MongoDB
```

---

## TÂCHE 1 — Refactorer `app.py` en point d'entrée multi-page + créer `components/`

### 1.1 Créer `apps/dashboard/components/__init__.py`

Fichier vide.

### 1.2 Créer `apps/dashboard/components/data_loaders.py`

Ce fichier centralise TOUS les accès MongoDB pour le dashboard. Chaque fonction utilise `@st.cache_data(ttl=60)` pour éviter les requêtes répétées.

```python
"""Cached data loaders for the CryoSwarm-Q dashboard.

All MongoDB queries pass through this module so pages never touch
the repository directly.  Results are cached with a 60-second TTL.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from packages.core.models import (
    AgentDecision,
    CampaignState,
    EvaluationResult,
    ExperimentGoal,
    MemoryRecord,
    RegisterCandidate,
    RobustnessReport,
    SequenceCandidate,
)
from packages.db.repositories import CryoSwarmRepository


def get_repository() -> CryoSwarmRepository:
    """Return a shared repository instance stored in session state."""
    if "repository" not in st.session_state:
        from packages.core.config import get_settings
        from packages.db.init_db import initialize_database
        settings = get_settings()
        if not settings.has_mongodb:
            st.error("MONGODB_URI is not configured.")
            st.stop()
        initialize_database()
        st.session_state["repository"] = CryoSwarmRepository(settings)
    return st.session_state["repository"]


@st.cache_data(ttl=60, show_spinner=False)
def load_latest_campaigns(limit: int = 20) -> list[dict[str, Any]]:
    repo = get_repository()
    campaigns = repo.list_latest_campaigns(limit=limit)
    return [c.model_dump(mode="json") for c in campaigns]


@st.cache_data(ttl=60, show_spinner=False)
def load_campaign(campaign_id: str) -> dict[str, Any] | None:
    repo = get_repository()
    campaign = repo.get_campaign(campaign_id)
    return campaign.model_dump(mode="json") if campaign else None


@st.cache_data(ttl=60, show_spinner=False)
def load_goal(goal_id: str) -> dict[str, Any] | None:
    repo = get_repository()
    goal = repo.get_goal(goal_id)
    return goal.model_dump(mode="json") if goal else None


@st.cache_data(ttl=60, show_spinner=False)
def load_ranked_candidates(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    results = repo.list_candidates_for_campaign(campaign_id)
    return [r.model_dump(mode="json") for r in results]


@st.cache_data(ttl=60, show_spinner=False)
def load_robustness_reports(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    reports = repo.list_robustness_reports(campaign_id)
    return [r.model_dump(mode="json") for r in reports]


@st.cache_data(ttl=60, show_spinner=False)
def load_agent_decisions(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    decisions = repo.list_agent_decisions(campaign_id)
    return [d.model_dump(mode="json") for d in decisions]


@st.cache_data(ttl=60, show_spinner=False)
def load_register_candidates(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    cursor = repo.collections["register_candidates"].find({"campaign_id": campaign_id})
    results = []
    for doc in cursor:
        doc.pop("_id", None)
        results.append(doc)
    return results


@st.cache_data(ttl=60, show_spinner=False)
def load_sequence_candidates(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    cursor = repo.collections["sequence_candidates"].find({"campaign_id": campaign_id})
    results = []
    for doc in cursor:
        doc.pop("_id", None)
        results.append(doc)
    return results


@st.cache_data(ttl=60, show_spinner=False)
def load_memory_records(campaign_id: str) -> list[dict[str, Any]]:
    repo = get_repository()
    records = repo.list_memory_records(campaign_id)
    return [r.model_dump(mode="json") for r in records]


@st.cache_data(ttl=60, show_spinner=False)
def load_all_memory(limit: int = 50) -> list[dict[str, Any]]:
    repo = get_repository()
    records = repo.list_recent_memory(limit=limit)
    return [r.model_dump(mode="json") for r in records]
```

**Important** : les fonctions retournent des `dict` (pas des modèles Pydantic) car `@st.cache_data` nécessite des objets sérialisables. Les pages reconstruisent les modèles Pydantic si nécessaire via `ModelClass.model_validate(data)`.

### 1.3 Créer `apps/dashboard/components/latex_panels.py`

Ce fichier contient toutes les formules LaTeX organisées par thème. Chaque fonction retourne une string LaTeX (pas d'appel Streamlit direct — la page choisit `st.latex()` ou `st.markdown()`).

Inclure les sections suivantes (chacune est une fonction retournant un `str` ou un `dict[str, str]` de formules nommées) :

**Fonctions requises** :

```python
def hamiltonian_formulas() -> dict[str, str]:
    """Rydberg Hamiltonian and interaction terms."""
    return {
        "full_hamiltonian": r"\hat{H} = \frac{\Omega}{2}\sum_{i=1}^{N} \hat{\sigma}_x^{(i)} - \delta\sum_{i=1}^{N} \hat{n}_i + \sum_{i<j} \frac{C_6}{|\mathbf{r}_i - \mathbf{r}_j|^6}\,\hat{n}_i\hat{n}_j",
        "interaction": r"U_{ij} = \frac{C_6}{r_{ij}^6}",
        "blockade_radius": r"R_b = \left(\frac{C_6}{\Omega}\right)^{\!1/6}",
        "c6_value": r"C_6 = 2\pi \times 862\,690 \;\text{rad}\cdot\mu\text{m}^6/\mu\text{s} \quad (^{87}\text{Rb},\; |70S_{1/2}\rangle)",
        "spectral_gap": r"\Delta E = E_1 - E_0",
        "ipr": r"\text{IPR} = \sum_{n} |\langle n | \psi_0 \rangle|^4",
        "adiabatic_condition": r"\frac{|\langle \psi_1 | \dot{H} | \psi_0 \rangle|}{(\Delta E)^2} \ll 1",
    }


def observable_formulas() -> dict[str, str]:
    """Quantum observables computed from state vectors."""
    return {
        "rydberg_density": r"\langle \hat{n}_i \rangle = \text{Tr}\!\left(\hat{n}_i |\psi\rangle\langle\psi|\right)",
        "connected_correlation": r"g_{ij}^{(2)} = \langle \hat{n}_i \hat{n}_j \rangle - \langle \hat{n}_i \rangle \langle \hat{n}_j \rangle",
        "antiferromagnetic_order": r"m_{\text{AF}} = \frac{1}{N}\left|\sum_{i=1}^{N} (-1)^i \bigl(2\langle \hat{n}_i \rangle - 1\bigr)\right|",
        "entanglement_entropy": r"S_A = -\text{Tr}\!\left(\rho_A \log_2 \rho_A\right)",
        "total_rydberg_fraction": r"\bar{n} = \frac{1}{N}\sum_{i=1}^{N} \langle \hat{n}_i \rangle",
        "state_fidelity": r"\mathcal{F} = |\langle \psi_1 | \psi_2 \rangle|^2",
        "bitstring_probability": r"P(b_0 b_1 \cdots b_{N-1}) = |\langle b_0 b_1 \cdots b_{N-1} | \psi \rangle|^2",
    }


def mis_formulas() -> dict[str, str]:
    """Maximum Independent Set on the blockade graph."""
    return {
        "mis_definition": r"\text{MIS}(G) = \arg\max_{S \subseteq V}\; |S| \quad \text{s.t.} \quad \forall\,(u,v)\in E,\; u \notin S \;\text{or}\; v \notin S",
        "mis_overlap": r"P_{\text{MIS}} = \sum_{s\,\in\,\text{MIS}} |\langle s | \psi_0 \rangle|^2",
        "mis_cost_function": r"C_{\text{MIS}} = -\sum_{i \in V} n_i + \alpha \sum_{(i,j) \in E} n_i n_j, \quad \alpha > 1",
    }


def pulse_formulas() -> dict[str, str]:
    """Pulse sequence waveforms and time evolution."""
    return {
        "blackman": r"\Omega_{\text{Blackman}}(t) = \Omega_{\max}\!\left[0.42 - 0.50\cos\!\left(\frac{2\pi t}{T}\right) + 0.08\cos\!\left(\frac{4\pi t}{T}\right)\right]",
        "linear_sweep": r"\delta(t) = \delta_{\text{start}} + \frac{\delta_{\text{end}} - \delta_{\text{start}}}{T}\,t",
        "pulse_area": r"\theta = \int_0^T \Omega(t)\,dt \qquad (\pi\text{-pulse:}\;\theta = \pi)",
        "time_evolution": r"|\psi(t)\rangle = \mathcal{T}\exp\!\left(-i\int_0^t \hat{H}(t')\,dt'\right)|\psi(0)\rangle",
        "trotter_suzuki_2": r"e^{-i(A+B)\Delta t} = e^{-iA\,\Delta t/2}\,e^{-iB\,\Delta t}\,e^{-iA\,\Delta t/2} + \mathcal{O}(\Delta t^3)",
        "ramp": r"\Omega_{\text{ramp}}(t) = \Omega_{\max} \cdot \frac{t}{T}",
        "constant": r"\Omega_{\text{const}}(t) = \Omega_{\max}, \quad \delta(t) = \delta_0",
    }


def robustness_formulas() -> dict[str, str]:
    """Robustness scoring and noise modeling."""
    return {
        "robustness_score": r"S_{\text{robust}} = w_n \, s_{\text{nom}} + w_a \, \bar{s}_{\text{pert}} + w_w \, s_{\text{worst}} + w_s \, b_{\text{stab}}",
        "robustness_weights": r"w_n = 0.25,\quad w_a = 0.35,\quad w_w = 0.30,\quad w_s = 0.10",
        "stability_bonus": r"b_{\text{stab}} = \max\!\left(0,\; 1 - \frac{\sigma_s}{\sigma_{\text{thresh}}}\right)",
        "penalty": r"\text{Penalty} = \max\!\left(0,\; (s_{\text{nom}} - s_{\text{worst}}) + \sigma_s\right)",
        "objective_score": r"S_{\text{obj}} = \alpha\, s_{\text{obs}} + \beta\, s_{\text{robust}} - \gamma\, c_{\text{exec}} - \delta\, \ell_{\text{latency}}",
        "weight_constraint": r"\alpha + \beta + \gamma + \delta = 1 \quad (\alpha{=}0.45,\;\beta{=}0.35,\;\gamma{=}0.10,\;\delta{=}0.10)",
        "noise_amplitude": r"\Omega \to \Omega\,(1 + \epsilon_\Omega), \quad \epsilon_\Omega \sim \mathcal{N}(0, \sigma_{\text{amp}}^2)",
        "noise_detuning": r"\delta \to \delta + \epsilon_\delta, \quad \epsilon_\delta \sim \mathcal{N}(0, \sigma_{\text{det}}^2 |\delta| + 0.1)",
        "dephasing_lindblad": r"\mathcal{L}_\phi[\rho] = \gamma_\phi \left(\hat{n}\,\rho\,\hat{n} - \tfrac{1}{2}\{\hat{n}^2, \rho\}\right)",
        "density_score": r"s_{\text{dens}} = \max\!\left(0,\; 1 - \frac{|\bar{n}_{\text{obs}} - \bar{n}_{\text{target}}|}{\max(\bar{n}_{\text{target}},\, 1 - \bar{n}_{\text{target}},\, 0.5)}\right)",
        "blockade_score": r"s_{\text{block}} = \max\!\left(0,\; 1 - \min\!\left(\frac{v_{\text{block}}}{0.20},\, 1\right)\right)",
    }


def ml_formulas() -> dict[str, str]:
    """Machine learning: PPO, surrogate, bandit."""
    return {
        "ucb1": r"\text{UCB1}(s) = \bar{r}_s + \sqrt{\frac{2\ln N}{n_s}}",
        "ppo_objective": r"L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}\!\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\;\text{clip}(r_t(\theta),\, 1 {\pm} \epsilon)\,\hat{A}_t\right)\right]",
        "importance_ratio": r"r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}",
        "gae": r"\hat{A}_t^{\text{GAE}} = \sum_{\ell=0}^{\infty}(\gamma\lambda)^\ell\,\delta_{t+\ell}, \quad \delta_t = r_t + \gamma\,V(s_{t+1}) - V(s_t)",
        "ensemble_mean": r"\hat{y}_{\text{ens}} = \frac{1}{M}\sum_{m=1}^{M} f_m(\mathbf{x})",
        "epistemic_uncertainty": r"\sigma_{\text{epist}}^2 = \frac{1}{M}\sum_{m=1}^{M} \bigl(f_m(\mathbf{x}) - \hat{y}_{\text{ens}}\bigr)^2",
        "weighted_mse": r"L = \frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{K} w_k\,\bigl(\hat{y}_{b,k} - y_{b,k}\bigr)^2",
    }


def campaign_formulas() -> dict[str, str]:
    """Campaign analytics: Pareto, regret, ranking."""
    return {
        "pareto_optimal": r"\mathbf{x}^* \text{ is Pareto optimal if } \nexists\,\mathbf{x} : f_i(\mathbf{x}) \geq f_i(\mathbf{x}^*)\;\forall i,\; f_j(\mathbf{x}) > f_j(\mathbf{x}^*)\;\exists j",
        "cumulative_regret": r"R_T = \sum_{t=1}^{T}\!\left(r^* - r_t\right)",
        "acquisition_ucb": r"\alpha(\mathbf{x}) = \mu(\mathbf{x}) + \kappa\,\sigma(\mathbf{x})",
        "ranking_key": r"\text{rank}(c) = \text{sort}\!\left(-S_{\text{obj}},\, -s_{\text{worst}},\, -S_{\text{robust}},\, -s_{\text{nom}}\right)",
    }
```

### 1.4 Créer `apps/dashboard/components/plotly_charts.py`

Ce fichier contient TOUTES les fonctions de visualisation Plotly. Chaque fonction retourne un `plotly.graph_objects.Figure` — elle ne touche JAMAIS Streamlit directement. La page appelante fait `st.plotly_chart(fig, use_container_width=True)`.

**Fonctions requises (signatures exactes à implémenter)** :

```python
"""Reusable Plotly chart builders for CryoSwarm-Q dashboard pages.

Every function returns a ``plotly.graph_objects.Figure``.
Pages call ``st.plotly_chart(fig, use_container_width=True)``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Page 1: Campaign Control ──────────────────────────────────────────

def pipeline_gantt(decisions: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart showing agent phase durations.

    Uses decision timestamps to reconstruct phase timing.
    Each agent gets its own colored bar. x-axis = relative time (seconds).
    """
    ...


def candidate_funnel(campaign_data: dict[str, Any], n_registers: int, n_sequences: int, n_evaluated: int, n_ranked: int) -> go.Figure:
    """Funnel chart showing how many candidates survive each pipeline phase.

    Phases: Registers → Sequences → Evaluated → Ranked.
    Show attrition percentage at each step.
    """
    ...


# ── Page 2: Register Physics ──────────────────────────────────────────

def register_scatter_2d(coordinates: list[tuple[float, float]], blockade_radius: float, label: str, rydberg_densities: list[float] | None = None) -> go.Figure:
    """Interactive 2D atom register layout.

    - Atom positions as scatter points
    - Blockade radius circles (dashed red, semi-transparent)
    - If rydberg_densities provided, color atoms by excitation probability
    - Hover: (x, y) in µm, atom index, Rydberg density
    - Equal aspect ratio, grid
    """
    ...


def vdw_interaction_heatmap(coordinates: list[tuple[float, float]], c6: float) -> go.Figure:
    """Van der Waals interaction matrix U_ij as annotated heatmap.

    - Log-scale colorbar (Viridis)
    - Hover shows U_ij value in rad/µs and distance in µm
    - Axis labels: atom indices q0, q1, ...
    """
    ...


def blockade_graph(coordinates: list[tuple[float, float]], adjacency: list[list[bool]], mis_sets: list[list[int]] | None = None) -> go.Figure:
    """Blockade graph as network visualization.

    - Nodes at atom positions (x, y)
    - Edges between blockaded pairs
    - Node size proportional to degree
    - If mis_sets provided, highlight first MIS solution nodes in gold
    - Hover: atom index, degree, MIS membership
    """
    ...


def distance_histogram(coordinates: list[tuple[float, float]], blockade_radius: float) -> go.Figure:
    """Histogram of pairwise distances with blockade radius marker.

    - Vertical dashed line at R_b
    - Annotation: "Blockade radius" with arrow
    - Color bars below R_b in red (blockaded), above in blue
    """
    ...


# ── Page 3: Hamiltonian Lab ───────────────────────────────────────────

def energy_spectrum(eigenvalues: list[float], n_show: int = 20) -> go.Figure:
    """Bar chart of lowest eigenvalues with spectral gap annotation.

    - x-axis: eigenstate index
    - y-axis: energy (rad/µs)
    - Gap ΔE annotated between E0 and E1 with double-arrow
    - Ground state bar in gold, rest in blue
    """
    ...


def bitstring_bar_chart(bitstrings: list[tuple[str, float]], mis_bitstrings: list[str] | None = None) -> go.Figure:
    """Bar chart of top-k bitstring probabilities.

    - x-axis: bitstring labels
    - y-axis: probability
    - MIS bitstrings highlighted in gold
    - Hover: decimal index, Hamming weight, probability
    """
    ...


def parametric_spectrum(delta_values: list[float], eigenvalue_curves: list[list[float]], omega: float) -> go.Figure:
    """Parametric energy level diagram (delta sweep).

    - x-axis: detuning delta (rad/µs)
    - y-axis: energy (rad/µs)
    - One line per eigenstate (first 6-8)
    - Avoided crossing regions highlighted
    - Subtitle shows Omega value
    """
    ...


# ── Page 4: Pulse Studio ─────────────────────────────────────────────

def pulse_waveform(time_ns: list[float], omega_t: list[float], delta_t: list[float], label: str) -> go.Figure:
    """Dual-axis pulse waveform plot.

    - Left y-axis: Ω(t) in rad/µs (blue)
    - Right y-axis: δ(t) in rad/µs (red)
    - x-axis: time in ns
    - Filled area under Ω(t) with alpha=0.15 for pulse area visualization
    - Title includes sequence label and family
    """
    ...


def time_evolution_traces(time_ns: list[float], densities_per_site: list[list[float]], total_fraction: list[float]) -> go.Figure:
    """Rydberg density time evolution per atom site.

    - One line per atom (q0, q1, ...) with distinct colors
    - Bold black dashed line for total fraction
    - x-axis: time (ns), y-axis: ⟨n_i⟩
    """
    ...


def parameter_space_scatter(candidates: list[dict[str, Any]]) -> go.Figure:
    """2D scatter: amplitude vs detuning, colored by objective score.

    - Each point = one sequence candidate
    - Color: objective_score (Viridis)
    - Size: proportional to duration_ns
    - Hover: all parameters + scores
    - Shape encodes sequence_family
    """
    ...


# ── Page 5: Robustness Arena ─────────────────────────────────────────

def robustness_grouped_bar(reports: list[dict[str, Any]]) -> go.Figure:
    """Grouped bar chart: nominal / average / worst per candidate.

    - x-axis: candidate labels (truncated IDs)
    - 3 bars per candidate: nominal (blue), average (orange), worst (red)
    - y-axis: [0, 1] score
    - Sort by robustness_score descending
    """
    ...


def noise_radar(reports: list[dict[str, Any]], max_candidates: int = 4) -> go.Figure:
    """Radar (polar) chart of noise sensitivity.

    - Axes: amplitude_jitter_sensitivity, detuning_sensitivity, dephasing, atom_loss, SPAM, temperature
    - One polygon per candidate (up to max_candidates)
    - Derived from scenario_scores difference: sensitivity = nominal - scenario_score
    - Overlaid semi-transparent fills
    """
    ...


def score_degradation_waterfall(report: dict[str, Any]) -> go.Figure:
    """Waterfall chart showing score drop from nominal through each noise scenario.

    - Start at nominal_score
    - Each bar = drop introduced by one noise level (low, medium, stressed)
    - Final bar = worst_case_score
    - Red bars for negative drops, green for recovery (rare)
    """
    ...


def robustness_violin(reports: list[dict[str, Any]]) -> go.Figure:
    """Violin plot of scenario score distributions per candidate.

    - One violin per candidate
    - Points overlay for individual scenario scores
    - Median line + quartile indicators
    """
    ...


def perturbation_heatmap(sweep_data: dict[str, Any]) -> go.Figure:
    """2D contour heatmap: amplitude_jitter × detuning_jitter → robustness_score.

    - If sweep_data available (from metadata), render actual grid
    - Else show placeholder with note "Run parameter sweep to populate"
    - Optimal region annotated
    """
    ...


# ── Page 6: ML Observatory ───────────────────────────────────────────

def training_loss_curves(history: dict[str, list[float]]) -> go.Figure:
    """Train/val loss curves for surrogate model.

    - Dual lines: train_loss (blue), val_loss (orange)
    - x-axis: epoch, y-axis: weighted MSE (log scale option)
    - Annotation at minimum val_loss point
    """
    ...


def prediction_vs_actual(predictions: list[float], actuals: list[float], target_name: str = "robustness") -> go.Figure:
    """Scatter: predicted vs actual with 45-degree reference line.

    - Each point is one test sample
    - Diagonal line y=x in dashed black
    - R² annotation in corner
    - Color by absolute error
    """
    ...


def ppo_training_dashboard(history: dict[str, list[float]]) -> go.Figure:
    """Multi-panel PPO training plot (2x2 subplots).

    - Top-left: episode reward + 50-update rolling average
    - Top-right: policy loss
    - Bottom-left: value loss
    - Bottom-right: entropy (if available, else empty)
    """
    ...


def strategy_ucb_evolution(strategy_data: dict[str, list[float]]) -> go.Figure:
    """UCB1 scores per strategy over time.

    - One line per strategy (heuristic, rl, hybrid)
    - x-axis: trial number, y-axis: UCB1 score
    - Annotation when strategy switches
    """
    ...


# ── Page 7: Campaign Analytics ───────────────────────────────────────

def campaign_timeline(campaigns: list[dict[str, Any]]) -> go.Figure:
    """Horizontal timeline of all campaigns.

    - Each campaign as a horizontal bar from created_at to completed_at
    - Color by status (completed=green, failed=red, no_candidates=yellow)
    - Hover: campaign_id, goal_id, candidate_count, top_score
    """
    ...


def cross_campaign_score_evolution(campaigns: list[dict[str, Any]]) -> go.Figure:
    """Line chart: best objective score per campaign over time.

    - x-axis: campaign date (created_at)
    - y-axis: best objective_score from top candidate
    - Trend line (5-campaign rolling average)
    - Point size proportional to candidate_count
    """
    ...


def backend_distribution_stacked(campaigns: list[dict[str, Any]]) -> go.Figure:
    """Stacked area/bar chart: backend mix per campaign.

    - x-axis: campaign index
    - Stacked areas: LOCAL_PULSER, EMU_SV, EMU_MPS proportions
    - Shows routing preference evolution
    """
    ...


def parameter_space_3d(candidates: list[dict[str, Any]]) -> go.Figure:
    """3D scatter: amplitude × detuning × duration, colored by robustness_score.

    - All candidates across all campaigns
    - Pareto frontier candidates highlighted in gold outline
    - Hover: campaign_id, sequence_family, all scores
    """
    ...


def memory_tag_cloud(memory_records: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart of most frequent reusable_tags.

    - Sorted by frequency descending
    - Color by lesson_type (candidate_pattern vs failure_pattern)
    """
    ...
```

**IMPORTANT** : Chaque fonction doit avoir un corps complet et fonctionnel. Pas de `...` ou `pass`. Utilise les bonnes pratiques Plotly :
- `fig.update_layout(template="plotly_white", ...)` pour un look clean
- `fig.update_traces(hovertemplate=...)` pour des tooltips informatifs
- Couleurs cohérentes : bleu primaire `#1f77b4`, rouge `#d62728`, vert `#2ca02c`, or `#ffd700`
- `config={"displayModeBar": True}` côté page (pas dans la figure)

---

## TÂCHE 2 — Page 1 : Campaign Control Center

### Fichier : `apps/dashboard/pages/1_Campaign_Control.py`

**But** : Lancer de nouvelles campagnes et suivre leur progression pipeline en temps réel.

### Layout précis

```
┌──────────────────────────────────────────────────────────────────┐
│ st.title("Campaign Control Center")                              │
│ st.caption("Launch, monitor, and inspect experiment campaigns.") │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Objective Score Formula (collapsible st.expander)      │
│  st.latex: S_obj = α·s_obs + β·s_robust - γ·c_exec - δ·ℓ_lat  │
│  st.markdown: weight meanings, constraint α+β+γ+δ = 1           │
├──────────────────────────────────────────────────────────────────┤
│ [FORM] New Campaign                                              │
│  Row 1: title (text_input) | scientific_objective (text_area)    │
│  Row 2: desired_atom_count (slider 2-50) |                       │
│          preferred_geometry (selectbox: mixed/square/line/2d/     │
│          triangular/ring/zigzag/honeycomb)                        │
│  Row 3: target_observable (selectbox) | priority (selectbox)     │
│  Row 4: Priority weights (4 sliders α,β,γ,δ) with sum display   │
│  [RUN CAMPAIGN] (form_submit_button)                             │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Recent Campaigns Table                                 │
│  st.dataframe with columns: campaign_id, goal_id, status,       │
│  candidate_count, top_candidate_id, created_at                   │
│  Clickable campaign_id → sets selected_campaign in session_state │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Pipeline Inspector (for selected campaign)             │
│  Left col (40%):                                                 │
│    Pipeline phase status badges                                  │
│    6 phases: Problem Framing ✓ → Geometry ✓ → Sequence ✓ →      │
│    Evaluation ✓ → Ranking ✓ → Memory ✓                          │
│    Each phase shows: agent_name, status, candidate count         │
│  Right col (60%):                                                │
│    pipeline_gantt() chart                                        │
│    candidate_funnel() chart                                      │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Agent Decision Log (for selected campaign)             │
│  st.dataframe of all decisions: agent, decision_type, status,    │
│  reasoning_summary (truncated), created_at                       │
│  Expandable rows for full structured_output JSON                 │
└──────────────────────────────────────────────────────────────────┘
```

### Données sources

- `DemoGoalRequest` depuis `packages.core.models` pour le formulaire
- `run_demo_campaign` depuis `packages.orchestration.runner` pour lancer
- `load_latest_campaigns`, `load_agent_decisions`, `load_ranked_candidates` depuis `data_loaders`
- `candidate_funnel`, `pipeline_gantt` depuis `plotly_charts`
- `robustness_formulas()["objective_score"]` et `["weight_constraint"]` depuis `latex_panels`

### Comportement

- Au submit du formulaire, exécuter `run_demo_campaign()` avec un spinner `st.spinner("Running pipeline...")`
- Après complétion, `st.cache_data.clear()` pour rafraîchir les données
- Le sélecteur de campagne utilise `st.session_state["selected_campaign_id"]`

---

## TÂCHE 3 — Page 2 : Register Physics Lab

### Fichier : `apps/dashboard/pages/2_Register_Physics.py`

**But** : Explorer les géométries atomiques, les interactions van der Waals, et le graphe de blockade pour chaque candidat registre.

### Layout précis

```
┌──────────────────────────────────────────────────────────────────┐
│ st.title("Register Physics Lab")                                 │
│ st.caption("Atom register geometry, interactions, blockade.")    │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Physics Primer (st.expander, collapsed by default)     │
│  LaTeX: U_ij = C6/r_ij^6                                        │
│  LaTeX: R_b = (C6/Ω)^{1/6}                                     │
│  LaTeX: C6 value for 87-Rb                                      │
│  st.markdown: explanation of blockade mechanism                  │
├──────────────────────────────────────────────────────────────────┤
│ Campaign selector | Register candidate selector (from dropdown)  │
├──────────────────────────────────────────────────────────────────┤
│ Left col (50%):                         Right col (50%):         │
│  register_scatter_2d() chart             vdw_interaction_heatmap │
│  - atoms with blockade circles           - log-scale colorbar   │
│  - color by Rydberg density              - annotated values      │
│    if available from reports                                     │
├──────────────────────────────────────────────────────────────────┤
│ Left col (50%):                         Right col (50%):         │
│  blockade_graph() chart                  distance_histogram()    │
│  - network with MIS highlighted          - R_b vertical marker  │
│  - node size by degree                                           │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Register Metrics Summary                               │
│  4 metric cards (st.metric):                                     │
│    Atom Count | Min Distance (µm) | Blockade Pairs | Feasibility│
│  + JSON expander for full register data                          │
└──────────────────────────────────────────────────────────────────┘
```

### Calculs côté page

Pour les graphes de cette page, effectuer les calculs directement en important les fonctions de simulation (elles sont rapides pour ≤12 atomes) :

```python
from packages.simulation.hamiltonian import (
    van_der_waals_matrix,
    interaction_graph,
    find_maximum_independent_sets,
    blockade_radius,
)
```

- Calculer `blockade_radius(omega_default, c6)` avec omega=5.0 rad/µs par défaut
- Calculer `interaction_graph(coords, omega)` pour le blockade graph
- Calculer `find_maximum_independent_sets(adjacency)` pour les MIS
- Calculer `van_der_waals_matrix(coords, c6)` pour le heatmap

Le slider Omega doit être interactif : changer Omega recalcule le blockade radius et le graphe en temps réel.

---

## TÂCHE 4 — Page 3 : Hamiltonian Spectroscopy

### Fichier : `apps/dashboard/pages/3_Hamiltonian_Lab.py`

**But** : Analyser le spectre énergétique du Hamiltonien many-body, les états propres, et le comportement paramétrique.

### Layout précis

```
┌──────────────────────────────────────────────────────────────────┐
│ st.title("Hamiltonian Spectroscopy")                             │
│ st.caption("Energy spectrum, eigenstates, parametric analysis.") │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Rydberg Hamiltonian (st.expander, expanded by default) │
│  st.latex: Full H = Ω/2 Σ σ_x - δ Σ n_i + Σ U_ij n_i n_j     │
│  st.latex: spectral gap ΔE = E1 - E0                            │
│  st.latex: IPR = Σ |<n|ψ0>|^4                                   │
│  st.latex: adiabatic condition                                   │
│  st.markdown: physical interpretation of each term              │
├──────────────────────────────────────────────────────────────────┤
│ [CONTROLS]                                                       │
│  Register selector (from selected campaign)                      │
│  Ω slider (0.1 - 15.8 rad/µs, default 5.0)                     │
│  δ slider (-126.0 - +126.0 rad/µs, default -10.0)              │
│  [Auto from sequence] checkbox → populate from best sequence     │
├──────────────────────────────────────────────────────────────────┤
│ Left col (60%):                         Right col (40%):         │
│  energy_spectrum() chart                 3 st.metric cards:      │
│  - ground state in gold                  - Spectral gap ΔE      │
│  - ΔE annotation                         - Hilbert dim 2^N      │
│  - first 20 eigenvalues                  - Ground state IPR     │
├──────────────────────────────────────────────────────────────────┤
│ Left col (50%):                         Right col (50%):         │
│  bitstring_bar_chart()                   State composition:      │
│  - top-20 bitstrings                     - Rydberg fraction      │
│  - MIS solutions highlighted             - Entanglement entropy  │
│  - from ground state                     - AF order parameter    │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Parametric Detuning Sweep                              │
│  parametric_spectrum() chart                                     │
│  - x: δ from -20 to +20 (or wider)                              │
│  - y: first 8 eigenvalues                                        │
│  - Interactive Ω slider updates in real-time                     │
│  st.markdown: phase transition interpretation                    │
│  ⚠ Limité à ≤10 atomes (exécution temps réel)                  │
└──────────────────────────────────────────────────────────────────┘
```

### Calculs côté page

```python
from packages.simulation.hamiltonian import (
    build_hamiltonian_matrix,
    ground_state,
    spectral_gap,
    mis_bitstrings,
)
from packages.simulation.observables import (
    rydberg_density,
    entanglement_entropy,
    antiferromagnetic_order,
    bitstring_probabilities,
    total_rydberg_fraction,
)
```

- Construire `H = build_hamiltonian_matrix(coords, omega, delta)` → `np.linalg.eigh(H)` pour le spectre complet
- `ground_state(coords, omega, delta)` pour les observables
- `bitstring_probabilities(psi, n_atoms, top_k=20)` pour le bar chart
- `mis_bitstrings(coords, omega)` pour les solutions MIS

**Pour le sweep paramétrique** :
- Boucle sur `delta_values = np.linspace(-20, 20, 50)` (ou configurable)
- Pour chaque δ, calculer les 8 premiers eigenvalues via `np.linalg.eigvalsh(H)[:8]`
- Limiter à ≤10 atomes pour l'interactivité (2^10 = 1024 dim max)
- Afficher un `st.warning` si atom_count > 10

**Cache** : Utiliser `@st.cache_data` sur la fonction de sweep.

---

## TÂCHE 5 — Page 4 : Pulse Sequence Studio

### Fichier : `apps/dashboard/pages/4_Pulse_Studio.py`

**But** : Visualiser les profils temporels des séquences, comparer les familles, explorer l'espace paramétrique.

### Layout précis

```
┌──────────────────────────────────────────────────────────────────┐
│ st.title("Pulse Sequence Studio")                                │
│ st.caption("Waveform profiles, time evolution, parameter space.")│
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Waveform Mathematics (st.expander)                     │
│  LaTeX: Blackman formula                                         │
│  LaTeX: linear sweep δ(t)                                        │
│  LaTeX: pulse area θ = ∫ Ω(t) dt                                │
│  LaTeX: time evolution operator                                  │
│  LaTeX: Trotter-Suzuki 2nd order                                 │
│  st.markdown: table of 5 sequence families with descriptions     │
├──────────────────────────────────────────────────────────────────┤
│ Campaign & sequence selectors | Family filter multiselect        │
├──────────────────────────────────────────────────────────────────┤
│ pulse_waveform() chart                                           │
│ - Dual axis: Ω(t) and δ(t)                                      │
│ - Shaded pulse area                                              │
├──────────────────────────────────────────────────────────────────┤
│ Left col (50%):                         Right col (50%):         │
│  Sequence comparison table               parameter_space_scatter │
│  st.dataframe:                           - amplitude vs detuning │
│    family, amplitude, detuning,          - colored by objective  │
│    duration_ns, predicted_cost,          - sized by duration     │
│    objective_score (if available)         - shaped by family     │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Waveform Generator (interactive)                       │
│  Family selector | Ω_max slider | δ_start slider | δ_end slider │
│  Duration slider | [Generate] button                             │
│  → compute and show waveform profile for arbitrary parameters    │
│  → no simulation, just the Ω(t) and δ(t) curves                │
└──────────────────────────────────────────────────────────────────┘
```

### Calcul des waveforms

Générer les courbes Ω(t) et δ(t) pour chaque famille :

```python
def generate_waveform(family: str, omega_max: float, delta_start: float, delta_end: float, duration_ns: float, n_points: int = 200) -> tuple[list[float], list[float], list[float]]:
    """Return (time_ns, omega_values, delta_values) for a given family."""
    t = np.linspace(0, duration_ns, n_points)
    T = duration_ns

    if family == "constant_drive":
        omega = np.full_like(t, omega_max)
        delta = np.full_like(t, delta_start)
    elif family == "global_ramp":
        omega = omega_max * t / T
        delta = delta_start + (delta_end - delta_start) * t / T
    elif family == "detuning_scan":
        omega = np.full_like(t, omega_max)
        delta = delta_start + (delta_end - delta_start) * t / T
    elif family == "adiabatic_sweep":
        omega = omega_max * np.sin(np.pi * t / T) ** 2
        delta = delta_start + (delta_end - delta_start) * t / T
    elif family == "blackman_sweep":
        omega = omega_max * (0.42 - 0.50 * np.cos(2 * np.pi * t / T) + 0.08 * np.cos(4 * np.pi * t / T))
        delta = delta_start + (delta_end - delta_start) * t / T
    else:
        omega = np.full_like(t, omega_max)
        delta = np.full_like(t, delta_start)

    return t.tolist(), omega.tolist(), delta.tolist()
```

Placer cette fonction dans `logic.py` (extension de l'existant).

---

## TÂCHE 6 — Page 5 : Robustness Arena

### Fichier : `apps/dashboard/pages/5_Robustness_Arena.py`

**But** : Analyse de sensibilité au bruit, comparaison de robustesse entre candidats, diagnostic de dégradation.

### Layout précis

```
┌──────────────────────────────────────────────────────────────────┐
│ st.title("Robustness Arena")                                     │
│ st.caption("Noise sensitivity analysis and robustness scoring.") │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Robustness Mathematics (st.expander)                   │
│  LaTeX: S_robust = w_n·s_nom + w_a·s̄_pert + w_w·s_worst + ...  │
│  LaTeX: weights (0.25, 0.35, 0.30, 0.10)                        │
│  LaTeX: stability bonus formula                                  │
│  LaTeX: penalty formula                                          │
│  LaTeX: noise model for amplitude, detuning, dephasing           │
│  LaTeX: density score and blockade score formulas                │
│  st.markdown: interpretation of each scoring component          │
│  st.markdown: table of noise scenario parameters:                │
│    LOW:      σ_amp=0.01, σ_det=0.01, γ_φ=0.001, spatial=2%     │
│    MEDIUM:   σ_amp=0.05, σ_det=0.05, γ_φ=0.005, spatial=5%     │
│    STRESSED: σ_amp=0.10, σ_det=0.10, γ_φ=0.010, spatial=8%     │
├──────────────────────────────────────────────────────────────────┤
│ Campaign selector                                                │
├──────────────────────────────────────────────────────────────────┤
│ robustness_grouped_bar() chart                                   │
│ - All candidates, nominal/average/worst bars                     │
├──────────────────────────────────────────────────────────────────┤
│ Left col (50%):                         Right col (50%):         │
│  noise_radar() chart                     robustness_violin()     │
│  - Overlaid polygons per candidate       - Score distributions   │
│  - Up to 4 candidates                    - With scenario points  │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Deep Dive (for selected candidate)                     │
│  Candidate selector (from ranked list)                           │
│  Left col (50%):                         Right col (50%):         │
│    score_degradation_waterfall()          Observable comparison  │
│    - Nominal → Low → Med → Stressed      st.dataframe:          │
│    - Score drop contribution              rydberg_density,       │
│                                           blockade_violation,    │
│                                           entanglement_entropy   │
│                                           per noise scenario     │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Robustness Summary Metrics                             │
│  4 metric cards for selected candidate:                          │
│    Nominal Score | Perturbation Avg | Worst Case | Score Std     │
│  + Robustness penalty value                                      │
│  + JSON expander for full report                                 │
└──────────────────────────────────────────────────────────────────┘
```

### Données sources

- `load_robustness_reports(campaign_id)` depuis `data_loaders`
- `load_ranked_candidates(campaign_id)` pour le mapping sequence_candidate_id
- Chaque `RobustnessReport` contient :
  - `nominal_score`, `perturbation_average`, `robustness_score`, `worst_case_score`, `score_std`, `robustness_penalty`
  - `scenario_scores`: `{"low_noise": 0.72, "medium_noise": 0.58, "stressed_noise": 0.41}`
  - `scenario_observables`: `{"low_noise": {"rydberg_density": ..., "blockade_violation": ..., ...}, ...}`
  - `nominal_observables`: `{"rydberg_density": ..., "entanglement_entropy": ..., ...}`
  - `hamiltonian_metrics`: `{"dimension": 64, "frobenius_norm": ..., "spectral_radius": ...}`

---

## TÂCHE 7 — Page 6 : ML Observatory

### Fichier : `apps/dashboard/pages/6_ML_Observatory.py`

**But** : Monitorer les modèles ML (surrogate et PPO), leur performance, et l'influence de la stratégie bandit sur les décisions.

### Layout précis

```
┌──────────────────────────────────────────────────────────────────┐
│ st.title("ML Observatory")                                       │
│ st.caption("Surrogate models, PPO training, strategy bandit.")   │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] ML Theory (st.expander)                                │
│  LaTeX: PPO clipped objective L^CLIP                             │
│  LaTeX: importance ratio r_t(θ)                                  │
│  LaTeX: GAE formula                                              │
│  LaTeX: ensemble prediction and epistemic uncertainty            │
│  LaTeX: UCB1 formula                                             │
│  LaTeX: weighted MSE loss                                        │
│  st.markdown: architecture descriptions:                         │
│    - SurrogateV2: Input → Linear(128) → GELU →                  │
│      [ResidualBlock(LayerNorm→Linear→GELU→Dropout→Linear)]×3    │
│      → LayerNorm → Linear(64) → GELU → Linear(4) → Sigmoid     │
│    - ActorCritic: obs(16) → Tanh(128) → Tanh(128) →            │
│      Actor: Tanh(Linear(4)) + learnable log_std                  │
│      Critic: Tanh(64) → Tanh(32) → Linear(1)                   │
├──────────────────────────────────────────────────────────────────┤
│ [TAB 1] Surrogate Model                                          │
│  File selector for checkpoint (.pt files in runs/)               │
│  If checkpoint found:                                            │
│    training_loss_curves() chart (if history available)            │
│    prediction_vs_actual() chart (if test data available)          │
│  Model architecture summary table:                               │
│    input_dim, output_dim, hidden, n_blocks, dropout, params count│
│  Normalizer status: means/stds range display                     │
├──────────────────────────────────────────────────────────────────┤
│ [TAB 2] PPO Training                                             │
│  File selector for PPO checkpoint (.pt in runs/)                 │
│  ppo_training_dashboard() 2x2 chart                              │
│  PPO config display: lr_actor, lr_critic, gamma, epsilon_clip,   │
│    entropy_coeff, rollout_steps, batch_size                      │
│  Action space summary: amplitude, detuning, duration, family     │
│  Observation space summary: 16-dim feature vector description    │
├──────────────────────────────────────────────────────────────────┤
│ [TAB 3] Strategy Bandit                                          │
│  Campaign selector → extract strategy decisions from             │
│  agent_decisions where decision_type == "candidate_generation"   │
│  and structured_output contains strategy_used                    │
│  strategy_ucb_evolution() chart                                  │
│  Strategy distribution pie chart (st.plotly_chart)               │
│  Per-strategy metrics table: n_trials, avg_reward, best_reward   │
├──────────────────────────────────────────────────────────────────┤
│ [TAB 4] RL vs Heuristic                                          │
│  If strategy data available:                                     │
│    Paired bar chart: RL candidates vs heuristic candidates       │
│    scored by robustness_score                                    │
│  Else: st.info("Run campaigns with adaptive strategy to compare")│
└──────────────────────────────────────────────────────────────────┘
```

### Sources de données ML

Les données ML proviennent de deux sources :

1. **Fichiers checkpoint** dans `runs/` :
   - `runs/surrogate_v2.pt` ou `runs/ensemble/member_0.pt` etc.
   - `runs/ppo_checkpoint.pt`
   - Ces fichiers contiennent `{"model": state_dict, "config": {...}, "version": "..."}`
   - Pour les history, chercher `runs/surrogate_history.json` et `runs/ppo_history.json` (à créer par le training runner)

2. **Agent decisions** dans MongoDB :
   - Filtrer `agent_decisions` par `decision_type == "candidate_generation"`
   - Le `structured_output` contient `strategy_used`, `strategy_reason`, `rl_candidates_count`, `heuristic_candidates_count`

**IMPORTANT** : Cette page doit fonctionner même si aucun checkpoint ML n'existe (mode dégradé avec `st.info`).

---

## TÂCHE 8 — Page 7 : Campaign Analytics

### Fichier : `apps/dashboard/pages/7_Campaign_Analytics.py`

**But** : Analyse cross-campaign, tendances temporelles, système de mémoire, exploration de l'espace paramétrique global.

### Layout précis

```
┌──────────────────────────────────────────────────────────────────┐
│ st.title("Campaign Analytics")                                   │
│ st.caption("Cross-campaign trends, memory system, Pareto front.")│
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Analytics Theory (st.expander)                         │
│  LaTeX: Pareto optimality definition                             │
│  LaTeX: cumulative regret R_T                                    │
│  LaTeX: acquisition function (UCB)                               │
│  LaTeX: ranking key sort order                                   │
├──────────────────────────────────────────────────────────────────┤
│ campaign_timeline() chart (all campaigns)                        │
│ - Horizontal bars colored by status                              │
├──────────────────────────────────────────────────────────────────┤
│ Left col (50%):                         Right col (50%):         │
│  cross_campaign_score_evolution()        backend_distribution_   │
│  - Best score per campaign over time     stacked() chart         │
│  - Trend line                            - Backend mix evolution │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Parameter Space Explorer                               │
│  parameter_space_3d() chart                                      │
│  - All candidates across campaigns                               │
│  - Pareto frontier highlighted                                   │
│  - Axis selectors: choose any 3 params for x/y/z                │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Memory System                                          │
│  memory_tag_cloud() chart                                        │
│  Memory records table with filters:                              │
│    - lesson_type filter (candidate_pattern / failure_pattern)    │
│    - campaign_id filter                                          │
│    - tag search                                                  │
│  Expandable rows: full signals dict, reasoning                   │
├──────────────────────────────────────────────────────────────────┤
│ [SECTION] Campaign Statistics Summary                            │
│  4 metric cards: Total Campaigns | Avg Candidates/Campaign |     │
│  Best Score Ever | Most Used Backend                             │
│  Success rate: completed / total campaigns                       │
└──────────────────────────────────────────────────────────────────┘
```

### Agrégation cross-campaign

Pour construire le scatter 3D et le Pareto front, charger les candidats de TOUTES les campagnes :

```python
campaigns = load_latest_campaigns(limit=50)
all_candidates = []
for campaign in campaigns:
    candidates = load_ranked_candidates(campaign["id"])
    for c in candidates:
        c["campaign_id_ref"] = campaign["id"]
        c["campaign_created_at"] = campaign["created_at"]
    all_candidates.extend(candidates)
```

Pour le Pareto front (2D simplifié : objective_score vs worst_case_score) :

```python
def compute_pareto_front(candidates: list[dict]) -> list[int]:
    """Return indices of Pareto-optimal candidates (maximize both objectives)."""
    pareto_indices = []
    for i, c in enumerate(candidates):
        dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if (other["objective_score"] >= c["objective_score"] and
                other["worst_case_score"] >= c["worst_case_score"] and
                (other["objective_score"] > c["objective_score"] or
                 other["worst_case_score"] > c["worst_case_score"])):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    return pareto_indices
```

Placer cette fonction dans `logic.py`.

---

## TÂCHE 9 — Page 8 : Theory Reference

### Fichier : `apps/dashboard/pages/8_Theory_Reference.py`

**But** : Référence mathématique complète pour les chercheurs PhD. Chaque section combine formules LaTeX, descriptions physiques, et références aux modules du code.

### Layout précis

```
┌──────────────────────────────────────────────────────────────────┐
│ st.title("Theory Reference")                                     │
│ st.caption("Complete mathematical reference for CryoSwarm-Q.")   │
│ st.markdown("*This page serves as a self-contained reference     │
│  for the physics, scoring, and ML behind CryoSwarm-Q.*")         │
├──────────────────────────────────────────────────────────────────┤
│ [8 expandable sections, one per topic]                           │
└──────────────────────────────────────────────────────────────────┘
```

### Section 1 : Rydberg Hamiltonian

Utiliser `hamiltonian_formulas()` depuis `latex_panels`. Afficher TOUTES les formules.

Ajouter un `st.markdown` d'interprétation :

```markdown
**Physical interpretation:**
- The **Rabi coupling** Ω/2 drives coherent transitions between ground |g⟩ and Rydberg |r⟩ states via σ_x.
- The **detuning** δ controls the energy offset of the Rydberg state relative to the drive frequency.
- The **van der Waals interaction** C₆/r⁶ between Rydberg-excited atoms creates the blockade: if two atoms are closer than R_b, simultaneous excitation is strongly suppressed.

**Implementation:** `packages/simulation/hamiltonian.py`
- Dense construction via Kronecker products for N ≤ 14 atoms
- Sparse CSC construction via scipy for larger systems
- C₆ coefficient calibrated for ⁸⁷Rb |70S₁/₂⟩
```

### Section 2 : Quantum Observables

Utiliser `observable_formulas()`. Afficher chaque formule avec interprétation.

```markdown
**Rydberg density** ⟨n_i⟩: probability that atom i is in the Rydberg state.
**Connected correlation** g_ij: measures genuine quantum correlations beyond mean-field.
  Negative values → anti-bunching (antiferromagnetic order).
  Positive values → bunching (ferromagnetic-like).
**Antiferromagnetic order** m_AF: staggered magnetization. 1 = perfect Néel order, 0 = disordered.
**Entanglement entropy** S_A: von Neumann entropy of the reduced density matrix.
  Computed via Schmidt decomposition (SVD) of the bipartite state.

**Implementation:** `packages/simulation/observables.py`
**Bitstring convention:** MSB — atom 0 is the most significant bit. Consistent with Pulser.
```

### Section 3 : Maximum Independent Set

Utiliser `mis_formulas()`.

```markdown
**Physical connection:** The ground state of the Rydberg Hamiltonian in the blockade regime
(large δ > 0, strong interactions) approximates solutions to the Maximum Independent Set problem
on the blockade graph. This is the basis for quantum optimization on neutral-atom hardware.

**Algorithm:**
- Exact enumeration for N ≤ 15 atoms (brute-force over all subsets)
- Greedy heuristic with randomised restarts for 15 < N ≤ 50

**Implementation:** `packages/simulation/hamiltonian.py` → `find_maximum_independent_sets()`
```

### Section 4 : Pulse Sequence Families

Utiliser `pulse_formulas()`. Afficher chaque formule de waveform.

Ajouter un tableau markdown des 5 familles :

```markdown
| Family | Ω(t) Profile | δ(t) Profile | Use Case |
|--------|-------------|-------------|----------|
| `constant_drive` | Constant Ω_max | Constant δ₀ | Simple Rabi oscillation |
| `global_ramp` | Linear ramp 0→Ω_max | Linear sweep δ_start→δ_end | Gradual excitation |
| `detuning_scan` | Constant Ω_max | Linear sweep δ_start→δ_end | Resonance scanning |
| `adiabatic_sweep` | sin²(πt/T) envelope | Linear sweep | Adiabatic state preparation |
| `blackman_sweep` | Blackman window | Linear sweep | Low-sidelobe adiabatic prep |
```

### Section 5 : Robustness Scoring

Utiliser `robustness_formulas()`. Afficher TOUTES les formules.

Ajouter :

```markdown
**Scoring pipeline:**
1. Simulate nominal (noiseless) → s_nom
2. Simulate under LOW, MEDIUM, STRESSED noise → perturbed scores
3. Compute: perturbation_average, worst_case, score_std
4. Aggregate: S_robust = weighted combination
5. Compute penalty for sharp degradation

**Noise model parameters:**
| Scenario | σ_amp | σ_det | γ_φ | γ_loss | T (µK) | SPAM | Spatial |
|----------|-------|-------|-----|--------|---------|------|---------|
| LOW | 0.01 | 0.01 | 0.001 | 0.001 | 50 | 0.5% | 2% |
| MEDIUM | 0.05 | 0.05 | 0.005 | 0.005 | 50 | 0.5% | 5% |
| STRESSED | 0.10 | 0.10 | 0.010 | 0.010 | 50 | 0.5% | 8% |

**Implementation:** `packages/scoring/robustness.py`, `packages/simulation/noise_profiles.py`
```

### Section 6 : Objective Score & Ranking

```markdown
**Objective score** combines observable alignment, robustness, execution cost, and latency.

**Ranking** is lexicographic: sort by (objective_score ↓, worst_case ↓, robustness ↓, nominal ↓).

**Implementation:** `packages/scoring/objective.py`, `packages/scoring/ranking.py`
```

### Section 7 : PPO & Reinforcement Learning

Utiliser `ml_formulas()` : `ppo_objective`, `importance_ratio`, `gae`.

```markdown
**Environment:** `PulseDesignEnv` (Gymnasium-compatible)
- Observation: 16-dim vector (atom count, spacing, blockade radius, feasibility,
  target density, best robustness so far, best params, step progress)
- Action: 4-dim continuous [-1,1] → rescaled to (amplitude, detuning, duration, family)
- Reward: shaped reward = (1-w)·raw_robustness + w·improvement·scale + terminal bonus

**Training:** Standard PPO with:
- Separate actor/critic learning rates
- GAE advantage estimation (γ=0.99, λ=0.95)
- Clipped surrogate objective (ε=0.2)
- Entropy regularization (c_ent=0.01)
- Gradient clipping (max_norm=0.5)

**Implementation:** `packages/ml/ppo.py`, `packages/ml/rl_env.py`
```

### Section 8 : Surrogate Ensemble & Uncertainty

Utiliser `ml_formulas()` : `ensemble_mean`, `epistemic_uncertainty`, `weighted_mse`.

```markdown
**Architecture:** SurrogateModelV2
- Input: 18-dim physics-informed features (normalized)
- 3 residual blocks with LayerNorm + GELU + Dropout
- Output: 4 targets → Sigmoid [0,1]
  (robustness_score, nominal_score, worst_case_score, observable_score)
- Target weights: [2.0, 1.0, 1.5, 1.0] (prioritize robustness prediction)

**Ensemble:** M=3 independently trained models
- Epistemic uncertainty = inter-model variance
- Used for active learning: high uncertainty → prioritize for re-simulation

**Strategy selection:** UCB1 bandit chooses between heuristic, RL, hybrid modes
based on observed robustness scores per problem class.

**Implementation:** `packages/ml/surrogate.py`, `packages/agents/sequence_strategy.py`
```

---

## TÂCHE 10 — Refactorer `app.py` comme entry point

### Fichier : `apps/dashboard/app.py`

Réécrire `app.py` comme point d'entrée minimal. Il doit :

1. Appeler `st.set_page_config(page_title="CryoSwarm-Q Dashboard", layout="wide", page_icon="⚛")`
2. Initialiser la connexion MongoDB via `data_loaders.get_repository()`
3. Afficher la sidebar avec :
   - Logo / titre du projet
   - Statut de connexion MongoDB (vert/rouge)
   - Compteur de campagnes stockées
   - Lien vers la page Theory Reference
4. Afficher la page d'accueil avec :
   - Description du projet (2-3 lignes)
   - Quick stats : nombre de campagnes, meilleur score, dernier run
   - Navigation cards vers les 8 pages (emojis autorisés dans les titres de page Streamlit)
5. **Conserver la rétro-compatibilité** : l'ancien code single-page doit rester fonctionnel. Si `pages/` n'existe pas, fallback sur le comportement existant.

**IMPORTANT** : Streamlit détecte automatiquement les fichiers dans `pages/` pour la navigation sidebar. Le nommage `1_Campaign_Control.py`, `2_Register_Physics.py`, etc. contrôle l'ordre.

---

## TÂCHE 11 — Tests

### 11.1 `tests/test_dashboard_components.py`

Tester toutes les fonctions de `plotly_charts.py` et `latex_panels.py` SANS Streamlit runtime :

```python
"""Tests for dashboard Plotly charts and LaTeX panels.

These tests verify that chart functions return valid Plotly figures
and that LaTeX panels produce non-empty strings, without requiring
a running Streamlit server.
"""
```

**Tests requis** (au minimum) :

```python
# latex_panels
def test_hamiltonian_formulas_all_keys():
    """All expected formula keys are present and non-empty."""

def test_observable_formulas_all_keys(): ...
def test_robustness_formulas_all_keys(): ...
def test_ml_formulas_all_keys(): ...
def test_pulse_formulas_all_keys(): ...
def test_campaign_formulas_all_keys(): ...

# plotly_charts — chaque fonction doit retourner un go.Figure valide
def test_register_scatter_2d_returns_figure():
    coords = [(0, 0), (7, 0), (3.5, 6)]
    fig = register_scatter_2d(coords, blockade_radius=8.0, label="test")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

def test_vdw_interaction_heatmap_returns_figure():
    coords = [(0, 0), (7, 0), (0, 7)]
    fig = vdw_interaction_heatmap(coords, c6=862690.0)
    assert isinstance(fig, go.Figure)

def test_energy_spectrum_returns_figure():
    eigenvalues = [-5.2, -3.1, -1.0, 0.5, 2.3, 4.1]
    fig = energy_spectrum(eigenvalues)
    assert isinstance(fig, go.Figure)

def test_robustness_grouped_bar_returns_figure():
    reports = [
        {"sequence_candidate_id": "seq_abc", "nominal_score": 0.8, "perturbation_average": 0.6, "worst_case_score": 0.4, "robustness_score": 0.65},
        {"sequence_candidate_id": "seq_def", "nominal_score": 0.7, "perturbation_average": 0.55, "worst_case_score": 0.35, "robustness_score": 0.55},
    ]
    fig = robustness_grouped_bar(reports)
    assert isinstance(fig, go.Figure)

def test_pulse_waveform_returns_figure():
    fig = pulse_waveform([0, 500, 1000], [5.0, 5.0, 5.0], [-10.0, 0.0, 10.0], "test-seq")
    assert isinstance(fig, go.Figure)

def test_noise_radar_returns_figure(): ...
def test_score_degradation_waterfall_returns_figure(): ...
def test_bitstring_bar_chart_returns_figure(): ...
def test_parameter_space_scatter_returns_figure(): ...
def test_candidate_funnel_returns_figure(): ...
def test_campaign_timeline_returns_figure(): ...
def test_memory_tag_cloud_returns_figure(): ...

# Edge cases
def test_robustness_grouped_bar_empty_list():
    fig = robustness_grouped_bar([])
    assert isinstance(fig, go.Figure)

def test_energy_spectrum_single_value():
    fig = energy_spectrum([-1.0])
    assert isinstance(fig, go.Figure)
```

### 11.2 `tests/test_data_loaders.py`

Tester les fonctions `data_loaders.py` avec des mocks MongoDB :

```python
"""Tests for dashboard data loaders with mocked MongoDB."""

# Utiliser monkeypatch pour mocker get_repository()
# Vérifier que les fonctions retournent des listes de dicts
# Vérifier que le cache fonctionne (appeler 2 fois, vérifier 1 seul appel DB)
```

### 11.3 Mettre à jour `tests/test_dashboard_logic.py`

Ajouter des tests pour les nouvelles fonctions ajoutées à `logic.py` :

```python
def test_generate_waveform_constant():
    t, omega, delta = generate_waveform("constant_drive", 5.0, -10.0, 10.0, 1000.0)
    assert len(t) == 200
    assert all(o == 5.0 for o in omega)

def test_generate_waveform_blackman():
    t, omega, delta = generate_waveform("blackman_sweep", 5.0, -10.0, 10.0, 1000.0)
    assert len(t) == 200
    assert omega[0] < 0.01  # Blackman starts near zero
    assert max(omega) <= 5.0

def test_compute_pareto_front():
    candidates = [
        {"objective_score": 0.8, "worst_case_score": 0.3},
        {"objective_score": 0.5, "worst_case_score": 0.7},
        {"objective_score": 0.6, "worst_case_score": 0.5},
    ]
    pareto = compute_pareto_front(candidates)
    assert 0 in pareto  # (0.8, 0.3) is Pareto optimal
    assert 1 in pareto  # (0.5, 0.7) is Pareto optimal
    assert 2 not in pareto  # (0.6, 0.5) is dominated by neither → actually check logic
```

---

## TÂCHE 12 — Mise à jour de `pyproject.toml`

Ajouter les dépendances dans `[project.dependencies]` :

```toml
"plotly>=5.22",
"networkx>=3.2",
```

Mettre à jour `[tool.coverage.run]` pour NE PAS omettre `apps/dashboard/components/` (supprimer `apps/dashboard/*` du omit et remplacer par `apps/dashboard/app.py` et `apps/dashboard/pages/*` uniquement — les components doivent être testés).

---

## Résumé des fichiers à créer / modifier

### Fichiers à CRÉER :

| Fichier | Lignes estimées | Description |
|---------|----------------|-------------|
| `apps/dashboard/components/__init__.py` | 1 | Package init |
| `apps/dashboard/components/data_loaders.py` | ~120 | Cached MongoDB loaders |
| `apps/dashboard/components/latex_panels.py` | ~150 | LaTeX formula collections |
| `apps/dashboard/components/plotly_charts.py` | ~800 | All Plotly chart functions |
| `apps/dashboard/pages/1_Campaign_Control.py` | ~180 | Campaign launch & monitor |
| `apps/dashboard/pages/2_Register_Physics.py` | ~200 | Atom register analysis |
| `apps/dashboard/pages/3_Hamiltonian_Lab.py` | ~220 | Spectral analysis |
| `apps/dashboard/pages/4_Pulse_Studio.py` | ~180 | Waveform visualization |
| `apps/dashboard/pages/5_Robustness_Arena.py` | ~220 | Noise analysis |
| `apps/dashboard/pages/6_ML_Observatory.py` | ~250 | ML monitoring |
| `apps/dashboard/pages/7_Campaign_Analytics.py` | ~200 | Cross-campaign analytics |
| `apps/dashboard/pages/8_Theory_Reference.py` | ~300 | Full math reference |
| `tests/test_dashboard_components.py` | ~150 | Plotly & LaTeX tests |
| `tests/test_data_loaders.py` | ~80 | Loader tests |

### Fichiers à MODIFIER :

| Fichier | Modification |
|---------|-------------|
| `apps/dashboard/app.py` | Refactorer en entry point multi-page |
| `apps/dashboard/logic.py` | Ajouter `generate_waveform()`, `compute_pareto_front()` |
| `tests/test_dashboard_logic.py` | Ajouter tests pour nouvelles fonctions |
| `pyproject.toml` | Ajouter `plotly`, `networkx` aux deps |

### Fichiers à NE PAS toucher :

- `packages/` — aucune modification du backend
- `apps/api/` — aucune modification de l'API
- Tests existants — ne pas casser

---

## Ordre d'exécution

1. **TÂCHE 12** — `pyproject.toml` deps (pour que les imports fonctionnent)
2. **TÂCHE 1.1-1.3** — Components `__init__`, `data_loaders`, `latex_panels`
3. **TÂCHE 1.4** — `plotly_charts.py` (le plus gros fichier)
4. **TÂCHE 10** — Refactorer `app.py`
5. **TÂCHE 2** — Page 1 (Campaign Control)
6. **TÂCHE 3** — Page 2 (Register Physics)
7. **TÂCHE 4** — Page 3 (Hamiltonian Lab)
8. **TÂCHE 5** — Page 4 (Pulse Studio) + extension `logic.py`
9. **TÂCHE 6** — Page 5 (Robustness Arena)
10. **TÂCHE 7** — Page 6 (ML Observatory)
11. **TÂCHE 8** — Page 7 (Campaign Analytics) + extension `logic.py`
12. **TÂCHE 9** — Page 8 (Theory Reference)
13. **TÂCHE 11** — Tests
14. `pytest tests/ -x --tb=short` — validation finale

---

## Validation finale

Le dashboard est considéré comme terminé quand :

1. `streamlit run apps/dashboard/app.py` ouvre la navigation multi-page
2. Chaque page charge indépendamment sans erreur (même sans données)
3. Les graphes Plotly sont interactifs (hover, zoom, pan)
4. Les formules LaTeX s'affichent correctement
5. Les pages dégradent gracieusement si MongoDB n'est pas connecté
6. `pytest tests/test_dashboard_components.py tests/test_data_loaders.py tests/test_dashboard_logic.py -x --tb=short` passe
7. Le pipeline reste fonctionnel : `pytest tests/ -x --tb=short` passe
