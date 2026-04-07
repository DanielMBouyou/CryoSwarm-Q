# CryoSwarm-Q

**Hardware-aware multi-agent orchestration for autonomous neutral-atom experiment design.**

CryoSwarm-Q is a research-grade software prototype that sits between human scientific intent and pulse-level neutral-atom programming. Given a structured experimental objective, it autonomously generates candidate atom registers and pulse sequences, evaluates them under realistic noise, ranks them by robustness and feasibility, and remembers lessons for future campaigns.

The system targets Pasqal-style neutral-atom quantum processors using optically trapped $^{87}\text{Rb}$ atoms driven into Rydberg states. All candidate evaluation is grounded in the driven Rydberg Hamiltonian, not in abstract or toy models.

---

## Table of Contents

- [Why CryoSwarm-Q](#why-cryoswarm-q)
- [Mathematical Foundations](#mathematical-foundations)
- [Multi-Agent Architecture](#multi-agent-architecture)
- [ML Pipeline](#ml-pipeline)
- [Interactive Dashboard](#interactive-dashboard-streamlit)
- [REST API](#rest-api)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Programmatic Usage](#programmatic-usage)
- [Test Suite](#test-suite)
- [Repository Structure](#repository-structure)
- [Current Limitations](#current-limitations)
- [License](#license)

---

## Why CryoSwarm-Q

Neutral-atom quantum computing platforms offer programmable atom placement and global/local pulse control, but designing good experiments is hard. The parameter space is large (atom positions, pulse shapes, timing, detuning ramps), noise is unavoidable (laser intensity fluctuations, dephasing, SPAM errors, position disorder), and hardware constraints are strict (blockade radius, Rabi frequency bounds, device geometry).

CryoSwarm-Q addresses this by automating the full experiment-design loop:

1. **Problem framing** — translate a scientific objective into a concrete experiment specification.
2. **Geometry generation** — propose atom registers that respect hardware constraints (minimum spacing, blockade conditions, device bounds).
3. **Pulse sequence design** — generate candidate pulse sequences across 5 waveform families, using heuristics, reinforcement learning, or hybrid strategies.
4. **Noise-aware evaluation** — simulate each candidate under nominal and perturbed conditions (amplitude noise, detuning drift, dephasing, atom loss, SPAM, spatial inhomogeneity).
5. **Robustness ranking** — score and rank candidates by a composite objective that balances observable alignment, robustness, execution cost, and latency.
6. **Memory** — store lessons learned so future campaigns start from prior knowledge.

The purpose is not to replace the physicist, but to systematically explore the design space, surface non-obvious trade-offs, and provide transparent, reproducible reasoning for every decision.

---

## Mathematical Foundations

### The Rydberg Hamiltonian

All physics in CryoSwarm-Q is grounded in the driven Rydberg Hamiltonian for $N$ atoms:

$$
\hat{H} = \frac{\Omega}{2}\sum_{i=1}^{N} \hat{\sigma}_x^{(i)} - \delta\sum_{i=1}^{N} \hat{n}_i + \sum_{i<j} \frac{C_6}{|\mathbf{r}_i - \mathbf{r}_j|^6}\,\hat{n}_i\hat{n}_j
$$

where:

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| $\Omega$ | Rabi frequency (drive amplitude) | 1 – 15 rad/μs |
| $\delta$ | Detuning (energy offset of Rydberg state) | −40 to +25 rad/μs |
| $\hat{n}_i = \|r_i\rangle\langle r_i\|$ | Rydberg number operator for atom $i$ | — |
| $C_6$ | Van der Waals coefficient | $2\pi \times 862\,690$ rad·μm⁶/μs |
| $\mathbf{r}_i$ | Position of atom $i$ in the tweezer array | Spacing: 4 – 15 μm |

The atom species is $^{87}\text{Rb}$ in the $|70S_{1/2}\rangle$ Rydberg state.

### Blockade Radius

When two atoms are closer than the blockade radius $R_b$, simultaneous Rydberg excitation is energetically suppressed:

$$
R_b = \left(\frac{C_6}{\Omega}\right)^{1/6}
$$

At $\Omega = 5$ rad/μs this gives $R_b \approx 9.8$ μm. The blockade underpins the connection between Rydberg physics and combinatorial optimization (MIS).

### Maximum Independent Set (MIS)

The ground state of the Rydberg Hamiltonian in the blockade regime (large positive $\delta$, strong interactions) approximates solutions to the Maximum Independent Set problem on the blockade graph:

$$
\text{MIS}(G) = \arg\max_{S \subseteq V}\; |S| \quad \text{s.t.} \quad \forall\,(u,v)\in E,\; u \notin S \;\text{or}\; v \notin S
$$

CryoSwarm-Q computes exact MIS for $N \leq 15$ atoms (brute-force enumeration) and uses greedy heuristics with randomized restarts for larger systems.

### Quantum Observables

The simulation layer computes these observables from the ground-state wavefunction $|\psi\rangle$:

| Observable | Formula | Physical Meaning |
|-----------|---------|------------------|
| Rydberg density | $\langle \hat{n}_i \rangle$ | Excitation probability of atom $i$ |
| Total Rydberg fraction | $\bar{n} = \frac{1}{N}\sum_i \langle \hat{n}_i \rangle$ | Average excitation across the register |
| Connected correlation | $g_{ij}^{(2)} = \langle \hat{n}_i \hat{n}_j \rangle - \langle \hat{n}_i \rangle \langle \hat{n}_j \rangle$ | Genuine quantum correlations beyond mean-field |
| Antiferromagnetic order | $m_{\text{AF}} = \frac{1}{N}\left\|\sum_i (-1)^i (2\langle \hat{n}_i \rangle - 1)\right\|$ | Staggered magnetization (1 = perfect Néel order) |
| Entanglement entropy | $S_A = -\text{Tr}(\rho_A \log_2 \rho_A)$ | Bipartite entanglement via Schmidt decomposition |
| Bitstring probability | $P(b) = \|\langle b \| \psi \rangle\|^2$ | Probability of measuring each computational basis state |

### Pulse Sequence Families

CryoSwarm-Q generates pulse schedules from 5 waveform families:

| Family | $\Omega(t)$ | $\delta(t)$ | Use Case |
|--------|------------|------------|----------|
| `constant_drive` | Constant $\Omega_{\max}$ | Constant $\delta_0$ | Simple Rabi oscillation |
| `global_ramp` | Linear ramp $0 \to \Omega_{\max}$ | Linear sweep | Gradual excitation |
| `detuning_scan` | Constant $\Omega_{\max}$ | Linear sweep | Resonance scanning |
| `adiabatic_sweep` | $\sin^2(\pi t/T)$ envelope | Linear sweep | Adiabatic state preparation |
| `blackman_sweep` | Blackman window | Linear sweep | Low-sidelobe adiabatic prep |

Time evolution follows $|\psi(t)\rangle = \mathcal{T}\exp\left(-i\int_0^t \hat{H}(t')\,dt'\right)|\psi(0)\rangle$, discretized via second-order Trotter-Suzuki decomposition.

### Scoring and Ranking

Each candidate receives a composite objective score:

$$
S_{\text{obj}} = \alpha\, s_{\text{obs}} + \beta\, S_{\text{robust}} - \gamma\, c_{\text{exec}} - \delta_w\, \ell_{\text{latency}}
$$

with default weights $\alpha = 0.45$, $\beta = 0.35$, $\gamma = 0.10$, $\delta_w = 0.10$.

Robustness scoring aggregates nominal and perturbed simulations:

$$
S_{\text{robust}} = w_n \, s_{\text{nom}} + w_a \, \bar{s}_{\text{pert}} + w_w \, s_{\text{worst}} + w_s \, b_{\text{stab}}
$$

with weights $w_n = 0.25$, $w_a = 0.35$, $w_w = 0.30$, $w_s = 0.10$.

The stability bonus rewards low variance across noise scenarios, while a penalty flags candidates with sharp degradation from nominal to worst-case.

### Noise Model

Three perturbation scenarios are applied to every candidate:

| Scenario | $\sigma_{\text{amp}}$ | $\sigma_{\text{det}}$ | $\gamma_\phi$ | $\gamma_{\text{loss}}$ | $T$ (μK) | SPAM | Spatial |
|----------|----------------------|----------------------|---------------|----------------------|----------|------|---------|
| LOW | 0.01 | 0.01 | 0.001 | 0.001 | 50 | 0.5% | 2% |
| MEDIUM | 0.05 | 0.05 | 0.005 | 0.005 | 50 | 0.5% | 5% |
| STRESSED | 0.10 | 0.10 | 0.010 | 0.010 | 50 | 0.5% | 8% |

Noise channels include amplitude fluctuation ($\Omega \to \Omega(1+\epsilon_\Omega)$), detuning drift, Lindblad dephasing, atom loss, SPAM errors, and spatial drive inhomogeneity.

---

## Multi-Agent Architecture

CryoSwarm-Q uses specialized agents with explicit responsibilities, orchestrated through a deterministic pipeline:

```
ExperimentGoal
      │
      ▼
┌─────────────────┐
│ ProblemFraming   │  Converts goal → ExperimentSpec
│     Agent        │  (atom count, geometry, target observable)
└────────┬────────┘
         ▼
┌─────────────────┐
│  GeometryAgent   │  Proposes hardware-feasible registers
│                  │  (spacing, blockade, device bounds)
└────────┬────────┘
         ▼
┌─────────────────┐
│ SequenceStrategy │  Generates pulse candidates
│ (heuristic/RL/   │  (5 families × parameter variations)
│  hybrid/bandit)  │
└────────┬────────┘
         ▼
┌─────────────────┐
│ SurrogateFilter  │  Pre-filters candidates (optional)
│                  │  (ensemble prediction + uncertainty)
└────────┬────────┘
         ▼
┌─────────────────┐
│ NoiseRobustness  │  Evaluates nominal + 3 noise scenarios
│     Agent        │  (parallel evaluation supported)
└────────┬────────┘
         ▼
┌─────────────────┐
│ CampaignAgent    │  Ranks candidates by composite score
│                  │  (lexicographic: obj > worst > robust > nom)
└────────┬────────┘
         ▼
┌─────────────────┐
│  MemoryAgent     │  Extracts and stores reusable lessons
│                  │  (tagged by problem class for retrieval)
└────────┬────────┘
         ▼
    PipelineSummary
```

Each agent produces an `AgentDecision` with structured output, reasoning, and timestamps — fully traceable.

### Sequence Strategy Modes

The `SequenceStrategy` selects how pulse candidates are generated:

| Mode | Description |
|------|-------------|
| `heuristic_only` | Parameter sweeps across 5 waveform families |
| `rl_only` | PPO policy proposes candidates directly |
| `hybrid` | Evaluates heuristic and RL candidates together |
| `adaptive` | Switches strategy per problem class based on history |
| `bandit` | UCB1 multi-armed bandit selects the best-performing strategy |

---

## ML Pipeline

### Surrogate Ensemble

Three independently trained `SurrogateModelV2` networks predict robustness scores without full simulation:

```
Input(18) → Linear(128) → GELU →
  [ResidualBlock(LayerNorm → Linear → GELU → Dropout → Linear)] × 3
→ LayerNorm → Linear(64) → GELU → Linear(4) → Sigmoid
```

The 18-dim input is a physics-informed feature vector (atom count, spacing ratios, blockade metrics, pulse parameters). The 4 outputs are `(robustness_score, nominal_score, worst_case_score, observable_score)`.

Epistemic uncertainty is estimated as inter-model variance:

$$
\sigma_{\text{epist}}^2 = \frac{1}{M}\sum_{m=1}^{M} (f_m(\mathbf{x}) - \hat{y}_{\text{ens}})^2
$$

High uncertainty candidates are prioritized for re-simulation (active learning).

### PPO Reinforcement Learning

An `ActorCritic` network learns to propose pulse parameters directly:

- **Observation**: 16-dim vector (atom count, spacing, blockade radius, feasibility, target density, best robustness so far, best parameters, step progress)
- **Action**: 4-dim continuous $[-1, 1]$ rescaled to (amplitude, detuning, duration, family)
- **Objective**: Clipped PPO with GAE advantage estimation

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1 \pm \epsilon)\hat{A}_t\right)\right]
$$

### Active Learning Loop

The system iterates between surrogate training and RL training:

1. Train / fine-tune the surrogate ensemble on simulation data
2. Train PPO using surrogate predictions as fast reward proxy
3. Collect strongest RL configurations
4. Select diverse high-uncertainty points
5. Re-simulate with the real evaluator
6. Grow the dataset and repeat

### Strategy Bandit

A UCB1 multi-armed bandit tracks per-strategy performance across campaigns:

$$
\text{UCB1}(s) = \bar{r}_s + \sqrt{\frac{2\ln N}{n_s}}
$$

This automatically balances exploration of new strategies versus exploitation of known-good ones.

---

## Interactive Dashboard (Streamlit)

The dashboard provides 8 specialized pages for inspecting every stage of the pipeline. Launch it with:

```bash
streamlit run apps/dashboard/app.py
```

### Page 1 — Campaign Control Center

The main operational page. Launch new campaigns by specifying a title, scientific objective, atom count (2–50), preferred geometry, target observable, and priority. During execution, a live event bus streams agent decisions in real time.

After execution:
- **Recent Campaigns** table showing up to 20 past campaigns
- **Pipeline Inspector** with agent phase status indicators and a Gantt timeline
- **Candidate Funnel** visualization (Registers → Sequences → Evaluated → Ranked)
- **Agent Decision Log** with expandable structured outputs for every agent call

### Page 2 — Register Physics Lab

Explore the physics of atom register geometries:
- **2D scatter plot** of atom positions with blockade radius circles (colored by Rydberg density when available)
- **Van der Waals interaction heatmap** showing $C_6 / r_{ij}^6$ pair interactions
- **Blockade graph** visualization with Maximum Independent Sets highlighted
- **Distance histogram** with the blockade radius threshold marked
- **Interactive $\Omega$ slider** that dynamically recalculates $R_b$
- Metrics: atom count, minimum distance, blockade pairs, feasibility score

### Page 3 — Hamiltonian Spectroscopy

Real-time Hamiltonian diagonalization and spectral analysis (up to 14 atoms):
- **Energy spectrum** of the first 20 levels
- **Bitstring probability distribution** with MIS bitstrings highlighted in gold
- **Spectral gap** ($\Delta E = E_1 - E_0$), Hilbert dimension ($2^N$), Inverse Participation Ratio
- State composition: Rydberg fraction, entanglement entropy, antiferromagnetic order parameter
- **Parametric detuning sweep** revealing avoided crossings and phase transitions (for $N \leq 10$)
- Auto-load controls from the best sequence candidate

### Page 4 — Pulse Sequence Studio

Inspect and design pulse waveforms:
- **Waveform visualization** ($\Omega(t)$ and $\delta(t)$ over time) for each candidate sequence
- **Waveform Mathematics** panel with Blackman, linear sweep, Trotter-Suzuki formulas
- **Sequence comparison table** across families, amplitudes, detunings, durations, and predicted costs
- **Parameter space scatter** (sequences colored by objective score)
- **Waveform Generator** — design arbitrary waveforms with free controls (family, $\Omega_{\max}$, $\delta_{\text{start}}$, $\delta_{\text{end}}$, duration)

### Page 5 — Robustness Arena

Noise sensitivity analysis and scoring comparison:
- **Robustness grouped bar chart** showing nominal, perturbation average, and worst-case scores across candidates
- **Noise radar chart** for multi-candidate comparison (up to 4 candidates)
- **Robustness violin plot** for score distribution visualization
- **Deep dive** per candidate: score degradation waterfall chart + observable comparison table (nominal vs LOW/MEDIUM/STRESSED)
- Summary metrics: nominal score, perturbation average, worst case, score std, robustness penalty
- Full robustness report export (JSON)

### Page 6 — ML Observatory

Monitor the machine learning subsystem across 4 tabs:
- **Surrogate Model** — training loss curves, architecture summary (18→128→GELU→3 residual blocks→4), checkpoint status
- **PPO Training** — reward curves, policy loss, value loss, entropy, PPO hyperparameters, action/observation space descriptions
- **Strategy Bandit** — UCB1 score evolution, strategy distribution pie chart, per-strategy reward statistics
- **RL vs Heuristic** — grouped bar chart comparing RL versus heuristic candidate counts per trial

### Page 7 — Campaign Analytics

Cross-campaign trend analysis:
- **Campaign Timeline** (up to 50 campaigns)
- **Score evolution** line chart across campaigns
- **Backend distribution** stacked bar chart
- **3D Parameter Space Explorer** with Pareto front highlighted
- **Memory System** — tag cloud, filterable memory records (by lesson type, campaign, tag), expandable JSON detail
- Campaign statistics: total campaigns, average candidates per campaign, best score ever, most-used backend, success rate

### Page 8 — Theory Reference

Self-contained mathematical reference with 8 expandable sections:
1. **Rydberg Hamiltonian** — full Hamiltonian, spectral gap, IPR, adiabatic condition
2. **Quantum Observables** — Rydberg density, correlations, AF order, entanglement entropy, fidelity
3. **Maximum Independent Set** — MIS definition, overlap measure, cost function
4. **Pulse Sequence Families** — 5 families with profile table
5. **Robustness Scoring** — score formula, weights, stability bonus, penalty, noise parameters
6. **Objective Score & Ranking** — composite scoring, lexicographic ranking key
7. **PPO & Reinforcement Learning** — PPO objective, importance ratio, GAE, environment details
8. **Surrogate Ensemble & Uncertainty** — architecture, epistemic uncertainty, UCB1 strategy

Every section links to the corresponding implementation file.

---

## REST API

CryoSwarm-Q exposes a FastAPI backend for programmatic access:

```bash
uvicorn apps.api.main:app --reload
```

Base URL: `/api/v1/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check and MongoDB connectivity status |
| `/api/v1/campaigns/` | POST | Launch a new campaign from an experiment goal |
| `/api/v1/campaigns/` | GET | List campaigns (with pagination) |
| `/api/v1/campaigns/{id}` | GET | Get full campaign detail |
| `/api/v1/candidates/{campaign_id}` | GET | List ranked candidates for a campaign |

---

## Installation

### Requirements

- Python 3.10+
- MongoDB (optional — the system falls back to in-memory storage)

### Editable install

```bash
pip install -e ".[dev]"
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Optional integrations

| Package | Purpose |
|---------|---------|
| `pulser`, `pulser-simulation` | Pasqal-native pulse sequence construction |
| `pasqal-cloud` | Cloud execution adapter |
| `qoolqit` | Quantum optimization toolkit |
| `torch` | ML modules (surrogate, PPO, active learning) |

---

## Quick Start

### Run the test suite

```bash
pytest tests -q
```

### Launch the dashboard

```bash
streamlit run apps/dashboard/app.py
```

### Launch the API

```bash
uvicorn apps.api.main:app --reload
```

### Run a demo campaign (script)

```bash
python -m scripts.run_demo_pipeline
```

### ML training commands

Generate a training dataset:
```bash
python -m scripts.train_ml --phase generate_v2 --n-samples 1000 --workers 2 --sampling lhs
```

Train the surrogate ensemble:
```bash
python -m scripts.train_ml --phase surrogate --data data/generated/dataset.npz --epochs 100
```

Train PPO:
```bash
python -m scripts.train_ml --phase rl --updates 500
```

Full training (surrogate + PPO):
```bash
python -m scripts.train_ml --phase full --data data/generated/dataset.npz --epochs 100 --updates 500
```

Active learning loop:
```bash
python -m scripts.train_ml --phase active --data data/generated/dataset.npz --al-iterations 5 --al-top-k 200
```

Benchmarks and ablations:
```bash
python -m scripts.benchmark --full --checkpoint-dir checkpoints --test-data data/generated/dataset.npz
python -m scripts.ablation --ablation all --data data/generated/dataset.npz
```

---

## Programmatic Usage

```python
from packages.core.models import ExperimentGoal
from packages.orchestration.pipeline import CryoSwarmPipeline

goal = ExperimentGoal(
    title="Robust neutral-atom sweep",
    scientific_objective="Search robust Rydberg-density protocols.",
    target_observable="rydberg_density",
    desired_atom_count=6,
    preferred_geometry="mixed",
)

pipeline = CryoSwarmPipeline(
    sequence_strategy_mode="adaptive",
    rl_checkpoint_path="checkpoints/ppo_latest.pt",
)

summary = pipeline.run(goal)
print(summary.status, summary.top_candidate_id)
```

The pipeline returns a `PipelineSummary` containing the campaign state, all ranked candidates, agent decisions, robustness reports, and memory records.

---

## Test Suite

The repository maintains **372 tests** covering:

| Area | Coverage |
|------|----------|
| Hamiltonian construction (dense + sparse) | Physics correctness, eigenvalues, symmetry |
| Quantum observables | Rydberg density, correlations, entanglement |
| Geometry agent | Spacing constraints, blockade conditions, device bounds |
| Sequence agent | All 5 waveform families, parameter validation |
| Noise profiles | 3 scenarios, perturbation application |
| Robustness scoring | Aggregation formula, stability bonus, penalty |
| Objective scoring | Composite scoring, weight normalization |
| Pipeline integration | End-to-end campaign execution |
| Parallel pipeline | Concurrent noise evaluation |
| API endpoints | Health check, campaign CRUD, error handling |
| Surrogate model | Training, inference, ensemble uncertainty |
| PPO + RL environment | Action space, observation, reward shaping |
| ML dataset | Feature generation, normalization |
| Backend routing | Emulator routing logic |
| Memory agent | Lesson extraction, tag generation |

Run with coverage:

```bash
pytest tests -q --tb=short
```

---

## Repository Structure

```text
packages/
├── agents/             8 specialized agents + strategy switching
│   ├── problem_agent   Goal → ExperimentSpec
│   ├── geometry_agent  Hardware-feasible register generation
│   ├── sequence_agent  Heuristic pulse sequence generation
│   ├── noise_agent     Nominal + perturbed evaluation
│   ├── routing_agent   Backend selection
│   ├── campaign_agent  Candidate ranking
│   ├── results_agent   Summary aggregation
│   └── memory_agent    Lesson extraction and storage
├── core/               Models, enums, logging, PhysicsParameterSpace
├── orchestration/      Pipeline, phases, event bus, runner
├── simulation/         Hamiltonian, observables, evaluators, noise profiles
├── scoring/            Robustness scoring, objective scoring, ranking
├── ml/                 Surrogate V2, PPO, RL env, dataset, active learning
├── pasqal_adapters/    Pulser, Pasqal Cloud, QoolQit adapters
└── db/                 MongoDB persistence layer

apps/
├── api/                FastAPI REST API
└── dashboard/          Streamlit dashboard (8 pages)

scripts/                Training, benchmark, ablation, demo
tests/                  372 tests across all modules
```

---

## Current Limitations

- Research prototype — not calibrated against a specific physical device.
- Dense Hamiltonian diagonalization limited to $N \leq 14$ atoms (sparse methods extend this range).
- Surrogate and PPO models require training before use; untrained models default to heuristic strategies.
- Large-scale active learning campaigns require significant compute (GPU recommended).
- Pipeline decisions are principled but not yet validated against experimental data.

---

## License

Research prototype. See repository contents for project terms.
