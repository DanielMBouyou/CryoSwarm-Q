# CryoSwarm-Q

Hardware-aware multi-agent orchestration for neutral-atom experiment design.

CryoSwarm-Q turns a structured scientific goal into:

- neutral-atom register candidates
- pulse-sequence candidates compatible with Pulser / Pasqal-style constraints
- robustness evaluations under noise
- backend routing and ranking
- reusable memory across campaigns

The repository now includes the full P0/P1/P2 stack:

- configurable `PhysicsParameterSpace`
- heuristic / RL `SequenceStrategy`
- large-scale dataset generation
- surrogate V1 and residual surrogate V2
- ensemble uncertainty estimates
- multi-step PPO with reward shaping
- curriculum learning
- active learning loop
- reproducible benchmark and ablation scripts

## System Overview

Core orchestration path:

1. `ProblemFramingAgent` converts an `ExperimentGoal` into an `ExperimentSpec`.
2. `GeometryAgent` proposes hardware-feasible neutral-atom registers.
3. `SequenceStrategy` selects heuristic, RL, hybrid, adaptive, or bandit sequence generation.
4. `NoiseRobustnessAgent` evaluates nominal and perturbed performance.
5. `BackendRoutingAgent` recommends the execution backend.
6. `CampaignAgent` ranks candidates.
7. `MemoryAgent` stores reusable lessons for future campaigns.

Main implementation areas:

```text
packages/
|-- agents/            Agent implementations and strategy switching
|-- core/              Models, enums, logging, parameter space
|-- orchestration/     Pipeline orchestration
|-- simulation/        Hamiltonians, observables, evaluators, noise
|-- scoring/           Robustness and campaign scoring
|-- ml/                Dataset, surrogate, PPO, curriculum, active learning
|-- pasqal_adapters/   Pulser and Pasqal-facing adapters
`-- db/                Persistence layer
scripts/
|-- train_ml.py
|-- benchmark.py
`-- ablation.py
tests/
`-- physics, pipeline, ML, routing, robustness, and integration coverage
```

## Physics Model

The driven Rydberg Hamiltonian is:

```math
H = \frac{\Omega}{2}\sum_i \sigma_x^{(i)} - \delta\sum_i n_i + \sum_{i<j}\frac{C_6}{r_{ij}^6} n_i n_j
```

Default calibration targets `^{87}Rb` with:

- `C6 = 862690 rad.um^6/us`
- inter-atom spacing in the `4-15 um` design range
- pulse amplitudes and detunings aligned with Pasqal AnalogDevice-style bounds

The full parameterization is centralized in
[parameter_space.py](c:\Users\danie\Documents\CRYOSWARRM-Q\packages\core\parameter_space.py).

## P0 / P1 / P2 Features

### Configurable Physics

`PhysicsParameterSpace` replaces hardcoded physical constants across:

- pulse families
- geometry generation
- noise profiles
- robustness aggregation
- observable scoring
- RL action scaling

It supports:

- random sampling
- grid search
- Latin Hypercube Sampling
- serialization for reproducibility

### Sequence Generation Strategy

`SequenceStrategy` lives in
[sequence_strategy.py](c:\Users\danie\Documents\CRYOSWARRM-Q\packages\agents\sequence_strategy.py)
and supports:

- `heuristic_only`
- `rl_only`
- `hybrid`
- `adaptive`
- `bandit`

Hybrid mode evaluates heuristic and RL candidates together. Adaptive and bandit modes track per-problem-class performance over time.

### Dataset Generation

The dataset generator in
[data_generator.py](c:\Users\danie\Documents\CRYOSWARRM-Q\packages\ml\data_generator.py)
builds large training sets directly from simulation, not from self-referential pipeline summaries.

It supports:

- `lhs`, `grid`, `random`, and `sobol` sampling
- checkpoint / resume
- streaming saves
- multi-process evaluation
- feature V2 generation with physics-informed ratios

### Surrogate Stack

The surrogate stack in
[surrogate.py](c:\Users\danie\Documents\CRYOSWARRM-Q\packages\ml\surrogate.py)
contains:

- `SurrogateModel` for backward-compatible V1 checkpoints
- `SurrogateModelV2` with residual blocks, LayerNorm, GELU, and dropout
- integrated feature normalization in save / load
- `SurrogateEnsemble` for uncertainty-aware prediction
- `EnsembleTrainer`

### RL Stack

The PPO environment in
[rl_env.py](c:\Users\danie\Documents\CRYOSWARRM-Q\packages\ml\rl_env.py)
is now multi-step by default:

- default `max_steps = 5`
- reward shaping from episode improvement
- episode-best feedback in the observation
- compatibility path with `max_steps=1, reward_shaping=False`

The PPO trainer in
[ppo.py](c:\Users\danie\Documents\CRYOSWARRM-Q\packages\ml\ppo.py)
supports curriculum integration via
[curriculum.py](c:\Users\danie\Documents\CRYOSWARRM-Q\packages\ml\curriculum.py).

### Active Learning

The iterative surrogate <-> RL loop is implemented in
[active_learning.py](c:\Users\danie\Documents\CRYOSWARRM-Q\packages\ml\active_learning.py).

Per iteration it:

1. trains or fine-tunes the surrogate
2. trains PPO with surrogate rewards
3. collects strong RL configurations
4. selects diverse points
5. re-simulates them with the real evaluator
6. grows the dataset

### Benchmarks and Ablations

Research reproducibility scripts:

- [benchmark.py](c:\Users\danie\Documents\CRYOSWARRM-Q\scripts\benchmark.py)
- [ablation.py](c:\Users\danie\Documents\CRYOSWARRM-Q\scripts\ablation.py)

They cover surrogate metrics, RL reward metrics, pipeline ranking behavior, and controlled ablation configurations.

## Installation

Editable install:

```bash
pip install -e ".[dev]"
```

Or:

```bash
pip install -r requirements.txt
```

Optional runtime integrations:

- `pulser`
- `pulser-simulation`
- `pasqal-cloud`
- `qoolqit`
- PyTorch for ML modules

## Quick Start

Run the full test suite:

```bash
pytest tests -q
```

Generate a V2 dataset:

```bash
python -m scripts.train_ml --phase generate_v2 --n-samples 1000 --workers 2 --sampling lhs
```

Train the surrogate:

```bash
python -m scripts.train_ml --phase surrogate --data data/generated/dataset.npz --epochs 100
```

Train PPO:

```bash
python -m scripts.train_ml --phase rl --updates 500
```

Train surrogate then PPO:

```bash
python -m scripts.train_ml --phase full --data data/generated/dataset.npz --epochs 100 --updates 500
```

Run active learning:

```bash
python -m scripts.train_ml --phase active --data data/generated/dataset.npz --al-iterations 5 --al-top-k 200
```

Run the benchmark suite:

```bash
python -m scripts.benchmark --full --checkpoint-dir checkpoints --test-data data/generated/dataset.npz
```

Run ablations:

```bash
python -m scripts.ablation --ablation all --data data/generated/dataset.npz
```

## Programmatic Usage

Pipeline example:

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

## Noise and Robustness

Default perturbation scenarios include:

- low noise
- medium noise
- stressed noise

The noise stack now also tracks spatial drive inhomogeneity in the `2%` to `8%` range for tweezer-array realism. Robustness aggregation and observable scoring are both parameterized through the physics parameter space rather than hardcoded constants.

## Current Limitations

- This is still a research codebase, not a lab-calibrated control stack.
- Large-system simulation remains expensive; approximation paths are used for larger atom counts.
- Pipeline decisions are stronger than before, but they are not yet backed by full experimental calibration data.
- Benchmark and ablation scripts are reproducible, but long-running studies still require significant compute.

## Validation

The repository includes broad automated coverage across:

- Hamiltonian and observable physics
- geometry and sequence generation
- robustness scoring and routing
- surrogate training and filtering
- PPO environment and policy code
- curriculum learning
- active learning
- benchmark metric computation

Run:

```bash
pytest tests -q
```

## License

Research prototype. See repository contents for project terms.
