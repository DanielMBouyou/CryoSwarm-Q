# P4 — Training Pipeline & Hardcoded Parameter Validation

## Context

You are working on **CryoSwarm-Q**, a hardware-aware multi-agent software system for neutral-atom quantum experimentation. The codebase lives at the workspace root. Python 3.14+, PyTorch 2.11 CPU locally, ROCm on droplet.

The ML pipeline is fully coded but **has never been trained for real**. All 372 tests pass (unit tests on synthetic micro-data, not actual training runs). There are **135+ hardcoded magic numbers** across the ML code that need validation.

---

## Hardware budget

| Machine | GPU | Hours | PyTorch backend |
|---------|-----|-------|-----------------|
| Local Windows | NVIDIA RTX 3060/3070/3080 (8-12 GB VRAM) | 200h | CUDA |
| DigitalOcean Droplet | AMD MI300X (192 GB HBM3) | 50h | ROCm |

The local machine currently has `torch==2.11.0+cpu`. CUDA torch must be installed first:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## What needs to happen — 2 work streams

### Stream A: Fix the 135+ hardcoded parameters (tests + config extraction)
### Stream B: Run the actual training pipeline (5 phases)

Both streams are interdependent: Stream A validates the defaults used by Stream B.

---

## Stream A — Hardcoded Parameter Validation

### A1. Feature normalization constants in `packages/ml/dataset.py`

The function `build_feature_vector_v2()` (line ~115) normalizes 18 features using hardcoded divisors. These MUST match the physical parameter ranges defined in `packages/core/parameter_space.py`.

Current hardcoded values and what they should derive from:

| Feature | Hardcoded divisor | Should be | Source |
|---------|-------------------|-----------|--------|
| `amplitude / 15.8` | 15.8 | `max(family.amplitude.max_val for all families)` = 15.0 | `parameter_space.py` pulse families |
| `(detuning + 126.0) / 252.0` | 126.0, 252.0 | `abs(min detuning_start)` = 40.0, range = 65.0 | Detuning spans [-40, +25] |
| `duration_ns / 6000.0` | 6000.0 | `max(family.duration_ns.max_val)` = 6000.0 | Correct but should be derived |
| `layout_encoding / 5.0` | 5.0 | `len(LAYOUT_ENCODING) - 1` = 5 | OK |
| `family_encoding / 4.0` | 4.0 | `len(SequenceFamily) - 1` = 4 | OK |
| `blockade_radius / 15.0` | 15.0 | `default_blockade_radius_um` = 9.5, but max can be ~20 | Needs validation |
| `min_distance / 15.0` | 15.0 | `spacing_um.max_val` = 15.0 | Correct but should be derived |
| `omega_over_interaction / 10.0` | 10.0 | Empirical — needs ablation | No physics source |
| `detuning_over_omega / 20.0` | 20.0 | Empirical — needs ablation | No physics source |
| `adiabaticity / 100.0` | 100.0 | `max_duration_us × max_amplitude` = 6.0 × 15.0 = 90.0 | Close but imprecise |
| `atom_count / 25.0` | 25.0 | `geometry.atom_count.max_val` = 25.0 | Correct but should be derived |
| `sweep_span / 50.0` | 50.0 | `max(|detuning_end - detuning_start|)` = 65.0 | Wrong |

**Task A1**: Write tests in `tests/test_feature_normalization.py` that:
1. Import `PhysicsParameterSpace.default()` and all family configs
2. For each normalization constant, compute the correct value from the parameter space
3. Assert the hardcoded value ≥ the physical maximum (so features stay in [0,1])
4. Assert that `build_feature_vector_v2()` output is always in [0, 1] for any valid parameter combination sampled from the parameter space (use random sampling, 1000 samples)
5. Flag any constant that doesn't match its derivation source

Then **replace the hardcoded constants** with values derived from `PhysicsParameterSpace.default()` at module level.

---

### A2. Surrogate training hyperparameters in `packages/ml/surrogate.py`

Current hardcoded defaults in `SurrogateTrainer.__init__()`:
- `lr: float = 1e-3` 
- `weight_decay: float = 1e-5`
- `target_weights: list[float] = [2.0, 1.0, 1.5, 1.0]` (robustness, nominal, worst_case, observable)
- `patience: int = 10` (ReduceLROnPlateau)
- `factor: float = 0.5` (LR reduction factor)

Architecture defaults in `SurrogateModelV2.__init__()`:
- `hidden: int = 128`
- `n_blocks: int = 3`
- `dropout: float = 0.1`

Ensemble defaults in `SurrogateEnsemble.__init__()`:
- `n_models: int = 3`

**Task A2**: Write tests in `tests/test_surrogate_hyperparams.py` that:
1. Test that `target_weights` sum is documented (currently 5.5 — is that intentional?)
2. Test that the model trains and val loss decreases with default hyperparameters on synthetic data
3. Test sensitivity: for each of `lr`, `weight_decay`, `dropout`, train with 3 values and assert val loss is finite and < 1.0 for all
4. Test that `target_weights=[2.0, 1.0, 1.5, 1.0]` gives better val loss on robustness prediction than uniform `[1,1,1,1]` (justification test)

---

### A3. PPO hyperparameters in `packages/ml/ppo.py`

Current defaults in `PPOConfig`:
```python
lr_actor: float = 3e-4
lr_critic: float = 1e-3
gamma: float = 0.99
gae_lambda: float = 0.95
epsilon_clip: float = 0.2
entropy_coeff: float = 0.01
value_coeff: float = 0.5
max_grad_norm: float = 0.5
epochs_per_update: int = 4
batch_size: int = 64
rollout_steps: int = 256
total_updates: int = 500
reward_clip_value: float = 5.0
replay_capacity: int = 4096
max_replay_policy_lag: int = 2
```

**Task A3**: Write tests in `tests/test_ppo_hyperparams.py` that:
1. Assert `lr_critic / lr_actor` ratio is within [2, 5] (standard practice, ensures critic learns faster)
2. Assert `epsilon_clip` is in [0.1, 0.3] (PPO paper recommendation)
3. Assert `gamma` is in [0.95, 0.999] 
4. Assert `gae_lambda` is in [0.9, 0.99]
5. Run a 10-update training and assert average reward improves (or at least doesn't diverge)
6. Test that `rollout_steps × total_updates` gives enough total environment interactions (≥50K)

---

### A4. RL environment constants in `packages/ml/rl_env.py`

Current defaults in `PulseDesignEnv.__init__()`:
- `max_steps: int = 5`
- `improvement_weight: float = 0.5`
- `improvement_scale: float = 5.0`
- Seed: `np.random.default_rng(42)`

Reward computation in `step()`:
- `raw_robustness` as base reward
- `improvement_bonus = improvement_scale * improvement` if shaping enabled
- `terminal_bonus = best_robustness * 0.5` on last step
- `shaped_reward = (1 - improvement_weight) * raw + improvement_weight * bonus`

**Task A4**: Write tests in `tests/test_rl_env_params.py` that:
1. Assert `max_steps` is in [3, 10] (too few = can't explore, too many = dilutes signal)
2. Assert reward is bounded: for any valid action, shaped reward is in [-1, 10] (prevent gradient explosion)
3. Test that `improvement_scale=5.0` doesn't dominate: assert `improvement_bonus < 2 × raw_robustness` for typical cases
4. Test that `terminal_bonus` doesn't exceed episode cumulative reward by more than 50%
5. Test observation normalization: all 16 observation features are in [-1, 2] for reasonable inputs

---

### A5. Surrogate filter thresholds in `packages/ml/surrogate_filter.py`

Current defaults:
- `top_k: int = 20`
- `min_score: float = 0.1`
- `max_uncertainty: float = 0.15`

**Task A5**: Write tests in `tests/test_filter_thresholds.py` that:
1. Assert filtering never removes ALL candidates (at least 1 survives)
2. Assert `min_score=0.1` is below the median predicted score on typical data
3. Assert `max_uncertainty=0.15` is above the median ensemble uncertainty
4. Test with 100 random candidates: assert 10% < filtered_fraction < 90% (filter is neither too aggressive nor too permissive)

---

### A6. Active learning and curriculum defaults

Active learning config (`packages/ml/active_learning.py`):
- `n_iterations: int = 5`
- `top_k_per_iteration: int = 200`
- `diversity_fraction: float = 0.5`
- `uncertainty_weight: float = 0.3`
- `surrogate_epochs: int = 50`
- `rl_updates: int = 100`
- `max_atoms_for_resim: int = 12`

Curriculum stages (`packages/ml/curriculum.py`):
- Stage 1 "warm-up": 3-5 atoms, square/line, min_performance=0.3, min_episodes=50
- Stage 2 "expansion": 4-8 atoms, +triangular/ring, min_performance=0.35, min_episodes=100
- Stage 3 "full": 3-15 atoms, all layouts, min_performance=0.0, min_episodes=0

**Task A6**: Write tests in `tests/test_al_curriculum_params.py` that:
1. Assert curriculum stages are progressive: stage N+1 has wider atom range than stage N
2. Assert stage performance thresholds are non-decreasing (except last which is 0)
3. Assert `max_atoms_for_resim=12` equals `parameter_space.max_atoms_dense` (consistency)
4. Assert `uncertainty_weight` is in (0, 1) 
5. Assert `diversity_fraction` is in (0, 1)
6. Assert `top_k_per_iteration × n_iterations` < reasonable dataset growth bound

---

### A7. Physics constants in simulation

GPU backend (`packages/ml/gpu_backend.py`):
- `krylov_dim: int = 20` (Lanczos subspace dimension)
- `n_steps: int = 200` (time evolution steps)

Hamiltonian (`packages/simulation/hamiltonian.py`):
- `C6 = 862690.0` (rad·μm⁶/μs for Rb-87 Rydberg atoms)
- `_MIS_EXACT_THRESHOLD = 15` (switch to greedy above this)

NumPy backend (`packages/simulation/numpy_backend.py`):
- Number of Trotter steps (varies by duration)

Noise profiles (`packages/simulation/noise_profiles.py`):
- Spatial inhomogeneity: [0.02, 0.05, 0.08] for LOW/MEDIUM/STRESSED

**Task A7**: Write tests in `tests/test_physics_constants.py` that:
1. Assert `C6 = 862690.0` matches literature value for ⁸⁷Rb |70S⟩ state (known value)
2. Assert `krylov_dim=20` gives < 1e-6 error vs exact expm for a 4-atom system
3. Assert `_MIS_EXACT_THRESHOLD=15` means 2^15 = 32768 states — confirm this is the practical limit for brute-force enumeration
4. Assert spatial inhomogeneity values are ordered: LOW < MEDIUM < STRESSED
5. Assert noise profile values are within `PhysicsParameterSpace.default().noise` ranges

---

### A8. Extract all validated constants to config

**Task A8**: After completing A1-A7:
1. Create `configs/training_defaults.yaml` with all validated parameter values, grouped by category
2. Create `packages/core/training_config.py` that loads this YAML with Pydantic validation
3. Update all files to read from the config instead of hardcoding
4. Write a test `tests/test_training_config.py` that verifies the YAML loads correctly and all values are within their validated ranges

---

## Stream B — Training Pipeline Execution

### B1. Install CUDA PyTorch locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must show `True` and the RTX GPU name.

---

### B2. Generate training dataset (LOCAL, ~8h)

**Phase 1 — Heuristic pipeline data:**
```bash
python -m scripts.train_ml --phase generate --runs 100 --data data/candidates_v1.npz
```
Expected: ~500-1500 samples (each run produces ~5-15 scored candidates)

**Phase 2 — Systematic LHS data:**
```bash
python -m scripts.train_ml --phase generate_v2 --n-samples 100000 --sampling lhs --output-dir data/generated --workers 4
```
Expected: 100K (features, targets) pairs in `data/generated/`

---

### B3. Train surrogate ensemble (LOCAL, ~2.5h)

```bash
# Single model first (sanity check)
python -m scripts.train_ml --phase surrogate --data data/generated/dataset.npz --epochs 200 --checkpoint-dir checkpoints/surrogate_v2

# Then ensemble with k-fold CV (needs custom script or extension)
```

**Success criteria:**
- Val loss < 0.05 after 200 epochs
- 1-σ coverage > 0.6 on held-out test set
- Calibration error < 0.1

---

### B4. Train PPO policy (LOCAL, ~10h)

```bash
# Short run first
python -m scripts.train_ml --phase rl --updates 500 --checkpoint-dir checkpoints/ppo_short

# Long run
python -m scripts.train_ml --phase rl --updates 2000 --checkpoint-dir checkpoints/ppo_long
```

**Success criteria:**
- Average episode reward > 0.3 (last 100 episodes)
- Policy entropy > 0.5 (not collapsed)
- Action std > 0.1 (still exploring)

---

### B5. Active learning loop (LOCAL, ~12h)

```bash
python -m scripts.train_ml --phase active --data data/generated/dataset.npz --al-iterations 5 --al-top-k 200 --checkpoint-dir checkpoints/active
```

**Success criteria:**
- Dataset grows by ≥500 new samples total
- Surrogate val loss decreases each iteration
- RL reward improves iteration over iteration

---

### B6. Heavy training on MI300X Droplet (~46h)

> **⚠️ DO NOT attempt to configure, provision, or run anything on the Droplet autonomously.**
>
> The Droplet phase will be done **together with the user, interactively**. When all local work (Stream A + B1-B5 + B7) is complete, **ask the user** to set up the Droplet session. The user will:
> - Create the droplet
> - Provide SSH access
> - Confirm the ROCm environment is ready
>
> Only then should you proceed with Droplet commands. Until the user explicitly says "let's do the droplet", treat B6 as **blocked/deferred**.

Once the user initiates the Droplet session:

```bash
# Full simulation dataset (100K with actual Schrödinger solver, not heuristic)
python -m scripts.train_ml --phase generate_v2 --n-samples 100000 --sampling sobol --output-dir data/generated_full --workers 16

# Long PPO with surrogate as fast evaluator
python -m scripts.train_ml --phase full --data data/generated_full/dataset.npz --epochs 300 --updates 5000 --checkpoint-dir checkpoints/full_mi300x

# Extended active learning
python -m scripts.train_ml --phase active --data data/generated_full/dataset.npz --al-iterations 10 --al-top-k 500 --checkpoint-dir checkpoints/active_mi300x
```

---

### B7. Ablation sweeps (LOCAL, ~80h)

Create `scripts/ablation_sweep.py` that:
1. Takes a parameter name, a list of values, and a training phase
2. Runs the phase for each value
3. Records the metric (val loss for surrogate, avg reward for RL)
4. Saves results to `data/ablation_results/{param_name}.json`

Priority ablations:

| Parameter | Values to sweep | Metric | Estimated time |
|-----------|-----------------|--------|----------------|
| `surrogate_lr` | [5e-4, 1e-3, 2e-3, 5e-3, 1e-2] | val loss | 3h |
| `surrogate_target_weights` | [[1,1,1,1], [2,1,1.5,1], [3,1,2,1], [1,1,1,2]] | val loss per target | 3h |
| `ppo_lr_actor` | [1e-4, 3e-4, 1e-3] | avg reward | 12h |
| `ppo_lr_critic` | [3e-4, 1e-3, 3e-3] | avg reward | 12h |
| `ppo_entropy_coeff` | [0.001, 0.01, 0.05, 0.1] | avg reward + entropy | 16h |
| `rl_improvement_weight` | [0.0, 0.25, 0.5, 0.75, 1.0] | avg reward | 10h |
| `rl_improvement_scale` | [1.0, 3.0, 5.0, 10.0] | avg reward | 8h |
| `filter_min_score` | [0.0, 0.05, 0.1, 0.2, 0.3] | pipeline end-to-end score | 3h |
| `krylov_dim` | [10, 15, 20, 30, 50] | simulation error vs exact | 2h |

---

## File map — Where things live

```
packages/ml/
├── surrogate.py          # SurrogateModel, SurrogateModelV2, SurrogateEnsemble, SurrogateTrainer, EnsembleTrainer
├── ppo.py                # PPOConfig, ActorCritic, RolloutBuffer, RunningMeanStd, PPOTrainer
├── rl_env.py             # PulseDesignEnv, rescale_action, inverse_rescale, OBS_DIM=16, ACT_DIM=4
├── rl_sequence_agent.py  # RLSequenceAgent (inference only, loads checkpoint)
├── training_runner.py    # TrainingRunner, TrainingConfig (orchestrates all phases)
├── dataset.py            # build_feature_vector, build_feature_vector_v2, CandidateDatasetBuilder
├── data_generator.py     # DatasetGenerator, GenerationConfig (LHS/Sobol/Grid sampling)
├── surrogate_filter.py   # SurrogateFilter (pre-filters candidates before simulation)
├── active_learning.py    # ActiveLearningLoop, ActiveLearningConfig
├── curriculum.py         # CurriculumScheduler, CurriculumStage
├── gpu_backend.py        # GPU sparse Hamiltonian, Lanczos time evolution
└── normalizer.py         # DatasetNormalizer (mean/std normalization)

packages/core/
├── parameter_space.py    # PhysicsParameterSpace — THE source of truth for all physical ranges
├── models.py             # All Pydantic domain models
├── enums.py              # SequenceFamily, NoiseLevel, etc.
└── config.py             # App-level config

packages/scoring/
├── robustness.py         # robustness_score() with weight defaults from parameter_space
├── ranking.py            # Candidate ranking
└── objective.py          # Objective scoring

packages/simulation/
├── numpy_backend.py      # CPU Schrödinger simulation
├── hamiltonian.py        # Rydberg Hamiltonian, MIS, C6 constant
├── noise_profiles.py     # LOW/MEDIUM/STRESSED noise scenarios
├── evaluators.py         # evaluate_candidate_robustness()
└── observables.py        # Quantum observables (density, correlations, entropy)

scripts/
├── train_ml.py           # CLI: --phase generate|generate_v2|surrogate|rl|full|active
└── run_demo_pipeline.py  # Quick demo (heuristic only, no ML)
```

---

## Physical parameter ranges (from `PhysicsParameterSpace.default()`)

These are the **source of truth** for normalization constants:

| Parameter | Min | Max | Default | Unit |
|-----------|-----|-----|---------|------|
| amplitude | 1.0 | 15.0 | varies by family | rad/μs |
| detuning_start | -40.0 | 0.0 | varies | rad/μs |
| detuning_end | 0.0 | 25.0 | varies | rad/μs |
| duration_ns | 500.0 | 6000.0 | varies | ns |
| spacing_um | 4.0 | 15.0 | 7.0 | μm |
| atom_count | 2 | 25 | 6 | — |
| blockade_radius (derived) | ~5 | ~20 | 9.5 | μm |
| C6 coefficient | — | — | 862690.0 | rad·μm⁶/μs |

---

## Test suite status

- **372 passed, 0 failed, 5 skipped** (MongoDB tests need server)
- All tests CPU-only on synthetic micro-data
- No integration tests for actual training loops
- No validation of hardcoded constants against physical ranges

---

## Priority order

1. **A1** (feature normalization) — highest risk, affects ALL downstream training
2. **A2-A3** (surrogate + PPO hyperparams) — needed before any training
3. **B1** (install CUDA torch) — unblocks GPU training
4. **B2** (generate dataset) — unblocks all training
5. **A4-A6** (env, filter, curriculum) — can run in parallel with B2
6. **B3** (train surrogate) — needs B2 data
7. **A7** (physics constants) — independent
8. **B4-B5** (PPO + active learning) — needs B3 surrogate
9. **B7** (ablations) — needs working training pipeline
10. **A8** (YAML config extraction) — after ablations decide final values
11. **B6** (droplet heavy training) — **BLOCKED until user says go**. Do NOT proceed with this until all local work (A1-A8 + B1-B5 + B7) is finished AND the user explicitly initiates the Droplet session. The user will configure the droplet together with you interactively.

---

## Critical rules

- **Never train without first running the test suite** (`pytest tests/ --ignore=tests/test_mongodb_config.py`)
- **Always save checkpoints** with metadata (epoch, val_loss, config used)
- **Log everything to TensorBoard** (`--log-dir` flag)
- **Validate feature ranges before training**: if any feature > 1.0, normalization is broken
- **Monitor GPU memory**: RTX 3060 has 12GB max; batch_size may need reduction
- **On droplet**: always use `tmux` or `screen` so training survives SSH disconnect
