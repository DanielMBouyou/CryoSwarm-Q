from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.dashboard.components.latex_panels import (
    campaign_formulas,
    hamiltonian_formulas,
    mis_formulas,
    ml_formulas,
    observable_formulas,
    pulse_formulas,
    robustness_formulas,
)

st.title("Theory Reference")
st.caption("Complete mathematical reference for CryoSwarm-Q.")
st.markdown(
    "*This page serves as a self-contained reference for the physics, "
    "scoring, and ML behind CryoSwarm-Q.*"
)

# --- Section 1: Rydberg Hamiltonian -----------------------------------------

hf = hamiltonian_formulas()
with st.expander("1. Rydberg Hamiltonian", expanded=False):
    for key, formula in hf.items():
        st.latex(formula)
    st.markdown(
        "**Physical interpretation:**\n"
        "- The **Rabi coupling** Omega/2 drives coherent transitions between "
        "ground |g> and Rydberg |r> states via sigma_x.\n"
        "- The **detuning** delta controls the energy offset of the Rydberg state "
        "relative to the drive frequency.\n"
        "- The **van der Waals interaction** C6/r^6 between Rydberg-excited atoms "
        "creates the blockade: if two atoms are closer than R_b, simultaneous "
        "excitation is strongly suppressed.\n\n"
        "**Implementation:** `packages/simulation/hamiltonian.py`\n"
        "- Dense construction via Kronecker products for N <= 14 atoms\n"
        "- Sparse CSC construction via scipy for larger systems\n"
        "- C6 coefficient calibrated for 87-Rb |70S_1/2>"
    )

# --- Section 2: Quantum Observables ----------------------------------------

of = observable_formulas()
with st.expander("2. Quantum Observables", expanded=False):
    for key, formula in of.items():
        st.latex(formula)
    st.markdown(
        "**Rydberg density** <n_i>: probability that atom i is in the Rydberg state.\n\n"
        "**Connected correlation** g_ij: measures genuine quantum correlations beyond mean-field. "
        "Negative values indicate anti-bunching (antiferromagnetic order). "
        "Positive values indicate bunching (ferromagnetic-like).\n\n"
        "**Antiferromagnetic order** m_AF: staggered magnetization. "
        "1 = perfect Neel order, 0 = disordered.\n\n"
        "**Entanglement entropy** S_A: von Neumann entropy of the reduced density matrix. "
        "Computed via Schmidt decomposition (SVD) of the bipartite state.\n\n"
        "**Implementation:** `packages/simulation/observables.py`\n\n"
        "**Bitstring convention:** MSB: atom 0 is the most significant bit. Consistent with Pulser."
    )

# --- Section 3: Maximum Independent Set ------------------------------------

mf = mis_formulas()
with st.expander("3. Maximum Independent Set", expanded=False):
    for key, formula in mf.items():
        st.latex(formula)
    st.markdown(
        "**Physical connection:** The ground state of the Rydberg Hamiltonian in the "
        "blockade regime (large delta > 0, strong interactions) approximates solutions "
        "to the Maximum Independent Set problem on the blockade graph. This is the basis "
        "for quantum optimization on neutral-atom hardware.\n\n"
        "**Algorithm:**\n"
        "- Exact enumeration for N <= 15 atoms (brute-force over all subsets)\n"
        "- Greedy heuristic with randomised restarts for 15 < N <= 50\n\n"
        "**Implementation:** `packages/simulation/hamiltonian.py` -> `find_maximum_independent_sets()`"
    )

# --- Section 4: Pulse Sequence Families -------------------------------------

pf = pulse_formulas()
with st.expander("4. Pulse Sequence Families", expanded=False):
    for key, formula in pf.items():
        st.latex(formula)
    st.markdown(
        "| Family | Omega(t) Profile | delta(t) Profile | Use Case |\n"
        "|--------|-----------------|-----------------|----------|\n"
        "| `constant_drive` | Constant Omega_max | Constant delta_0 | Simple Rabi oscillation |\n"
        "| `global_ramp` | Linear ramp 0->Omega_max | Linear sweep | Gradual excitation |\n"
        "| `detuning_scan` | Constant Omega_max | Linear sweep | Resonance scanning |\n"
        "| `adiabatic_sweep` | sin^2(pi*t/T) envelope | Linear sweep | Adiabatic state preparation |\n"
        "| `blackman_sweep` | Blackman window | Linear sweep | Low-sidelobe adiabatic prep |\n"
    )

# --- Section 5: Robustness Scoring -----------------------------------------

rf = robustness_formulas()
with st.expander("5. Robustness Scoring", expanded=False):
    for key, formula in rf.items():
        st.latex(formula)
    st.markdown(
        "**Scoring pipeline:**\n"
        "1. Simulate nominal (noiseless) -> s_nom\n"
        "2. Simulate under LOW, MEDIUM, STRESSED noise -> perturbed scores\n"
        "3. Compute: perturbation_average, worst_case, score_std\n"
        "4. Aggregate: S_robust = weighted combination\n"
        "5. Compute penalty for sharp degradation\n\n"
        "**Noise model parameters:**\n\n"
        "| Scenario | sigma_amp | sigma_det | gamma_phi | gamma_loss | T (uK) | SPAM | Spatial |\n"
        "|----------|-----------|-----------|-----------|------------|--------|------|---------|\n"
        "| LOW | 0.01 | 0.01 | 0.001 | 0.001 | 50 | 0.5% | 2% |\n"
        "| MEDIUM | 0.05 | 0.05 | 0.005 | 0.005 | 50 | 0.5% | 5% |\n"
        "| STRESSED | 0.10 | 0.10 | 0.010 | 0.010 | 50 | 0.5% | 8% |\n\n"
        "**Implementation:** `packages/scoring/robustness.py`, `packages/simulation/noise_profiles.py`"
    )

# --- Section 6: Objective Score & Ranking -----------------------------------

with st.expander("6. Objective Score & Ranking", expanded=False):
    st.latex(rf["objective_score"])
    st.latex(rf["weight_constraint"])
    cf = campaign_formulas()
    st.latex(cf["ranking_key"])
    st.markdown(
        "**Objective score** combines observable alignment, robustness, "
        "execution cost, and latency.\n\n"
        "**Ranking** is lexicographic: sort by "
        "(objective_score desc, worst_case desc, robustness desc, nominal desc).\n\n"
        "**Implementation:** `packages/scoring/objective.py`, `packages/scoring/ranking.py`"
    )

# --- Section 7: PPO & Reinforcement Learning --------------------------------

mlf = ml_formulas()
with st.expander("7. PPO & Reinforcement Learning", expanded=False):
    st.latex(mlf["ppo_objective"])
    st.latex(mlf["importance_ratio"])
    st.latex(mlf["gae"])
    st.markdown(
        "**Environment:** `PulseDesignEnv` (Gymnasium-compatible)\n"
        "- Observation: 16-dim vector (atom count, spacing, blockade radius, feasibility, "
        "target density, best robustness so far, best params, step progress)\n"
        "- Action: 4-dim continuous [-1,1] -> rescaled to (amplitude, detuning, duration, family)\n"
        "- Reward: shaped reward = (1-w)*raw_robustness + w*improvement*scale + terminal bonus\n\n"
        "**Training:** Standard PPO with:\n"
        "- Separate actor/critic learning rates\n"
        "- GAE advantage estimation (gamma=0.99, lambda=0.95)\n"
        "- Clipped surrogate objective (epsilon=0.2)\n"
        "- Entropy regularization (c_ent=0.01)\n"
        "- Gradient clipping (max_norm=0.5)\n\n"
        "**Implementation:** `packages/ml/ppo.py`, `packages/ml/rl_env.py`"
    )

# --- Section 8: Surrogate Ensemble & Uncertainty ----------------------------

with st.expander("8. Surrogate Ensemble & Uncertainty", expanded=False):
    st.latex(mlf["ensemble_mean"])
    st.latex(mlf["epistemic_uncertainty"])
    st.latex(mlf["weighted_mse"])
    st.latex(mlf["ucb1"])
    st.markdown(
        "**Architecture:** SurrogateModelV2\n"
        "- Input: 18-dim physics-informed features (normalized)\n"
        "- 3 residual blocks with LayerNorm + GELU + Dropout\n"
        "- Output: 4 targets -> Sigmoid [0,1] "
        "(robustness_score, nominal_score, worst_case_score, observable_score)\n"
        "- Target weights: [2.0, 1.0, 1.5, 1.0] (prioritize robustness prediction)\n\n"
        "**Ensemble:** M=3 independently trained models\n"
        "- Epistemic uncertainty = inter-model variance\n"
        "- Used for active learning: high uncertainty -> prioritize for re-simulation\n\n"
        "**Strategy selection:** UCB1 bandit chooses between heuristic, RL, hybrid modes "
        "based on observed robustness scores per problem class.\n\n"
        "**Implementation:** `packages/ml/surrogate.py`, `packages/agents/sequence_strategy.py`"
    )
