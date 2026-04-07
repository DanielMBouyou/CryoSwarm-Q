from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.dashboard.components.data_loaders import (
    load_agent_decisions,
    load_latest_campaigns,
)
from apps.dashboard.components.latex_panels import ml_formulas
from apps.dashboard.components.plotly_charts import (
    ppo_training_dashboard,
    prediction_vs_actual,
    strategy_ucb_evolution,
    training_loss_curves,
)

st.title("ML Observatory")
st.caption("Surrogate models, PPO training, strategy bandit.")

# --- ML Theory --------------------------------------------------------------

mf = ml_formulas()
with st.expander("ML Theory"):
    st.latex(mf["ppo_objective"])
    st.latex(mf["importance_ratio"])
    st.latex(mf["gae"])
    st.latex(mf["ensemble_mean"])
    st.latex(mf["epistemic_uncertainty"])
    st.latex(mf["ucb1"])
    st.latex(mf["weighted_mse"])
    st.markdown(
        "**SurrogateV2:** Input(18) -> Linear(128) -> GELU -> "
        "[ResidualBlock(LayerNorm->Linear->GELU->Dropout->Linear)]x3 "
        "-> LayerNorm -> Linear(64) -> GELU -> Linear(4) -> Sigmoid\n\n"
        "**ActorCritic:** obs(16) -> Tanh(128) -> Tanh(128) -> "
        "Actor: Tanh(Linear(4)) + learnable log_std | "
        "Critic: Tanh(64) -> Tanh(32) -> Linear(1)"
    )

# --- Tabs -------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Surrogate Model", "PPO Training", "Strategy Bandit", "RL vs Heuristic"])

RUNS_DIR = ROOT_DIR / "runs"

# --- Tab 1: Surrogate Model -------------------------------------------------

with tab1:
    st.subheader("Surrogate Model")
    surrogate_history_path = RUNS_DIR / "surrogate_history.json"

    if surrogate_history_path.exists():
        with open(surrogate_history_path) as f:
            history = json.load(f)
        fig = training_loss_curves(history)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No surrogate training history found at `runs/surrogate_history.json`.")

    # Model architecture summary
    st.markdown("**Model Architecture Summary**")
    arch_table = [
        {"Parameter": "input_dim", "Value": "18"},
        {"Parameter": "output_dim", "Value": "4"},
        {"Parameter": "hidden_dim", "Value": "128"},
        {"Parameter": "n_residual_blocks", "Value": "3"},
        {"Parameter": "dropout", "Value": "0.1"},
        {"Parameter": "activation", "Value": "GELU"},
        {"Parameter": "output_activation", "Value": "Sigmoid"},
    ]
    st.dataframe(arch_table, use_container_width=True)

    # Checkpoint info
    checkpoint_path = RUNS_DIR / "surrogate_v2.pt"
    if checkpoint_path.exists():
        st.success(f"Checkpoint found: {checkpoint_path.name}")
    else:
        st.info("No surrogate checkpoint found.")

# --- Tab 2: PPO Training ----------------------------------------------------

with tab2:
    st.subheader("PPO Training")
    ppo_history_path = RUNS_DIR / "ppo_history.json"

    if ppo_history_path.exists():
        with open(ppo_history_path) as f:
            ppo_hist = json.load(f)
        fig = ppo_training_dashboard(ppo_hist)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No PPO training history found at `runs/ppo_history.json`.")

    st.markdown("**PPO Configuration**")
    ppo_config = [
        {"Parameter": "lr_actor", "Value": "3e-4"},
        {"Parameter": "lr_critic", "Value": "1e-3"},
        {"Parameter": "gamma", "Value": "0.99"},
        {"Parameter": "epsilon_clip", "Value": "0.2"},
        {"Parameter": "entropy_coeff", "Value": "0.01"},
        {"Parameter": "rollout_steps", "Value": "256"},
        {"Parameter": "batch_size", "Value": "64"},
    ]
    st.dataframe(ppo_config, use_container_width=True)

    st.markdown(
        "**Action space:** 4-dim continuous [-1,1] -> (amplitude, detuning, duration, family)\n\n"
        "**Observation space:** 16-dim feature vector (atom count, spacing, blockade radius, "
        "feasibility, target density, best robustness, best params, step progress)"
    )

    ppo_ckpt = RUNS_DIR / "ppo_checkpoint.pt"
    if ppo_ckpt.exists():
        st.success(f"PPO checkpoint found: {ppo_ckpt.name}")
    else:
        st.info("No PPO checkpoint found.")

# --- Tab 3: Strategy Bandit -------------------------------------------------

with tab3:
    st.subheader("Strategy Bandit")
    campaigns = load_latest_campaigns(limit=20)
    if not campaigns:
        st.info("No campaigns available.")
    else:
        campaign_ids = [c["id"] for c in campaigns]
        default_idx = 0
        if "selected_campaign_id" in st.session_state:
            sid = st.session_state["selected_campaign_id"]
            if sid in campaign_ids:
                default_idx = campaign_ids.index(sid)

        sel_cid = st.selectbox("Campaign", campaign_ids, index=default_idx, key="ml_bandit_campaign")
        decisions = load_agent_decisions(sel_cid)

        strategy_decisions = [
            d for d in decisions
            if d.get("decision_type") == "candidate_generation"
            and "strategy_used" in d.get("structured_output", {})
        ]

        if strategy_decisions:
            # Build strategy data
            strategy_data: dict[str, list[float]] = {}
            strategy_counts: dict[str, int] = {}
            strategy_rewards: dict[str, list[float]] = {}

            for d in strategy_decisions:
                so = d.get("structured_output", {})
                strat = so.get("strategy_used", "unknown")
                strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
                reward = so.get("reward", 0.5)
                strategy_rewards.setdefault(strat, []).append(reward)

            import math

            total_n = sum(strategy_counts.values())
            for strat, rewards_list in strategy_rewards.items():
                ucb_scores = []
                for i, r in enumerate(rewards_list):
                    n_s = i + 1
                    mean_r = sum(rewards_list[:n_s]) / n_s
                    ucb = mean_r + math.sqrt(2 * math.log(max(total_n, 1)) / n_s)
                    ucb_scores.append(ucb)
                strategy_data[strat] = ucb_scores

            fig = strategy_ucb_evolution(strategy_data)
            st.plotly_chart(fig, use_container_width=True)

            # Pie chart
            import plotly.graph_objects as go

            labels = list(strategy_counts.keys())
            values = list(strategy_counts.values())
            fig_pie = go.Figure(go.Pie(labels=labels, values=values, hole=0.3))
            fig_pie.update_layout(template="plotly_white", title="Strategy Distribution", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Per-strategy table
            table = []
            for strat, rewards_list in strategy_rewards.items():
                table.append({
                    "Strategy": strat,
                    "Trials": len(rewards_list),
                    "Avg Reward": f"{sum(rewards_list)/len(rewards_list):.3f}",
                    "Best Reward": f"{max(rewards_list):.3f}",
                })
            st.dataframe(table, use_container_width=True)
        else:
            st.info("No strategy decisions found for this campaign. Run campaigns with adaptive strategy to populate.")

# --- Tab 4: RL vs Heuristic -------------------------------------------------

with tab4:
    st.subheader("RL vs Heuristic Comparison")
    campaigns = load_latest_campaigns(limit=20)
    if not campaigns:
        st.info("No campaigns available.")
    else:
        campaign_ids = [c["id"] for c in campaigns]
        default_idx = 0
        if "selected_campaign_id" in st.session_state:
            sid = st.session_state["selected_campaign_id"]
            if sid in campaign_ids:
                default_idx = campaign_ids.index(sid)

        sel_cid = st.selectbox("Campaign", campaign_ids, index=default_idx, key="ml_rl_campaign")
        decisions = load_agent_decisions(sel_cid)

        strat_decisions = [
            d for d in decisions
            if d.get("decision_type") == "candidate_generation"
            and "rl_candidates_count" in d.get("structured_output", {})
        ]

        if strat_decisions:
            import plotly.graph_objects as go

            rl_counts = [d["structured_output"].get("rl_candidates_count", 0) for d in strat_decisions]
            heur_counts = [d["structured_output"].get("heuristic_candidates_count", 0) for d in strat_decisions]
            labels = [f"Trial {i+1}" for i in range(len(strat_decisions))]

            fig = go.Figure(
                data=[
                    go.Bar(name="RL", x=labels, y=rl_counts, marker_color="#d62728"),
                    go.Bar(name="Heuristic", x=labels, y=heur_counts, marker_color="#1f77b4"),
                ]
            )
            fig.update_layout(
                template="plotly_white",
                title="RL vs Heuristic Candidates per Trial",
                barmode="group",
                yaxis_title="Candidate Count",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run campaigns with adaptive strategy to compare RL vs heuristic.")
