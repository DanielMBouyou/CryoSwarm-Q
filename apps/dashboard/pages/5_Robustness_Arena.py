from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.dashboard.components.data_loaders import (
    load_latest_campaigns,
    load_ranked_candidates,
    load_robustness_reports,
)
from apps.dashboard.components.latex_panels import robustness_formulas
from apps.dashboard.components.plotly_charts import (
    noise_radar,
    robustness_grouped_bar,
    robustness_violin,
    score_degradation_waterfall,
)

st.title("Robustness Arena")
st.caption("Noise sensitivity analysis and robustness scoring.")

# --- Robustness Mathematics -------------------------------------------------

rf = robustness_formulas()
with st.expander("Robustness Mathematics"):
    st.latex(rf["robustness_score"])
    st.latex(rf["robustness_weights"])
    st.latex(rf["stability_bonus"])
    st.latex(rf["penalty"])
    st.latex(rf["noise_amplitude"])
    st.latex(rf["noise_detuning"])
    st.latex(rf["dephasing_lindblad"])
    st.latex(rf["density_score"])
    st.latex(rf["blockade_score"])
    st.markdown(
        "Each scoring component captures a different aspect of noise resilience. "
        "The stability bonus rewards low variance across perturbations. "
        "The penalty flags sharp degradation from nominal to worst-case."
    )
    st.markdown(
        "| Scenario | sigma_amp | sigma_det | gamma_phi | gamma_loss | T (uK) | SPAM | Spatial |\n"
        "|----------|-----------|-----------|-----------|------------|--------|------|---------|\n"
        "| LOW | 0.01 | 0.01 | 0.001 | 0.001 | 50 | 0.5% | 2% |\n"
        "| MEDIUM | 0.05 | 0.05 | 0.005 | 0.005 | 50 | 0.5% | 5% |\n"
        "| STRESSED | 0.10 | 0.10 | 0.010 | 0.010 | 50 | 0.5% | 8% |\n"
    )

# --- Campaign selection -----------------------------------------------------

campaigns = load_latest_campaigns(limit=20)
if not campaigns:
    st.info("No campaigns available.")
    st.stop()

campaign_ids = [c["id"] for c in campaigns]
default_idx = 0
if "selected_campaign_id" in st.session_state:
    sid = st.session_state["selected_campaign_id"]
    if sid in campaign_ids:
        default_idx = campaign_ids.index(sid)

selected_cid = st.selectbox("Campaign", campaign_ids, index=default_idx, key="rob_campaign")
reports = load_robustness_reports(selected_cid)

if not reports:
    st.info("No robustness reports for this campaign.")
    st.stop()

# --- Grouped bar chart ------------------------------------------------------

fig = robustness_grouped_bar(reports)
st.plotly_chart(fig, use_container_width=True)

# --- Radar + Violin ---------------------------------------------------------

left, right = st.columns(2)
with left:
    fig = noise_radar(reports, max_candidates=4)
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = robustness_violin(reports)
    st.plotly_chart(fig, use_container_width=True)

# --- Deep Dive --------------------------------------------------------------

st.subheader("Deep Dive")
report_labels = [r.get("sequence_candidate_id", "?")[-12:] for r in reports]
dd_idx = st.selectbox("Select candidate for deep dive", range(len(report_labels)), format_func=lambda i: report_labels[i])
selected_report = reports[dd_idx]

left2, right2 = st.columns(2)
with left2:
    fig = score_degradation_waterfall(selected_report)
    st.plotly_chart(fig, use_container_width=True)

with right2:
    st.markdown("**Observable Comparison Across Noise Scenarios**")
    scenario_obs = selected_report.get("scenario_observables", {})
    nominal_obs = selected_report.get("nominal_observables", {})
    obs_table = [{"scenario": "nominal", **{k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in nominal_obs.items()}}]
    for scenario_name, obs_dict in scenario_obs.items():
        row = {"scenario": scenario_name}
        for k, v in obs_dict.items():
            row[k] = f"{v:.4f}" if isinstance(v, float) else str(v)
        obs_table.append(row)
    if obs_table:
        st.dataframe(obs_table, use_container_width=True)
    else:
        st.info("No observable data available.")

# --- Summary Metrics --------------------------------------------------------

st.subheader("Robustness Summary Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nominal Score", f"{selected_report.get('nominal_score', 0):.4f}")
c2.metric("Perturbation Avg", f"{selected_report.get('perturbation_average', 0):.4f}")
c3.metric("Worst Case", f"{selected_report.get('worst_case_score', 0):.4f}")
c4.metric("Score Std", f"{selected_report.get('score_std', 0):.4f}")

st.markdown(f"**Robustness Penalty:** {selected_report.get('robustness_penalty', 0):.4f}")

with st.expander("Full robustness report"):
    st.json(selected_report)
