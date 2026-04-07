from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.dashboard.components.data_loaders import (
    get_repository,
    load_agent_decisions,
    load_latest_campaigns,
    load_ranked_candidates,
    load_register_candidates,
    load_sequence_candidates,
)
from apps.dashboard.components.latex_panels import robustness_formulas
from apps.dashboard.components.plotly_charts import candidate_funnel, pipeline_gantt
from apps.dashboard.logic import build_campaign_table, build_decision_table
from packages.core.models import DemoGoalRequest
from packages.orchestration.events import EventBus, PipelineEvent
from packages.orchestration.runner import run_demo_campaign

st.title("Campaign Control Center")
st.caption("Launch, monitor, and inspect experiment campaigns.")

# --- Objective Score Formula ------------------------------------------------

formulas = robustness_formulas()
with st.expander("Objective Score Formula"):
    st.latex(formulas["objective_score"])
    st.latex(formulas["weight_constraint"])
    st.markdown(
        r"$\alpha$ = observable alignment, $\beta$ = robustness, "
        r"$\gamma$ = execution cost, $\delta$ = latency. "
        r"All weights sum to 1."
    )

# --- New Campaign Form ------------------------------------------------------

with st.form("new-campaign-form"):
    st.subheader("New Campaign")
    row1_l, row1_r = st.columns(2)
    with row1_l:
        title = st.text_input("Goal title", value="Robust neutral-atom benchmark sweep")
    with row1_r:
        scientific_objective = st.text_area(
            "Scientific objective",
            value="Design a small neutral-atom experiment campaign that balances robustness and execution feasibility.",
            height=80,
        )

    row2_l, row2_r = st.columns(2)
    with row2_l:
        desired_atom_count = st.slider("Desired atom count", min_value=2, max_value=50, value=6)
    with row2_r:
        preferred_geometry = st.selectbox(
            "Preferred geometry",
            ["mixed", "square", "line", "2d", "triangular", "ring", "zigzag", "honeycomb"],
        )

    row3_l, row3_r = st.columns(2)
    with row3_l:
        target_observable = st.selectbox(
            "Target observable",
            ["rydberg_density", "entanglement_entropy", "antiferromagnetic_order"],
        )
    with row3_r:
        priority = st.selectbox("Priority", ["balanced", "speed", "quality"])

    submitted = st.form_submit_button("Run Campaign")

if submitted:
    live_events: list[dict] = []
    status_placeholder = st.empty()

    from apps.dashboard.logic import build_event_table

    events_placeholder = st.empty()
    event_bus = EventBus()

    def _handle_event(event: PipelineEvent) -> None:
        live_events.append({"event": event.event_type, "phase": str(event.payload.get("phase", ""))})
        status_placeholder.caption(f"Pipeline event: {event.event_type}")
        events_placeholder.dataframe(live_events[-8:], use_container_width=True)

    event_bus.subscribe("*", _handle_event)

    repo = get_repository()
    with st.spinner("Running pipeline..."):
        summary = run_demo_campaign(
            request=DemoGoalRequest(
                title=title,
                scientific_objective=scientific_objective,
                target_observable=target_observable,
                desired_atom_count=desired_atom_count,
                preferred_geometry=preferred_geometry,
            ),
            repository=repo,
            event_bus=event_bus,
        )
    st.success(f"Campaign {summary.campaign.id} completed.")
    st.cache_data.clear()

# --- Recent Campaigns -------------------------------------------------------

st.subheader("Recent Campaigns")
campaigns = load_latest_campaigns(limit=20)
if not campaigns:
    st.info("No campaigns stored yet. Run a demo campaign to populate the dashboard.")
    st.stop()

from packages.core.models import CampaignState

campaign_models = [CampaignState.model_validate(c) for c in campaigns]
st.dataframe(build_campaign_table(campaign_models), use_container_width=True)

campaign_ids = [c["id"] for c in campaigns]
selected_idx = st.selectbox("Select campaign to inspect", range(len(campaign_ids)), format_func=lambda i: campaign_ids[i])
selected_campaign_id = campaign_ids[selected_idx]
st.session_state["selected_campaign_id"] = selected_campaign_id

# --- Pipeline Inspector -----------------------------------------------------

st.subheader("Pipeline Inspector")

decisions = load_agent_decisions(selected_campaign_id)
ranked = load_ranked_candidates(selected_campaign_id)
registers = load_register_candidates(selected_campaign_id)
sequences = load_sequence_candidates(selected_campaign_id)

left, right = st.columns([0.4, 0.6])

with left:
    phases = [
        "Problem Framing", "Geometry", "Sequence",
        "Evaluation", "Ranking", "Memory",
    ]
    for phase_name in phases:
        matching = [d for d in decisions if phase_name.lower().replace(" ", "_") in d.get("agent_name", "")]
        status_icon = "done" if matching else "pending"
        st.markdown(f"**{phase_name}** — {status_icon}")

with right:
    fig_gantt = pipeline_gantt(decisions)
    st.plotly_chart(fig_gantt, use_container_width=True)

    campaign_data = next((c for c in campaigns if c["id"] == selected_campaign_id), {})
    fig_funnel = candidate_funnel(
        campaign_data,
        n_registers=len(registers),
        n_sequences=len(sequences),
        n_evaluated=len(ranked),
        n_ranked=len([r for r in ranked if r.get("final_rank") is not None]),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

# --- Agent Decision Log -----------------------------------------------------

st.subheader("Agent Decision Log")
if decisions:
    from packages.core.models import AgentDecision

    decision_models = [AgentDecision.model_validate(d) for d in decisions]
    st.dataframe(build_decision_table(decision_models), use_container_width=True)

    with st.expander("Full decision details"):
        for d in decisions:
            st.json(d.get("structured_output", {}))
else:
    st.info("No decisions recorded for this campaign.")
