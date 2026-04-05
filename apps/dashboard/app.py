from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from packages.core.config import get_settings
from packages.core.models import DemoGoalRequest
from packages.db.init_db import initialize_database
from packages.db.repositories import CryoSwarmRepository
from packages.orchestration.runner import run_demo_campaign


st.set_page_config(page_title="CryoSwarm-Q", layout="wide")

settings = get_settings()
st.title("CryoSwarm-Q")
st.caption(
    "Hardware-aware multi-agent orchestration for neutral-atom experiment campaigns."
)
st.write(
    "This dashboard runs the initial research-grade prototype locally, stores campaign "
    "artifacts in MongoDB Atlas, and exposes ranked experiment candidates and agent decisions."
)

if not settings.has_mongodb:
    st.error("MONGODB_URI is not configured. The dashboard cannot load campaign data.")
    st.stop()

initialize_database()
repository = CryoSwarmRepository(settings)

with st.form("demo-goal-form"):
    title = st.text_input(
        "Goal title",
        value="Robust neutral-atom benchmark sweep",
    )
    scientific_objective = st.text_area(
        "Scientific objective",
        value=(
            "Design a small neutral-atom experiment campaign that balances robustness "
            "and execution feasibility."
        ),
    )
    desired_atom_count = st.slider("Desired atom count", min_value=4, max_value=12, value=6)
    preferred_geometry = st.selectbox("Preferred geometry", ["mixed", "square", "line", "2d"])
    submitted = st.form_submit_button("Run demo campaign")

if submitted:
    summary = run_demo_campaign(
        request=DemoGoalRequest(
            title=title,
            scientific_objective=scientific_objective,
            desired_atom_count=desired_atom_count,
            preferred_geometry=preferred_geometry,
        ),
        repository=repository,
    )
    st.success(f"Campaign {summary.campaign.id} completed.")

latest_campaigns = repository.list_latest_campaigns(limit=5)
if not latest_campaigns:
    st.info("No campaigns stored yet. Run a demo campaign to populate the dashboard.")
    st.stop()

campaign_rows = [
    {
        "campaign_id": campaign.id,
        "goal_id": campaign.goal_id,
        "status": campaign.status.value,
        "candidate_count": campaign.candidate_count,
        "top_candidate_id": campaign.top_candidate_id,
    }
    for campaign in latest_campaigns
]
st.subheader("Latest Campaigns")
st.dataframe(campaign_rows, use_container_width=True)

selected_campaign_id = st.selectbox(
    "Select campaign",
    options=[campaign.id for campaign in latest_campaigns],
)

selected_campaign = next(item for item in latest_campaigns if item.id == selected_campaign_id)
ranked_candidates = repository.list_candidates_for_campaign(selected_campaign_id)
decisions = repository.list_agent_decisions(selected_campaign_id)
reports = repository.list_robustness_reports(selected_campaign_id)

left_col, right_col = st.columns(2)
with left_col:
    st.subheader("Campaign Summary")
    st.json(selected_campaign.model_dump(mode="json"))
with right_col:
    st.subheader("Ranked Candidates")
    st.dataframe(
        [
            {
                "rank": candidate.final_rank,
                "sequence_candidate_id": candidate.sequence_candidate_id,
                "objective_score": candidate.objective_score,
                "robustness_score": candidate.robustness_score,
                "backend_choice": candidate.backend_choice.value,
            }
            for candidate in ranked_candidates
        ],
        use_container_width=True,
    )

st.subheader("Agent Decisions")
st.dataframe(
    [
        {
            "agent": decision.agent_name.value,
            "subject_id": decision.subject_id,
            "decision_type": decision.decision_type.value,
            "status": decision.status,
            "reasoning_summary": decision.reasoning_summary,
        }
        for decision in decisions
    ],
    use_container_width=True,
)

st.subheader("Robustness Reports")
st.dataframe(
    [
        {
            "sequence_candidate_id": report.sequence_candidate_id,
            "nominal_score": report.nominal_score,
            "robustness_score": report.robustness_score,
            "perturbation_average": report.perturbation_average,
        }
        for report in reports
    ],
    use_container_width=True,
)
