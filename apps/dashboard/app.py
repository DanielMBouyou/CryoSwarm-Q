from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.dashboard.logic import (
    build_campaign_table,
    build_decision_table,
    build_ranked_table,
    build_register_lookup_from_documents,
    select_noise_sensitivity_data,
    select_robustness_chart_data,
)
from packages.core.config import get_settings
from packages.core.models import DemoGoalRequest, RegisterCandidate, RobustnessReport
from packages.db.init_db import initialize_database
from packages.db.repositories import CryoSwarmRepository
from packages.orchestration.runner import run_demo_campaign


def _load_register_candidates(repository: CryoSwarmRepository, campaign_id: str) -> dict[str, RegisterCandidate]:
    cursor = repository.collections["register_candidates"].find({"campaign_id": campaign_id})
    return build_register_lookup_from_documents(list(cursor))


def _render_register_plot(candidate: RegisterCandidate) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs, ys = zip(*candidate.coordinates)
    ax.scatter(xs, ys, s=100, c="royalblue")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal")
    ax.set_title(f"Register: {candidate.label}")
    for x_value, y_value in candidate.coordinates:
        circle = plt.Circle(
            (x_value, y_value),
            candidate.blockade_radius_um,
            fill=False,
            color="crimson",
            linestyle="--",
            alpha=0.25,
        )
        ax.add_patch(circle)
    st.pyplot(fig)
    plt.close(fig)


def _render_robustness_chart(reports: list[RobustnessReport]) -> None:
    labels, nominal, average, worst = select_robustness_chart_data(reports)
    if not labels:
        return

    x_positions = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([x - width for x in x_positions], nominal, width=width, label="Nominal")
    ax.bar(x_positions, average, width=width, label="Average")
    ax.bar([x + width for x in x_positions], worst, width=width, label="Worst")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Robustness Comparison Across Top Candidates")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def _render_noise_sensitivity(report: RobustnessReport) -> None:
    labels, values = select_noise_sensitivity_data(report)
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(labels, values, marker="o", linewidth=2, color="darkgreen")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Observable score")
    ax.set_title(f"Noise Sensitivity for {report.sequence_candidate_id}")
    st.pyplot(fig)
    plt.close(fig)


st.set_page_config(page_title="CryoSwarm-Q", layout="wide")

settings = get_settings()
st.title("CryoSwarm-Q")
st.caption(
    "Hardware-aware multi-agent orchestration for neutral-atom experiment campaigns."
)
st.write(
    "This dashboard runs the research-grade prototype locally, stores campaign "
    "artifacts in MongoDB Atlas, and exposes ranked candidates, agent decisions, "
    "robustness metrics, and register-level physics views."
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

campaign_rows = build_campaign_table(latest_campaigns)
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
register_lookup = _load_register_candidates(repository, selected_campaign_id)
report_lookup = {report.sequence_candidate_id: report for report in reports}

left_col, right_col = st.columns(2)
with left_col:
    st.subheader("Campaign Summary")
    st.json(selected_campaign.model_dump(mode="json"))
with right_col:
    st.subheader("Ranked Candidates")
    st.dataframe(build_ranked_table(ranked_candidates), use_container_width=True)

st.subheader("Register Visualisation")
if ranked_candidates:
    candidate_options = {
        f"#{candidate.final_rank or '?'} {candidate.sequence_candidate_id}": candidate
        for candidate in ranked_candidates[:5]
    }
    selected_candidate_label = st.selectbox("Candidate for register view", list(candidate_options.keys()))
    selected_candidate = candidate_options[selected_candidate_label]
    register_candidate = register_lookup.get(selected_candidate.register_candidate_id)
    if register_candidate:
        _render_register_plot(register_candidate)
    else:
        st.info("No register geometry stored for this candidate yet.")
else:
    st.info("No ranked candidates available for visualisation.")

st.subheader("Robustness Comparison")
_render_robustness_chart(reports)

st.subheader("Noise Sensitivity")
if ranked_candidates:
    top_report = report_lookup.get(ranked_candidates[0].sequence_candidate_id)
    if top_report:
        _render_noise_sensitivity(top_report)
    else:
        st.info("No robustness report found for the top candidate.")
else:
    st.info("Run a campaign to generate robustness reports.")

st.subheader("Agent Decisions")
st.dataframe(build_decision_table(decisions), use_container_width=True)

st.subheader("Robustness Reports")
st.dataframe(
    [
        {
            "sequence_candidate_id": report.sequence_candidate_id,
            "nominal_score": report.nominal_score,
            "robustness_score": report.robustness_score,
            "perturbation_average": report.perturbation_average,
            "worst_case_score": report.worst_case_score,
        }
        for report in reports
    ],
    use_container_width=True,
)
