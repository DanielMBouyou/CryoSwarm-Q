from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.dashboard.components.data_loaders import (
    load_all_memory,
    load_latest_campaigns,
    load_ranked_candidates,
    load_robustness_reports,
    load_sequence_candidates,
)
from apps.dashboard.components.latex_panels import campaign_formulas
from apps.dashboard.components.plotly_charts import (
    backend_distribution_stacked,
    campaign_timeline,
    cross_campaign_score_evolution,
    memory_tag_cloud,
    parameter_space_3d,
)
from apps.dashboard.logic import compute_pareto_front

st.title("Campaign Analytics")
st.caption("Cross-campaign trends, memory system, Pareto front.")

# --- Analytics Theory -------------------------------------------------------

cf = campaign_formulas()
with st.expander("Analytics Theory"):
    st.latex(cf["pareto_optimal"])
    st.latex(cf["cumulative_regret"])
    st.latex(cf["acquisition_ucb"])
    st.latex(cf["ranking_key"])

# --- Load all campaigns -----------------------------------------------------

campaigns = load_latest_campaigns(limit=50)
if not campaigns:
    st.info("No campaigns available. Run campaigns to populate analytics.")
    st.stop()

# --- Campaign Timeline ------------------------------------------------------

fig = campaign_timeline(campaigns)
st.plotly_chart(fig, use_container_width=True)

# --- Score Evolution + Backend Distribution ---------------------------------

left, right = st.columns(2)
with left:
    fig = cross_campaign_score_evolution(campaigns)
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = backend_distribution_stacked(campaigns)
    st.plotly_chart(fig, use_container_width=True)

# --- Parameter Space Explorer -----------------------------------------------

st.subheader("Parameter Space Explorer")

all_candidates: list[dict] = []
for campaign in campaigns:
    cid = campaign["id"]
    ranked = load_ranked_candidates(cid)
    seqs = load_sequence_candidates(cid)
    reports = load_robustness_reports(cid)

    report_lookup = {r.get("sequence_candidate_id"): r for r in reports}
    seq_lookup = {s.get("id"): s for s in seqs}

    for r in ranked:
        seq_id = r.get("sequence_candidate_id", "")
        seq = seq_lookup.get(seq_id, {})
        rob_report = report_lookup.get(seq_id, {})
        candidate = {
            "amplitude": seq.get("amplitude", 0),
            "detuning": seq.get("detuning", 0),
            "duration_ns": seq.get("duration_ns", 500),
            "sequence_family": seq.get("sequence_family", "unknown"),
            "objective_score": r.get("objective_score", 0),
            "robustness_score": r.get("robustness_score", 0),
            "worst_case_score": r.get("worst_case_score", 0),
            "campaign_id_ref": cid,
        }
        all_candidates.append(candidate)

if all_candidates:
    # Pareto front
    pareto_indices = compute_pareto_front(all_candidates)
    for i in pareto_indices:
        all_candidates[i]["is_pareto"] = True

    fig = parameter_space_3d(all_candidates)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"{len(pareto_indices)} Pareto-optimal candidates out of {len(all_candidates)} total.")
else:
    st.info("No candidates available across campaigns.")

# --- Memory System ----------------------------------------------------------

st.subheader("Memory System")
memory_records = load_all_memory(limit=50)

if memory_records:
    fig = memory_tag_cloud(memory_records)
    st.plotly_chart(fig, use_container_width=True)

    # Filters
    filter_l, filter_m, filter_r = st.columns(3)
    with filter_l:
        lesson_types = list(set(r.get("lesson_type", "unknown") for r in memory_records))
        selected_type = st.multiselect("Lesson type", lesson_types, default=lesson_types)
    with filter_m:
        mem_campaign_ids = list(set(r.get("campaign_id", "") for r in memory_records))
        selected_mem_cid = st.multiselect("Campaign", mem_campaign_ids, default=mem_campaign_ids)
    with filter_r:
        tag_search = st.text_input("Tag search", "")

    filtered_memory = [
        r for r in memory_records
        if r.get("lesson_type", "unknown") in selected_type
        and r.get("campaign_id", "") in selected_mem_cid
        and (not tag_search or tag_search.lower() in str(r.get("reusable_tags", [])).lower())
    ]

    for rec in filtered_memory:
        with st.expander(f"{rec.get('lesson_type', '?')}: {rec.get('summary', '')[:80]}"):
            st.json(rec)
else:
    st.info("No memory records available.")

# --- Campaign Statistics Summary --------------------------------------------

st.subheader("Campaign Statistics")
total = len(campaigns)
completed = sum(1 for c in campaigns if c.get("status") == "completed")
avg_candidates = sum(c.get("candidate_count", 0) for c in campaigns) / max(total, 1)

best_ever = 0.0
for c in campaigns:
    sr = c.get("summary_report", {})
    bs = sr.get("best_objective_score", 0)
    if bs and float(bs) > best_ever:
        best_ever = float(bs)

# Most used backend
from collections import Counter

backend_counter: Counter[str] = Counter()
for c in campaigns:
    mix = c.get("backend_mix", c.get("summary_report", {}).get("backend_mix", {}))
    for bt, count in mix.items():
        backend_counter[bt] += count
most_used = backend_counter.most_common(1)[0][0] if backend_counter else "N/A"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Campaigns", total)
c2.metric("Avg Candidates/Campaign", f"{avg_candidates:.1f}")
c3.metric("Best Score Ever", f"{best_ever:.3f}" if best_ever > 0 else "N/A")
c4.metric("Most Used Backend", most_used.replace("_", " ").title()[:20])

st.metric("Success Rate", f"{completed}/{total} ({100*completed/max(total,1):.0f}%)")
