from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.dashboard.components.data_loaders import (
    load_latest_campaigns,
    load_register_candidates,
    load_robustness_reports,
)
from apps.dashboard.components.latex_panels import hamiltonian_formulas
from apps.dashboard.components.plotly_charts import (
    blockade_graph,
    distance_histogram,
    register_scatter_2d,
    vdw_interaction_heatmap,
)
from packages.simulation.hamiltonian import (
    C6_RB87_70S,
    blockade_radius,
    find_maximum_independent_sets,
    interaction_graph,
    van_der_waals_matrix,
)

st.title("Register Physics Lab")
st.caption("Atom register geometry, interactions, blockade.")

# --- Physics Primer ---------------------------------------------------------

formulas = hamiltonian_formulas()
with st.expander("Physics Primer", expanded=False):
    st.latex(formulas["interaction"])
    st.latex(formulas["blockade_radius"])
    st.latex(formulas["c6_value"])
    st.markdown(
        "The **Rydberg blockade** prevents simultaneous excitation of two atoms "
        "closer than $R_b$. Within the blockade radius, the van der Waals "
        "interaction energy exceeds the Rabi coupling, suppressing double excitation."
    )

# --- Campaign & Register selection ------------------------------------------

campaigns = load_latest_campaigns(limit=20)
if not campaigns:
    st.info("No campaigns available. Run a campaign first.")
    st.stop()

campaign_ids = [c["id"] for c in campaigns]
default_idx = 0
if "selected_campaign_id" in st.session_state:
    sid = st.session_state["selected_campaign_id"]
    if sid in campaign_ids:
        default_idx = campaign_ids.index(sid)

selected_cid = st.selectbox("Campaign", campaign_ids, index=default_idx, key="reg_campaign")
registers = load_register_candidates(selected_cid)

if not registers:
    st.info("No register candidates for this campaign.")
    st.stop()

reg_labels = [r.get("label", r.get("id", f"reg_{i}")) for i, r in enumerate(registers)]
reg_idx = st.selectbox("Register candidate", range(len(reg_labels)), format_func=lambda i: reg_labels[i])
reg = registers[reg_idx]
coords = [tuple(c) for c in reg["coordinates"]]

# --- Interactive Omega slider -----------------------------------------------

omega = st.slider("Omega (rad/us)", min_value=0.1, max_value=15.8, value=5.0, step=0.1, key="reg_omega")
rb = blockade_radius(omega, C6_RB87_70S)

# --- Rydberg densities from reports -----------------------------------------

reports = load_robustness_reports(selected_cid)
rydberg_densities = None
for report in reports:
    nom_obs = report.get("nominal_observables", {})
    if "rydberg_density" in nom_obs:
        rd = nom_obs["rydberg_density"]
        if isinstance(rd, list) and len(rd) == len(coords):
            rydberg_densities = rd
            break

# --- Charts row 1 ----------------------------------------------------------

left, right = st.columns(2)
with left:
    fig = register_scatter_2d(coords, rb, reg.get("label", "register"), rydberg_densities)
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = vdw_interaction_heatmap(coords, C6_RB87_70S)
    st.plotly_chart(fig, use_container_width=True)

# --- Charts row 2 ----------------------------------------------------------

adj = interaction_graph(coords, omega, C6_RB87_70S)
mis_sets = find_maximum_independent_sets(adj)

left2, right2 = st.columns(2)
with left2:
    adj_list = adj.tolist()
    fig = blockade_graph(coords, adj_list, mis_sets if mis_sets else None)
    st.plotly_chart(fig, use_container_width=True)

with right2:
    fig = distance_histogram(coords, rb)
    st.plotly_chart(fig, use_container_width=True)

# --- Metrics ----------------------------------------------------------------

st.subheader("Register Metrics")
import numpy as np

pts = np.array(coords)
n = len(coords)
dists = []
for i in range(n):
    for j in range(i + 1, n):
        dists.append(float(np.sqrt(np.sum((pts[i] - pts[j]) ** 2))))
min_dist = min(dists) if dists else 0.0
blockade_pairs = int(adj.sum()) // 2

c1, c2, c3, c4 = st.columns(4)
c1.metric("Atom Count", n)
c2.metric("Min Distance (um)", f"{min_dist:.2f}")
c3.metric("Blockade Pairs", blockade_pairs)
c4.metric("Feasibility", f"{reg.get('feasibility_score', 0):.2f}")

with st.expander("Full register data"):
    st.json(reg)
