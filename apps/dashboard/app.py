from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(page_title="CryoSwarm-Q Dashboard", layout="wide", page_icon="atom_symbol")

# --- Sidebar ---------------------------------------------------------------

st.sidebar.title("CryoSwarm-Q")
st.sidebar.caption("Hardware-aware multi-agent orchestration for neutral-atom experiment design.")

from packages.core.config import get_settings

settings = get_settings()

if settings.has_mongodb:
    try:
        from apps.dashboard.components.data_loaders import get_repository

        repo = get_repository()
        campaigns = repo.list_latest_campaigns(limit=100)
        st.sidebar.success(f"MongoDB connected ({len(campaigns)} campaigns)")
    except Exception as exc:
        st.sidebar.error(f"MongoDB error: {exc}")
        campaigns = []
else:
    st.sidebar.warning("MONGODB_URI not configured")
    campaigns = []

st.sidebar.markdown("---")
st.sidebar.markdown("[Theory Reference](8_Theory_Reference)")

# --- Main page --------------------------------------------------------------

st.title("CryoSwarm-Q Dashboard")
st.markdown(
    "CryoSwarm-Q is a hardware-aware multi-agent system for autonomous "
    "experiment design in neutral-atom quantum computing. This dashboard "
    "exposes every stage of the pipeline: from register geometry and "
    "Hamiltonian spectroscopy to robustness analysis and ML monitoring."
)

# Quick stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Campaigns", len(campaigns))
with col2:
    best_score = 0.0
    for c in campaigns:
        sr = getattr(c, "summary_report", {}) or {}
        bs = sr.get("best_objective_score", 0)
        if bs and float(bs) > best_score:
            best_score = float(bs)
    st.metric("Best Score", f"{best_score:.3f}" if best_score > 0 else "N/A")
with col3:
    if campaigns:
        latest = campaigns[0]
        st.metric("Latest Campaign", latest.id[-12:])
    else:
        st.metric("Latest Campaign", "N/A")

st.markdown("---")
st.subheader("Navigate")

pages = [
    ("1. Campaign Control", "Launch, monitor, and inspect experiment campaigns."),
    ("2. Register Physics", "Atom geometry, van der Waals interactions, blockade graph."),
    ("3. Hamiltonian Lab", "Energy spectrum, eigenstates, parametric analysis."),
    ("4. Pulse Studio", "Waveform profiles, time evolution, parameter space."),
    ("5. Robustness Arena", "Noise sensitivity analysis and robustness scoring."),
    ("6. ML Observatory", "Surrogate models, PPO training, strategy bandit."),
    ("7. Campaign Analytics", "Cross-campaign trends, memory system, Pareto front."),
    ("8. Theory Reference", "Complete mathematical reference for CryoSwarm-Q."),
]

cols = st.columns(2)
for i, (title, desc) in enumerate(pages):
    with cols[i % 2]:
        st.markdown(f"**{title}**  \n{desc}")
