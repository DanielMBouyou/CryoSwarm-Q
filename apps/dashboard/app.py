from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import apps.dashboard.components  # noqa: F401 — injects st.secrets into os.environ

st.set_page_config(page_title="CryoSwarm-Q Dashboard", layout="wide", page_icon="atom_symbol")

# ---------------------------------------------------------------------------
# Custom CSS — dark, modern, research-grade aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* ---- Header area ---- */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 800px;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 24px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00d4aa;
        box-shadow: 0 0 20px rgba(0, 212, 170, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #00d4aa !important;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* ---- Navigation cards ---- */
    .nav-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .nav-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, #00d4aa, #00b4d8);
        opacity: 0;
        transition: opacity 0.25s ease;
    }
    .nav-card:hover {
        border-color: #30363d;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .nav-card:hover::before {
        opacity: 1;
    }
    .nav-card .card-icon {
        font-size: 1.5rem;
        margin-bottom: 6px;
    }
    .nav-card .card-title {
        color: #e6edf3;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 4px;
    }
    .nav-card .card-desc {
        color: #8b949e;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    /* ---- Section dividers ---- */
    .section-label {
        color: #484f58;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }

    /* ---- Status badge ---- */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .status-connected {
        background: rgba(0, 212, 170, 0.12);
        color: #00d4aa;
        border: 1px solid rgba(0, 212, 170, 0.25);
    }
    .status-warning {
        background: rgba(255, 183, 77, 0.12);
        color: #ffb74d;
        border: 1px solid rgba(255, 183, 77, 0.25);
    }

    /* ---- Expanders ---- */
    details {
        border: 1px solid #21262d !important;
        border-radius: 10px !important;
        background: #161b22 !important;
    }
    details summary {
        font-weight: 600;
    }

    /* ---- DataFrames ---- */
    .stDataFrame {
        border: 1px solid #21262d;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 16px rgba(0, 212, 170, 0.15);
    }

    /* ---- Hide default decoration ---- */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

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
        st.sidebar.markdown(
            f'<span class="status-badge status-connected">MongoDB connected &middot; {len(campaigns)} campaigns</span>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.sidebar.error(f"MongoDB error: {exc}")
        campaigns = []
else:
    st.sidebar.markdown(
        '<span class="status-badge status-warning">MongoDB not configured</span>',
        unsafe_allow_html=True,
    )
    campaigns = []

st.sidebar.markdown("---")
st.sidebar.markdown("[Theory Reference](8_Theory_Reference)")

# --- Hero section -----------------------------------------------------------

st.markdown('<div class="hero-title">CryoSwarm-Q Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    "Hardware-aware multi-agent system for autonomous experiment design "
    "in neutral-atom quantum computing. From register geometry and "
    "Hamiltonian spectroscopy to robustness analysis and ML monitoring."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("")

# --- Quick stats ------------------------------------------------------------

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

# --- Navigation cards -------------------------------------------------------

st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)

pages = [
    ("Campaign Control", "Launch, monitor, and inspect experiment campaigns.", "&#x1F680;"),
    ("Register Physics", "Atom geometry, van der Waals interactions, blockade graph.", "&#x269B;"),
    ("Hamiltonian Lab", "Energy spectrum, eigenstates, parametric analysis.", "&#x1F9EA;"),
    ("Pulse Studio", "Waveform profiles, time evolution, parameter space.", "&#x1F4C8;"),
    ("Robustness Arena", "Noise sensitivity analysis and robustness scoring.", "&#x1F6E1;"),
    ("ML Observatory", "Surrogate models, PPO training, strategy bandit.", "&#x1F916;"),
    ("Campaign Analytics", "Cross-campaign trends, memory system, Pareto front.", "&#x1F4CA;"),
    ("Theory Reference", "Complete mathematical reference for CryoSwarm-Q.", "&#x1F4D6;"),
    ("Training Tracker", "Live training runs, checkpoints, and convergence.", "&#x23F1;"),
    ("Neutral Atom Academy", "Interactive physics course with live demos.", "&#x1F393;"),
]

cols = st.columns(2)
for i, (title, desc, icon) in enumerate(pages):
    with cols[i % 2]:
        st.markdown(
            f"""
            <div class="nav-card">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
