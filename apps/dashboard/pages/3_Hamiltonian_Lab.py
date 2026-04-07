from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.dashboard.components.data_loaders import (
    load_latest_campaigns,
    load_register_candidates,
    load_sequence_candidates,
)
from apps.dashboard.components.latex_panels import hamiltonian_formulas, observable_formulas
from apps.dashboard.components.plotly_charts import (
    bitstring_bar_chart,
    energy_spectrum,
    parametric_spectrum,
)
from packages.simulation.hamiltonian import (
    C6_RB87_70S,
    build_hamiltonian_matrix,
    ground_state,
    mis_bitstrings,
)
from packages.simulation.observables import (
    antiferromagnetic_order,
    bitstring_probabilities,
    entanglement_entropy,
    rydberg_density,
)

st.title("Hamiltonian Spectroscopy")
st.caption("Energy spectrum, eigenstates, parametric analysis.")

# --- Hamiltonian formulas ---------------------------------------------------

hf = hamiltonian_formulas()
with st.expander("Rydberg Hamiltonian", expanded=True):
    st.latex(hf["full_hamiltonian"])
    st.latex(hf["spectral_gap"])
    st.latex(hf["ipr"])
    st.latex(hf["adiabatic_condition"])
    st.markdown(
        "**Rabi coupling** drives ground-Rydberg transitions. "
        "**Detuning** offsets the Rydberg energy. "
        "**Van der Waals interaction** creates the blockade."
    )

# --- Campaign & Register selection ------------------------------------------

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

selected_cid = st.selectbox("Campaign", campaign_ids, index=default_idx, key="ham_campaign")
registers = load_register_candidates(selected_cid)

if not registers:
    st.info("No register candidates for this campaign.")
    st.stop()

reg_labels = [r.get("label", r.get("id", f"reg_{i}")) for i, r in enumerate(registers)]
reg_idx = st.selectbox("Register", range(len(reg_labels)), format_func=lambda i: reg_labels[i])
reg = registers[reg_idx]
coords = [tuple(c) for c in reg["coordinates"]]
n_atoms = len(coords)

# --- Controls ---------------------------------------------------------------

ctrl_l, ctrl_m, ctrl_r = st.columns(3)
with ctrl_l:
    omega = st.slider("Omega (rad/us)", 0.1, 15.8, 5.0, 0.1, key="ham_omega")
with ctrl_m:
    delta = st.slider("delta (rad/us)", -126.0, 126.0, -10.0, 0.5, key="ham_delta")
with ctrl_r:
    auto_from_seq = st.checkbox("Auto from best sequence", value=False)

if auto_from_seq:
    sequences = load_sequence_candidates(selected_cid)
    if sequences:
        best_seq = sequences[0]
        omega = float(best_seq.get("amplitude", omega))
        delta = float(best_seq.get("detuning", delta))
        st.info(f"Using best sequence params: Omega={omega:.2f}, delta={delta:.2f}")

# --- Diagonalisation --------------------------------------------------------

if n_atoms > 14:
    st.warning(f"Dense Hamiltonian limited to 14 atoms. Current register has {n_atoms}.")
    st.stop()

H = build_hamiltonian_matrix(coords, omega, delta, C6_RB87_70S)
eigvals_all = np.sort(np.linalg.eigvalsh(H).real)
eigvals, eigvecs = np.linalg.eigh(H)
order = np.argsort(eigvals.real)
eigvals = eigvals.real[order]
eigvecs = eigvecs[:, order]
psi0 = eigvecs[:, 0]

gap = float(eigvals[1] - eigvals[0]) if len(eigvals) >= 2 else 0.0
dim = 2 ** n_atoms
ipr = float(np.sum(np.abs(psi0) ** 4))

# --- Spectrum + Metrics -----------------------------------------------------

left, right = st.columns([0.6, 0.4])
with left:
    fig = energy_spectrum(eigvals_all[:20].tolist())
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.metric("Spectral Gap (dE)", f"{gap:.4f} rad/us")
    st.metric("Hilbert Dimension", f"2^{n_atoms} = {dim}")
    st.metric("Ground State IPR", f"{ipr:.4f}")

# --- Bitstrings + Observables -----------------------------------------------

bs_probs = bitstring_probabilities(psi0, n_atoms, top_k=20)
mis_bs = mis_bitstrings(coords, omega, C6_RB87_70S)

left2, right2 = st.columns(2)
with left2:
    fig = bitstring_bar_chart(bs_probs, mis_bs)
    st.plotly_chart(fig, use_container_width=True)

with right2:
    st.markdown("**State Composition**")
    rd = rydberg_density(psi0, n_atoms)
    total_frac = float(rd.mean())
    st.metric("Rydberg Fraction", f"{total_frac:.4f}")

    ee = entanglement_entropy(psi0, n_atoms)
    st.metric("Entanglement Entropy", f"{ee:.4f}")

    af = antiferromagnetic_order(psi0, n_atoms)
    st.metric("AF Order Parameter", f"{af:.4f}")

# --- Parametric Detuning Sweep ----------------------------------------------

st.subheader("Parametric Detuning Sweep")
if n_atoms > 10:
    st.warning(f"Parametric sweep limited to 10 atoms for interactivity. Current: {n_atoms}.")
else:
    n_levels = min(8, dim)
    delta_vals = np.linspace(-20, 20, 50)

    @st.cache_data(show_spinner="Computing parametric sweep...")
    def _compute_sweep(_coords_tuple: tuple, _omega: float, _n_levels: int) -> tuple[list[float], list[list[float]]]:
        coords_list = [tuple(c) for c in _coords_tuple]
        curves: list[list[float]] = [[] for _ in range(_n_levels)]
        d_vals = np.linspace(-20, 20, 50)
        for d in d_vals:
            H_d = build_hamiltonian_matrix(coords_list, _omega, float(d), C6_RB87_70S)
            ev = np.sort(np.linalg.eigvalsh(H_d).real)[:_n_levels]
            for k in range(len(ev)):
                curves[k].append(float(ev[k]))
            for k in range(len(ev), _n_levels):
                curves[k].append(float("nan"))
        return d_vals.tolist(), curves

    d_list, curves = _compute_sweep(tuple(tuple(c) for c in coords), omega, n_levels)
    fig = parametric_spectrum(d_list, curves, omega)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "The parametric sweep reveals avoided crossings and phase transitions. "
        "At large positive detuning, the ground state approaches MIS solutions."
    )
