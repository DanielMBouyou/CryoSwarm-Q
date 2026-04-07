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
    load_sequence_candidates,
)
from apps.dashboard.components.latex_panels import pulse_formulas
from apps.dashboard.components.plotly_charts import parameter_space_scatter, pulse_waveform
from apps.dashboard.logic import generate_waveform

st.title("Pulse Sequence Studio")
st.caption("Waveform profiles, time evolution, parameter space.")

# --- Waveform Mathematics ---------------------------------------------------

pf = pulse_formulas()
with st.expander("Waveform Mathematics"):
    st.latex(pf["blackman"])
    st.latex(pf["linear_sweep"])
    st.latex(pf["pulse_area"])
    st.latex(pf["time_evolution"])
    st.latex(pf["trotter_suzuki_2"])
    st.markdown(
        "| Family | Omega(t) | delta(t) | Use Case |\n"
        "|--------|----------|----------|----------|\n"
        "| `constant_drive` | Constant | Constant | Rabi oscillation |\n"
        "| `global_ramp` | Linear ramp | Linear sweep | Gradual excitation |\n"
        "| `detuning_scan` | Constant | Linear sweep | Resonance scanning |\n"
        "| `adiabatic_sweep` | sin^2 envelope | Linear sweep | Adiabatic state prep |\n"
        "| `blackman_sweep` | Blackman window | Linear sweep | Low-sidelobe adiabatic |\n"
    )

# --- Campaign & Sequence selection ------------------------------------------

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

selected_cid = st.selectbox("Campaign", campaign_ids, index=default_idx, key="pulse_campaign")
sequences = load_sequence_candidates(selected_cid)

if sequences:
    families_present = list(set(s.get("sequence_family", "unknown") for s in sequences))
    family_filter = st.multiselect("Family filter", families_present, default=families_present)
    filtered = [s for s in sequences if s.get("sequence_family", "unknown") in family_filter]

    if filtered:
        seq_labels = [s.get("label", s.get("id", f"seq_{i}")) for i, s in enumerate(filtered)]
        seq_idx = st.selectbox("Sequence candidate", range(len(seq_labels)), format_func=lambda i: seq_labels[i])
        seq = filtered[seq_idx]

        family = seq.get("sequence_family", "constant_drive")
        amp = seq.get("amplitude", 5.0)
        det = seq.get("detuning", -10.0)
        dur = seq.get("duration_ns", 1000)

        t, omega_vals, delta_vals = generate_waveform(family, amp, -abs(det), abs(det), float(dur))
        fig = pulse_waveform(t, omega_vals, delta_vals, seq.get("label", "sequence"))
        st.plotly_chart(fig, use_container_width=True)

        # --- Sequence comparison table + parameter space ---

        left, right = st.columns(2)
        with left:
            st.markdown("**Sequence Comparison**")
            table_data = [
                {
                    "family": s.get("sequence_family", "?"),
                    "amplitude": s.get("amplitude", 0),
                    "detuning": s.get("detuning", 0),
                    "duration_ns": s.get("duration_ns", 0),
                    "predicted_cost": s.get("predicted_cost", 0),
                }
                for s in filtered
            ]
            st.dataframe(table_data, use_container_width=True)

        with right:
            ranked = load_ranked_candidates(selected_cid)
            score_lookup = {r.get("sequence_candidate_id"): r.get("objective_score", 0) for r in ranked}
            for s in filtered:
                s["objective_score"] = score_lookup.get(s.get("id", ""), s.get("predicted_cost", 0))
            fig = parameter_space_scatter(filtered)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sequences match the selected families.")
else:
    st.info("No sequence candidates for this campaign.")

# --- Waveform Generator -----------------------------------------------------

st.subheader("Waveform Generator")
st.markdown("Generate waveforms for arbitrary parameters (no simulation, just profiles).")

gen_l, gen_m, gen_r = st.columns(3)
with gen_l:
    gen_family = st.selectbox(
        "Family",
        ["constant_drive", "global_ramp", "detuning_scan", "adiabatic_sweep", "blackman_sweep"],
        key="gen_family",
    )
    gen_omega = st.slider("Omega_max (rad/us)", 0.1, 15.8, 5.0, 0.1, key="gen_omega")
with gen_m:
    gen_d_start = st.slider("delta_start (rad/us)", -126.0, 126.0, -10.0, 0.5, key="gen_dstart")
    gen_d_end = st.slider("delta_end (rad/us)", -126.0, 126.0, 10.0, 0.5, key="gen_dend")
with gen_r:
    gen_dur = st.slider("Duration (ns)", 100, 5000, 1000, 50, key="gen_dur")

if st.button("Generate"):
    t, omega_vals, delta_vals = generate_waveform(gen_family, gen_omega, gen_d_start, gen_d_end, float(gen_dur))
    fig = pulse_waveform(t, omega_vals, delta_vals, f"Generated ({gen_family})")
    st.plotly_chart(fig, use_container_width=True)
