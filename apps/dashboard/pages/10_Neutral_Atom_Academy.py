"""Neutral Atom Physics Academy -- Interactive course for researchers.

A comprehensive, PhD-grade walkthrough of neutral-atom quantum computing
physics, from single trapped atoms to combinatorial optimisation via MIS.
Each section pairs rigorous LaTeX theory with live Plotly demos backed by
the real CryoSwarm-Q simulation stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.dashboard.components.latex_panels import (
    hamiltonian_formulas,
    mis_formulas,
    observable_formulas,
    pulse_formulas,
)
from apps.dashboard.logic import generate_waveform
from packages.simulation.hamiltonian import (
    C6_RB87_70S,
    blockade_radius,
    build_hamiltonian_matrix,
    find_maximum_independent_sets,
    interaction_graph,
    pairwise_distances,
    van_der_waals_matrix,
)
from packages.simulation.observables import (
    antiferromagnetic_order,
    bitstring_probabilities,
    connected_correlation,
    entanglement_entropy,
    pair_correlation,
    rydberg_density,
    total_rydberg_fraction,
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_BLUE = "#1f77b4"
_RED = "#d62728"
_GREEN = "#2ca02c"
_GOLD = "#ff7f0e"
_PURPLE = "#9467bd"
_CYAN = "#17becf"
_PINK = "#e377c2"
_TEMPLATE = "plotly_white"

st.set_page_config(page_title="Neutral Atom Academy", layout="wide", page_icon="🔬")

st.title("Neutral Atom Physics Academy")
st.caption(
    "An interactive course from trapped atoms to quantum optimisation. "
    "Every demo runs real CryoSwarm-Q physics."
)

# ===================================================================
#  HELPERS -- geometry builders
# ===================================================================


def _make_coords(layout: str, n_atoms: int, spacing: float) -> list[tuple[float, float]]:
    """Generate atom coordinates for common register geometries."""
    if layout == "line":
        return [(i * spacing, 0.0) for i in range(n_atoms)]
    if layout == "square":
        side = int(np.ceil(np.sqrt(n_atoms)))
        coords = []
        for idx in range(n_atoms):
            row, col = divmod(idx, side)
            coords.append((col * spacing, row * spacing))
        return coords
    if layout == "triangular":
        coords = []
        idx = 0
        row = 0
        while idx < n_atoms:
            cols_in_row = int(np.ceil(np.sqrt(n_atoms)))
            for col in range(cols_in_row):
                if idx >= n_atoms:
                    break
                x = col * spacing + (row % 2) * spacing / 2
                y = row * spacing * np.sqrt(3) / 2
                coords.append((x, y))
                idx += 1
            row += 1
        return coords
    if layout == "ring":
        radius = spacing * n_atoms / (2 * np.pi) if n_atoms > 1 else 0.0
        return [
            (radius * np.cos(2 * np.pi * i / n_atoms), radius * np.sin(2 * np.pi * i / n_atoms))
            for i in range(n_atoms)
        ]
    if layout == "zigzag":
        return [(i * spacing, (i % 2) * spacing * 0.6) for i in range(n_atoms)]
    if layout == "honeycomb":
        coords = []
        idx = 0
        row = 0
        while idx < n_atoms:
            for col in range(int(np.ceil(np.sqrt(n_atoms)))):
                if idx >= n_atoms:
                    break
                x = col * spacing * 1.5
                y = row * spacing * np.sqrt(3) / 2
                if col % 2 == 1:
                    y += spacing * np.sqrt(3) / 4
                coords.append((x, y))
                idx += 1
            row += 1
        return coords
    return [(i * spacing, 0.0) for i in range(n_atoms)]


# ===================================================================
#  COURSE 1.1 -- What is a trapped neutral atom?
# ===================================================================

st.divider()
st.header("1.1 -- What is a trapped neutral atom?")

col_text, col_demo = st.columns([3, 2])
with col_text:
    st.markdown("""
**The actor: Rubidium-87** ($^{87}$Rb)

Neutral-atom quantum computing uses individual atoms of **Rubidium-87** held in place
by tightly focused laser beams called **optical tweezers**. Each atom acts as a qubit
with two key states:

| State | Label | Physical meaning |
|---|---|---|
| **Ground** $\\vert g \\rangle$ | $\\vert 0 \\rangle$ | Electron close to the nucleus, atom is "quiet" |
| **Rydberg** $\\vert r \\rangle$ | $\\vert 1 \\rangle$ | Electron far from the nucleus, atom is "excited" |

**Why Rubidium?**
- Well-characterised energy levels (decades of spectroscopy data)
- Strong Rydberg-Rydberg interactions for entanglement
- Mature laser-cooling technology (Nobel Prize 1997)
- Long-lived Rydberg states (lifetime $\\sim 100\\,\\mu$s for $\\vert 70S_{1/2} \\rangle$)

**The Rydberg state** is special: the electron orbits at a distance that scales as
$r_n \\propto n^2 a_0$, where $n$ is the principal quantum number and $a_0 \\approx 0.053\\,$nm
is the Bohr radius. At $n=70$, the electron is $\\sim 0.26\\,\\mu$m from the nucleus -- that's
nearly **5000 times** the ground-state orbit.
""")

with col_demo:
    st.subheader("Demo: Electron orbit vs. excitation level")
    n_level = st.slider(
        "Principal quantum number $n$",
        min_value=1,
        max_value=70,
        value=5,
        step=1,
        key="demo_1_1_n",
    )

    # Orbital radius in Bohr radii
    r_bohr = n_level**2
    r_nm = r_bohr * 0.0529  # in nm

    theta = np.linspace(0, 2 * np.pi, 300)
    r_plot = r_bohr * (1 + 0.15 * np.sin(3 * theta))  # slight shape for visual interest

    fig_orbit = go.Figure()
    # nucleus
    fig_orbit.add_trace(go.Scatterpolar(
        r=[0],
        theta=[0],
        mode="markers",
        marker=dict(size=14, color=_RED, symbol="circle"),
        name="Nucleus",
        showlegend=True,
    ))
    # electron orbit
    fig_orbit.add_trace(go.Scatterpolar(
        r=r_plot,
        theta=np.degrees(theta),
        mode="lines",
        line=dict(color=_BLUE, width=2),
        name=f"Electron orbit (n={n_level})",
        fill="toself",
        fillcolor="rgba(31,119,180,0.08)",
        showlegend=True,
    ))
    # electron position marker
    fig_orbit.add_trace(go.Scatterpolar(
        r=[r_bohr],
        theta=[45],
        mode="markers",
        marker=dict(size=10, color=_BLUE, symbol="circle"),
        name="Electron",
        showlegend=True,
    ))
    fig_orbit.update_layout(
        template=_TEMPLATE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(r_bohr * 1.3, 10)]),
            angularaxis=dict(showticklabels=False),
        ),
        height=380,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig_orbit, use_container_width=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Radius", f"{r_bohr} $a_0$")
    m2.metric("Physical size", f"{r_nm:.1f} nm")
    m3.metric("Energy", f"$-13.6/{n_level**2:.0f}$ eV" if n_level > 0 else "")

    if n_level >= 50:
        st.success(
            f"At $n={n_level}$, the electron is {r_nm:.0f} nm from the nucleus -- "
            "this is the **Rydberg regime**. The atom becomes enormous and interacts "
            "strongly with its neighbours."
        )
    elif n_level >= 10:
        st.info(f"At $n={n_level}$, the orbit is already {r_bohr}x the ground state.")

# ===================================================================
#  COURSE 1.2 -- Tweezer arrays
# ===================================================================

st.divider()
st.header("1.2 -- The optical tweezer array")

col_text2, col_demo2 = st.columns([3, 2])
with col_text2:
    st.markdown("""
**Placing atoms one by one with lasers**

An **optical tweezer** is a tightly focused laser beam that creates a potential well
deep enough to trap a single atom. By arranging many tweezers in a pattern, we build
a **register** -- the quantum equivalent of a circuit board.

**Key hardware constraints:**
- **Minimum spacing**: $\\sim 4\\,\\mu$m between traps (limited by optical diffraction)
- **Maximum atoms**: current devices support 100--1000 atoms (Pasqal, QuEra)
- **Geometry**: tweezers can be arranged in **any 2D pattern** -- line, square,
  triangular, ring, zigzag, honeycomb

**Why geometry matters:**

The physical arrangement determines which atoms interact with each other.
Two atoms separated by $4\\,\\mu$m experience a van der Waals interaction of
$U \\approx 211\\,$rad/$\\mu$s, while atoms at $8\\,\\mu$m feel only $U \\approx 3.3\\,$rad/$\\mu$s
(a factor of 64 weaker due to the $1/r^6$ scaling). The geometry **is** the problem
Hamiltonian.

**The recipe:**
1. Load atoms from a magneto-optical trap (MOT)
2. Image the array to detect which sites are filled
3. Rearrange atoms to fill gaps (atom sorting)
4. Execute the quantum program (pulse sequence)
5. Measure by fluorescence imaging
""")

with col_demo2:
    st.subheader("Demo: Build your register")
    demo_layout = st.selectbox(
        "Geometry",
        ["line", "square", "triangular", "ring", "zigzag", "honeycomb"],
        index=1,
        key="demo_1_2_layout",
    )
    demo_n = st.slider("Number of atoms", 2, 15, 6, key="demo_1_2_n")
    demo_spacing = st.slider("Spacing ($\\mu$m)", 4.0, 15.0, 7.0, step=0.5, key="demo_1_2_sp")

    coords_12 = _make_coords(demo_layout, demo_n, demo_spacing)
    xs = [c[0] for c in coords_12]
    ys = [c[1] for c in coords_12]

    dists = pairwise_distances(coords_12)
    min_dist = float(np.min(dists[dists > 0])) if len(coords_12) > 1 else demo_spacing
    feasible = min_dist >= 4.0

    fig_reg = go.Figure()
    # Tweezer beams (circles)
    for i, (x, y) in enumerate(coords_12):
        fig_reg.add_shape(
            type="circle",
            x0=x - 1.5, y0=y - 1.5, x1=x + 1.5, y1=y + 1.5,
            line=dict(color="rgba(31,119,180,0.3)", width=1),
            fillcolor="rgba(31,119,180,0.06)",
        )
    fig_reg.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(size=16, color=_BLUE if feasible else _RED),
        text=[str(i) for i in range(len(coords_12))],
        textposition="top center",
        textfont=dict(size=10),
        name="Atoms",
    ))
    x_range = max(xs) - min(xs) if xs else 10
    y_range = max(ys) - min(ys) if ys else 10
    pad = max(x_range, y_range, 10) * 0.25
    fig_reg.update_layout(
        template=_TEMPLATE,
        xaxis=dict(
            title="x (um)",
            range=[min(xs) - pad, max(xs) + pad],
            scaleanchor="y",
        ),
        yaxis=dict(title="y (um)", range=[min(ys) - pad, max(ys) + pad]),
        height=400,
        margin=dict(l=40, r=40, t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Atoms", demo_n)
    r2.metric("Min distance", f"{min_dist:.1f} um")
    r3.metric("Feasible", "Yes" if feasible else "No (< 4 um)")

    if not feasible:
        st.error(
            f"Minimum inter-atom distance is {min_dist:.1f} um, below the 4 um hardware limit. "
            "Increase spacing or reduce atom count."
        )
    else:
        st.success(
            f"Register is **hardware-feasible**. {demo_n} atoms in a {demo_layout} "
            f"arrangement with {min_dist:.1f} um minimum spacing."
        )

# ===================================================================
#  COURSE 1.3 -- Rydberg blockade
# ===================================================================

st.divider()
st.header("1.3 -- The Rydberg blockade mechanism")

col_text3, col_demo3 = st.columns([3, 2])
with col_text3:
    formulas_h = hamiltonian_formulas()
    st.markdown("""
**The quantum exclusion principle of Rydberg atoms**

When two atoms are close enough, the van der Waals interaction energy $U_{ij}$ exceeds
the laser coupling strength $\\Omega$. In this regime, **only one of the two atoms
can be excited** to the Rydberg state at a time -- this is the **Rydberg blockade**.

The critical distance is the **blockade radius**:
""")
    st.latex(formulas_h["blockade_radius"])
    st.markdown("""
**Physical intuition:** Think of two people trying to sit in the same seat on a
crowded bus. If they're within the blockade radius, only one can "sit" (get excited).
This creates a natural **constraint graph** where edges connect blockaded atom pairs.

**Why this matters for quantum computing:**
- The blockade creates **entanglement** between atoms -- measuring one tells you about the other
- The constraint graph maps directly to **optimisation problems** (see Course 1.7)
- By tuning $\\Omega$, you control the blockade radius and thus the problem structure

For $^{87}$Rb $\\vert 70S_{1/2}\\rangle$ with $C_6 = 862\\,690$ rad$\\cdot\\mu$m$^6/\\mu$s:
- $\\Omega = 1$ rad/$\\mu$s $\\Rightarrow R_b \\approx 9.8\\,\\mu$m
- $\\Omega = 5$ rad/$\\mu$s $\\Rightarrow R_b \\approx 7.5\\,\\mu$m
- $\\Omega = 15$ rad/$\\mu$s $\\Rightarrow R_b \\approx 6.2\\,\\mu$m
""")

with col_demo3:
    st.subheader("Demo: Watch the blockade graph form")
    bl_layout = st.selectbox(
        "Geometry", ["line", "square", "triangular", "ring"],
        index=1, key="demo_1_3_layout",
    )
    bl_n = st.slider("Atoms", 3, 10, 5, key="demo_1_3_n")
    bl_spacing = st.slider("Spacing ($\\mu$m)", 4.0, 15.0, 7.0, step=0.5, key="demo_1_3_sp")
    bl_omega = st.slider("Rabi frequency $\\Omega$ (rad/$\\mu$s)", 0.5, 20.0, 5.0, step=0.5, key="demo_1_3_omega")

    coords_13 = _make_coords(bl_layout, bl_n, bl_spacing)
    r_b = blockade_radius(bl_omega)
    adj = interaction_graph(coords_13, bl_omega)
    dists_13 = pairwise_distances(coords_13)

    xs13 = [c[0] for c in coords_13]
    ys13 = [c[1] for c in coords_13]

    fig_bl = go.Figure()

    # Blockade radius circles (transparent)
    for i, (x, y) in enumerate(coords_13):
        fig_bl.add_shape(
            type="circle",
            x0=x - r_b, y0=y - r_b, x1=x + r_b, y1=y + r_b,
            line=dict(color="rgba(214,39,40,0.2)", width=1, dash="dot"),
            fillcolor="rgba(214,39,40,0.04)",
        )

    # Blockade edges
    edge_x, edge_y = [], []
    n_edges = 0
    for i in range(bl_n):
        for j in range(i + 1, bl_n):
            if adj[i, j]:
                edge_x.extend([coords_13[i][0], coords_13[j][0], None])
                edge_y.extend([coords_13[i][1], coords_13[j][1], None])
                n_edges += 1
    if edge_x:
        fig_bl.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color=_RED, width=2),
            name="Blockade edges",
            hoverinfo="skip",
        ))

    # Atoms
    fig_bl.add_trace(go.Scatter(
        x=xs13, y=ys13,
        mode="markers+text",
        marker=dict(size=18, color=_BLUE, line=dict(width=2, color="white")),
        text=[str(i) for i in range(bl_n)],
        textposition="middle center",
        textfont=dict(size=9, color="white"),
        name="Atoms",
    ))

    x_range13 = max(xs13) - min(xs13) if xs13 else 10
    y_range13 = max(ys13) - min(ys13) if ys13 else 10
    pad13 = max(x_range13, y_range13, 10) * 0.3 + r_b
    fig_bl.update_layout(
        template=_TEMPLATE,
        xaxis=dict(
            title="x (um)", scaleanchor="y",
            range=[min(xs13) - pad13, max(xs13) + pad13],
        ),
        yaxis=dict(
            title="y (um)",
            range=[min(ys13) - pad13, max(ys13) + pad13],
        ),
        height=450,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )
    st.plotly_chart(fig_bl, use_container_width=True)

    b1, b2, b3 = st.columns(3)
    b1.metric("Blockade radius", f"{r_b:.2f} um")
    b2.metric("Blockade edges", n_edges)
    b3.metric("Max possible edges", bl_n * (bl_n - 1) // 2)

    if n_edges == 0:
        st.warning("No blockade edges -- atoms are too far apart or $\\Omega$ is too large. "
                   "Decrease spacing or increase $\\Omega$.")
    elif n_edges == bl_n * (bl_n - 1) // 2:
        st.info("**Fully connected** blockade graph -- every atom blocks every other. "
                "This is the extreme blockade regime.")

# ===================================================================
#  COURSE 1.4 -- The Rydberg Hamiltonian
# ===================================================================

st.divider()
st.header("1.4 -- The Rydberg Hamiltonian: the boss final")

col_text4, col_demo4 = st.columns([3, 2])
with col_text4:
    st.markdown("""
**The equation that describes EVERYTHING**

In quantum mechanics, the **Hamiltonian** $\\hat{H}$ is the operator that encodes all
the physics of a system. For neutral atoms, it has exactly **three terms**:
""")
    st.latex(formulas_h["full_hamiltonian"])
    st.markdown("""
| Term | Symbol | Role | Analogy |
|---|---|---|---|
| **Drive** | $\\frac{\\Omega}{2} \\sum_i \\hat{\\sigma}_x^{(i)}$ | Laser coupling ground $\\leftrightarrow$ Rydberg | The **fire** under the pot -- drives excitation |
| **Detuning** | $-\\delta \\sum_i \\hat{n}_i$ | Energy cost of being in Rydberg state | The **seasoning** -- controls which state is favoured |
| **Interaction** | $\\sum_{i<j} U_{ij} \\hat{n}_i \\hat{n}_j$ | Van der Waals repulsion between Rydberg atoms | The **chemistry** between ingredients |

**How to read the spectrum:**
- The **eigenvalues** $E_k$ are the allowed energy levels of the system
- The **ground state** $\\vert \\psi_0 \\rangle$ (lowest energy) is what the system naturally relaxes to
- The **spectral gap** $\\Delta E = E_1 - E_0$ tells you how "protected" the ground state is

When $\\delta \\gg \\Omega$: the Rydberg state is energetically favourable, atoms want to excite
(but are constrained by the blockade).

When $\\delta \\ll -\\Omega$: the ground state is favoured, atoms stay de-excited.

When $\\delta \\approx 0$: the system is in a superposition -- this is the quantum regime.
""")
    st.latex(formulas_h["spectral_gap"])

with col_demo4:
    st.subheader("Demo: Tune the Hamiltonian, see the spectrum")
    h_n = st.slider("Atoms (warning: $2^N$ Hilbert space)", 2, 8, 4, key="demo_1_4_n")
    h_layout = st.selectbox("Layout", ["line", "square", "ring"], index=0, key="demo_1_4_layout")
    h_spacing = st.slider("Spacing ($\\mu$m)", 5.0, 12.0, 7.0, step=0.5, key="demo_1_4_sp")
    h_omega = st.slider("$\\Omega$ (rad/$\\mu$s)", 0.1, 15.0, 5.0, step=0.1, key="demo_1_4_omega")
    h_delta = st.slider("$\\delta$ (rad/$\\mu$s)", -30.0, 30.0, 0.0, step=0.5, key="demo_1_4_delta")

    coords_14 = _make_coords(h_layout, h_n, h_spacing)
    dim_14 = 2 ** h_n

    if h_n <= 8:
        H_mat = build_hamiltonian_matrix(coords_14, h_omega, h_delta, max_atoms_dense=8)
        eigvals = np.linalg.eigvalsh(H_mat.real)
        eigvals_sorted = np.sort(eigvals)

        n_show = min(20, len(eigvals_sorted))

        fig_spec = go.Figure()
        # Energy levels as horizontal lines
        for k in range(n_show):
            color = _RED if k == 0 else (_GOLD if k == 1 else "rgba(100,100,100,0.4)")
            width = 3 if k <= 1 else 1
            fig_spec.add_trace(go.Scatter(
                x=[0.2, 0.8],
                y=[eigvals_sorted[k], eigvals_sorted[k]],
                mode="lines",
                line=dict(color=color, width=width),
                name=f"$E_{{{k}}}$ = {eigvals_sorted[k]:.2f}" if k < 5 else None,
                showlegend=k < 5,
                hovertemplate=f"E_{k} = {eigvals_sorted[k]:.4f} rad/us",
            ))

        gap = eigvals_sorted[1] - eigvals_sorted[0] if len(eigvals_sorted) > 1 else 0.0
        # Gap annotation
        if gap > 0.01:
            fig_spec.add_annotation(
                x=0.85, y=(eigvals_sorted[0] + eigvals_sorted[1]) / 2,
                text=f"Gap = {gap:.3f}",
                showarrow=False,
                font=dict(size=12, color=_GOLD),
            )

        fig_spec.update_layout(
            template=_TEMPLATE,
            xaxis=dict(visible=False, range=[0, 1.2]),
            yaxis=dict(title="Energy (rad/us)"),
            height=420,
            margin=dict(l=60, r=40, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.12),
        )
        st.plotly_chart(fig_spec, use_container_width=True)

        s1, s2, s3 = st.columns(3)
        s1.metric("Hilbert space dim", f"$2^{{{h_n}}}$ = {dim_14}")
        s2.metric("Ground state energy", f"{eigvals_sorted[0]:.3f}")
        s3.metric("Spectral gap", f"{gap:.4f} rad/$\\mu$s")

        if gap < 0.1:
            st.warning(
                "Spectral gap is very small -- the ground state is **nearly degenerate**. "
                "Adiabatic protocols will struggle here."
            )
        elif gap > 5.0:
            st.success(f"Large spectral gap ({gap:.2f}) -- ground state is well-protected.")

# ===================================================================
#  COURSE 1.5 -- Pulse sequences
# ===================================================================

st.divider()
st.header("1.5 -- Pulse sequences: controlling atoms in time")

col_text5, col_demo5 = st.columns([3, 2])
with col_text5:
    formulas_p = pulse_formulas()
    st.markdown("""
**Not just ON/OFF -- sculpted control in time**

A pulse sequence describes how the laser parameters $\\Omega(t)$ and $\\delta(t)$
change during the experiment. This is the **program** that runs on the quantum computer.

The system evolves according to the time-dependent Schrodinger equation:
""")
    st.latex(formulas_p["time_evolution"])
    st.markdown("""
**The 5 waveform families and WHY each exists:**

| Family | $\\Omega(t)$ profile | Purpose | Analogy |
|---|---|---|---|
| **constant_drive** | Flat | Rabi oscillations -- the "Hello World" | Leaving the stove on constant heat |
| **global_ramp** | Linear ramp | Gradual turn-on to avoid shocks | Slowly turning up the heat |
| **detuning_scan** | Flat $\\Omega$, swept $\\delta$ | Scan through resonance | Tuning a radio dial |
| **adiabatic_sweep** | $\\sin^2$ envelope | Follow the ground state slowly | Heating water slowly to avoid boiling over |
| **blackman_sweep** | Blackman window | Premium adiabatic with minimal spectral leakage | Precision oven with PID control |

**The Blackman window** is particularly important:
""")
    st.latex(formulas_p["blackman"])
    st.markdown("""
It has the special property that its Fourier transform has very small sidelobes,
meaning it excites fewer unwanted transitions than a simple ramp.

**The adiabatic condition** -- when can we follow the ground state?
""")
    st.latex(formulas_p["pulse_area"])

with col_demo5:
    st.subheader("Demo: Design your pulse sequence")
    p_family = st.selectbox(
        "Waveform family",
        ["constant_drive", "adiabatic_sweep", "blackman_sweep", "global_ramp", "detuning_scan"],
        index=1,
        key="demo_1_5_family",
    )
    p_omega = st.slider("$\\Omega_{\\max}$ (rad/$\\mu$s)", 0.5, 15.0, 5.0, step=0.5, key="demo_1_5_omega")
    p_d_start = st.slider("$\\delta_{\\text{start}}$ (rad/$\\mu$s)", -30.0, 10.0, -15.0, step=0.5, key="demo_1_5_ds")
    p_d_end = st.slider("$\\delta_{\\text{end}}$ (rad/$\\mu$s)", -10.0, 30.0, 15.0, step=0.5, key="demo_1_5_de")
    p_duration = st.slider("Duration (ns)", 500, 6000, 3000, step=100, key="demo_1_5_dur")

    t_vals, omega_vals, delta_vals = generate_waveform(
        p_family, p_omega, p_d_start, p_d_end, float(p_duration), n_points=300,
    )

    fig_pulse = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Drive $\\Omega(t)$", "Detuning $\\delta(t)$"],
    )
    fig_pulse.add_trace(go.Scatter(
        x=t_vals, y=omega_vals,
        mode="lines",
        line=dict(color=_BLUE, width=2.5),
        name="Omega",
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.1)",
    ), row=1, col=1)
    fig_pulse.add_trace(go.Scatter(
        x=t_vals, y=delta_vals,
        mode="lines",
        line=dict(color=_RED, width=2.5),
        name="Delta",
        fill="tozeroy",
        fillcolor="rgba(214,39,40,0.1)",
    ), row=2, col=1)
    fig_pulse.update_xaxes(title_text="Time (ns)", row=2, col=1)
    fig_pulse.update_yaxes(title_text="rad/us", row=1, col=1)
    fig_pulse.update_yaxes(title_text="rad/us", row=2, col=1)
    fig_pulse.update_layout(
        template=_TEMPLATE,
        height=420,
        margin=dict(l=50, r=30, t=40, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_pulse, use_container_width=True)

    # Pulse area
    dt_ns = t_vals[1] - t_vals[0] if len(t_vals) > 1 else 1.0
    pulse_area = sum(omega_vals) * dt_ns / 1000.0  # convert ns to us
    pi_pulses = pulse_area / np.pi

    pa1, pa2, pa3 = st.columns(3)
    pa1.metric("Pulse area", f"{pulse_area:.2f} rad$\\cdot\\mu$s")
    pa2.metric("$\\pi$-pulses", f"{pi_pulses:.2f}")
    pa3.metric("Sweep span", f"{abs(p_d_end - p_d_start):.1f} rad/$\\mu$s")

# ===================================================================
#  COURSE 1.6 -- Observables
# ===================================================================

st.divider()
st.header("1.6 -- Quantum observables: what do we measure?")

col_text6, col_demo6 = st.columns([2, 3])
with col_text6:
    formulas_o = observable_formulas()
    st.markdown("""
**After running the experiment, what do we learn?**

Quantum mechanics doesn't give us a single answer -- it gives us **probabilities**.
Here are the key observables we extract from the final quantum state $\\vert \\psi \\rangle$:

**1. Rydberg density** $\\langle \\hat{n}_i \\rangle$ -- the probability each atom
is excited. This is our primary readout.
""")
    st.latex(formulas_o["rydberg_density"])
    st.markdown("**2. Pair correlations** -- are two atoms *jointly* excited?")
    st.latex(formulas_o["connected_correlation"])
    st.markdown("""
Negative $g_{ij}^{(2)}$ means **anti-bunching**: if atom $i$ is excited,
atom $j$ tends *not* to be. This is the signature of the Rydberg blockade.

**3. Entanglement entropy** -- how "quantum" is the state?
""")
    st.latex(formulas_o["entanglement_entropy"])
    st.markdown("""
$S_A = 0$ means the state is a simple product state (classical-like).
$S_A > 0$ means the subsystems are **entangled** -- measuring one part
affects the other, no matter how far apart.

**4. Antiferromagnetic order** -- is there an alternating pattern?
""")
    st.latex(formulas_o["antiferromagnetic_order"])
    st.markdown("""
$m_{AF} = 1$: perfect alternating excited/ground pattern (|1010...> or |0101...>).
$m_{AF} = 0$: completely disordered.

**5. Bitstring probabilities** -- the raw measurement outcomes.
""")
    st.latex(formulas_o["bitstring_probability"])

with col_demo6:
    st.subheader("Demo: Simulate a register and see all observables")
    obs_layout = st.selectbox(
        "Layout", ["line", "square", "ring", "triangular"],
        index=0, key="demo_1_6_layout",
    )
    obs_n = st.slider("Atoms", 2, 7, 4, key="demo_1_6_n")
    obs_sp = st.slider("Spacing ($\\mu$m)", 5.0, 12.0, 7.0, step=0.5, key="demo_1_6_sp")
    obs_omega = st.slider("$\\Omega$", 0.5, 15.0, 5.0, step=0.5, key="demo_1_6_omega")
    obs_delta = st.slider("$\\delta$", -20.0, 20.0, 5.0, step=0.5, key="demo_1_6_delta")

    coords_16 = _make_coords(obs_layout, obs_n, obs_sp)

    if obs_n <= 7:
        H_16 = build_hamiltonian_matrix(coords_16, obs_omega, obs_delta, max_atoms_dense=8)
        eigvals_16, eigvecs_16 = np.linalg.eigh(H_16.real)
        ground_state = eigvecs_16[:, 0].astype(np.complex128)

        # Compute all observables
        densities = rydberg_density(ground_state, obs_n)
        tot_frac = total_rydberg_fraction(ground_state, obs_n)
        corr = connected_correlation(ground_state, obs_n)
        entropy = entanglement_entropy(ground_state, obs_n)
        af_order = antiferromagnetic_order(ground_state, obs_n)
        top_bits = bitstring_probabilities(ground_state, obs_n, top_k=8)

        # Summary metrics
        om1, om2, om3, om4 = st.columns(4)
        om1.metric("Rydberg fraction", f"{tot_frac:.3f}")
        om2.metric("AF order", f"{af_order:.3f}")
        om3.metric("Entanglement $S_A$", f"{entropy:.3f}")
        om4.metric("Hilbert dim", f"{2**obs_n}")

        # --- Plots ---
        obs_tab1, obs_tab2, obs_tab3 = st.tabs(["Rydberg Density", "Correlations", "Bitstrings"])

        with obs_tab1:
            fig_dens = go.Figure()
            fig_dens.add_trace(go.Bar(
                x=[f"Atom {i}" for i in range(obs_n)],
                y=densities.tolist(),
                marker_color=[_RED if d > 0.5 else _BLUE for d in densities],
                text=[f"{d:.3f}" for d in densities],
                textposition="outside",
            ))
            fig_dens.update_layout(
                template=_TEMPLATE,
                yaxis=dict(title="$\\langle n_i \\rangle$", range=[0, 1.05]),
                height=320,
                margin=dict(l=50, r=30, t=20, b=40),
            )
            st.plotly_chart(fig_dens, use_container_width=True)

        with obs_tab2:
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.tolist(),
                x=[f"Atom {i}" for i in range(obs_n)],
                y=[f"Atom {i}" for i in range(obs_n)],
                colorscale="RdBu_r",
                zmid=0,
                text=[[f"{corr[i, j]:.4f}" for j in range(obs_n)] for i in range(obs_n)],
                texttemplate="%{text}",
                colorbar=dict(title="$g_{ij}$"),
            ))
            fig_corr.update_layout(
                template=_TEMPLATE,
                height=350,
                margin=dict(l=60, r=30, t=20, b=40),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption(
                "**Negative values** (blue) = anti-bunching (blockade signature). "
                "**Positive values** (red) = bunching."
            )

        with obs_tab3:
            if top_bits:
                bs_labels = [b[0] for b in top_bits]
                bs_probs = [b[1] for b in top_bits]
                fig_bits = go.Figure()
                fig_bits.add_trace(go.Bar(
                    x=bs_labels,
                    y=bs_probs,
                    marker_color=_PURPLE,
                    text=[f"{p:.3f}" for p in bs_probs],
                    textposition="outside",
                ))
                fig_bits.update_layout(
                    template=_TEMPLATE,
                    xaxis=dict(title="Bitstring $|b_0 b_1 \\cdots\\rangle$"),
                    yaxis=dict(title="Probability", range=[0, max(bs_probs) * 1.2 + 0.01]),
                    height=320,
                    margin=dict(l=50, r=30, t=20, b=40),
                )
                st.plotly_chart(fig_bits, use_container_width=True)
                st.caption(
                    "Each bitstring shows which atoms are excited (1) or in ground state (0). "
                    "The most probable bitstrings reveal the system's preferred configurations."
                )
            else:
                st.info("No significant bitstrings found.")

# ===================================================================
#  COURSE 1.7 -- Maximum Independent Set (MIS)
# ===================================================================

st.divider()
st.header("1.7 -- Maximum Independent Set: from physics to optimisation")

col_text7, col_demo7 = st.columns([3, 2])
with col_text7:
    formulas_m = mis_formulas()
    st.markdown("""
**The bridge between quantum physics and computer science**

The **Maximum Independent Set (MIS)** problem asks: given a graph $G = (V, E)$,
find the largest subset $S \\subseteq V$ of vertices such that **no two vertices
in $S$ are connected by an edge**.
""")
    st.latex(formulas_m["mis_definition"])
    st.markdown("""
**Why it matters:**
- MIS is **NP-hard** on classical computers -- no known polynomial-time algorithm
- It appears in wireless network scheduling, protein folding, circuit design, social network analysis
- It is equivalent to Maximum Clique on the complement graph

**The quantum connection:**

When we tune the Rydberg Hamiltonian to the regime $\\delta > 0$ (favouring excitation)
with the blockade active, the **ground state of the Hamiltonian naturally encodes
the MIS solution**:
""")
    st.latex(formulas_m["mis_cost_function"])
    st.markdown("""
The first term rewards excitation ($-\\sum n_i \\leftrightarrow -\\delta \\sum \\hat{n}_i$).
The second term penalises adjacent excitations ($\\alpha \\sum n_i n_j \\leftrightarrow U_{ij} \\hat{n}_i \\hat{n}_j$).

When $U_{ij} \\gg \\delta$, minimising $C_{MIS}$ is equivalent to finding the MIS.
The quantum system does this "for free" by relaxing to its ground state.

**MIS overlap** measures how well the quantum state solves the problem:
""")
    st.latex(formulas_m["mis_overlap"])
    st.markdown("""
$P_{MIS} = 1$ means the quantum state is a perfect superposition of MIS solutions.
$P_{MIS} \\approx 0$ means the system didn't find the optimal solution.
""")

with col_demo7:
    st.subheader("Demo: Draw a graph, watch Rydberg atoms solve MIS")
    mis_layout = st.selectbox(
        "Geometry", ["line", "square", "ring", "triangular", "zigzag"],
        index=1, key="demo_1_7_layout",
    )
    mis_n = st.slider("Atoms", 3, 10, 6, key="demo_1_7_n")
    mis_sp = st.slider("Spacing ($\\mu$m)", 5.0, 12.0, 7.0, step=0.5, key="demo_1_7_sp")
    mis_omega = st.slider("$\\Omega$", 1.0, 15.0, 5.0, step=0.5, key="demo_1_7_omega")

    coords_17 = _make_coords(mis_layout, mis_n, mis_sp)
    adj_17 = interaction_graph(coords_17, mis_omega)

    # Find MIS
    mis_solutions = find_maximum_independent_sets(adj_17, max_results=10)
    mis_size = len(mis_solutions[0]) if mis_solutions else 0

    # Also diagonalise to get quantum MIS overlap
    mis_delta_val = 10.0  # positive delta to favour excitation
    if mis_n <= 8:
        H_17 = build_hamiltonian_matrix(coords_17, mis_omega, mis_delta_val, max_atoms_dense=8)
        eigvals_17, eigvecs_17 = np.linalg.eigh(H_17.real)
        gs_17 = eigvecs_17[:, 0].astype(np.complex128)
        top_bits_17 = bitstring_probabilities(gs_17, mis_n, top_k=10)

        # MIS overlap
        mis_bitstring_set = set()
        for sol in mis_solutions:
            bits = ["0"] * mis_n
            for idx in sol:
                bits[idx] = "1"
            mis_bitstring_set.add("".join(bits))

        mis_overlap_val = sum(p for bs, p in top_bits_17 if bs in mis_bitstring_set)

    xs17 = [c[0] for c in coords_17]
    ys17 = [c[1] for c in coords_17]

    # Build the visualisation
    fig_mis = go.Figure()

    # Edges
    edge_x17, edge_y17 = [], []
    for i in range(mis_n):
        for j in range(i + 1, mis_n):
            if adj_17[i, j]:
                edge_x17.extend([coords_17[i][0], coords_17[j][0], None])
                edge_y17.extend([coords_17[i][1], coords_17[j][1], None])
    if edge_x17:
        fig_mis.add_trace(go.Scatter(
            x=edge_x17, y=edge_y17, mode="lines",
            line=dict(color="rgba(150,150,150,0.5)", width=1.5),
            name="Blockade edges",
            hoverinfo="skip",
        ))

    # Colour atoms by MIS membership (first solution)
    first_mis = set(mis_solutions[0]) if mis_solutions else set()
    colors_17 = [_RED if i in first_mis else _BLUE for i in range(mis_n)]
    labels_17 = [f"{i} (MIS)" if i in first_mis else str(i) for i in range(mis_n)]

    fig_mis.add_trace(go.Scatter(
        x=xs17, y=ys17,
        mode="markers+text",
        marker=dict(
            size=22,
            color=colors_17,
            line=dict(width=2, color="white"),
        ),
        text=labels_17,
        textposition="top center",
        textfont=dict(size=9),
        name="Atoms",
    ))

    x_range17 = max(xs17) - min(xs17) if xs17 else 10
    y_range17 = max(ys17) - min(ys17) if ys17 else 10
    pad17 = max(x_range17, y_range17, 10) * 0.25
    fig_mis.update_layout(
        template=_TEMPLATE,
        xaxis=dict(title="x (um)", scaleanchor="y",
                   range=[min(xs17) - pad17, max(xs17) + pad17]),
        yaxis=dict(title="y (um)",
                   range=[min(ys17) - pad17, max(ys17) + pad17]),
        height=420,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )
    st.plotly_chart(fig_mis, use_container_width=True)

    mi1, mi2, mi3 = st.columns(3)
    mi1.metric("MIS size", mis_size)
    mi2.metric("Solutions found", len(mis_solutions))
    if mis_n <= 8:
        mi3.metric("Quantum MIS overlap", f"{mis_overlap_val:.3f}")

    # Show MIS solutions
    if mis_solutions:
        with st.expander(f"All MIS solutions (size {mis_size})", expanded=False):
            for idx, sol in enumerate(mis_solutions):
                bits = ["0"] * mis_n
                for s in sol:
                    bits[s] = "1"
                bitstring = "".join(bits)
                st.code(f"Solution {idx + 1}: atoms {sol}  -->  |{bitstring}>")

    # Quantum state analysis
    if mis_n <= 8:
        st.markdown("---")
        st.markdown("**Quantum ground state analysis** ($\\delta = 10$ rad/$\\mu$s to favour excitation):")

        if top_bits_17:
            qcol1, qcol2 = st.columns(2)
            with qcol1:
                fig_qbits = go.Figure()
                qbs = [b[0] for b in top_bits_17[:8]]
                qps = [b[1] for b in top_bits_17[:8]]
                bar_colors = [_RED if bs in mis_bitstring_set else _PURPLE for bs in qbs]
                fig_qbits.add_trace(go.Bar(
                    x=qbs, y=qps,
                    marker_color=bar_colors,
                    text=[f"{p:.3f}" for p in qps],
                    textposition="outside",
                ))
                fig_qbits.update_layout(
                    template=_TEMPLATE,
                    xaxis=dict(title="Bitstring"),
                    yaxis=dict(title="Probability", range=[0, max(qps) * 1.3 + 0.01]),
                    height=280,
                    margin=dict(l=50, r=30, t=20, b=60),
                )
                st.plotly_chart(fig_qbits, use_container_width=True)
                st.caption("Red bars = MIS solutions. Purple bars = non-MIS states.")

            with qcol2:
                densities_17 = rydberg_density(gs_17, mis_n)
                fig_dens17 = go.Figure()
                fig_dens17.add_trace(go.Bar(
                    x=[f"Atom {i}" for i in range(mis_n)],
                    y=densities_17.tolist(),
                    marker_color=[_RED if i in first_mis else _BLUE for i in range(mis_n)],
                    text=[f"{d:.3f}" for d in densities_17],
                    textposition="outside",
                ))
                fig_dens17.update_layout(
                    template=_TEMPLATE,
                    yaxis=dict(title="Excitation prob.", range=[0, 1.05]),
                    height=280,
                    margin=dict(l=50, r=30, t=20, b=40),
                )
                st.plotly_chart(fig_dens17, use_container_width=True)
                st.caption("Excitation probability per atom. MIS atoms (red) should be highly excited.")

        if mis_overlap_val > 0.7:
            st.success(
                f"Quantum MIS overlap = **{mis_overlap_val:.3f}** -- the quantum system "
                f"strongly favours the MIS solution!"
            )
        elif mis_overlap_val > 0.3:
            st.info(
                f"Quantum MIS overlap = **{mis_overlap_val:.3f}** -- partial solution. "
                "Try increasing $\\delta$ or adjusting geometry."
            )
        else:
            st.warning(
                f"Quantum MIS overlap = **{mis_overlap_val:.3f}** -- the ground state "
                "doesn't align well with MIS. The blockade structure may not match."
            )

# ===================================================================
#  Footer
# ===================================================================

st.divider()
st.markdown("""
---
**Neutral Atom Academy** -- Part of the CryoSwarm-Q Dashboard.

All simulations use the CryoSwarm-Q physics stack:
[`packages.simulation.hamiltonian`](packages/simulation/hamiltonian.py) for Hamiltonian construction,
[`packages.simulation.observables`](packages/simulation/observables.py) for quantum measurements.

Constants: $C_6 = 862\\,690$ rad$\\cdot\\mu$m$^6/\\mu$s for $^{87}$Rb $|70S_{1/2}\\rangle$.
Hilbert space dimension: $2^N$ (exact diagonalisation up to 8 atoms in demos).
""")
