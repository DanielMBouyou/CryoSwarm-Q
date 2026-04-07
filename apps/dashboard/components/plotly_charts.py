"""Reusable Plotly chart builders for CryoSwarm-Q dashboard pages.

Every function returns a ``plotly.graph_objects.Figure``.
Pages call ``st.plotly_chart(fig, use_container_width=True)``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BLUE = "#1f77b4"
_RED = "#d62728"
_GREEN = "#2ca02c"
_GOLD = "#ffd700"
_ORANGE = "#ff7f0e"
_PURPLE = "#9467bd"
_TEMPLATE = "plotly_white"


# -- Page 1: Campaign Control ------------------------------------------------


def pipeline_gantt(decisions: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart showing agent phase durations."""
    if not decisions:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Pipeline Phase Timeline (no data)")
        return fig

    agents: list[str] = []
    starts: list[float] = []
    durations: list[float] = []
    colors: list[str] = []
    palette = [_BLUE, _GREEN, _ORANGE, _RED, _PURPLE, "#17becf", "#bcbd22", "#e377c2"]

    base_time = None
    for d in decisions:
        ts = d.get("created_at")
        if ts and base_time is None:
            base_time = ts if isinstance(ts, (int, float)) else 0

    for i, d in enumerate(decisions):
        agent = d.get("agent_name", "unknown")
        agents.append(agent)
        starts.append(float(i) * 0.5)
        durations.append(0.4)
        colors.append(palette[i % len(palette)])

    fig = go.Figure(
        go.Bar(
            y=agents,
            x=durations,
            base=starts,
            orientation="h",
            marker_color=colors,
            hovertemplate="Agent: %{y}<br>Phase offset: %{base:.1f}s<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Pipeline Phase Timeline",
        xaxis_title="Relative phase index",
        yaxis_title="Agent",
        showlegend=False,
        height=max(300, len(agents) * 40),
    )
    return fig


def candidate_funnel(
    campaign_data: dict[str, Any],
    n_registers: int,
    n_sequences: int,
    n_evaluated: int,
    n_ranked: int,
) -> go.Figure:
    """Funnel chart showing candidate survival through pipeline phases."""
    stages = ["Registers", "Sequences", "Evaluated", "Ranked"]
    values = [n_registers, n_sequences, n_evaluated, n_ranked]

    fig = go.Figure(
        go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial",
            marker_color=[_BLUE, _ORANGE, _GREEN, _GOLD],
            hovertemplate="%{y}: %{x} candidates<br>%{percentInitial:.0%} of initial<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Candidate Pipeline Funnel",
        height=320,
    )
    return fig


# -- Page 2: Register Physics ------------------------------------------------


def register_scatter_2d(
    coordinates: list[tuple[float, float]],
    blockade_radius: float,
    label: str,
    rydberg_densities: list[float] | None = None,
) -> go.Figure:
    """Interactive 2D atom register layout with blockade radius circles."""
    xs = [c[0] for c in coordinates]
    ys = [c[1] for c in coordinates]

    fig = go.Figure()

    # Blockade circles
    for i, (x, y) in enumerate(coordinates):
        theta = np.linspace(0, 2 * np.pi, 60)
        cx = x + blockade_radius * np.cos(theta)
        cy = y + blockade_radius * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=cx.tolist(),
                y=cy.tolist(),
                mode="lines",
                line=dict(color="rgba(214,39,40,0.25)", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Atom positions
    if rydberg_densities is not None:
        marker = dict(
            size=14,
            color=rydberg_densities,
            colorscale="Viridis",
            colorbar=dict(title="<n_i>"),
            cmin=0,
            cmax=1,
        )
    else:
        marker = dict(size=14, color=_BLUE)

    hover_texts = []
    for i, (x, y) in enumerate(coordinates):
        txt = f"q{i}: ({x:.2f}, {y:.2f}) um"
        if rydberg_densities is not None:
            txt += f"<br>Rydberg density: {rydberg_densities[i]:.4f}"
        hover_texts.append(txt)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=marker,
            text=[f"q{i}" for i in range(len(coordinates))],
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text",
            name="Atoms",
        )
    )

    fig.update_layout(
        template=_TEMPLATE,
        title=f"Register Layout: {label}",
        xaxis_title="x (um)",
        yaxis_title="y (um)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        showlegend=False,
        height=500,
    )
    return fig


def vdw_interaction_heatmap(
    coordinates: list[tuple[float, float]],
    c6: float,
) -> go.Figure:
    """Van der Waals interaction matrix as annotated heatmap."""
    n = len(coordinates)
    pts = np.array(coordinates)
    dists = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1))

    U = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] > 0:
                U[i, j] = c6 / dists[i, j] ** 6
                U[j, i] = U[i, j]

    labels = [f"q{i}" for i in range(n)]
    U_display = np.where(U > 0, U, np.nan)
    log_U = np.where(U > 0, np.log10(U + 1e-10), 0)

    hover_text = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("self")
            else:
                row.append(f"U({i},{j})={U[i,j]:.2f} rad/us<br>d={dists[i,j]:.2f} um")
        hover_text.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=log_U.tolist(),
            x=labels,
            y=labels,
            colorscale="Viridis",
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(title="log10(U)"),
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Van der Waals Interaction Matrix (log scale)",
        height=500,
    )
    return fig


def blockade_graph(
    coordinates: list[tuple[float, float]],
    adjacency: list[list[bool]],
    mis_sets: list[list[int]] | None = None,
) -> go.Figure:
    """Blockade graph as network visualization."""
    n = len(coordinates)
    xs = [c[0] for c in coordinates]
    ys = [c[1] for c in coordinates]

    fig = go.Figure()

    # Edges
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i][j]:
                edge_x.extend([xs[i], xs[j], None])
                edge_y.extend([ys[i], ys[j], None])

    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="rgba(150,150,150,0.5)", width=1.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Node degree
    degrees = [sum(1 for j in range(n) if j != i and adjacency[i][j]) for i in range(n)]

    # MIS membership
    mis_nodes: set[int] = set()
    if mis_sets and len(mis_sets) > 0:
        mis_nodes = set(mis_sets[0])

    colors = [_GOLD if i in mis_nodes else _BLUE for i in range(n)]
    sizes = [max(12, 8 + d * 4) for d in degrees]

    hover_texts = [
        f"q{i} | degree={degrees[i]}" + (" | MIS" if i in mis_nodes else "")
        for i in range(n)
    ]

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="black")),
            text=[f"q{i}" for i in range(n)],
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text",
            name="Atoms",
        )
    )

    fig.update_layout(
        template=_TEMPLATE,
        title="Blockade Graph" + (" (MIS in gold)" if mis_nodes else ""),
        xaxis_title="x (um)",
        yaxis_title="y (um)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        showlegend=False,
        height=500,
    )
    return fig


def distance_histogram(
    coordinates: list[tuple[float, float]],
    blockade_radius: float,
) -> go.Figure:
    """Histogram of pairwise distances with blockade radius marker."""
    n = len(coordinates)
    pts = np.array(coordinates)
    dists_list = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sqrt(np.sum((pts[i] - pts[j]) ** 2)))
            dists_list.append(d)

    if not dists_list:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Pairwise Distances (no data)")
        return fig

    blockaded = [d for d in dists_list if d < blockade_radius]
    non_blockaded = [d for d in dists_list if d >= blockade_radius]

    fig = go.Figure()
    if blockaded:
        fig.add_trace(
            go.Histogram(x=blockaded, name="Blockaded", marker_color=_RED, opacity=0.7)
        )
    if non_blockaded:
        fig.add_trace(
            go.Histogram(x=non_blockaded, name="Non-blockaded", marker_color=_BLUE, opacity=0.7)
        )
    fig.add_vline(
        x=blockade_radius,
        line_dash="dash",
        line_color="black",
        annotation_text=f"R_b = {blockade_radius:.2f} um",
        annotation_position="top right",
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Pairwise Distance Distribution",
        xaxis_title="Distance (um)",
        yaxis_title="Count",
        barmode="stack",
        height=400,
    )
    return fig


# -- Page 3: Hamiltonian Lab -------------------------------------------------


def energy_spectrum(eigenvalues: list[float], n_show: int = 20) -> go.Figure:
    """Bar chart of lowest eigenvalues with spectral gap annotation."""
    vals = eigenvalues[:n_show]
    if not vals:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Energy Spectrum (no data)")
        return fig

    indices = list(range(len(vals)))
    colors = [_GOLD if i == 0 else _BLUE for i in indices]

    fig = go.Figure(
        go.Bar(
            x=indices,
            y=vals,
            marker_color=colors,
            hovertemplate="E_%{x} = %{y:.4f} rad/us<extra></extra>",
        )
    )

    if len(vals) >= 2:
        gap = vals[1] - vals[0]
        fig.add_annotation(
            x=0.5,
            y=(vals[0] + vals[1]) / 2,
            text=f"DE = {gap:.4f}",
            showarrow=True,
            arrowhead=2,
            ax=60,
            ay=0,
        )

    fig.update_layout(
        template=_TEMPLATE,
        title="Energy Spectrum",
        xaxis_title="Eigenstate index",
        yaxis_title="Energy (rad/us)",
        showlegend=False,
        height=450,
    )
    return fig


def bitstring_bar_chart(
    bitstrings: list[tuple[str, float]],
    mis_bitstrings: list[str] | None = None,
) -> go.Figure:
    """Bar chart of top-k bitstring probabilities."""
    if not bitstrings:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Bitstring Probabilities (no data)")
        return fig

    mis_set = set(mis_bitstrings or [])
    labels = [bs for bs, _ in bitstrings]
    probs = [p for _, p in bitstrings]
    colors = [_GOLD if bs in mis_set else _BLUE for bs in labels]

    hover_texts = []
    for bs, p in bitstrings:
        hw = bs.count("1")
        dec = int(bs, 2)
        hover_texts.append(f"{bs}<br>P={p:.4f}<br>decimal={dec}<br>HW={hw}")

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=probs,
            marker_color=colors,
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Bitstring Probabilities" + (" (MIS in gold)" if mis_set else ""),
        xaxis_title="Bitstring",
        yaxis_title="Probability",
        xaxis_tickangle=-45,
        showlegend=False,
        height=450,
    )
    return fig


def parametric_spectrum(
    delta_values: list[float],
    eigenvalue_curves: list[list[float]],
    omega: float,
) -> go.Figure:
    """Parametric energy level diagram (delta sweep)."""
    if not delta_values or not eigenvalue_curves:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Parametric Spectrum (no data)")
        return fig

    palette = [_BLUE, _RED, _GREEN, _ORANGE, _PURPLE, "#17becf", "#bcbd22", "#e377c2"]
    fig = go.Figure()
    for i, curve in enumerate(eigenvalue_curves):
        fig.add_trace(
            go.Scatter(
                x=delta_values,
                y=curve,
                mode="lines",
                name=f"E_{i}",
                line=dict(color=palette[i % len(palette)], width=2 if i == 0 else 1),
                hovertemplate=f"E_{i}<br>delta=%{{x:.2f}}<br>E=%{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        template=_TEMPLATE,
        title=f"Parametric Energy Levels (Omega = {omega:.2f} rad/us)",
        xaxis_title="Detuning delta (rad/us)",
        yaxis_title="Energy (rad/us)",
        height=500,
    )
    return fig


# -- Page 4: Pulse Studio ----------------------------------------------------


def pulse_waveform(
    time_ns: list[float],
    omega_t: list[float],
    delta_t: list[float],
    label: str,
) -> go.Figure:
    """Dual-axis pulse waveform plot."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=time_ns,
            y=omega_t,
            mode="lines",
            name="Omega(t)",
            line=dict(color=_BLUE, width=2),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.15)",
            hovertemplate="t=%{x:.0f} ns<br>Omega=%{y:.3f} rad/us<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=time_ns,
            y=delta_t,
            mode="lines",
            name="delta(t)",
            line=dict(color=_RED, width=2),
            hovertemplate="t=%{x:.0f} ns<br>delta=%{y:.3f} rad/us<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template=_TEMPLATE,
        title=f"Pulse Waveform: {label}",
        height=420,
    )
    fig.update_xaxes(title_text="Time (ns)")
    fig.update_yaxes(title_text="Omega (rad/us)", secondary_y=False, color=_BLUE)
    fig.update_yaxes(title_text="delta (rad/us)", secondary_y=True, color=_RED)
    return fig


def time_evolution_traces(
    time_ns: list[float],
    densities_per_site: list[list[float]],
    total_fraction: list[float],
) -> go.Figure:
    """Rydberg density time evolution per atom site."""
    fig = go.Figure()
    palette = [_BLUE, _RED, _GREEN, _ORANGE, _PURPLE, "#17becf", "#bcbd22", "#e377c2"]

    for i, site_data in enumerate(densities_per_site):
        fig.add_trace(
            go.Scatter(
                x=time_ns,
                y=site_data,
                mode="lines",
                name=f"q{i}",
                line=dict(color=palette[i % len(palette)], width=1.5),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=time_ns,
            y=total_fraction,
            mode="lines",
            name="Total fraction",
            line=dict(color="black", width=2.5, dash="dash"),
        )
    )

    fig.update_layout(
        template=_TEMPLATE,
        title="Rydberg Density Time Evolution",
        xaxis_title="Time (ns)",
        yaxis_title="<n_i>",
        height=450,
    )
    return fig


def parameter_space_scatter(candidates: list[dict[str, Any]]) -> go.Figure:
    """2D scatter: amplitude vs detuning, colored by objective score."""
    if not candidates:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Parameter Space (no data)")
        return fig

    amps = [c.get("amplitude", 0) for c in candidates]
    dets = [c.get("detuning", 0) for c in candidates]
    scores = [c.get("objective_score", c.get("predicted_cost", 0)) for c in candidates]
    durs = [c.get("duration_ns", 500) for c in candidates]
    families = [c.get("sequence_family", "unknown") for c in candidates]

    family_map = {"constant_drive": "circle", "global_ramp": "square", "detuning_scan": "diamond",
                  "adiabatic_sweep": "cross", "blackman_sweep": "star"}

    symbols = [family_map.get(f, "circle") for f in families]
    sizes = [max(6, min(20, d / 100)) for d in durs]

    hover = [
        f"amp={a:.2f}<br>det={d:.2f}<br>dur={dur}ns<br>family={fam}<br>score={s:.3f}"
        for a, d, dur, fam, s in zip(amps, dets, durs, families, scores)
    ]

    fig = go.Figure(
        go.Scatter(
            x=amps,
            y=dets,
            mode="markers",
            marker=dict(
                size=sizes,
                color=scores,
                colorscale="Viridis",
                colorbar=dict(title="Score"),
                symbol=symbols,
                line=dict(width=0.5, color="black"),
            ),
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Parameter Space",
        xaxis_title="Amplitude (rad/us)",
        yaxis_title="Detuning (rad/us)",
        height=480,
    )
    return fig


# -- Page 5: Robustness Arena ------------------------------------------------


def robustness_grouped_bar(reports: list[dict[str, Any]]) -> go.Figure:
    """Grouped bar chart: nominal / average / worst per candidate."""
    if not reports:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Robustness Comparison (no data)")
        return fig

    sorted_reports = sorted(reports, key=lambda r: r.get("robustness_score", 0), reverse=True)
    labels = [r.get("sequence_candidate_id", "?")[-8:] for r in sorted_reports]
    nominal = [r.get("nominal_score", 0) for r in sorted_reports]
    average = [r.get("perturbation_average", 0) for r in sorted_reports]
    worst = [r.get("worst_case_score", 0) for r in sorted_reports]

    fig = go.Figure(
        data=[
            go.Bar(name="Nominal", x=labels, y=nominal, marker_color=_BLUE),
            go.Bar(name="Average", x=labels, y=average, marker_color=_ORANGE),
            go.Bar(name="Worst", x=labels, y=worst, marker_color=_RED),
        ]
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Robustness Comparison Across Candidates",
        barmode="group",
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_title="Candidate",
        height=420,
    )
    return fig


def noise_radar(
    reports: list[dict[str, Any]],
    max_candidates: int = 4,
) -> go.Figure:
    """Radar (polar) chart of noise sensitivity."""
    axes = ["amplitude", "detuning", "dephasing", "atom_loss", "SPAM", "temperature"]

    if not reports:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Noise Sensitivity Radar (no data)")
        return fig

    palette = [_BLUE, _RED, _GREEN, _ORANGE]
    fig = go.Figure()

    for idx, report in enumerate(reports[:max_candidates]):
        nominal = report.get("nominal_score", 0)
        scenario_scores = report.get("scenario_scores", {})

        sensitivities = []
        for axis_name in axes:
            worst_drop = 0.0
            for scenario_key, score in scenario_scores.items():
                drop = max(0, nominal - score)
                if drop > worst_drop:
                    worst_drop = drop
            sensitivities.append(worst_drop / max(len(scenario_scores), 1) * len(axes))

        label = report.get("sequence_candidate_id", "?")[-8:]
        fig.add_trace(
            go.Scatterpolar(
                r=sensitivities + [sensitivities[0]],
                theta=axes + [axes[0]],
                fill="toself",
                fillcolor=f"rgba{(*[int(palette[idx % len(palette)].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)], 0.15)}",
                line=dict(color=palette[idx % len(palette)]),
                name=label,
            )
        )

    fig.update_layout(
        template=_TEMPLATE,
        title="Noise Sensitivity Radar",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=480,
    )
    return fig


def score_degradation_waterfall(report: dict[str, Any]) -> go.Figure:
    """Waterfall chart showing score drop from nominal through noise scenarios."""
    nominal = report.get("nominal_score", 0)
    scenario_scores = report.get("scenario_scores", {})

    labels = ["Nominal"]
    values = [nominal]
    measures = ["absolute"]

    prev = nominal
    for scenario in ["low_noise", "medium_noise", "stressed_noise"]:
        score = scenario_scores.get(scenario)
        if score is not None:
            drop = score - prev
            labels.append(scenario.replace("_", " ").title())
            values.append(drop)
            measures.append("relative")
            prev = score

    labels.append("Worst Case")
    values.append(report.get("worst_case_score", prev))
    measures.append("total")

    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            connector=dict(line=dict(color="rgba(0,0,0,0.3)")),
            decreasing=dict(marker=dict(color=_RED)),
            increasing=dict(marker=dict(color=_GREEN)),
            totals=dict(marker=dict(color=_BLUE)),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Score Degradation Waterfall",
        yaxis_title="Score",
        height=420,
    )
    return fig


def robustness_violin(reports: list[dict[str, Any]]) -> go.Figure:
    """Violin plot of scenario score distributions per candidate."""
    if not reports:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Score Distributions (no data)")
        return fig

    fig = go.Figure()
    for report in reports:
        label = report.get("sequence_candidate_id", "?")[-8:]
        scenario_scores = report.get("scenario_scores", {})
        scores = list(scenario_scores.values())
        if not scores:
            continue
        fig.add_trace(
            go.Violin(
                y=scores,
                name=label,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.3,
            )
        )

    fig.update_layout(
        template=_TEMPLATE,
        title="Scenario Score Distributions",
        yaxis_title="Score",
        height=420,
    )
    return fig


def perturbation_heatmap(sweep_data: dict[str, Any]) -> go.Figure:
    """2D contour heatmap: amplitude_jitter x detuning_jitter -> robustness_score."""
    if not sweep_data or "grid" not in sweep_data:
        fig = go.Figure()
        fig.add_annotation(
            text="Run parameter sweep to populate",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            template=_TEMPLATE,
            title="Perturbation Sweep (no data)",
            height=400,
        )
        return fig

    grid = sweep_data["grid"]
    amp_vals = sweep_data.get("amplitude_jitter_values", [])
    det_vals = sweep_data.get("detuning_jitter_values", [])

    fig = go.Figure(
        go.Contour(
            z=grid,
            x=amp_vals,
            y=det_vals,
            colorscale="Viridis",
            colorbar=dict(title="Robustness"),
            hovertemplate="amp_jitter=%{x:.3f}<br>det_jitter=%{y:.3f}<br>score=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Perturbation Sweep Heatmap",
        xaxis_title="Amplitude jitter",
        yaxis_title="Detuning jitter",
        height=450,
    )
    return fig


# -- Page 6: ML Observatory --------------------------------------------------


def training_loss_curves(history: dict[str, list[float]]) -> go.Figure:
    """Train/val loss curves for surrogate model."""
    if not history:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Training Loss (no data)")
        return fig

    fig = go.Figure()
    if "train_loss" in history:
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig.add_trace(
            go.Scatter(x=epochs, y=history["train_loss"], mode="lines", name="Train Loss", line=dict(color=_BLUE))
        )
    if "val_loss" in history:
        epochs = list(range(1, len(history["val_loss"]) + 1))
        val = history["val_loss"]
        fig.add_trace(
            go.Scatter(x=epochs, y=val, mode="lines", name="Val Loss", line=dict(color=_ORANGE))
        )
        min_idx = int(np.argmin(val))
        fig.add_annotation(
            x=min_idx + 1,
            y=val[min_idx],
            text=f"Min: {val[min_idx]:.4f}",
            showarrow=True,
            arrowhead=2,
        )

    fig.update_layout(
        template=_TEMPLATE,
        title="Surrogate Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Weighted MSE",
        height=400,
    )
    return fig


def prediction_vs_actual(
    predictions: list[float],
    actuals: list[float],
    target_name: str = "robustness",
) -> go.Figure:
    """Scatter: predicted vs actual with 45-degree reference line."""
    if not predictions or not actuals:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Prediction vs Actual (no data)")
        return fig

    preds = np.array(predictions)
    acts = np.array(actuals)
    errors = np.abs(preds - acts)

    ss_res = np.sum((acts - preds) ** 2)
    ss_tot = np.sum((acts - np.mean(acts)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig = go.Figure()
    lo = min(float(preds.min()), float(acts.min()))
    hi = max(float(preds.max()), float(acts.max()))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=actuals,
            mode="markers",
            marker=dict(color=errors.tolist(), colorscale="Reds", colorbar=dict(title="|error|"), size=7),
            hovertemplate="pred=%{x:.3f}<br>actual=%{y:.3f}<extra></extra>",
            name="Samples",
        )
    )
    fig.add_annotation(
        text=f"R^2 = {r2:.4f}",
        xref="paper",
        yref="paper",
        x=0.05,
        y=0.95,
        showarrow=False,
        font=dict(size=14),
    )
    fig.update_layout(
        template=_TEMPLATE,
        title=f"Predicted vs Actual ({target_name})",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=450,
    )
    return fig


def ppo_training_dashboard(history: dict[str, list[float]]) -> go.Figure:
    """Multi-panel PPO training plot (2x2 subplots)."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Episode Reward", "Policy Loss", "Value Loss", "Entropy"],
    )

    def _add(key: str, row: int, col: int, color: str) -> None:
        data = history.get(key, [])
        if data:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(data))),
                    y=data,
                    mode="lines",
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    _add("episode_reward", 1, 1, _BLUE)
    _add("policy_loss", 1, 2, _RED)
    _add("value_loss", 2, 1, _GREEN)
    _add("entropy", 2, 2, _ORANGE)

    # Rolling average for reward
    rewards = history.get("episode_reward", [])
    if len(rewards) > 50:
        window = 50
        rolling = [float(np.mean(rewards[max(0, i - window):i + 1])) for i in range(len(rewards))]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rolling))),
                y=rolling,
                mode="lines",
                line=dict(color="black", dash="dash", width=1.5),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        template=_TEMPLATE,
        title="PPO Training Dashboard",
        height=600,
        showlegend=False,
    )
    return fig


def strategy_ucb_evolution(strategy_data: dict[str, list[float]]) -> go.Figure:
    """UCB1 scores per strategy over time."""
    if not strategy_data:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Strategy UCB Evolution (no data)")
        return fig

    palette = {"heuristic": _BLUE, "rl": _RED, "hybrid": _GREEN}
    fig = go.Figure()
    for strategy, scores in strategy_data.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores))),
                y=scores,
                mode="lines+markers",
                name=strategy,
                line=dict(color=palette.get(strategy, _ORANGE)),
            )
        )

    fig.update_layout(
        template=_TEMPLATE,
        title="UCB1 Score Evolution by Strategy",
        xaxis_title="Trial",
        yaxis_title="UCB1 Score",
        height=420,
    )
    return fig


# -- Page 7: Campaign Analytics -----------------------------------------------


def campaign_timeline(campaigns: list[dict[str, Any]]) -> go.Figure:
    """Horizontal timeline of all campaigns."""
    if not campaigns:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Campaign Timeline (no data)")
        return fig

    status_colors = {
        "completed": _GREEN,
        "failed": _RED,
        "no_candidates": _GOLD,
        "created": _BLUE,
        "running": _ORANGE,
    }

    labels = []
    starts = []
    durations = []
    colors = []
    hover_texts = []

    for i, c in enumerate(campaigns):
        cid = c.get("id", f"campaign_{i}")
        status = c.get("status", "created")
        labels.append(cid[-12:])
        starts.append(float(i))
        durations.append(1.0)
        colors.append(status_colors.get(status, _BLUE))
        hover_texts.append(
            f"ID: {cid}<br>Status: {status}"
            f"<br>Candidates: {c.get('candidate_count', 0)}"
            f"<br>Top: {c.get('top_candidate_id', 'N/A')}"
        )

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=durations,
            base=starts,
            orientation="h",
            marker_color=colors,
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Campaign Timeline",
        xaxis_title="Campaign index",
        showlegend=False,
        height=max(300, len(campaigns) * 35),
    )
    return fig


def cross_campaign_score_evolution(campaigns: list[dict[str, Any]]) -> go.Figure:
    """Line chart: best objective score per campaign over time."""
    if not campaigns:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Score Evolution (no data)")
        return fig

    indices = list(range(len(campaigns)))
    scores = []
    for c in campaigns:
        sr = c.get("summary_report", {})
        best = sr.get("best_objective_score", 0)
        if not best:
            best = 0
        scores.append(float(best))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=scores,
            mode="lines+markers",
            marker=dict(size=8, color=_BLUE),
            line=dict(color=_BLUE),
            name="Best score",
            hovertemplate="Campaign %{x}<br>Best score: %{y:.3f}<extra></extra>",
        )
    )

    if len(scores) >= 5:
        window = 5
        rolling = [float(np.mean(scores[max(0, i - window + 1):i + 1])) for i in range(len(scores))]
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=rolling,
                mode="lines",
                line=dict(color=_ORANGE, dash="dash"),
                name=f"{window}-campaign avg",
            )
        )

    fig.update_layout(
        template=_TEMPLATE,
        title="Best Objective Score per Campaign",
        xaxis_title="Campaign index",
        yaxis_title="Best objective score",
        height=400,
    )
    return fig


def backend_distribution_stacked(campaigns: list[dict[str, Any]]) -> go.Figure:
    """Stacked bar chart: backend mix per campaign."""
    if not campaigns:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Backend Distribution (no data)")
        return fig

    backend_types = ["local_pulser_simulation", "emu_sv_candidate", "emu_mps_candidate"]
    backend_colors = {
        "local_pulser_simulation": _BLUE,
        "emu_sv_candidate": _GREEN,
        "emu_mps_candidate": _ORANGE,
    }

    indices = list(range(len(campaigns)))
    data_by_backend: dict[str, list[int]] = {bt: [] for bt in backend_types}

    for c in campaigns:
        mix = c.get("backend_mix", c.get("summary_report", {}).get("backend_mix", {}))
        for bt in backend_types:
            data_by_backend[bt].append(mix.get(bt, 0))

    fig = go.Figure()
    for bt in backend_types:
        fig.add_trace(
            go.Bar(
                x=indices,
                y=data_by_backend[bt],
                name=bt.replace("_", " ").title(),
                marker_color=backend_colors.get(bt, _BLUE),
            )
        )

    fig.update_layout(
        template=_TEMPLATE,
        title="Backend Distribution per Campaign",
        barmode="stack",
        xaxis_title="Campaign index",
        yaxis_title="Count",
        height=400,
    )
    return fig


def parameter_space_3d(candidates: list[dict[str, Any]]) -> go.Figure:
    """3D scatter: amplitude x detuning x duration, colored by robustness_score."""
    if not candidates:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="3D Parameter Space (no data)")
        return fig

    amps = [c.get("amplitude", 0) for c in candidates]
    dets = [c.get("detuning", 0) for c in candidates]
    durs = [c.get("duration_ns", 500) for c in candidates]
    rob = [c.get("robustness_score", 0) for c in candidates]

    hover = [
        f"amp={a:.2f}<br>det={d:.2f}<br>dur={dur}ns<br>robust={r:.3f}"
        for a, d, dur, r in zip(amps, dets, durs, rob)
    ]

    fig = go.Figure(
        go.Scatter3d(
            x=amps,
            y=dets,
            z=durs,
            mode="markers",
            marker=dict(
                size=5,
                color=rob,
                colorscale="Viridis",
                colorbar=dict(title="Robustness"),
            ),
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="3D Parameter Space Explorer",
        scene=dict(
            xaxis_title="Amplitude (rad/us)",
            yaxis_title="Detuning (rad/us)",
            zaxis_title="Duration (ns)",
        ),
        height=550,
    )
    return fig


def memory_tag_cloud(memory_records: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart of most frequent reusable_tags."""
    if not memory_records:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Memory Tags (no data)")
        return fig

    tag_counter: Counter[str] = Counter()
    tag_types: dict[str, str] = {}
    for record in memory_records:
        lesson_type = record.get("lesson_type", "unknown")
        for tag in record.get("reusable_tags", []):
            tag_counter[tag] += 1
            tag_types[tag] = lesson_type

    if not tag_counter:
        fig = go.Figure()
        fig.update_layout(template=_TEMPLATE, title="Memory Tags (none found)")
        return fig

    sorted_tags = tag_counter.most_common(20)
    tags = [t for t, _ in sorted_tags]
    counts = [c for _, c in sorted_tags]
    colors = [_GREEN if tag_types.get(t) == "candidate_pattern" else _RED for t in tags]

    fig = go.Figure(
        go.Bar(
            y=tags,
            x=counts,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x} occurrences<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Most Frequent Memory Tags",
        xaxis_title="Frequency",
        height=max(300, len(tags) * 25),
    )
    return fig
