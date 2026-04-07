"""Tests for dashboard Plotly charts and LaTeX panels.

These tests verify that chart functions return valid Plotly figures
and that LaTeX panels produce non-empty strings, without requiring
a running Streamlit server.
"""
from __future__ import annotations

import plotly.graph_objects as go

from apps.dashboard.components.latex_panels import (
    campaign_formulas,
    hamiltonian_formulas,
    mis_formulas,
    ml_formulas,
    observable_formulas,
    pulse_formulas,
    robustness_formulas,
)
from apps.dashboard.components.plotly_charts import (
    bitstring_bar_chart,
    blockade_graph,
    campaign_timeline,
    candidate_funnel,
    cross_campaign_score_evolution,
    distance_histogram,
    energy_spectrum,
    memory_tag_cloud,
    noise_radar,
    parameter_space_3d,
    parameter_space_scatter,
    perturbation_heatmap,
    pipeline_gantt,
    ppo_training_dashboard,
    prediction_vs_actual,
    pulse_waveform,
    register_scatter_2d,
    robustness_grouped_bar,
    robustness_violin,
    score_degradation_waterfall,
    strategy_ucb_evolution,
    time_evolution_traces,
    training_loss_curves,
    vdw_interaction_heatmap,
)


# ---- LaTeX panels ----------------------------------------------------------


def test_hamiltonian_formulas_all_keys() -> None:
    formulas = hamiltonian_formulas()
    expected = {"full_hamiltonian", "interaction", "blockade_radius", "c6_value", "spectral_gap", "ipr", "adiabatic_condition"}
    assert expected == set(formulas.keys())
    for v in formulas.values():
        assert isinstance(v, str) and len(v) > 0


def test_observable_formulas_all_keys() -> None:
    formulas = observable_formulas()
    expected = {"rydberg_density", "connected_correlation", "antiferromagnetic_order", "entanglement_entropy", "total_rydberg_fraction", "state_fidelity", "bitstring_probability"}
    assert expected == set(formulas.keys())
    for v in formulas.values():
        assert isinstance(v, str) and len(v) > 0


def test_mis_formulas_all_keys() -> None:
    formulas = mis_formulas()
    expected = {"mis_definition", "mis_overlap", "mis_cost_function"}
    assert expected == set(formulas.keys())


def test_robustness_formulas_all_keys() -> None:
    formulas = robustness_formulas()
    expected = {
        "robustness_score", "robustness_weights", "stability_bonus", "penalty",
        "objective_score", "weight_constraint", "noise_amplitude", "noise_detuning",
        "dephasing_lindblad", "density_score", "blockade_score",
    }
    assert expected == set(formulas.keys())
    for v in formulas.values():
        assert isinstance(v, str) and len(v) > 0


def test_ml_formulas_all_keys() -> None:
    formulas = ml_formulas()
    expected = {"ucb1", "ppo_objective", "importance_ratio", "gae", "ensemble_mean", "epistemic_uncertainty", "weighted_mse"}
    assert expected == set(formulas.keys())


def test_pulse_formulas_all_keys() -> None:
    formulas = pulse_formulas()
    expected = {"blackman", "linear_sweep", "pulse_area", "time_evolution", "trotter_suzuki_2", "ramp", "constant"}
    assert expected == set(formulas.keys())


def test_campaign_formulas_all_keys() -> None:
    formulas = campaign_formulas()
    expected = {"pareto_optimal", "cumulative_regret", "acquisition_ucb", "ranking_key"}
    assert expected == set(formulas.keys())


# ---- Plotly charts ---------------------------------------------------------


def test_register_scatter_2d_returns_figure() -> None:
    coords = [(0, 0), (7, 0), (3.5, 6)]
    fig = register_scatter_2d(coords, blockade_radius=8.0, label="test")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_register_scatter_2d_with_densities() -> None:
    coords = [(0, 0), (7, 0), (3.5, 6)]
    fig = register_scatter_2d(coords, blockade_radius=8.0, label="test", rydberg_densities=[0.2, 0.5, 0.8])
    assert isinstance(fig, go.Figure)


def test_vdw_interaction_heatmap_returns_figure() -> None:
    coords = [(0, 0), (7, 0), (0, 7)]
    fig = vdw_interaction_heatmap(coords, c6=862690.0)
    assert isinstance(fig, go.Figure)


def test_blockade_graph_returns_figure() -> None:
    coords = [(0, 0), (7, 0), (0, 7)]
    adj = [[False, True, True], [True, False, False], [True, False, False]]
    fig = blockade_graph(coords, adj)
    assert isinstance(fig, go.Figure)


def test_blockade_graph_with_mis() -> None:
    coords = [(0, 0), (7, 0), (0, 7)]
    adj = [[False, True, True], [True, False, False], [True, False, False]]
    fig = blockade_graph(coords, adj, mis_sets=[[1, 2]])
    assert isinstance(fig, go.Figure)


def test_distance_histogram_returns_figure() -> None:
    coords = [(0, 0), (7, 0), (3.5, 6)]
    fig = distance_histogram(coords, blockade_radius=8.0)
    assert isinstance(fig, go.Figure)


def test_energy_spectrum_returns_figure() -> None:
    eigenvalues = [-5.2, -3.1, -1.0, 0.5, 2.3, 4.1]
    fig = energy_spectrum(eigenvalues)
    assert isinstance(fig, go.Figure)


def test_energy_spectrum_single_value() -> None:
    fig = energy_spectrum([-1.0])
    assert isinstance(fig, go.Figure)


def test_bitstring_bar_chart_returns_figure() -> None:
    bitstrings = [("0011", 0.3), ("1100", 0.25), ("1010", 0.15)]
    fig = bitstring_bar_chart(bitstrings)
    assert isinstance(fig, go.Figure)


def test_bitstring_bar_chart_with_mis() -> None:
    bitstrings = [("0011", 0.3), ("1100", 0.25)]
    fig = bitstring_bar_chart(bitstrings, mis_bitstrings=["1100"])
    assert isinstance(fig, go.Figure)


def test_parametric_spectrum_returns_figure() -> None:
    from apps.dashboard.components.plotly_charts import parametric_spectrum

    delta_vals = [-10.0, -5.0, 0.0, 5.0, 10.0]
    curves = [[-5, -3, -1, 0, 1], [-3, -1, 1, 3, 5]]
    fig = parametric_spectrum(delta_vals, curves, omega=5.0)
    assert isinstance(fig, go.Figure)


def test_pulse_waveform_returns_figure() -> None:
    fig = pulse_waveform([0, 500, 1000], [5.0, 5.0, 5.0], [-10.0, 0.0, 10.0], "test-seq")
    assert isinstance(fig, go.Figure)


def test_time_evolution_traces_returns_figure() -> None:
    fig = time_evolution_traces(
        [0, 100, 200],
        [[0.0, 0.3, 0.5], [0.0, 0.2, 0.4]],
        [0.0, 0.25, 0.45],
    )
    assert isinstance(fig, go.Figure)


def test_parameter_space_scatter_returns_figure() -> None:
    candidates = [
        {"amplitude": 5.0, "detuning": -10.0, "duration_ns": 1000, "sequence_family": "constant_drive", "objective_score": 0.7},
    ]
    fig = parameter_space_scatter(candidates)
    assert isinstance(fig, go.Figure)


def test_parameter_space_scatter_empty() -> None:
    fig = parameter_space_scatter([])
    assert isinstance(fig, go.Figure)


def test_robustness_grouped_bar_returns_figure() -> None:
    reports = [
        {"sequence_candidate_id": "seq_abc", "nominal_score": 0.8, "perturbation_average": 0.6, "worst_case_score": 0.4, "robustness_score": 0.65},
        {"sequence_candidate_id": "seq_def", "nominal_score": 0.7, "perturbation_average": 0.55, "worst_case_score": 0.35, "robustness_score": 0.55},
    ]
    fig = robustness_grouped_bar(reports)
    assert isinstance(fig, go.Figure)


def test_robustness_grouped_bar_empty_list() -> None:
    fig = robustness_grouped_bar([])
    assert isinstance(fig, go.Figure)


def test_noise_radar_returns_figure() -> None:
    reports = [
        {"sequence_candidate_id": "seq_abc", "nominal_score": 0.8, "scenario_scores": {"low_noise": 0.7, "medium_noise": 0.5}},
    ]
    fig = noise_radar(reports)
    assert isinstance(fig, go.Figure)


def test_noise_radar_empty() -> None:
    fig = noise_radar([])
    assert isinstance(fig, go.Figure)


def test_score_degradation_waterfall_returns_figure() -> None:
    report = {
        "nominal_score": 0.8,
        "worst_case_score": 0.4,
        "scenario_scores": {"low_noise": 0.7, "medium_noise": 0.55, "stressed_noise": 0.4},
    }
    fig = score_degradation_waterfall(report)
    assert isinstance(fig, go.Figure)


def test_robustness_violin_returns_figure() -> None:
    reports = [
        {"sequence_candidate_id": "seq_abc", "scenario_scores": {"low_noise": 0.7, "medium_noise": 0.5, "stressed_noise": 0.3}},
    ]
    fig = robustness_violin(reports)
    assert isinstance(fig, go.Figure)


def test_robustness_violin_empty() -> None:
    fig = robustness_violin([])
    assert isinstance(fig, go.Figure)


def test_perturbation_heatmap_no_data() -> None:
    fig = perturbation_heatmap({})
    assert isinstance(fig, go.Figure)


def test_perturbation_heatmap_with_data() -> None:
    data = {
        "grid": [[0.5, 0.6], [0.4, 0.3]],
        "amplitude_jitter_values": [0.01, 0.05],
        "detuning_jitter_values": [0.01, 0.05],
    }
    fig = perturbation_heatmap(data)
    assert isinstance(fig, go.Figure)


def test_training_loss_curves_returns_figure() -> None:
    history = {"train_loss": [1.0, 0.5, 0.3], "val_loss": [1.1, 0.6, 0.35]}
    fig = training_loss_curves(history)
    assert isinstance(fig, go.Figure)


def test_training_loss_curves_empty() -> None:
    fig = training_loss_curves({})
    assert isinstance(fig, go.Figure)


def test_prediction_vs_actual_returns_figure() -> None:
    fig = prediction_vs_actual([0.5, 0.6, 0.7], [0.55, 0.58, 0.72])
    assert isinstance(fig, go.Figure)


def test_prediction_vs_actual_empty() -> None:
    fig = prediction_vs_actual([], [])
    assert isinstance(fig, go.Figure)


def test_ppo_training_dashboard_returns_figure() -> None:
    history = {
        "episode_reward": list(range(100)),
        "policy_loss": [0.1] * 100,
        "value_loss": [0.2] * 100,
        "entropy": [0.5] * 100,
    }
    fig = ppo_training_dashboard(history)
    assert isinstance(fig, go.Figure)


def test_ppo_training_dashboard_empty() -> None:
    fig = ppo_training_dashboard({})
    assert isinstance(fig, go.Figure)


def test_strategy_ucb_evolution_returns_figure() -> None:
    data = {"heuristic": [1.0, 1.2, 1.1], "rl": [0.8, 1.0, 1.3]}
    fig = strategy_ucb_evolution(data)
    assert isinstance(fig, go.Figure)


def test_strategy_ucb_evolution_empty() -> None:
    fig = strategy_ucb_evolution({})
    assert isinstance(fig, go.Figure)


def test_campaign_timeline_returns_figure() -> None:
    campaigns = [
        {"id": "campaign_abc123", "status": "completed", "candidate_count": 5, "top_candidate_id": "seq_1"},
    ]
    fig = campaign_timeline(campaigns)
    assert isinstance(fig, go.Figure)


def test_campaign_timeline_empty() -> None:
    fig = campaign_timeline([])
    assert isinstance(fig, go.Figure)


def test_cross_campaign_score_evolution_returns_figure() -> None:
    campaigns = [
        {"id": "c1", "summary_report": {"best_objective_score": 0.7}},
        {"id": "c2", "summary_report": {"best_objective_score": 0.75}},
    ]
    fig = cross_campaign_score_evolution(campaigns)
    assert isinstance(fig, go.Figure)


def test_backend_distribution_stacked_returns_figure() -> None:
    from apps.dashboard.components.plotly_charts import backend_distribution_stacked

    campaigns = [
        {"id": "c1", "backend_mix": {"local_pulser_simulation": 3, "emu_sv_candidate": 1}},
    ]
    fig = backend_distribution_stacked(campaigns)
    assert isinstance(fig, go.Figure)


def test_parameter_space_3d_returns_figure() -> None:
    candidates = [
        {"amplitude": 5.0, "detuning": -10.0, "duration_ns": 1000, "robustness_score": 0.7},
    ]
    fig = parameter_space_3d(candidates)
    assert isinstance(fig, go.Figure)


def test_parameter_space_3d_empty() -> None:
    fig = parameter_space_3d([])
    assert isinstance(fig, go.Figure)


def test_memory_tag_cloud_returns_figure() -> None:
    records = [
        {"lesson_type": "candidate_pattern", "reusable_tags": ["high_robustness", "adiabatic"]},
        {"lesson_type": "failure_pattern", "reusable_tags": ["high_robustness"]},
    ]
    fig = memory_tag_cloud(records)
    assert isinstance(fig, go.Figure)


def test_memory_tag_cloud_empty() -> None:
    fig = memory_tag_cloud([])
    assert isinstance(fig, go.Figure)


def test_candidate_funnel_returns_figure() -> None:
    fig = candidate_funnel({"id": "c1"}, n_registers=10, n_sequences=30, n_evaluated=25, n_ranked=5)
    assert isinstance(fig, go.Figure)


def test_pipeline_gantt_returns_figure() -> None:
    decisions = [
        {"agent_name": "problem_framing_agent", "created_at": 0},
        {"agent_name": "geometry_agent", "created_at": 1},
    ]
    fig = pipeline_gantt(decisions)
    assert isinstance(fig, go.Figure)


def test_pipeline_gantt_empty() -> None:
    fig = pipeline_gantt([])
    assert isinstance(fig, go.Figure)
