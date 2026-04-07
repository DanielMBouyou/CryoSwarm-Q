from __future__ import annotations

"""Tests for pure dashboard data-shaping helpers."""

from datetime import UTC, datetime

from apps.dashboard.logic import (
    build_campaign_table,
    build_decision_table,
    build_event_table,
    build_ranked_table,
    build_register_lookup_from_documents,
    select_noise_sensitivity_data,
    select_robustness_chart_data,
)
from packages.core.enums import AgentName, BackendType, CampaignStatus, CandidateStatus, DecisionType
from packages.core.models import AgentDecision, CampaignState, EvaluationResult, RegisterCandidate, RobustnessReport
from packages.orchestration.events import PipelineEvent


def test_build_campaign_table_empty() -> None:
    assert build_campaign_table([]) == []


def test_build_campaign_table() -> None:
    campaign = CampaignState(
        goal_id="goal_1",
        status=CampaignStatus.COMPLETED,
        candidate_count=5,
        top_candidate_id="seq_abc",
    )

    rows = build_campaign_table([campaign])

    assert len(rows) == 1
    assert rows[0]["status"] == CampaignStatus.COMPLETED.value
    assert rows[0]["candidate_count"] == 5


def test_build_ranked_table() -> None:
    result = EvaluationResult(
        campaign_id="c1",
        sequence_candidate_id="seq_1",
        register_candidate_id="reg_1",
        nominal_score=0.8,
        robustness_score=0.7,
        worst_case_score=0.55,
        observable_score=0.78,
        objective_score=0.72,
        backend_choice=BackendType.LOCAL_PULSER_SIMULATION,
        estimated_cost=0.12,
        estimated_latency=0.15,
        final_rank=1,
        status=CandidateStatus.RANKED,
        reasoning_summary="test",
    )

    rows = build_ranked_table([result])

    assert rows[0]["rank"] == 1
    assert rows[0]["objective_score"] == 0.72


def test_build_decision_table() -> None:
    decision = AgentDecision(
        campaign_id="campaign_1",
        agent_name=AgentName.PROBLEM_FRAMING,
        subject_id="spec_1",
        decision_type=DecisionType.SPECIFICATION,
        status="completed",
        reasoning_summary="Framed successfully.",
    )

    rows = build_decision_table([decision])

    assert rows == [
        {
            "agent": AgentName.PROBLEM_FRAMING.value,
            "subject_id": "spec_1",
            "decision_type": DecisionType.SPECIFICATION.value,
            "status": "completed",
            "reasoning_summary": "Framed successfully.",
        }
    ]


def test_build_event_table() -> None:
    events = [
        PipelineEvent(event_type="phase.started", payload={"phase": "geometry_generation", "status": "RUNNING"}),
        PipelineEvent(event_type="geometry.completed", payload={"phase": "geometry_generation", "status": "RUNNING"}),
        PipelineEvent(event_type="pipeline.completed", payload={"summary_status": "COMPLETED"}),
    ]

    rows = build_event_table(events, limit=2)

    assert len(rows) == 2
    assert rows[0]["event"] == "geometry.completed"
    assert rows[0]["phase"] == "geometry_generation"
    assert rows[1]["status"] == "COMPLETED"


def test_robustness_chart_data_empty() -> None:
    labels, nominal, average, worst = select_robustness_chart_data([])

    assert labels == []
    assert nominal == []
    assert average == []
    assert worst == []


def test_robustness_chart_data() -> None:
    report = RobustnessReport(
        campaign_id="c1",
        sequence_candidate_id="seq_123456",
        nominal_score=0.8,
        perturbation_average=0.72,
        robustness_penalty=0.08,
        robustness_score=0.75,
        worst_case_score=0.6,
        score_std=0.05,
        target_observable="rydberg_density",
        scenario_scores={"low_noise": 0.78, "medium_noise": 0.72},
        reasoning_summary="test",
    )

    labels, nominal, average, worst = select_robustness_chart_data([report])

    assert labels == ["123456"]
    assert nominal == [0.8]
    assert average == [0.72]
    assert worst == [0.6]


def test_noise_sensitivity_data() -> None:
    report = RobustnessReport(
        campaign_id="c1",
        sequence_candidate_id="seq_1",
        nominal_score=0.8,
        perturbation_average=0.72,
        robustness_penalty=0.08,
        robustness_score=0.75,
        worst_case_score=0.6,
        score_std=0.05,
        target_observable="rydberg_density",
        scenario_scores={
            "low_noise": 0.78,
            "medium_noise": 0.72,
            "stressed_noise": 0.6,
        },
        reasoning_summary="test",
    )

    labels, values = select_noise_sensitivity_data(report)

    assert labels == ["low_noise", "medium_noise", "stressed_noise"]
    assert values == [0.78, 0.72, 0.6]


def test_noise_sensitivity_data_missing_scenarios() -> None:
    report = RobustnessReport(
        campaign_id="c1",
        sequence_candidate_id="seq_1",
        nominal_score=0.8,
        perturbation_average=0.72,
        robustness_penalty=0.08,
        robustness_score=0.75,
        worst_case_score=0.6,
        score_std=0.05,
        target_observable="rydberg_density",
        scenario_scores={"medium_noise": 0.72},
        reasoning_summary="test",
    )

    labels, values = select_noise_sensitivity_data(report)

    assert labels == ["medium_noise"]
    assert values == [0.72]


def test_generate_waveform_constant() -> None:
    from apps.dashboard.logic import generate_waveform

    t, omega, delta = generate_waveform("constant_drive", 5.0, -10.0, 10.0, 1000.0)
    assert len(t) == 200
    assert all(abs(o - 5.0) < 1e-10 for o in omega)
    assert all(abs(d - (-10.0)) < 1e-10 for d in delta)


def test_generate_waveform_blackman() -> None:
    from apps.dashboard.logic import generate_waveform

    t, omega, delta = generate_waveform("blackman_sweep", 5.0, -10.0, 10.0, 1000.0)
    assert len(t) == 200
    assert omega[0] < 0.01  # Blackman starts near zero
    assert max(omega) <= 5.0 + 1e-10


def test_generate_waveform_adiabatic() -> None:
    from apps.dashboard.logic import generate_waveform

    t, omega, delta = generate_waveform("adiabatic_sweep", 5.0, -10.0, 10.0, 1000.0)
    assert len(t) == 200
    assert abs(omega[0]) < 1e-10  # sin^2(0) = 0
    assert abs(omega[-1]) < 1e-10  # sin^2(pi) = 0


def test_generate_waveform_global_ramp() -> None:
    from apps.dashboard.logic import generate_waveform

    t, omega, delta = generate_waveform("global_ramp", 5.0, -10.0, 10.0, 1000.0)
    assert len(t) == 200
    assert abs(omega[0]) < 1e-10  # starts at 0
    assert abs(omega[-1] - 5.0) < 1e-10  # ends at omega_max


def test_generate_waveform_detuning_scan() -> None:
    from apps.dashboard.logic import generate_waveform

    t, omega, delta = generate_waveform("detuning_scan", 5.0, -10.0, 10.0, 1000.0)
    assert len(t) == 200
    assert all(abs(o - 5.0) < 1e-10 for o in omega)
    assert abs(delta[0] - (-10.0)) < 1e-10
    assert abs(delta[-1] - 10.0) < 1e-10


def test_compute_pareto_front() -> None:
    from apps.dashboard.logic import compute_pareto_front

    candidates = [
        {"objective_score": 0.8, "worst_case_score": 0.3},
        {"objective_score": 0.5, "worst_case_score": 0.7},
        {"objective_score": 0.6, "worst_case_score": 0.5},
    ]
    pareto = compute_pareto_front(candidates)
    assert 0 in pareto  # (0.8, 0.3) is Pareto optimal
    assert 1 in pareto  # (0.5, 0.7) is Pareto optimal


def test_compute_pareto_front_single() -> None:
    from apps.dashboard.logic import compute_pareto_front

    candidates = [{"objective_score": 0.8, "worst_case_score": 0.5}]
    pareto = compute_pareto_front(candidates)
    assert pareto == [0]


def test_compute_pareto_front_empty() -> None:
    from apps.dashboard.logic import compute_pareto_front

    pareto = compute_pareto_front([])
    assert pareto == []


def test_build_register_lookup_from_documents() -> None:
    created_at = datetime.now(UTC)
    document = {
        "_id": "reg_doc_1",
        "id": "reg_doc_1",
        "campaign_id": "campaign_1",
        "spec_id": "spec_1",
        "label": "square-4",
        "layout_type": "square",
        "atom_count": 4,
        "coordinates": [(0.0, 0.0), (7.0, 0.0), (0.0, 7.0), (7.0, 7.0)],
        "device_constraints": {"min_spacing_um": 5.0},
        "min_distance_um": 7.0,
        "blockade_radius_um": 9.5,
        "blockade_pair_count": 4,
        "van_der_waals_matrix": [
            [0.0, 7.2, 7.2, 1.8],
            [7.2, 0.0, 1.8, 7.2],
            [7.2, 1.8, 0.0, 7.2],
            [1.8, 7.2, 7.2, 0.0],
        ],
        "feasibility_score": 0.9,
        "reasoning_summary": "Stored register.",
        "metadata": {"spacing_um": 7.0},
        "created_at": created_at,
    }

    lookup = build_register_lookup_from_documents([document])

    assert list(lookup) == ["reg_doc_1"]
    assert isinstance(lookup["reg_doc_1"], RegisterCandidate)
    assert lookup["reg_doc_1"].label == "square-4"
