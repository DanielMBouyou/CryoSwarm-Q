from __future__ import annotations

import pytest

from packages.core.models import ExperimentGoal, RobustnessReport
from packages.orchestration.events import EventBus
from packages.orchestration.phases import GeometryGenerationPhase, PipelinePhase, ProblemFramingPhase
from packages.orchestration.pipeline import CryoSwarmPipeline


def _goal() -> ExperimentGoal:
    return ExperimentGoal(
        title="Architecture test",
        scientific_objective="Verify composable pipeline phases and event emission.",
        desired_atom_count=4,
        preferred_geometry="line",
    )


def _deterministic_report(spec, register_candidate, sequence_candidate):  # type: ignore[no-untyped-def]
    normalized_amplitude = min(sequence_candidate.amplitude / 10.0, 1.0)
    nominal = round(max(0.1, min(0.95, 0.45 + 0.35 * normalized_amplitude)), 4)
    robustness = round(max(0.1, nominal - 0.08), 4)
    worst_case = round(max(0.05, robustness - 0.06), 4)
    return RobustnessReport(
        campaign_id=sequence_candidate.campaign_id,
        sequence_candidate_id=sequence_candidate.id,
        nominal_score=nominal,
        perturbation_average=robustness,
        robustness_penalty=round(nominal - worst_case, 4),
        robustness_score=robustness,
        worst_case_score=worst_case,
        score_std=0.03,
        target_observable=spec.target_observable,
        scenario_scores={
            "low_noise": robustness,
            "medium_noise": round(max(0.05, robustness - 0.03), 4),
            "stressed_noise": worst_case,
        },
        nominal_observables={"observable_score": nominal, "spectral_gap": 0.8},
        scenario_observables={},
        hamiltonian_metrics={"dimension": 2 ** register_candidate.atom_count, "spectral_gap": 0.8},
        reasoning_summary=f"Deterministic robustness for {sequence_candidate.label}.",
        metadata={"stubbed": True},
    )


def test_pipeline_exposes_phase_objects() -> None:
    pipeline = CryoSwarmPipeline(repository=None)

    assert pipeline.phase_names == [
        "problem_framing",
        "geometry_generation",
        "sequence_generation",
        "surrogate_filter",
        "evaluation",
        "ranking",
        "results_summary",
        "memory_capture",
    ]
    assert all(isinstance(phase, PipelinePhase) for phase in pipeline.phases)


def test_problem_and_geometry_phases_run_in_isolation() -> None:
    pipeline = CryoSwarmPipeline(repository=None)
    ctx = pipeline._init_context(_goal())

    ctx = ProblemFramingPhase(pipeline).execute(ctx)
    ctx = GeometryGenerationPhase(pipeline).execute(ctx)

    assert ctx.spec is not None
    assert ctx.registers
    assert ctx.should_stop is False


def test_pipeline_emits_phase_and_domain_events(monkeypatch: pytest.MonkeyPatch) -> None:
    event_bus = EventBus()
    seen_events: list[str] = []
    event_bus.subscribe("*", lambda event: seen_events.append(event.event_type))

    pipeline = CryoSwarmPipeline(repository=None, event_bus=event_bus)
    monkeypatch.setattr(pipeline.noise_agent, "run", _deterministic_report)

    summary = pipeline.run(_goal())

    assert summary.status == "COMPLETED"
    assert "pipeline.started" in seen_events
    assert "phase.started" in seen_events
    assert "problem_framing.completed" in seen_events
    assert "geometry.completed" in seen_events
    assert "sequence.completed" in seen_events
    assert "surrogate_filter.skipped" in seen_events
    assert "evaluation.completed" in seen_events
    assert "pipeline.completed" in seen_events


def test_pipeline_applies_surrogate_filter_before_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubSurrogateFilter:
        def filter_with_report(self, sequences, register_lookup):  # type: ignore[no-untyped-def]
            filtered = sequences[:1]
            return filtered, {
                "enabled": True,
                "applied": True,
                "reason": "stubbed_filter",
                "input_count": len(sequences),
                "kept_count": len(filtered),
                "rejected_count": len(sequences) - len(filtered),
                "use_ensemble": False,
            }

    pipeline = CryoSwarmPipeline(repository=None, surrogate_filter=_StubSurrogateFilter())
    monkeypatch.setattr(pipeline.noise_agent, "run", _deterministic_report)

    summary = pipeline.run(_goal())

    assert summary.status == "COMPLETED"
    assert len(summary.robustness_reports) == 1
    assert summary.campaign.summary_report["surrogate_filter"]["kept_count"] == 1
