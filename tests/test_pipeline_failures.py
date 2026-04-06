from __future__ import annotations

import pytest

from packages.core.models import ExperimentGoal, RobustnessReport
from packages.orchestration.pipeline import CryoSwarmPipeline

pytestmark = pytest.mark.integration


def _goal() -> ExperimentGoal:
    return ExperimentGoal(
        title="Pipeline failure test",
        scientific_objective="Exercise degraded orchestration paths.",
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


def test_pipeline_returns_failed_when_problem_agent_crashes(monkeypatch) -> None:
    pipeline = CryoSwarmPipeline(repository=None)
    monkeypatch.setattr(
        pipeline.problem_agent,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("problem failure")),
    )

    summary = pipeline.run(_goal())

    assert summary.status == "FAILED"
    assert any(decision.status == "failed" for decision in summary.decisions)


def test_pipeline_returns_no_candidates_when_all_sequences_fail(monkeypatch) -> None:
    pipeline = CryoSwarmPipeline(repository=None)
    monkeypatch.setattr(
        pipeline.sequence_agent,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sequence failure")),
    )

    summary = pipeline.run(_goal())

    assert summary.status == "NO_CANDIDATES"
    assert summary.ranked_candidates == []


def test_pipeline_continues_when_memory_agent_fails(monkeypatch) -> None:
    pipeline = CryoSwarmPipeline(repository=None)
    monkeypatch.setattr(pipeline.noise_agent, "run", _deterministic_report)
    monkeypatch.setattr(
        pipeline.memory_agent,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("memory failure")),
    )

    summary = pipeline.run(_goal())

    assert summary.status == "COMPLETED"
    assert any(
        decision.agent_name.value == "memory_agent" and decision.status == "failed"
        for decision in summary.decisions
    )


def test_pipeline_continues_after_one_noise_failure(monkeypatch) -> None:
    pipeline = CryoSwarmPipeline(repository=None)
    state = {"failed_once": False}

    def flaky_noise(*args, **kwargs):  # type: ignore[no-untyped-def]
        if not state["failed_once"]:
            state["failed_once"] = True
            raise RuntimeError("transient noise failure")
        return _deterministic_report(*args, **kwargs)

    monkeypatch.setattr(pipeline.noise_agent, "run", flaky_noise)

    summary = pipeline.run(_goal())

    assert summary.ranked_candidates
    assert any(decision.status == "failed" for decision in summary.decisions)
