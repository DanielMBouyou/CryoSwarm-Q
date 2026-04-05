from __future__ import annotations

from packages.core.models import ExperimentGoal, RobustnessReport
from packages.orchestration.pipeline import CryoSwarmPipeline


def _goal() -> ExperimentGoal:
    return ExperimentGoal(
        title="Pipeline failure test",
        scientific_objective="Exercise degraded orchestration paths.",
        desired_atom_count=4,
        preferred_geometry="line",
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
    monkeypatch.setattr(
        pipeline.memory_agent,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("memory failure")),
    )

    summary = pipeline.run(_goal())

    assert summary.status in {"COMPLETED", "FAILED"}
    assert any(
        decision.agent_name.value == "memory_agent" and decision.status == "failed"
        for decision in summary.decisions
    )


def test_pipeline_continues_after_one_noise_failure(monkeypatch) -> None:
    pipeline = CryoSwarmPipeline(repository=None)
    state = {"failed_once": False}
    original_run = pipeline.noise_agent.run

    def flaky_noise(*args, **kwargs):  # type: ignore[no-untyped-def]
        if not state["failed_once"]:
            state["failed_once"] = True
            raise RuntimeError("transient noise failure")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(pipeline.noise_agent, "run", flaky_noise)

    summary = pipeline.run(_goal())

    assert summary.ranked_candidates
    assert any(decision.status == "failed" for decision in summary.decisions)
