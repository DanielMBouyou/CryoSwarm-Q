from __future__ import annotations

import packages.orchestration.pipeline as pipeline_module
import pytest

from packages.core.models import ExperimentGoal, RobustnessReport
from packages.orchestration.pipeline import CryoSwarmPipeline

pytestmark = pytest.mark.slow


class _FakeFuture:
    def __init__(self, fn, *args, **kwargs) -> None:
        self._exception = None
        try:
            self._result = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - exercised through .result()
            self._result = None
            self._exception = exc

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result


class _FakeProcessPoolExecutor:
    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def submit(self, fn, *args, **kwargs):
        return _FakeFuture(fn, *args, **kwargs)


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


def _build_goal() -> ExperimentGoal:
    return ExperimentGoal(
        title="Parallel evaluation check",
        scientific_objective="Compare serial and parallel evaluation outputs.",
        desired_atom_count=4,
        priority="balanced",
        preferred_geometry="line",
    )


def test_parallel_and_serial_pipeline_produce_same_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    serial_pipeline = CryoSwarmPipeline(repository=None, parallel=False)
    parallel_pipeline = CryoSwarmPipeline(repository=None, parallel=True)

    monkeypatch.setattr(serial_pipeline.noise_agent, "run", _deterministic_report)
    monkeypatch.setattr(pipeline_module, "_evaluate_noise_task", _deterministic_report)
    monkeypatch.setattr(pipeline_module, "ProcessPoolExecutor", _FakeProcessPoolExecutor)
    monkeypatch.setattr(pipeline_module, "as_completed", lambda futures: list(futures))

    serial_summary = serial_pipeline.run(_build_goal())
    parallel_summary = parallel_pipeline.run(_build_goal())

    serial_scores = [candidate.objective_score for candidate in serial_summary.ranked_candidates]
    parallel_scores = [candidate.objective_score for candidate in parallel_summary.ranked_candidates]
    assert parallel_summary.status == serial_summary.status == "COMPLETED"
    assert parallel_summary.backend_mix == serial_summary.backend_mix
    assert len(parallel_scores) == len(serial_scores)
    assert parallel_scores == pytest.approx(serial_scores, rel=1e-9, abs=1e-9)
