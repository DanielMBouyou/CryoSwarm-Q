from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from packages.core.enums import CandidateStatus, GoalStatus, SequenceFamily
from packages.core.models import (
    EvaluationResult,
    ExperimentGoal,
    MemoryRecord,
    RegisterCandidate,
    SequenceCandidate,
)
from packages.db.repositories import CryoSwarmRepository


pytestmark = pytest.mark.skipif(
    not os.getenv("MONGODB_URI"),
    reason="MONGODB_URI is not configured.",
)


def _get_repository_or_skip() -> CryoSwarmRepository:
    try:
        repository = CryoSwarmRepository()
        repository.database.command("ping")
        return repository
    except Exception as exc:
        pytest.skip(f"MongoDB is not reachable in this environment: {exc}")


def test_save_and_retrieve_goal() -> None:
    repository = _get_repository_or_skip()
    goal = ExperimentGoal(
        title="Mongo repository goal",
        scientific_objective="Verify goal persistence in MongoDB.",
        desired_atom_count=4,
        status=GoalStatus.STORED,
    )

    repository.create_goal(goal)
    stored = repository.get_goal(goal.id)

    assert stored is not None
    assert stored.id == goal.id
    assert stored.title == goal.title


def test_save_register_candidate() -> None:
    repository = _get_repository_or_skip()
    candidate = RegisterCandidate(
        campaign_id="campaign_repo_test",
        spec_id="spec_repo_test",
        label="line-4-s7.0",
        layout_type="line",
        atom_count=4,
        coordinates=[(0.0, 0.0), (7.0, 0.0), (14.0, 0.0), (21.0, 0.0)],
        min_distance_um=7.0,
        blockade_radius_um=6.0,
        blockade_pair_count=0,
        van_der_waals_matrix=[[0.0] * 4 for _ in range(4)],
        feasibility_score=0.8,
        reasoning_summary="Repository persistence test.",
    )

    repository.insert_register_candidates([candidate])
    raw = repository.collections["register_candidates"].find_one({"_id": candidate.id})

    assert raw is not None
    assert raw["label"] == candidate.label


def test_save_sequence_candidate() -> None:
    repository = _get_repository_or_skip()
    candidate = SequenceCandidate(
        campaign_id="campaign_repo_test",
        spec_id="spec_repo_test",
        register_candidate_id="reg_repo_test",
        label="seq-repo-test",
        sequence_family=SequenceFamily.CONSTANT_DRIVE,
        duration_ns=1200,
        amplitude=5.0,
        detuning=0.0,
        phase=0.0,
        predicted_cost=0.2,
        reasoning_summary="Repository persistence test.",
    )

    repository.insert_sequence_candidates([candidate])
    raw = repository.collections["sequence_candidates"].find_one({"_id": candidate.id})

    assert raw is not None
    assert raw["label"] == candidate.label


def test_get_recent_memory_returns_latest_first() -> None:
    repository = _get_repository_or_skip()
    older = MemoryRecord(
        campaign_id="campaign_memory_test",
        source_candidate_id="seq_old",
        lesson_type="candidate_pattern",
        summary="Older record",
        created_at=datetime.now(UTC) - timedelta(days=1),
    )
    newer = MemoryRecord(
        campaign_id="campaign_memory_test",
        source_candidate_id="seq_new",
        lesson_type="candidate_pattern",
        summary="Newer record",
        created_at=datetime.now(UTC),
    )

    repository.insert_memory_record(older)
    repository.insert_memory_record(newer)
    records = repository.list_recent_memory(limit=2)

    assert len(records) >= 2
    assert records[0].created_at >= records[1].created_at


def test_get_campaign_candidates_filters_by_campaign_id() -> None:
    repository = _get_repository_or_skip()
    keep = EvaluationResult(
        campaign_id="campaign_keep",
        sequence_candidate_id="seq_keep",
        register_candidate_id="reg_keep",
        nominal_score=0.7,
        robustness_score=0.65,
        worst_case_score=0.6,
        observable_score=0.7,
        objective_score=0.66,
        backend_choice="local_pulser_simulation",
        estimated_cost=0.2,
        estimated_latency=0.2,
        final_rank=1,
        status=CandidateStatus.RANKED,
        reasoning_summary="Keep this candidate.",
    )
    other = EvaluationResult(
        campaign_id="campaign_other",
        sequence_candidate_id="seq_other",
        register_candidate_id="reg_other",
        nominal_score=0.6,
        robustness_score=0.55,
        worst_case_score=0.5,
        observable_score=0.6,
        objective_score=0.57,
        backend_choice="local_pulser_simulation",
        estimated_cost=0.3,
        estimated_latency=0.3,
        final_rank=1,
        status=CandidateStatus.RANKED,
        reasoning_summary="Other campaign candidate.",
    )

    repository.insert_evaluation_results([keep, other])
    results = repository.list_candidates_for_campaign("campaign_keep")

    assert results
    assert all(result.campaign_id == "campaign_keep" for result in results)
