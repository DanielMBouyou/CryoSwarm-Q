from __future__ import annotations

from packages.agents.geometry_agent import GeometryAgent
from packages.agents.memory_agent import MemoryAgent
from packages.agents.sequence_agent import SequenceAgent
from packages.core.enums import BackendType, CandidateStatus, SequenceFamily
from packages.core.models import EvaluationResult, MemoryRecord, RegisterCandidate, SequenceCandidate


def test_sequence_agent_reduces_variants_for_weak_family() -> None:
    agent = SequenceAgent()
    baseline = agent._family_variants(SequenceFamily.GLOBAL_RAMP, atom_count=4, memory_records=[])
    weak_memory = [
        MemoryRecord(
            campaign_id="campaign_old",
            source_candidate_id="seq_bad",
            lesson_type="failure_pattern",
            summary="Weak global ramp",
            signals={"sequence_family": SequenceFamily.GLOBAL_RAMP.value},
            reusable_tags=["weak", SequenceFamily.GLOBAL_RAMP.value],
        )
    ]

    reduced = agent._family_variants(SequenceFamily.GLOBAL_RAMP, atom_count=4, memory_records=weak_memory)

    assert len(reduced) < len(baseline)


def test_memory_agent_success_record_contains_spectral_gap() -> None:
    agent = MemoryAgent()
    register = RegisterCandidate(
        campaign_id="campaign_test",
        spec_id="spec_test",
        label="line-4-s7.0",
        layout_type="line",
        atom_count=4,
        coordinates=[(0.0, 0.0), (7.0, 0.0), (14.0, 0.0), (21.0, 0.0)],
        min_distance_um=7.0,
        blockade_radius_um=6.0,
        blockade_pair_count=0,
        van_der_waals_matrix=[[0.0] * 4 for _ in range(4)],
        feasibility_score=0.8,
        reasoning_summary="Register test.",
        metadata={"spacing_um": 7.0},
    )
    sequence = SequenceCandidate(
        campaign_id="campaign_test",
        spec_id="spec_test",
        register_candidate_id=register.id,
        label="seq-test",
        sequence_family=SequenceFamily.GLOBAL_RAMP,
        duration_ns=2000,
        amplitude=6.0,
        detuning=-12.0,
        phase=0.0,
        predicted_cost=0.2,
        reasoning_summary="Sequence test.",
    )
    result = EvaluationResult(
        campaign_id="campaign_test",
        sequence_candidate_id=sequence.id,
        register_candidate_id=register.id,
        nominal_score=0.82,
        robustness_score=0.78,
        worst_case_score=0.72,
        observable_score=0.8,
        objective_score=0.79,
        backend_choice=BackendType.LOCAL_PULSER_SIMULATION,
        estimated_cost=0.2,
        estimated_latency=0.2,
        final_rank=1,
        status=CandidateStatus.RANKED,
        reasoning_summary="Top candidate.",
        metadata={"score_std": 0.04, "nominal_observables": {"spectral_gap": 0.91}},
    )

    records = agent.run("campaign_test", [result], {sequence.id: sequence}, {register.id: register})

    success_record = next(record for record in records if record.lesson_type == "candidate_pattern")
    assert success_record.signals["spectral_gap"] == 0.91


def test_geometry_spacing_values_keep_baseline_without_memory() -> None:
    agent = GeometryAgent()

    spacings = agent._spacing_values([])

    assert spacings == [7.0, 8.5]
