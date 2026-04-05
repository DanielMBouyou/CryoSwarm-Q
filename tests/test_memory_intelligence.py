from __future__ import annotations

from packages.agents.geometry_agent import GeometryAgent
from packages.agents.memory_agent import MemoryAgent
from packages.agents.sequence_agent import SequenceAgent
from packages.core.enums import BackendType, CandidateStatus, SequenceFamily
from packages.core.models import EvaluationResult, MemoryRecord, RegisterCandidate, SequenceCandidate


def test_memory_agent_stores_failure_patterns() -> None:
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
    strong_sequence = SequenceCandidate(
        campaign_id="campaign_test",
        spec_id="spec_test",
        register_candidate_id=register.id,
        label="strong-seq",
        sequence_family=SequenceFamily.GLOBAL_RAMP,
        duration_ns=2000,
        amplitude=6.0,
        detuning=-12.0,
        phase=0.0,
        predicted_cost=0.2,
        reasoning_summary="Strong sequence.",
    )
    weak_sequence = SequenceCandidate(
        campaign_id="campaign_test",
        spec_id="spec_test",
        register_candidate_id=register.id,
        label="weak-seq",
        sequence_family=SequenceFamily.DETUNING_SCAN,
        duration_ns=2200,
        amplitude=4.0,
        detuning=-15.0,
        phase=0.0,
        predicted_cost=0.2,
        reasoning_summary="Weak sequence.",
    )
    ranked = [
        EvaluationResult(
            campaign_id="campaign_test",
            sequence_candidate_id=strong_sequence.id,
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
            metadata={"score_std": 0.04, "nominal_observables": {"spectral_gap": 0.9}},
        ),
        EvaluationResult(
            campaign_id="campaign_test",
            sequence_candidate_id=weak_sequence.id,
            register_candidate_id=register.id,
            nominal_score=0.5,
            robustness_score=0.32,
            worst_case_score=0.2,
            observable_score=0.45,
            objective_score=0.3,
            backend_choice=BackendType.LOCAL_PULSER_SIMULATION,
            estimated_cost=0.3,
            estimated_latency=0.3,
            final_rank=2,
            status=CandidateStatus.RANKED,
            reasoning_summary="Worst candidate.",
            metadata={"score_std": 0.08, "nominal_observables": {"spectral_gap": 0.2}},
        ),
    ]

    records = agent.run(
        "campaign_test",
        ranked,
        {strong_sequence.id: strong_sequence, weak_sequence.id: weak_sequence},
        {register.id: register},
    )

    assert any(record.lesson_type == "failure_pattern" for record in records)
    failure_record = next(record for record in records if record.lesson_type == "failure_pattern")
    assert "weak" in failure_record.reusable_tags
    assert "high_noise_sensitivity" in failure_record.reusable_tags
    assert failure_record.signals["confidence"] >= 0.0


def test_geometry_spacing_values_use_confident_memory_and_avoid_failures() -> None:
    agent = GeometryAgent()
    memory_records = [
        MemoryRecord(
            campaign_id="campaign_old",
            source_candidate_id="seq_good",
            lesson_type="candidate_pattern",
            summary="Confident spacing",
            signals={"spacing_um": 7.4, "confidence": 0.82},
            reusable_tags=["strong"],
        ),
        MemoryRecord(
            campaign_id="campaign_old",
            source_candidate_id="seq_bad",
            lesson_type="failure_pattern",
            summary="Avoid this spacing",
            signals={"spacing_um": 8.5, "confidence": 0.2},
            reusable_tags=["weak", "high_noise_sensitivity"],
        ),
    ]

    spacings = agent._spacing_values(memory_records)

    assert 7.4 in spacings
    assert 6.9 in spacings or 7.9 in spacings
    assert 8.5 not in spacings


def test_sequence_agent_adds_refined_variants_from_memory() -> None:
    agent = SequenceAgent()
    memory_records = [
        MemoryRecord(
            campaign_id="campaign_old",
            source_candidate_id="seq_good",
            lesson_type="candidate_pattern",
            summary="Strong global ramp",
            signals={
                "sequence_family": SequenceFamily.GLOBAL_RAMP.value,
                "amplitude": 6.0,
                "detuning": -12.0,
                "duration_ns": 2300,
                "confidence": 0.85,
            },
            reusable_tags=["strong", SequenceFamily.GLOBAL_RAMP.value],
        )
    ]

    variants = agent._family_variants(SequenceFamily.GLOBAL_RAMP, atom_count=4, memory_records=memory_records)
    refined = [variant for variant in variants if variant[4].startswith("refined_memory")]

    assert refined
    amplitudes = {variant[1] for variant in refined}
    detunings = {variant[2] for variant in refined}
    assert 5.4 in amplitudes or 6.6 in amplitudes
    assert -10.8 in detunings or -13.2 in detunings
