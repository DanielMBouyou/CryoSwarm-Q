from __future__ import annotations

from packages.core.enums import BackendType
from packages.core.models import BackendChoice, ExperimentSpec, RobustnessReport, SequenceCandidate


def recommend_backend(
    spec: ExperimentSpec,
    sequence: SequenceCandidate,
    report: RobustnessReport,
) -> BackendChoice:
    atom_count = int(sequence.metadata.get("atom_count", spec.max_atoms))
    state_dimension = 2**atom_count
    hamiltonian_dimension = int(report.hamiltonian_metrics.get("dimension", state_dimension))

    if atom_count <= 8 and hamiltonian_dimension <= 256 and report.worst_case_score >= 0.55:
        backend = BackendType.EMU_SV_CANDIDATE
        estimated_cost = round(0.12 + state_dimension / 1024.0, 4)
        estimated_latency = 0.20
        rationale = "State dimension remains tractable and worst-case behavior stays credible."
    elif atom_count <= 16 and report.robustness_score >= 0.35:
        backend = BackendType.EMU_MPS_CANDIDATE
        estimated_cost = round(0.20 + atom_count / 100.0, 4)
        estimated_latency = 0.28
        rationale = "Candidate is larger but still promising enough for approximate tensor-network emulation."
    else:
        backend = BackendType.LOCAL_PULSER_SIMULATION
        estimated_cost = 0.15
        estimated_latency = 0.12
        rationale = "Candidate should remain in local Pulser simulation until robustness or scale improves."

    return BackendChoice(
        campaign_id=sequence.campaign_id,
        sequence_candidate_id=sequence.id,
        recommended_backend=backend,
        state_dimension=state_dimension,
        estimated_cost=estimated_cost,
        estimated_latency=estimated_latency,
        rationale=rationale,
        metadata={
            "objective_class": spec.objective_class,
            "atom_count": atom_count,
            "hamiltonian_dimension": hamiltonian_dimension,
        },
    )
