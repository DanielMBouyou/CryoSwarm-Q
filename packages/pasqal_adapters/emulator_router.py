from __future__ import annotations

from packages.core.enums import BackendType
from packages.core.models import BackendChoice, ExperimentSpec, RobustnessReport, SequenceCandidate


def recommend_backend(
    spec: ExperimentSpec,
    sequence: SequenceCandidate,
    report: RobustnessReport,
) -> BackendChoice:
    atom_count = int(sequence.metadata.get("atom_count", spec.max_atoms))

    if atom_count <= 8 and report.robustness_score >= 0.60:
        backend = BackendType.EMU_SV_CANDIDATE
        estimated_cost = 0.35
        estimated_latency = 0.20
        rationale = "Small candidate with solid robustness. Exact-style emulation is justified."
    elif atom_count <= 16:
        backend = BackendType.EMU_MPS_CANDIDATE
        estimated_cost = 0.28
        estimated_latency = 0.25
        rationale = "Medium-scale candidate. Tensor-network style emulation is the most plausible next step."
    else:
        backend = BackendType.LOCAL_PULSER_SIMULATION
        estimated_cost = 0.15
        estimated_latency = 0.12
        rationale = "Candidate should remain in local simulation until scale and robustness improve."

    return BackendChoice(
        campaign_id=sequence.campaign_id,
        sequence_candidate_id=sequence.id,
        recommended_backend=backend,
        estimated_cost=estimated_cost,
        estimated_latency=estimated_latency,
        rationale=rationale,
        metadata={"objective_class": spec.objective_class, "atom_count": atom_count},
    )
