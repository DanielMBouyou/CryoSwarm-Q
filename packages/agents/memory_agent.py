from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.core.enums import AgentName
from packages.core.models import EvaluationResult, MemoryRecord, RegisterCandidate, SequenceCandidate


class MemoryAgent(BaseAgent):
    agent_name = AgentName.MEMORY

    def run(
        self,
        campaign_id: str,
        ranked_candidates: list[EvaluationResult],
        sequence_lookup: dict[str, SequenceCandidate],
        register_lookup: dict[str, RegisterCandidate],
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for result in ranked_candidates[:3]:
            sequence = sequence_lookup[result.sequence_candidate_id]
            register = register_lookup[result.register_candidate_id]
            score_std = float(result.metadata.get("score_std", 0.0))
            confidence = round(max(0.0, result.robustness_score * (1.0 - score_std)), 4)
            robustness_band = "strong" if result.robustness_score >= 0.70 else "moderate"
            nominal_observables = result.metadata.get("nominal_observables", {})
            spectral_gap = float(
                nominal_observables.get(
                    "spectral_gap",
                    result.metadata.get("hamiltonian_metrics", {}).get("spectral_gap", 0.0),
                )
            )
            records.append(
                MemoryRecord(
                    campaign_id=campaign_id,
                    source_candidate_id=result.sequence_candidate_id,
                    lesson_type="candidate_pattern",
                    summary=(
                        f"Candidate {result.sequence_candidate_id} showed {robustness_band} "
                        f"robustness with backend {result.backend_choice.value}."
                    ),
                    signals={
                        "objective_score": result.objective_score,
                        "robustness_score": result.robustness_score,
                        "worst_case_score": result.worst_case_score,
                        "backend_choice": result.backend_choice.value,
                        "sequence_family": sequence.sequence_family.value,
                        "layout_type": register.layout_type,
                        "atom_count": register.atom_count,
                        "spacing_um": register.metadata.get("spacing_um"),
                        "amplitude": sequence.amplitude,
                        "detuning": sequence.detuning,
                        "duration_ns": sequence.duration_ns,
                        "confidence": confidence,
                        "spectral_gap": spectral_gap,
                    },
                    reusable_tags=[
                        robustness_band,
                        result.backend_choice.value,
                        sequence.sequence_family.value,
                        register.layout_type,
                    ],
                )
            )

        if ranked_candidates:
            worst_result = ranked_candidates[-1]
            worst_sequence = sequence_lookup[worst_result.sequence_candidate_id]
            worst_register = register_lookup[worst_result.register_candidate_id]
            score_std = float(worst_result.metadata.get("score_std", 0.0))
            confidence = round(max(0.0, worst_result.robustness_score * (1.0 - score_std)), 4)
            degradation = max(0.0, worst_result.nominal_score - worst_result.worst_case_score)
            failure_tags = [
                "weak",
                "high_noise_sensitivity" if degradation >= 0.05 else "failure_pattern",
                worst_sequence.sequence_family.value,
                worst_register.layout_type,
            ]
            records.append(
                MemoryRecord(
                    campaign_id=campaign_id,
                    source_candidate_id=worst_result.sequence_candidate_id,
                    lesson_type="failure_pattern",
                    summary=(
                        f"Candidate {worst_result.sequence_candidate_id} underperformed with "
                        f"robustness {worst_result.robustness_score:.3f} and worst-case "
                        f"{worst_result.worst_case_score:.3f}."
                    ),
                    signals={
                        "objective_score": worst_result.objective_score,
                        "robustness_score": worst_result.robustness_score,
                        "worst_case_score": worst_result.worst_case_score,
                        "spacing_um": worst_register.metadata.get("spacing_um"),
                        "amplitude": worst_sequence.amplitude,
                        "detuning": worst_sequence.detuning,
                        "duration_ns": worst_sequence.duration_ns,
                        "layout_type": worst_register.layout_type,
                        "sequence_family": worst_sequence.sequence_family.value,
                        "confidence": confidence,
                        "spectral_gap": float(
                            worst_result.metadata.get("nominal_observables", {}).get(
                                "spectral_gap",
                                worst_result.metadata.get("hamiltonian_metrics", {}).get("spectral_gap", 0.0),
                            )
                        ),
                        "noise_degradation": round(degradation, 4),
                    },
                    reusable_tags=failure_tags,
                )
            )
        return records
