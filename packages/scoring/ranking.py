from __future__ import annotations

from packages.core.enums import CandidateStatus
from packages.core.models import EvaluationResult


def rank_evaluations(results: list[EvaluationResult]) -> list[EvaluationResult]:
    ranked = sorted(
        results,
        key=lambda item: (
            item.objective_score,
            item.robustness_score,
            item.nominal_score,
        ),
        reverse=True,
    )

    output: list[EvaluationResult] = []
    for index, result in enumerate(ranked, start=1):
        output.append(
            result.model_copy(
                update={
                    "final_rank": index,
                    "status": CandidateStatus.RANKED,
                }
            )
        )
    return output
