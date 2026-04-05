from packages.core.models import ScoringWeights
from packages.scoring.objective import compute_objective_score


def test_objective_score_uses_weighted_terms() -> None:
    score = compute_objective_score(
        observable_score=0.8,
        robustness=0.7,
        cost=0.2,
        latency=0.1,
        weights=ScoringWeights(alpha=0.5, beta=0.3, gamma=0.1, delta=0.1),
    )

    assert score == 0.58
