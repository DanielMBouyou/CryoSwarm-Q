from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from packages.core.enums import GoalStatus
from packages.core.models import ExperimentGoal
from packages.db.init_db import initialize_database
from packages.db.mongodb import close_mongo_client
from packages.db.repositories import CryoSwarmRepository


def main() -> int:
    initialize_database()
    repository = CryoSwarmRepository()
    goal = ExperimentGoal(
        title="Seeded neutral-atom geometry study",
        scientific_objective=(
            "Seed a compact experiment goal for geometry-aware neutral-atom candidate generation."
        ),
        desired_atom_count=6,
        preferred_geometry="mixed",
        status=GoalStatus.STORED,
    )
    repository.create_goal(goal)
    print(f"Stored seeded goal {goal.id}")
    close_mongo_client()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
