from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from packages.db.init_db import initialize_database
from packages.db.mongodb import close_mongo_client
from packages.db.repositories import CryoSwarmRepository
from packages.orchestration.runner import run_demo_campaign


def main() -> int:
    initialize_database()
    repository = CryoSwarmRepository()
    summary = run_demo_campaign(repository=repository)
    print(f"Campaign {summary.campaign.id} completed.")
    if summary.top_candidate:
        print(
            "Top candidate:",
            summary.top_candidate.sequence_candidate_id,
            summary.top_candidate.backend_choice.value,
            summary.top_candidate.objective_score,
        )
    close_mongo_client()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
