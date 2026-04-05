from __future__ import annotations

from packages.core.logging import get_logger
from packages.db.mongodb import COLLECTION_NAMES, get_database


logger = get_logger(__name__)


def initialize_database() -> None:
    database = get_database()
    existing = set(database.list_collection_names())
    for collection_name in COLLECTION_NAMES:
        if collection_name not in existing:
            database.create_collection(collection_name)
        collection = database[collection_name]
        collection.create_index("id", unique=True)
        collection.create_index("created_at")
        collection.create_index("campaign_id")
        collection.create_index("goal_id")
    logger.info("MongoDB collections initialized.")


if __name__ == "__main__":
    initialize_database()
