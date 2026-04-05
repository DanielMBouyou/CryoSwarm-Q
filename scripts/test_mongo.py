from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from packages.core.config import get_settings
from packages.db.init_db import initialize_database
from packages.db.mongodb import close_mongo_client, get_database, get_mongo_client


def main() -> int:
    settings = get_settings()
    if not settings.has_mongodb:
        raise RuntimeError("MONGODB_URI is not configured.")

    initialize_database()
    client = get_mongo_client(settings)

    try:
        client.admin.command("ping")
        collection = get_database(settings)[settings.default_collection]
        result = collection.insert_one(
            {
                "project": "CryoSwarm-Q",
                "event": "mongodb_connection_test",
                "status": "connected",
                "created_at": datetime.now(UTC),
            }
        )
        print("MongoDB Atlas connection succeeded.")
        print(f"Inserted test document with _id={result.inserted_id}")
        return 0
    finally:
        close_mongo_client()


if __name__ == "__main__":
    raise SystemExit(main())
