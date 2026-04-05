from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> int:
    load_dotenv(ENV_PATH)

    uri = require_env("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE", "cryoswarm_q")
    collection_name = os.getenv("MONGODB_COLLECTION", "connection_tests")

    client = MongoClient(uri, server_api=ServerApi("1"))

    try:
        client.admin.command("ping")

        collection = client[database_name][collection_name]
        document = {
            "project": "CryoSwarm-Q",
            "event": "mongodb_connection_test",
            "status": "connected",
            "created_at": datetime.now(UTC),
            "source": "scripts/test_mongo.py",
        }
        result = collection.insert_one(document)

        print("Pinged MongoDB Atlas successfully.")
        print(
            f"Inserted one document into {database_name}.{collection_name} "
            f"with _id={result.inserted_id}"
        )
        return 0
    except Exception as exc:
        print(f"MongoDB connection test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
