from __future__ import annotations

import threading

from pymongo import MongoClient
from pymongo.database import Database

from packages.core.config import Settings, get_settings


COLLECTION_NAMES = [
    "experiment_goals",
    "register_candidates",
    "sequence_candidates",
    "robustness_reports",
    "campaigns",
    "agent_decisions",
    "memory",
    "evaluation_results",
]

_CLIENT: MongoClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_mongo_client(settings: Settings | None = None) -> MongoClient:
    global _CLIENT
    settings = settings or get_settings()
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not configured.")
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = MongoClient(
                    settings.mongodb_uri,
                    connectTimeoutMS=settings.mongodb_connect_timeout_ms,
                    serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
                    socketTimeoutMS=settings.mongodb_socket_timeout_ms,
                    maxPoolSize=settings.mongodb_max_pool_size,
                )
    return _CLIENT


def get_database(settings: Settings | None = None) -> Database:
    settings = settings or get_settings()
    client = get_mongo_client(settings)
    return client[settings.mongodb_db]


def close_mongo_client() -> None:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.close()
            _CLIENT = None
