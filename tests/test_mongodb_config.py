from __future__ import annotations

"""MongoDB client configuration and timeout tests."""

import inspect

import apps.api.dependencies as deps_mod
import packages.db.mongodb as mongo_mod
from packages.core.config import Settings
from packages.db.mongodb import COLLECTION_NAMES


def test_default_timeouts_are_bounded() -> None:
    """Verify default MongoDB timeouts and pool sizes are finite and reasonable."""
    settings = Settings(mongodb_uri="mongodb://fake")

    assert settings.mongodb_connect_timeout_ms > 0
    assert settings.mongodb_connect_timeout_ms <= 30000
    assert settings.mongodb_server_selection_timeout_ms > 0
    assert settings.mongodb_server_selection_timeout_ms <= 30000
    assert settings.mongodb_socket_timeout_ms > 0
    assert settings.mongodb_socket_timeout_ms <= 60000
    assert settings.mongodb_max_pool_size > 0
    assert settings.mongodb_max_pool_size <= 200


def test_mongo_client_passes_timeouts(monkeypatch) -> None:
    """Verify that get_mongo_client forwards timeout settings to MongoClient."""
    monkeypatch.setattr(mongo_mod, "_CLIENT", None)

    captured_kwargs: dict[str, int] = {}

    class FakeMongoClient:
        def __init__(self, uri: str, **kwargs: int) -> None:
            self.uri = uri
            captured_kwargs.update(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr(mongo_mod, "MongoClient", FakeMongoClient)

    settings = Settings(
        mongodb_uri="mongodb://fake",
        mongodb_connect_timeout_ms=3000,
        mongodb_server_selection_timeout_ms=4000,
        mongodb_socket_timeout_ms=8000,
        mongodb_max_pool_size=15,
    )

    mongo_mod.get_mongo_client(settings)

    assert captured_kwargs["connectTimeoutMS"] == 3000
    assert captured_kwargs["serverSelectionTimeoutMS"] == 4000
    assert captured_kwargs["socketTimeoutMS"] == 8000
    assert captured_kwargs["maxPoolSize"] == 15

    monkeypatch.setattr(mongo_mod, "_CLIENT", None)


def test_close_mongo_client_is_idempotent(monkeypatch) -> None:
    """Verify that close_mongo_client can be called multiple times safely."""
    monkeypatch.setattr(mongo_mod, "_CLIENT", None)

    mongo_mod.close_mongo_client()
    mongo_mod.close_mongo_client()


def test_dependency_does_not_call_initialize_database() -> None:
    """Verify request-time repository creation does not initialize MongoDB."""
    source = inspect.getsource(deps_mod.get_repository)
    assert "initialize_database" not in source


def test_init_db_idempotent(monkeypatch) -> None:
    """Verify initialize_database performs remote work only on the first call."""
    import packages.db.init_db as init_mod

    init_mod.reset_initialization_flag()
    call_count = 0

    class FakeCollection:
        def create_index(self, *args, **kwargs) -> None:
            return None

    class FakeDatabase:
        def list_collection_names(self) -> list[str]:
            nonlocal call_count
            call_count += 1
            return list(COLLECTION_NAMES)

        def create_collection(self, name: str) -> None:
            return None

        def __getitem__(self, name: str) -> FakeCollection:
            return FakeCollection()

    monkeypatch.setattr(init_mod, "get_database", lambda: FakeDatabase())

    init_mod.initialize_database()
    init_mod.initialize_database()
    init_mod.initialize_database()

    assert call_count == 1
    init_mod.reset_initialization_flag()
