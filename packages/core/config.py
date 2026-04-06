from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from packages.core.enums import AppEnvironment


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=False)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    log_level: str = "INFO"
    mongodb_uri: str = ""
    mongodb_db: str = "cryoswarm_q"
    mongodb_connect_timeout_ms: int = 5000
    mongodb_server_selection_timeout_ms: int = 5000
    mongodb_socket_timeout_ms: int = 10000
    mongodb_max_pool_size: int = 20
    cors_origins: str = ""
    api_key: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    pasqal_cloud_username: str = ""
    pasqal_cloud_password: str = ""
    pasqal_token: str = ""
    pasqal_cloud_project_id: str = ""
    default_collection: str = Field(default="connection_tests")

    @property
    def has_mongodb(self) -> bool:
        return bool(self.mongodb_uri)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def has_pasqal_cloud_credentials(self) -> bool:
        has_token_auth = bool(self.pasqal_token and self.pasqal_cloud_project_id)
        has_user_password_auth = all(
            [
                self.pasqal_cloud_username,
                self.pasqal_cloud_password,
                self.pasqal_cloud_project_id,
            ]
        )
        return has_token_auth or has_user_password_auth


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_env=AppEnvironment(os.getenv("APP_ENV", "development")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        mongodb_uri=os.getenv("MONGODB_URI", ""),
        mongodb_db=os.getenv("MONGODB_DB")
        or os.getenv("MONGODB_DATABASE", "cryoswarm_q"),
        mongodb_connect_timeout_ms=int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "5000")),
        mongodb_server_selection_timeout_ms=int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000")),
        mongodb_socket_timeout_ms=int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", "10000")),
        mongodb_max_pool_size=int(os.getenv("MONGODB_MAX_POOL_SIZE", "20")),
        cors_origins=os.getenv("CORS_ORIGINS", ""),
        api_key=os.getenv("CRYOSWARM_API_KEY", ""),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
        pasqal_cloud_username=os.getenv("PASQAL_CLOUD_USERNAME", ""),
        pasqal_cloud_password=os.getenv("PASQAL_CLOUD_PASSWORD", ""),
        pasqal_token=os.getenv("PASQAL_TOKEN", ""),
        pasqal_cloud_project_id=os.getenv("PASQAL_CLOUD_PROJECT_ID", ""),
        default_collection=os.getenv("MONGODB_COLLECTION", "connection_tests"),
    )
