import os

from pydantic import (
    Field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        #env_file="../../.env",
        #extra="ignore",
        env_ignore_empty=True
    )
    PROJECT_NAME: str = "ai-movie-assistant"

    # Database
    DB_URI: str = "sqlite:///./test.db"
    ASYNC_DB_URI: str
    DROP_DB: bool = False
    VERBOSE_DB: bool = False

    # Telegram bot
    TELEGRAM_TOKEN: str
    USE_WEBHOOK: bool = False

    # LLM agent
    KP_API_KEY: str
    LLM_NAME: str
    OPENAI_API_KEY: str
    VERBOSE_AGENT: bool = False
    USER_HISTORY_LIMIT: int = 5

    ENCODER_MODEL_NAME: str = "text-embedding-3-small"
    INDEX_DB_HOST: str = "index_db"
    INDEX_DB_PORT: int = 8000
    MOVIES_COLLECTION_NAME: str = "movies_collection"


settings = Settings()
