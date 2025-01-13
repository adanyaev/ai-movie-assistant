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
    VERBOSE_AGENT: bool = False
    USER_HISTORY_LIMIT: int = 5


settings = Settings()
