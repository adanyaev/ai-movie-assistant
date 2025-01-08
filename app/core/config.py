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

    DB_URI: str = "sqlite:///./test.db"
    ASYNC_DB_URI: str
    DROP_DB: bool = False

    TELEGRAM_TOKEN: str
    USE_WEBHOOK: bool = False
    


settings = Settings()
