from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search",
        alias="DATABASE_URL",
    )
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")

    model_config = SettingsConfigDict(
        env_file=(Path(__file__).resolve().parents[1] / ".env", Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
