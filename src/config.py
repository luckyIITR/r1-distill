"""Centralized config loaded from environment variables / .env file.

Import the settings singleton from anywhere:
    from src.config import settings
    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

Fail-fast: if a required key is missing, this raises at import time with a clear message.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # DeepSeek API
    deepseek_api_key: str = Field(..., description="DeepSeek API key")
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # HuggingFace
    hf_token: str | None = None

    # Weights & Biases
    wandb_api_key: str | None = None
    wandb_project: str = "r1-distill-qwen"
    wandb_entity: str | None = None

    # Cache locations
    hf_home: str | None = None


# Singleton. Importing this validates env at startup.
settings = Settings()  # type: ignore[call-arg]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]