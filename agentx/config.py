"""Configuration settings for AgentX framework."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """AgentX system settings."""

    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    WORKFLOW_TIMEOUT_SECONDS: float = float(os.getenv("WORKFLOW_TIMEOUT_SECONDS", "30.0"))
    MEMORY_STORE_TYPE: str = os.getenv("MEMORY_STORE_TYPE", "in_memory")
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))


settings = Settings()
