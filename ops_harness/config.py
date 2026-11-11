"""Configuration loaded from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Settings:
    """Runtime settings; every value can be overridden via env vars."""

    provider: str = field(default_factory=lambda: os.getenv("AGENT_LLM_PROVIDER", "mock"))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "gpt-4o-mini"))

    max_steps: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "10")))
    time_limit_s: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TIME_LIMIT_SECONDS", "120"))
    )
    tool_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))
    )
    tool_max_retries: int = field(default_factory=lambda: int(os.getenv("TOOL_MAX_RETRIES", "2")))

    db_path: str = field(default_factory=lambda: os.getenv("OPS_DB_PATH", "./ops_harness.db"))
    data_dir: str = field(
        default_factory=lambda: os.getenv("OPS_DATA_DIR", str(_DEFAULT_DATA_DIR))
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


def load_settings() -> Settings:
    return Settings()
