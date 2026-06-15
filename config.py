from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


warnings.filterwarnings("ignore", message=".*TqdmWarning.*")
load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
    tavily_max_results: int = int(os.getenv("TAVILY_MAX_RESULTS", "2"))
    max_revisions: int = int(os.getenv("MAX_REVISIONS", "2"))
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int | None = int(os.getenv("PORT1")) if os.getenv("PORT1") else None
    share: bool = _as_bool(os.getenv("APP_SHARE"), default=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
