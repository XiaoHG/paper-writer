"""Runtime configuration helpers.

The project is configured entirely through environment variables so it can be
run locally, in notebooks, or in hosted environments without code changes.
``load_dotenv()`` is called on import to allow a local ``.env`` file to supply
those variables during development.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


# Suppress tqdm warnings emitted by dependencies so the terminal/UI output stays
# focused on application state rather than progress-bar compatibility noise.
warnings.filterwarnings("ignore", message=".*TqdmWarning.*")
# Load environment variables from a local ``.env`` file if present.
load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Convert a string environment variable into a boolean value.

    Args:
        value: Raw string value read from the environment.
        default: Value to use when the variable is missing.

    Returns:
        ``True`` for common truthy strings such as ``"1"`` or ``"true"``,
        otherwise ``False``.
    """

    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings shared by the whole application.

    Attributes:
        openai_model: Chat model name passed to ``ChatOpenAI``.
        tavily_api_key: API key used by the Tavily search client.
        tavily_max_results: Number of snippets requested per Tavily query.
        max_revisions: Maximum number of draft generations before stopping.
        max_iterations: Safety cap for UI-driven graph stepping.
        host: Network host/interface used by the Gradio server.
        port: Optional Gradio server port. ``None`` lets Gradio pick a port.
        share: Whether Gradio should create a public share link.
    """

    # Default model chosen for low-friction local usage; callers can override it
    # with ``OPENAI_MODEL`` when they need another model or version.
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    # Tavily is optional at type level but practically required by the agent.
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
    # Keep result counts small by default to control prompt size and cost.
    tavily_max_results: int = int(os.getenv("TAVILY_MAX_RESULTS", "2"))
    # The graph produces one new draft per cycle until this limit is reached.
    max_revisions: int = int(os.getenv("MAX_REVISIONS", "2"))
    # Extra guardrail in the UI so manual stepping cannot loop forever.
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    # ``0.0.0.0`` makes the app reachable from other machines when desired.
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    # The original project expects ``PORT1`` rather than the more common ``PORT``.
    port: int | None = int(os.getenv("PORT1")) if os.getenv("PORT1") else None
    share: bool = _as_bool(os.getenv("APP_SHARE"), default=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Caching avoids reparsing environment variables repeatedly and guarantees
    that the app, agent, and UI all see a consistent configuration snapshot.
    """

    return Settings()
