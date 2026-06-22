"""Application composition helpers.

This module wires together the three top-level pieces of the project:

- ``Settings``: runtime configuration loaded from environment variables
- ``EssayWriterAgent``: the LangGraph-based essay workflow
- ``WriterUI``: the Gradio interface used to operate and inspect the workflow
"""

from __future__ import annotations

from agent import EssayWriterAgent
from config import Settings, get_settings
from ui import WriterUI


def create_app(settings: Settings | None = None) -> WriterUI:
    """Build the UI application with a configured agent instance.

    Args:
        settings: Optional prebuilt settings object. Passing one is useful for
            tests or for callers that want explicit control over configuration.

    Returns:
        A fully constructed ``WriterUI`` object that owns the Gradio demo.
    """

    # Fall back to the cached environment-derived settings for normal runtime.
    settings = settings or get_settings()
    # The agent owns the LangGraph workflow and model/search integrations.
    agent = EssayWriterAgent(settings=settings)
    # The UI only needs the compiled graph and the same runtime settings.
    return WriterUI(agent.graph, settings=settings)


def main():
    """Create the application and launch the Gradio server."""

    app = create_app()
    app.launch()


if __name__ == "__main__":
    # Allow `python app.py` to start the Gradio app directly.
    main()
