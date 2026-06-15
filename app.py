from __future__ import annotations

from agent import EssayWriterAgent
from config import Settings, get_settings
from ui import WriterUI


def create_app(settings: Settings | None = None) -> WriterUI:
    settings = settings or get_settings()
    agent = EssayWriterAgent(settings=settings)
    return WriterUI(agent.graph, settings=settings)


def main():
    app = create_app()
    app.launch()


if __name__ == "__main__":
    # Allow `python app.py` to start the Gradio app directly.
    main()
