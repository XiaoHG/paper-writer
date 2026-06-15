# paper-writer

AI-assisted paper writer with a LangGraph agent and Gradio UI.

## Project structure

- `agent.py`: LangGraph workflow and model/search integration
- `ui.py`: Gradio interface and state controls
- `config.py`: environment-driven runtime settings
- `app.py`: application factory and startup entrypoint
- `__main__.py`: root-level module entrypoint

## Setup

```bash
pip install -e .
```

Required environment variables:

```bash
TAVILY_API_KEY=...
OPENAI_API_KEY=...
```

Optional environment variables:

```bash
OPENAI_MODEL=gpt-5.4-mini
MAX_REVISIONS=2
MAX_ITERATIONS=10
TAVILY_MAX_RESULTS=2
APP_HOST=0.0.0.0
PORT1=7860
APP_SHARE=false
```

## Run

```bash
python app.py
```

Or:

```bash
python -m app
```

After installation, you can also run:

```bash
paper-writer
```
