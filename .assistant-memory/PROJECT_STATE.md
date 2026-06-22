# Project State

Last updated: 2026-06-22

## Project summary

- Project name: `paper-writer`
- Type: small Python app for AI-assisted essay writing
- Main stack: Gradio, LangGraph, LangChain OpenAI, Tavily, Pydantic

## High-level architecture

- `app.py`
  - Builds `Settings`
  - Creates `EssayWriterAgent`
  - Injects the compiled graph into `WriterUI`
- `config.py`
  - Loads environment variables with `dotenv`
  - Exposes a cached `Settings` dataclass
- `agent.py`
  - Defines the LangGraph workflow
  - Calls OpenAI for planning, writing, critique, and search-query generation
  - Calls Tavily for research snippets
- `state.py`
  - Defines `AgentState` and `Queries`
- `ui.py`
  - Provides a Gradio interface for running, inspecting, rewinding, and editing
    graph state
- `prompts.py`
  - Stores prompt templates

## Runtime flow

Primary graph flow:

1. `planner`
2. `research_plan`
3. `generate`
4. If revisions remain: `reflect`
5. `research_critique`
6. Back to `generate`

Notes:

- Graph checkpoints are stored in in-memory SQLite only.
- The graph is compiled with `interrupt_after` on each main node.
- The UI advances the graph step by step via repeated `graph.invoke()` calls.

## Current analysis findings

- The project is compact and easy to reason about.
- The UI is more of an agent debugging console than a polished end-user app.
- Query generation has a useful fallback path when structured model output fails.

## Current risks and limitations

- `generation_node` does not explicitly include `critique` in the prompt, so
  revision quality depends mostly on extra search content rather than direct
  feedback incorporation.
- Research content is accumulated across cycles without deduplication or
  trimming, so prompt size may grow over time.
- Checkpoint memory is process-local because SQLite uses `:memory:`.
- UI state is held in instance variables, which may become fragile under more
  complex multi-user or concurrent usage.
- Environment parsing is direct and lightly validated.

## Current repository status

- No code changes made by the assistant before creating this memory directory.
- Initial project scan and architecture analysis completed.

## Suggested next actions

- If asked to continue managing the project, start from deeper review of
  `agent.py` and `ui.py`.
- If asked to improve behavior, prioritize revision quality, state durability,
  and research-content management.

