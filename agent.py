"""LangGraph agent for planning, researching, drafting, and revising essays.

Graph flow:

            START
              |
              v
      +---------------+
      |    planner    |
      +---------------+
              |
              v
      +---------------+
      | research_plan |
      +---------------+
              |
              |
              |------------------------------------------+
              |               revision loop              |
              v                                          |
      +---------------+      +-----------+     +-------------------+ 
      |    generate   |----->|  reflect  |---->| research_critique |
      +---------------+      +-----------+     +-------------------+  
              |                                                 
 (if revision_number >= max_revisions)
              | 
              v
             END

The main path enters the revision loop at ``generate``. If more revisions are
needed, the graph continues through ``reflect`` and ``research_critique`` and
then loops back to ``generate`` again.

The ``generate -> reflect -> research_critique -> generate`` section forms the
revision loop. Whenever the graph structure changes, this diagram must be
updated in the same change so the code comments stay accurate.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from tavily import TavilyClient

from config import Settings, get_settings
from prompts import (
    PLAN_PROMPT,
    REFLECTION_PROMPT,
    RESEARCH_CRITIQUE_PROMPT,
    RESEARCH_PLAN_PROMPT,
    WRITER_PROMPT,
)
from state import AgentState, Queries


class EssayWriterAgent:
    """Build and run the essay-writing workflow.

    The agent combines:

    - an OpenAI chat model for planning, drafting, critique, and query creation
    - Tavily search for lightweight external research
    - a LangGraph state machine to coordinate the multi-step loop
    """

    def __init__(self, settings: Settings | None = None):
        """Create model/search clients and compile the LangGraph workflow."""

        self.settings = settings or get_settings()
        # Temperature 0 favors deterministic, repeatable graph behavior.
        self.model = ChatOpenAI(model=self.settings.openai_model, temperature=0)
        self.tavily = TavilyClient(api_key=self._get_tavily_api_key())
        self.graph = self._build_graph()

    def _get_tavily_api_key(self) -> str:
        """Resolve the Tavily API key and fail early when it is missing."""

        api_key = self.settings.tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is required.")
        return api_key

    def _build_graph(self):
        """Assemble and compile the LangGraph state machine.

        The graph intentionally interrupts after every major node so the UI can
        inspect state, let the user edit values, and then continue execution.

        Graph diagram:

            planner -> research_plan -> generate
                                          |
                                          +-> END                if revisions are done
                                          |
                                          +-> reflect
                                              -> research_critique
                                              -> generate        revision loop
        """

        # In-memory checkpointing keeps thread history available during the
        # current process without creating any on-disk state files.
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        builder = StateGraph(AgentState)

        # Register each callable node under the name that will appear in UI
        # status displays and checkpoint history.
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.set_entry_point("planner")

        # After each draft, either stop or continue into critique/revision.
        builder.add_conditional_edges(
            "generate",
            self.should_continue,
            {END: END, "reflect": "reflect"},
        )

        # The rest of the flow is linear.
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        return builder.compile(
            checkpointer=memory,
            # Interrupting after every node gives the Gradio UI fine-grained
            # control over how far the workflow runs in one click.
            interrupt_after=[
                "planner",
                "generate",
                "reflect",
                "research_plan",
                "research_critique",
            ],
        )

    def _normalize_queries(self, raw_text: str | None, limit: int) -> list[str]:
        """Parse a model response into a clean list of search queries.

        The model may return valid JSON, a JSON-like dict, a bare list, bullet
        points, or even a single semicolon-delimited line. This helper turns
        those variants into a consistent ``list[str]``.
        """

        raw_text = (raw_text or "").strip()
        if not raw_text:
            return []

        try:
            # First try the strictest path: Pydantic validation from JSON text.
            parsed = Queries.model_validate_json(raw_text)
            cleaned = [query.strip() for query in parsed.queries if query.strip()]
            return cleaned[:limit]
        except Exception:
            pass

        try:
            # Next accept plain JSON that may not exactly match model_validate_json.
            parsed_json = json.loads(raw_text)
            if isinstance(parsed_json, dict):
                queries = parsed_json.get("queries", [])
            elif isinstance(parsed_json, list):
                queries = parsed_json
            else:
                queries = []
            cleaned = [str(query).strip() for query in queries if str(query).strip()]
            if cleaned:
                return cleaned[:limit]
        except Exception:
            pass

        lines = []
        for line in raw_text.splitlines():
            # Remove common list markers like ``- item`` or ``1. item``.
            cleaned_line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned_line:
                lines.append(cleaned_line)

        # Handle one-line outputs such as ``query A; query B``.
        if len(lines) == 1 and ";" in lines[0]:
            lines = [item.strip() for item in lines[0].split(";") if item.strip()]

        return lines[:limit]

    def _generate_queries(
        self,
        system_prompt: str,
        human_input: str,
        limit: int,
    ) -> list[str]:
        """Ask the model for research queries with a structured-output fallback.

        Args:
            system_prompt: The role instruction for the query-generation step.
            human_input: User task or critique used to derive queries.
            limit: Maximum number of queries to keep.
        """

        try:
            # Prefer structured output when the model/provider supports it well.
            structured_model = self.model.with_structured_output(Queries)
            queries = structured_model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_input),
                ]
            )
            cleaned = [query.strip() for query in queries.queries if query.strip()]
            if cleaned:
                return cleaned[:limit]
        except Exception:
            pass

        # If structured output fails, relax the requirement and parse the model's
        # free-form response into a best-effort query list.
        fallback_prompt = (
            f"{system_prompt}\n"
            "Return only search queries. Prefer JSON like "
            f'{{"queries": ["query 1", "query 2"]}}. Maximum {limit} queries.'
        )
        response = self.model.invoke(
            [
                SystemMessage(content=fallback_prompt),
                HumanMessage(content=human_input),
            ]
        )
        return self._normalize_queries(response.content, limit)

    def _search_content(self, queries: Iterable[str]) -> list[str]:
        """Run Tavily searches and collect snippet text for prompt context."""

        content: list[str] = []
        for query in queries:
            response = self.tavily.search(
                query=query,
                max_results=self.settings.tavily_max_results,
            )
            for result in response.get("results", []):
                # Only the text snippet is retained; URLs/titles are discarded.
                snippet = result.get("content")
                if snippet:
                    content.append(snippet)
        return content

    def plan_node(self, state: AgentState) -> dict:
        """Create an essay outline from the original user task."""

        response = self.model.invoke(
            [
                SystemMessage(content=PLAN_PROMPT),
                HumanMessage(content=state["task"]),
            ]
        )
        # ``lnode`` tracks the last node that completed, and ``count`` feeds the
        # additive progress indicator defined in ``AgentState``.
        return {"plan": response.content, "lnode": "planner", "count": 1}

    def research_plan_node(self, state: AgentState) -> dict:
        """Generate initial research queries and append search results."""

        queries = self._generate_queries(RESEARCH_PLAN_PROMPT, state["task"], 3)
        # Copy the list so node logic does not mutate shared state in place.
        content = list(state.get("content") or [])
        content.extend(self._search_content(queries))
        return {
            "content": content,
            "queries": queries,
            "lnode": "research_plan",
            "count": 1,
        }

    def generation_node(self, state: AgentState) -> dict:
        """Write the next draft using the task, outline, and research snippets."""

        # Research snippets are flattened into one prompt section for the model.
        content = "\n\n".join(state.get("content") or [])
        response = self.model.invoke(
            [
                SystemMessage(content=WRITER_PROMPT.format(content=content)),
                HumanMessage(
                    content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
                ),
            ]
        )
        return {
            "draft": response.content,
            # Each pass through this node counts as one produced draft revision.
            "revision_number": state.get("revision_number", 0) + 1,
            "lnode": "generate",
            "count": 1,
        }

    def reflection_node(self, state: AgentState) -> dict:
        """Critique the latest draft and suggest improvements."""

        response = self.model.invoke(
            [
                SystemMessage(content=REFLECTION_PROMPT),
                HumanMessage(content=state["draft"]),
            ]
        )
        return {"critique": response.content, "lnode": "reflect", "count": 1}

    def research_critique_node(self, state: AgentState) -> dict:
        """Search for information that can address the critique in the next draft."""

        queries = self._generate_queries(
            RESEARCH_CRITIQUE_PROMPT,
            state["critique"],
            2,
        )
        # Revision research is appended to the same content pool so later drafts
        # can use both the initial and follow-up research context.
        content = list(state.get("content") or [])
        content.extend(self._search_content(queries))
        return {
            "content": content,
            "queries": queries,
            "lnode": "research_critique",
            "count": 1,
        }

    def should_continue(self, state: AgentState):
        """Decide whether the graph should stop after the current draft."""

        # Stop once the configured number of drafts has been produced.
        if state["revision_number"] >= state["max_revisions"]:
            return END
        return "reflect"
