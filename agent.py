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
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model = ChatOpenAI(model=self.settings.openai_model, temperature=0)
        self.tavily = TavilyClient(api_key=self._get_tavily_api_key())
        self.graph = self._build_graph()

    def _get_tavily_api_key(self) -> str:
        api_key = self.settings.tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is required.")
        return api_key

    def _build_graph(self):
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate",
            self.should_continue,
            {END: END, "reflect": "reflect"},
        )
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        return builder.compile(
            checkpointer=memory,
            interrupt_after=[
                "planner",
                "generate",
                "reflect",
                "research_plan",
                "research_critique",
            ],
        )

    def _normalize_queries(self, raw_text: str | None, limit: int) -> list[str]:
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return []

        try:
            parsed = Queries.model_validate_json(raw_text)
            cleaned = [query.strip() for query in parsed.queries if query.strip()]
            return cleaned[:limit]
        except Exception:
            pass

        try:
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
            cleaned_line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned_line:
                lines.append(cleaned_line)

        if len(lines) == 1 and ";" in lines[0]:
            lines = [item.strip() for item in lines[0].split(";") if item.strip()]

        return lines[:limit]

    def _generate_queries(
        self,
        system_prompt: str,
        human_input: str,
        limit: int,
    ) -> list[str]:
        try:
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
        content: list[str] = []
        for query in queries:
            response = self.tavily.search(
                query=query,
                max_results=self.settings.tavily_max_results,
            )
            for result in response.get("results", []):
                snippet = result.get("content")
                if snippet:
                    content.append(snippet)
        return content

    def plan_node(self, state: AgentState) -> dict:
        response = self.model.invoke(
            [
                SystemMessage(content=PLAN_PROMPT),
                HumanMessage(content=state["task"]),
            ]
        )
        return {"plan": response.content, "lnode": "planner", "count": 1}

    def research_plan_node(self, state: AgentState) -> dict:
        queries = self._generate_queries(RESEARCH_PLAN_PROMPT, state["task"], 3)
        content = list(state.get("content") or [])
        content.extend(self._search_content(queries))
        return {
            "content": content,
            "queries": queries,
            "lnode": "research_plan",
            "count": 1,
        }

    def generation_node(self, state: AgentState) -> dict:
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
            "revision_number": state.get("revision_number", 0) + 1,
            "lnode": "generate",
            "count": 1,
        }

    def reflection_node(self, state: AgentState) -> dict:
        response = self.model.invoke(
            [
                SystemMessage(content=REFLECTION_PROMPT),
                HumanMessage(content=state["draft"]),
            ]
        )
        return {"critique": response.content, "lnode": "reflect", "count": 1}

    def research_critique_node(self, state: AgentState) -> dict:
        queries = self._generate_queries(
            RESEARCH_CRITIQUE_PROMPT,
            state["critique"],
            2,
        )
        content = list(state.get("content") or [])
        content.extend(self._search_content(queries))
        return {
            "content": content,
            "queries": queries,
            "lnode": "research_critique",
            "count": 1,
        }

    def should_continue(self, state: AgentState):
        # Stop once the configured number of drafts has been produced.
        if state["revision_number"] >= state["max_revisions"]:
            return END
        return "reflect"
