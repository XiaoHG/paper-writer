from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict):
    task: str
    lnode: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    queries: List[str]
    revision_number: int
    max_revisions: int
    count: Annotated[int, operator.add]


class Queries(BaseModel):
    queries: list[str] = Field(default_factory=list)
