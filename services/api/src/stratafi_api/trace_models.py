"""The shape of a trace, shared by eval runs' traces and the search playground's.
``stratafi_core.traces`` builds the content."""

from typing import Any

from pydantic import BaseModel, Field


class TraceEntryOut(BaseModel):
    rank: int
    chunk_id: str
    score: float
    relevant: bool | None = Field(description="Null when the query has no labels")


class TraceListOut(BaseModel):
    checkpoint: str
    length: int
    first_relevant_rank: int | None
    found: bool | None = Field(description="A relevant chunk is in the list; null without labels")
    top: list[TraceEntryOut] = Field(description="The first ten entries")


class SpanOut(BaseModel):
    id: str = Field(description="search, keyword, embed, dense, fuse, rerank or other")
    parent_id: str | None
    name: str
    kind: str
    start_ms: float = Field(description="When it started, from the search's start")
    duration_ms: float
    attributes: dict[str, Any]
    lists: list[TraceListOut] = Field(description="The lists this step made")


class TraceBody(BaseModel):
    total_ms: float
    spans: list[SpanOut] = Field(description="The search, then its steps in the order they ran")
    path: list[TraceListOut] = Field(
        description="Every checkpoint's list, in order: where the evidence is, without entries"
    )
    lost_at_checkpoint: str | None = Field(
        description="The first list that dropped evidence the list before it had"
    )
