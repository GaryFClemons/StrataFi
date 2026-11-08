"""Chat with a run's index: retrieve passages, then stream a grounded, cited answer.

``POST /chat`` takes the conversation so far and, optionally, a run whose indexes to search
(``run_id``), with a saved or unsaved retrieval config or a plain mode. The latest user message
is the search query; the passages go to the model numbered from 1 (``stratafi_core.chat``), and
the answer streams back as server-sent events:

- ``retrieval``: the passages the answer may cite, with the search's latency and the embedding
  revision, before the first token, so the screen can show its sources at once
- ``delta``: the next piece of the answer
- ``done``: the model's alias and revision, why it stopped, the passages the answer cited,
  token counts and timings (retrieval, time to first token, generation speed)
- ``error``: a code and a message; the stream ends after it

Without a ``run_id`` the conversation goes to the model alone, to compare models without
retrieval. Search problems (a missing run or index, the gateway down) are answered as problems
before the stream starts. Nothing is stored, and only counts and timings are logged.

``GET /chat/models`` lists the gateway's chat models for the screen's pickers.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from typing import Any, Literal

import httpx2
from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_api.routes.search import Hit, Mode, SearchIn, search
from stratafi_components.models import REVISION_HEADER
from stratafi_core.chat import PROMPT_VERSION, Passage, build_messages, citations_used
from stratafi_core.identity import HEADER as IDENTITY_HEADER

router = APIRouter(tags=["chat"])
logger = logging.getLogger("stratafi.api.chat")

ALIAS_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"
MAX_QUERY_CHARS = 2000
GATEWAY_TIMEOUT_SECONDS = 600.0


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(
        min_length=1, max_length=50, description="The conversation; the last is the question"
    )
    model: str = Field(
        default="chat-default", pattern=ALIAS_PATTERN, description="A chat model on the gateway"
    )
    run_id: uuid.UUID | None = Field(
        default=None, description="A run whose indexes to search; none chats with the model alone"
    )
    mode: Mode = "hybrid"
    retrieval_config_id: uuid.UUID | None = Field(
        default=None, description="A saved retrieval config: replaces mode and top_k"
    )
    retrieval_config: dict[str, Any] | None = Field(
        default=None, description="An unsaved stratafi.retrieval.v1 document"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Passages to give the model")
    as_of: date | None = Field(
        default=None,
        description="Search the versions in effect on this day (today by default), so a "
        "superseded or future version never answers",
    )
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=600, ge=16, le=4000)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    system_prompt: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _ends_with_a_question(self) -> "ChatIn":
        if self.messages[-1].role != "user":
            raise ValueError("the last message must be the user's")
        return self


class ChatPassage(BaseModel):
    number: int = Field(description="What the answer cites it as: [number]")
    chunk_id: str
    document_id: str
    entry: str | None
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    score: float
    text: str


class ChatModelOut(BaseModel):
    id: str
    revision: str | None = None
    served_model: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    context_length: int | None = None


class ChatModelsOut(BaseModel):
    models: list[ChatModelOut]
    default: str | None = Field(description="chat-default, when the gateway serves it")


def _gateway_url(request: Request) -> str:
    url = request.app.state.settings.models.gateway_url
    if not url:
        raise Problem(
            "unavailable", "No model gateway is configured.", code="models_not_configured"
        )
    return str(url).rstrip("/")


def _client(request: Request, url: str) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        base_url=url,
        timeout=httpx2.Timeout(GATEWAY_TIMEOUT_SECONDS, connect=5.0),
        trust_env=False,  # a proxy would see every passage
        transport=getattr(request.app.state, "gateway_transport", None),
    )


def _headers(request: Request) -> dict[str, str]:
    token = request.headers.get(IDENTITY_HEADER)
    return {IDENTITY_HEADER: token} if token else {}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _passages(hits: list[Hit]) -> list[ChatPassage]:
    return [
        ChatPassage(
            number=number,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            entry=hit.entry,
            heading_path=hit.heading_path,
            page_start=hit.page_start,
            page_end=hit.page_end,
            score=hit.score,
            text=hit.text,
        )
        for number, hit in enumerate(hits, start=1)
    ]


@router.get(
    "/chat/models",
    response_model=ChatModelsOut,
    summary="The model gateway's chat models",
    responses=problem_responses(401, 403, 503),
)
async def chat_models(
    request: Request, principal: Principal = Depends(require_role("viewer"))
) -> ChatModelsOut:
    url = _gateway_url(request)
    try:
        async with _client(request, url) as client:
            response = await client.get("/v1/models", headers=_headers(request))
    except httpx2.HTTPError:
        raise Problem("unavailable", code="gateway_unreachable") from None
    if response.status_code != 200:
        raise Problem("unavailable", code=f"gateway_refused_{response.status_code}")
    try:
        items = [item for item in response.json()["data"] if item.get("purpose") == "chat"]
        models = [ChatModelOut.model_validate(item) for item in items]
    except (ValueError, KeyError, TypeError):
        raise Problem("unavailable", code="gateway_bad_response") from None
    ids = {model.id for model in models}
    return ChatModelsOut(models=models, default="chat-default" if "chat-default" in ids else None)


@router.post(
    "/chat",
    summary="Ask about a run's documents: retrieve, then stream a cited answer as events",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events: retrieval, delta, done or error",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        **problem_responses(400, 401, 403, 404, 409, 503),
    },
)
async def chat(
    request: Request, body: ChatIn, principal: Principal = Depends(require_role("viewer"))
) -> StreamingResponse:
    url = _gateway_url(request)
    question = body.messages[-1].content
    started = time.perf_counter()
    retrieval: dict[str, Any] | None = None
    passages: list[ChatPassage] | None = None
    if body.run_id is not None:
        found = await run_in_threadpool(
            search,
            request,
            SearchIn(
                run_id=body.run_id,
                query=question[:MAX_QUERY_CHARS],
                mode=body.mode,
                top_k=body.top_k,
                retrieval_config_id=body.retrieval_config_id,
                retrieval_config=body.retrieval_config,
                as_of=body.as_of or datetime.now(UTC).date(),
            ),
            principal,
        )
        passages = _passages(found.results[: body.top_k])
        retrieval = {
            "run_id": str(body.run_id),
            "revision": found.revision,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "passages": [passage.model_dump(mode="json") for passage in passages],
        }
    messages = build_messages(
        [(message.role, message.content) for message in body.messages],
        [Passage(p.text, p.entry, tuple(p.heading_path)) for p in passages]
        if passages is not None
        else None,
        system_prompt=body.system_prompt,
    )
    payload: dict[str, Any] = {
        "model": body.model,
        "messages": messages,
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
        "stream": True,
    }
    if body.seed is not None:
        payload["seed"] = body.seed
    headers = _headers(request)

    async def events() -> AsyncIterator[str]:
        if retrieval is not None:
            yield _sse("retrieval", retrieval)
        asked = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        final: dict[str, Any] = {}
        revision: str | None = None
        try:
            async with (
                _client(request, url) as client,
                client.stream(
                    "POST", "/v1/chat/completions", json=payload, headers=headers
                ) as response,
            ):
                revision = response.headers.get(REVISION_HEADER)
                if response.status_code != 200:
                    code = (
                        "model_not_found"
                        if response.status_code == 404
                        else f"gateway_refused_{response.status_code}"
                    )
                    yield _sse("error", {"code": code, "detail": "The model gateway refused."})
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        error = chunk.get("error") or {}
                        yield _sse(
                            "error",
                            {
                                "code": str(error.get("type") or "gateway_failed"),
                                "detail": "The model stopped with an error.",
                            },
                        )
                        return
                    choice = (chunk.get("choices") or [{}])[0]
                    text = (choice.get("delta") or {}).get("content") or ""
                    if text:
                        first = first or time.perf_counter()
                        parts.append(text)
                        yield _sse("delta", {"text": text})
                    if choice.get("finish_reason"):
                        final = {
                            "finish_reason": choice["finish_reason"],
                            "usage": chunk.get("usage"),
                            "timings": chunk.get("timings") or {},
                        }
        except httpx2.TimeoutException:
            yield _sse("error", {"code": "gateway_timeout", "detail": "The model took too long."})
            return
        except (httpx2.HTTPError, ValueError):
            yield _sse(
                "error", {"code": "gateway_unreachable", "detail": "The model gateway is down."}
            )
            return
        answer = "".join(parts)
        finished = time.perf_counter()
        gateway_timings = final.get("timings") or {}
        timings = {
            "retrieval_ms": retrieval["latency_ms"] if retrieval else None,
            "ttft_ms": round((first - asked) * 1000, 1) if first else None,
            "generation_ms": round((finished - asked) * 1000, 1),
            "total_ms": round((finished - started) * 1000, 1),
            "load_ms": gateway_timings.get("load_ms"),
            "completion_tokens_per_second": gateway_timings.get("completion_tokens_per_second"),
        }
        logger.info(
            "chat answered",
            extra={
                "passages": len(passages or []),
                "answer_chars": len(answer),
                "total_ms": timings["total_ms"],
            },
        )
        yield _sse(
            "done",
            {
                "model": body.model,
                "revision": revision,
                "finish_reason": final.get("finish_reason", "stop"),
                "citations": citations_used(answer, len(passages or [])),
                "usage": final.get("usage"),
                "timings": timings,
                "prompt_version": PROMPT_VERSION,
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
