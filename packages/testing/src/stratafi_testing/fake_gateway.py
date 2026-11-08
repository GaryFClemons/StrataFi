"""A fake model gateway for tests and local development. Never deployed.

It speaks the OpenAI-compatible gateway API as far as Stratafi uses it:

- ``POST /v1/embeddings``: deterministic hashed vectors (``stratafi_core.fake_models``, as
  ``embed/fake-hash`` makes them), so texts that share words score higher
- ``POST /v1/rerank``: the Cohere/Jina shape, scored by word overlap (as ``rerank/fake-overlap``
  scores)
- ``GET /v1/models``: the aliases with their purpose, dimensions and revision
- ``POST /v1/chat/completions``: a deterministic echo of the last line's first words, or
  ``chat_reply`` when set; with ``refuse_chat``, a 403, as a gateway that refuses chat to the
  worker's service token answers
- every response names the alias's revision in ``X-Model-Revision``

Aliases are configurable; by default ``embed-default`` (384 dimensions), ``embed-local`` (768,
standing in for nomic-embed-text), ``rerank-default`` and ``chat-default``, like a gateway's
config.
``inject_fault(status, count)`` answers the next requests with an error, and ``records`` keeps
each request's endpoint, alias, input count and identity header, never content.

    uv run python -m stratafi_testing.fake_gateway    # on 127.0.0.1:8091
"""

import hashlib
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from stratafi_core.fake_models import EMBED_REVISION, RERANK_REVISION, hashed_vector, overlap_score

REVISION_HEADER = "X-Model-Revision"
_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class FakeAlias:
    name: str
    purpose: Literal["embed", "rerank", "chat"]
    dimensions: int | None = None
    served_name: str = "fake"
    weights_revision: str = "1"

    @property
    def revision(self) -> str:
        """A hash of what serves the alias: its name, size, weights and the fake algorithms."""
        material = (
            f"fake:{EMBED_REVISION}:{RERANK_REVISION}:"
            f"{self.served_name}:{self.dimensions}:{self.weights_revision}"
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]


DEFAULT_ALIASES = (
    FakeAlias("embed-default", "embed", 384, "fake-embed"),
    FakeAlias("embed-local", "embed", 768, "fake-nomic-embed-text"),
    FakeAlias("rerank-default", "rerank", served_name="fake-rerank"),
    FakeAlias("chat-default", "chat", served_name="fake-chat"),
)


@dataclass(frozen=True)
class GatewayCall:
    endpoint: str
    alias: str | None
    inputs: int
    identity: str | None


@dataclass
class FakeGateway:
    aliases: dict[str, FakeAlias] = field(
        default_factory=lambda: {alias.name: alias for alias in DEFAULT_ALIASES}
    )
    records: list[GatewayCall] = field(default_factory=list)
    # Chat answers with this text when set, instead of the echo.
    chat_reply: str | None = None
    # Chat refuses (403), as a gateway that refuses service tokens there does.
    refuse_chat: bool = False
    _faults: deque[int] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inject_fault(self, status: int, count: int = 1) -> None:
        with self._lock:
            self._faults.extend([status] * count)

    def take_fault(self) -> int | None:
        with self._lock:
            return self._faults.popleft() if self._faults else None

    def change_revision(self, alias: str, weights_revision: str) -> None:
        """Move an alias to other weights: its revision changes, as after a model swap."""
        current = self.aliases[alias]
        self.aliases[alias] = FakeAlias(
            current.name, current.purpose, current.dimensions, current.served_name, weights_revision
        )


def create_app(gateway: FakeGateway | None = None) -> FastAPI:
    gateway = gateway or FakeGateway()
    app = FastAPI(title="Fake model gateway")
    app.state.gateway = gateway

    def fault() -> JSONResponse | None:
        status = gateway.take_fault()
        if status is None:
            return None
        return JSONResponse(
            {"error": {"type": "fault"}}, status_code=status, headers={"Retry-After": "0"}
        )

    def record(request: Request, endpoint: str, alias: str | None, inputs: int) -> None:
        gateway.records.append(
            GatewayCall(endpoint, alias, inputs, request.headers.get("x-stratafi-identity"))
        )

    def alias_or_error(name: Any, purpose: str) -> FakeAlias | JSONResponse:
        alias = gateway.aliases.get(str(name))
        if alias is None or alias.purpose != purpose:
            return JSONResponse({"error": {"type": "not_found"}}, status_code=404)
        return alias

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        record(request, "models", None, 0)
        if (response := fault()) is not None:
            return response
        data = []
        for alias in gateway.aliases.values():
            item: dict[str, Any] = {
                "id": alias.name,
                "object": "model",
                "created": 0,
                "owned_by": "stratafi",
                "purpose": alias.purpose,
                "revision": alias.revision,
            }
            if alias.dimensions is not None:
                item["dimensions"] = alias.dimensions
            data.append(item)
        return JSONResponse({"object": "list", "data": data})

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        body = await request.json()
        raw = body.get("input")
        texts = [raw] if isinstance(raw, str) else list(raw or [])
        record(request, "embeddings", body.get("model"), len(texts))
        if (response := fault()) is not None:
            return response
        alias = alias_or_error(body.get("model"), "embed")
        if isinstance(alias, JSONResponse):
            return alias
        if not texts or not all(isinstance(text, str) for text in texts):
            return JSONResponse({"error": {"type": "invalid_request"}}, status_code=400)
        dimensions = alias.dimensions or 384
        data = [
            {"object": "embedding", "index": i, "embedding": hashed_vector(text, dimensions)}
            for i, text in enumerate(texts)
        ]
        tokens = sum(len(text.split()) for text in texts)
        return JSONResponse(
            {
                "object": "list",
                "data": data,
                "model": alias.served_name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
            headers={REVISION_HEADER: alias.revision},
        )

    @app.post("/v1/rerank")
    async def rerank(request: Request) -> JSONResponse:
        body = await request.json()
        documents = [
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in body.get("documents") or []
        ]
        record(request, "rerank", body.get("model"), len(documents))
        if (response := fault()) is not None:
            return response
        alias = alias_or_error(body.get("model"), "rerank")
        if isinstance(alias, JSONResponse):
            return alias
        query = str(body.get("query", ""))
        scored = sorted(
            ((i, overlap_score(query, doc)) for i, doc in enumerate(documents)),
            key=lambda pair: (-pair[1], pair[0]),
        )
        if body.get("top_n") is not None:
            scored = scored[: int(body["top_n"])]
        return JSONResponse(
            {
                "model": alias.served_name,
                "results": [{"index": i, "relevance_score": score} for i, score in scored],
            },
            headers={REVISION_HEADER: alias.revision},
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        """Deterministic: echoes the first words of the last user message's last line, so a
        test can tell which chunk an answer was for, and counts words as tokens. ``chat_reply``
        replaces the echo; ``refuse_chat`` refuses, as a gateway may refuse service tokens."""
        body = await request.json()
        messages = body.get("messages") or []
        record(request, "chat", body.get("model"), len(messages))
        if (response := fault()) is not None:
            return response
        if gateway.refuse_chat:
            return JSONResponse(
                {"error": {"code": "service_token_refused", "message": "chat needs a user"}},
                status_code=403,
            )
        alias = alias_or_error(body.get("model"), "chat")
        if isinstance(alias, JSONResponse):
            return alias
        last = str(messages[-1].get("content", "")) if messages else ""
        line = next((part for part in reversed(last.splitlines()) if part.strip()), "")
        content = gateway.chat_reply
        if content is None:
            content = "Context: " + " ".join(_WORD_RE.findall(line)[:8])
        prompt = sum(len(str(m.get("content", "")).split()) for m in messages)
        completion = len(content.split())
        return JSONResponse(
            {
                "id": "fake-chat",
                "object": "chat.completion",
                "model": alias.served_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                },
            },
            headers={REVISION_HEADER: alias.revision},
        )

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("STRATAFI_FAKE_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("STRATAFI_FAKE_GATEWAY_PORT", "8091")),
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
