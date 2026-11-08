"""The gateway's HTTP API: StrataFi's model gateway contract, served locally.

- ``GET /v1/models``: every model Ollama has pulled and every alias over one, with its
  ``purpose`` (``embed``, ``chat`` or ``rerank``), ``revision`` and, for embedders,
  ``dimensions``; plus the cross-encoders when the ``rerank`` extra is installed.
- ``POST /v1/embeddings``: OpenAI's shape, through Ollama's ``/api/embed``.
- ``POST /v1/chat/completions``: OpenAI's shape, through Ollama's ``/api/chat`` with the context
  window set (``num_ctx``), streamed as server-sent events when ``stream`` is true. Responses
  carry a ``timings`` object (load, prompt and completion milliseconds, tokens per second), and
  a stream's last chunk carries the usage and timings.
- ``POST /v1/rerank``: the Cohere/Jina shape, scored by a cross-encoder.
- ``GET /health``: whether Ollama answers, and its version.

Every model response names the model's revision in ``X-Model-Revision``. The gateway takes no
credentials and listens on 127.0.0.1 by default: it's for one machine. It logs endpoints,
models, counts and timings, never text.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from stratafi_gateway import REVISION_HEADER, __version__
from stratafi_gateway.ollama import Catalog, LocalModel, OllamaClient, OllamaError
from stratafi_gateway.rerank import RerankerPool
from stratafi_gateway.settings import GatewaySettings

log = logging.getLogger("stratafi_gateway")

MAX_INPUTS = 2048


class _Body(BaseModel):
    model_config = ConfigDict(extra="allow")


class EmbeddingsIn(_Body):
    model: str
    input: str | list[str]
    encoding_format: str | None = None


class ChatIn(_Body):
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    response_format: dict[str, Any] | None = None


class RerankIn(_Body):
    model: str
    query: str
    documents: list[str | dict[str, Any]]
    top_n: int | None = Field(default=None, ge=1)


def _error(status: int, code: str, message: str = "") -> JSONResponse:
    return JSONResponse({"error": {"type": code, "message": message}}, status_code=status)


def timings(data: dict[str, Any]) -> dict[str, float | None]:
    """Ollama's durations (nanoseconds) as milliseconds, with the generation speed."""

    def ms(key: str) -> float | None:
        value = data.get(key)
        return round(value / 1e6, 1) if isinstance(value, int | float) else None

    tokens, nanos = data.get("eval_count") or 0, data.get("eval_duration") or 0
    return {
        "load_ms": ms("load_duration"),
        "prompt_ms": ms("prompt_eval_duration"),
        "completion_ms": ms("eval_duration"),
        "total_ms": ms("total_duration"),
        "completion_tokens_per_second": round(tokens / (nanos / 1e9), 1) if nanos else None,
    }


def usage(data: dict[str, Any]) -> dict[str, int]:
    prompt = int(data.get("prompt_eval_count") or 0)
    completion = int(data.get("eval_count") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def finish_reason(data: dict[str, Any]) -> str:
    return "length" if data.get("done_reason") == "length" else "stop"


def chat_payload(body: ChatIn, model: LocalModel, settings: GatewaySettings) -> dict[str, Any]:
    """An OpenAI chat request as Ollama's ``/api/chat`` takes it."""
    options: dict[str, Any] = {"num_ctx": settings.num_ctx}
    if body.temperature is not None:
        options["temperature"] = body.temperature
    if body.top_p is not None:
        options["top_p"] = body.top_p
    if (limit := body.max_completion_tokens or body.max_tokens) is not None:
        options["num_predict"] = limit
    if body.seed is not None:
        options["seed"] = body.seed
    if body.stop:
        options["stop"] = [body.stop] if isinstance(body.stop, str) else list(body.stop)
    payload: dict[str, Any] = {
        "model": model.name,
        "messages": [
            {"role": str(m.get("role", "user")), "content": str(m.get("content") or "")}
            for m in body.messages
        ],
        "options": options,
    }
    if body.response_format:
        kind = body.response_format.get("type")
        if kind == "json_object":
            payload["format"] = "json"
        elif kind == "json_schema":
            schema = (body.response_format.get("json_schema") or {}).get("schema")
            payload["format"] = schema or "json"
    if settings.keep_alive:
        payload["keep_alive"] = settings.keep_alive
    return payload


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def create_app(
    settings: GatewaySettings | None = None,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    rerankers: RerankerPool | None = None,
) -> FastAPI:
    settings = settings or GatewaySettings.from_env()
    client = OllamaClient(
        settings.ollama_url, timeout=settings.timeout_seconds, transport=transport
    )
    catalog = Catalog(client, aliases=settings.aliases, ttl=settings.models_ttl_seconds)
    pool = rerankers or RerankerPool(settings.rerankers, device=settings.rerank_device)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await client.aclose()

    app = FastAPI(title="StrataFi local model gateway", version=__version__, lifespan=lifespan)
    app.state.settings, app.state.catalog, app.state.rerankers = settings, catalog, pool

    @app.exception_handler(OllamaError)
    async def ollama_error(_: Request, exc: OllamaError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Never echo the request: it holds document text.
        fields = sorted({".".join(str(p) for p in e.get("loc", ())[1:]) for e in exc.errors()})
        return _error(400, "invalid_request", "Check: " + ", ".join(fields))

    @app.get("/health")
    async def health() -> JSONResponse:
        try:
            version = await client.version()
        except OllamaError as exc:
            return JSONResponse(
                {"status": "degraded", "ollama": {"reachable": False, "error": exc.code}},
                status_code=503,
            )
        return JSONResponse({"status": "ok", "ollama": {"reachable": True, "version": version}})

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        data: list[dict[str, Any]] = []
        for name, model in await catalog.listing():
            item: dict[str, Any] = {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "ollama",
                "purpose": model.purpose,
                "revision": model.revision,
                "served_model": model.name,
                "context_length": model.context_length,
                "family": model.details.get("family"),
                "parameter_size": model.details.get("parameter_size"),
                "quantization": model.details.get("quantization_level"),
            }
            if model.dimensions is not None:
                item["dimensions"] = model.dimensions
            data.append(item)
        for alias, model_id in sorted(pool.aliases.items()):
            data.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "huggingface",
                    "purpose": "rerank",
                    "revision": pool.revision(alias),
                    "served_model": model_id,
                }
            )
        return JSONResponse({"object": "list", "data": data})

    @app.post("/v1/embeddings")
    async def embeddings(body: EmbeddingsIn) -> JSONResponse:
        texts = [body.input] if isinstance(body.input, str) else list(body.input)
        if not texts or len(texts) > MAX_INPUTS:
            return _error(400, "invalid_request", f"Send 1 to {MAX_INPUTS} inputs")
        if body.encoding_format not in (None, "float"):
            return _error(400, "invalid_request", "Only encoding_format=float is supported")
        model = await catalog.resolve(body.model, "embed")
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": model.name,
            "input": texts,
            "truncate": settings.truncate,
        }
        if settings.keep_alive:
            payload["keep_alive"] = settings.keep_alive
        data = await client.embed(payload)
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError(502, "ollama_bad_response")
        tokens = int(data.get("prompt_eval_count") or 0)
        log.info(
            "embeddings model=%s inputs=%d tokens=%d ms=%.0f",
            model.short_name,
            len(texts),
            tokens,
            (time.perf_counter() - started) * 1000,
        )
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": vector}
                    for i, vector in enumerate(vectors)
                ],
                "model": model.name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
            headers={REVISION_HEADER: model.revision},
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(body: ChatIn) -> JSONResponse | StreamingResponse:
        model = await catalog.resolve(body.model, "chat")
        payload = chat_payload(body, model, settings)
        completion_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        headers = {REVISION_HEADER: model.revision}

        if not body.stream:
            data = await client.chat(payload)
            content = (data.get("message") or {}).get("content")
            if not isinstance(content, str):
                raise OllamaError(502, "ollama_bad_response")
            log.info("chat model=%s tokens=%s", model.short_name, usage(data)["total_tokens"])
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": body.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": finish_reason(data),
                        }
                    ],
                    "usage": usage(data),
                    "timings": timings(data),
                },
                headers=headers,
            )

        def chunk(delta: dict[str, Any], reason: str | None, **extra: Any) -> str:
            return _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
                    **extra,
                }
            )

        async def events() -> AsyncIterator[str]:
            yield chunk({"role": "assistant", "content": ""}, None)
            try:
                async for part in client.chat_stream(payload):
                    if part.get("done"):
                        log.info(
                            "chat stream model=%s tokens=%s",
                            model.short_name,
                            usage(part)["total_tokens"],
                        )
                        yield chunk(
                            {}, finish_reason(part), usage=usage(part), timings=timings(part)
                        )
                        break
                    text = (part.get("message") or {}).get("content") or ""
                    if text:
                        yield chunk({"content": text}, None)
            except OllamaError as exc:
                yield _sse({"error": {"type": exc.code, "message": exc.message}})
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/rerank")
    async def rerank(body: RerankIn) -> JSONResponse:
        if not pool.has(body.model):
            return _error(404, "unknown_alias", f"No rerank model named {body.model!r}")
        documents = [
            item if isinstance(item, str) else str(item.get("text", "")) for item in body.documents
        ]
        if len(documents) > MAX_INPUTS:
            return _error(400, "invalid_request", f"Send at most {MAX_INPUTS} documents")
        started = time.perf_counter()
        scores = (
            await run_in_threadpool(pool.score, body.model, body.query, documents)
            if documents
            else []
        )
        ranked = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
        if body.top_n is not None:
            ranked = ranked[: body.top_n]
        log.info(
            "rerank model=%s documents=%d ms=%.0f",
            body.model,
            len(documents),
            (time.perf_counter() - started) * 1000,
        )
        return JSONResponse(
            {
                "model": body.model,
                "results": [{"index": i, "relevance_score": score} for i, score in ranked],
            },
            headers={REVISION_HEADER: pool.revision(body.model)},
        )

    return app
