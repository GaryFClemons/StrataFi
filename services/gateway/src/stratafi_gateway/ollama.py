"""Ollama, as the gateway uses it: an async client and a catalog of the models it has pulled.

The catalog turns ``/api/tags`` and ``/api/show`` into what the gateway contract needs for each
model: its purpose (``embed`` when Ollama reports the ``embedding`` capability, ``chat`` when it
reports ``completion``), its dimensions, its context length, and a **revision**: the first 16
hex digits of the weights' digest. Pulling a model again, or pointing an alias at another one,
changes the revision, so StrataFi's run cache never mixes vectors from two sets of weights.

Errors become ``OllamaError`` with the status the gateway answers: 503 when Ollama is down and
504 when it times out (callers retry both), 404 for an unknown model, 400 for a bad request.
Nothing here logs request or response text.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx2

Purpose = Literal["embed", "chat"]


class OllamaError(Exception):
    def __init__(self, status: int, code: str, message: str = "") -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LocalModel:
    name: str  # Ollama's full name, e.g. "nomic-embed-text:latest"
    digest: str
    purpose: Purpose
    dimensions: int | None = None
    context_length: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def short_name(self) -> str:
        return self.name.removesuffix(":latest")

    @property
    def revision(self) -> str:
        return self.digest.removeprefix("sha256:")[:16]


def describe(show: Mapping[str, Any]) -> tuple[Purpose | None, int | None, int | None]:
    """A model's purpose, dimensions and context length from ``/api/show``."""
    capabilities = show.get("capabilities") or []
    purpose: Purpose | None = (
        "embed" if "embedding" in capabilities else "chat" if "completion" in capabilities else None
    )
    info = show.get("model_info") or {}
    dimensions = context = None
    for key, value in info.items():
        if not isinstance(value, int):
            continue
        if key.endswith(".embedding_length"):
            dimensions = value
        elif key.endswith(".context_length"):
            context = value
    return purpose, (dimensions if purpose == "embed" else None), context


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 600.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx2.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx2.Timeout(timeout, connect=5.0),
            trust_env=False,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _call(self, method: str, path: str, body: Any = None) -> Any:
        try:
            response = await self._client.request(method, path, json=body)
        except httpx2.TimeoutException:
            raise OllamaError(504, "ollama_timeout", "Ollama took too long to answer") from None
        except httpx2.HTTPError:
            raise OllamaError(503, "ollama_unreachable", "Is Ollama running?") from None
        _raise_for(response.status_code, response.content)
        try:
            return response.json()
        except ValueError:
            raise OllamaError(502, "ollama_bad_response") from None

    async def version(self) -> str:
        return str((await self._call("GET", "/api/version")).get("version", ""))

    async def tags(self) -> list[dict[str, Any]]:
        return list((await self._call("GET", "/api/tags")).get("models") or [])

    async def show(self, name: str) -> dict[str, Any]:
        return dict(await self._call("POST", "/api/show", {"model": name}))

    async def embed(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(await self._call("POST", "/api/embed", payload))

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(await self._call("POST", "/api/chat", {**payload, "stream": False}))

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """``/api/chat`` with ``stream: true``: one parsed NDJSON line at a time."""
        try:
            async with self._client.stream(
                "POST", "/api/chat", json={**payload, "stream": True}
            ) as response:
                if response.status_code != 200:
                    _raise_for(response.status_code, await response.aread())
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        part = json.loads(line)
                    except ValueError:
                        raise OllamaError(502, "ollama_bad_response") from None
                    if isinstance(part, dict) and part.get("error"):
                        raise OllamaError(502, "ollama_failed", str(part["error"])[:200])
                    yield part
        except httpx2.TimeoutException:
            raise OllamaError(504, "ollama_timeout", "Ollama took too long to answer") from None
        except httpx2.HTTPError:
            raise OllamaError(503, "ollama_unreachable", "Is Ollama running?") from None


def _raise_for(status: int, content: bytes) -> None:
    if status == 200:
        return
    message = ""
    try:
        body = json.loads(content or b"{}")
        if isinstance(body, dict):
            message = str(body.get("error") or "")[:200]
    except ValueError:
        pass
    if status == 404:
        raise OllamaError(404, "model_not_found", message)
    if 400 <= status < 500:
        raise OllamaError(400, "invalid_request", message)
    raise OllamaError(502, "ollama_failed", message)


class Catalog:
    """The models Ollama has pulled, with the gateway's aliases over them.

    ``/api/tags`` is asked at most every ``ttl`` seconds, and ``/api/show`` once per digest. An
    unknown name refreshes the list once before it's refused, so a model pulled a moment ago
    works at once.
    """

    def __init__(
        self, client: OllamaClient, *, aliases: Mapping[str, str], ttl: float = 10.0
    ) -> None:
        self._client = client
        self._aliases = dict(aliases)
        self._ttl = ttl
        self._described: dict[str, tuple[Purpose | None, int | None, int | None]] = {}
        self._models: dict[str, LocalModel] = {}
        self._at = float("-inf")
        self._lock = asyncio.Lock()

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    async def refresh(self, *, force: bool = False) -> dict[str, LocalModel]:
        async with self._lock:
            if not force and time.monotonic() - self._at < self._ttl:
                return self._models
            models: dict[str, LocalModel] = {}
            for tag in await self._client.tags():
                name, digest = str(tag.get("name") or ""), str(tag.get("digest") or "")
                if not name or not digest:
                    continue
                if digest not in self._described:
                    self._described[digest] = describe(await self._client.show(name))
                purpose, dimensions, context = self._described[digest]
                if purpose is None:
                    continue
                model = LocalModel(
                    name, digest, purpose, dimensions, context, dict(tag.get("details") or {})
                )
                models[name] = model
                models.setdefault(model.short_name, model)
            self._models, self._at = models, time.monotonic()
            return models

    def _lookup(self, models: Mapping[str, LocalModel], alias: str) -> LocalModel | None:
        target = self._aliases.get(alias, alias)
        return models.get(target) or models.get(target.removesuffix(":latest"))

    async def resolve(self, alias: str, purpose: Purpose) -> LocalModel:
        model = self._lookup(await self.refresh(), alias)
        if model is None:
            model = self._lookup(await self.refresh(force=True), alias)
        if model is None or model.purpose != purpose:
            raise OllamaError(404, "unknown_alias", f"No {purpose} model named {alias!r}")
        return model

    async def listing(self) -> list[tuple[str, LocalModel]]:
        """Every (name, model) the gateway serves: each model by its short name, then each
        alias whose model is pulled."""
        models = await self.refresh()
        seen: dict[str, LocalModel] = {}
        for model in models.values():
            seen.setdefault(model.short_name, model)
        for alias in self._aliases:
            if alias not in seen and (model := self._lookup(models, alias)) is not None:
                seen[alias] = model
        return sorted(seen.items())
