"""Model providers: how components reach embedding and reranking models.

``GatewayProvider`` is the ``gateway`` kind: an OpenAI-compatible model gateway reached by
alias, such as ``services/gateway`` on this machine:

- ``POST /v1/embeddings`` in batches; every response names the alias's revision in
  ``X-Model-Revision``. A missing revision, or one that changes between batches of one call,
  is an error: vectors that can't be placed are never stored.
- ``GET /v1/models`` lists the aliases a tenant may use with their revisions; a run resolves its
  embedding model's revision there before it starts, so the revision is part of the cache key.
- ``POST /v1/rerank`` scores documents against a query (the Cohere/Jina shape); its revision is
  checked the same way.
- ``POST /v1/chat/completions`` writes text: LLM enrichment, listwise reranking and
  ``generate/gateway``.
- Transient failures (timeouts, refused connections, 429 and 5xx) are retried with exponential
  backoff, honoring ``Retry-After``; after ``max_attempts`` they're a retryable
  ``ComponentError``, so the job retries later.
- Each batch waits for the provider's rate limits first (the worker's ``ProviderLimiter``,
  through ``StageContext.provider``).
- Proxy environment variables are ignored: a proxy would see every chunk.

The caller supplies the identity headers: the worker's service identity for the workspace's
tenant.
"""

import math
import re
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import httpx2
import numpy as np

from stratafi_core.components import ComponentError

REVISION_HEADER = "X-Model-Revision"
GATEWAY = "model_gateway"
MAX_RETRY_AFTER_SECONDS = 30.0

Slot = Callable[[int], AbstractContextManager[Any]]


@dataclass(frozen=True)
class EmbedResult:
    vectors: np.ndarray  # float32 [n, d]
    alias: str
    revision: str
    served_model: str | None
    tokens: int


@dataclass(frozen=True)
class RerankResult:
    ranked: list[tuple[int, float]]  # (document index, relevance score), best first
    alias: str
    revision: str
    tokens: int  # estimated: characters / 4


@dataclass(frozen=True)
class ChatResult:
    text: str
    alias: str
    revision: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ModelInfo:
    alias: str
    purpose: str | None
    revision: str | None
    dimensions: int | None


_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}")


def _valid_revision(value: str | None) -> bool:
    return isinstance(value, str) and _REVISION_RE.fullmatch(value) is not None


class GatewayProvider:
    name = GATEWAY

    def __init__(
        self,
        base_url: str,
        *,
        headers: Callable[[], dict[str, str]] = dict,
        timeout: float = 60.0,
        max_attempts: int = 4,
        backoff_seconds: float = 0.5,
        client: httpx2.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx2.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, trust_env=False
        )
        self._headers = headers
        self._max_attempts = max_attempts
        self._backoff = backoff_seconds
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> httpx2.Response:
        for attempt in range(1, self._max_attempts + 1):
            retry_after: float | None = None
            try:
                response = self._client.request(method, path, json=body, headers=self._headers())
            except httpx2.TimeoutException:
                code = "gateway_timeout"
            except httpx2.HTTPError:
                code = "gateway_unreachable"
            else:
                if response.status_code == 200:
                    return response
                if response.status_code != 429 and response.status_code < 500:
                    raise ComponentError(f"gateway_refused_{response.status_code}")
                code = "gateway_rate_limited" if response.status_code == 429 else "gateway_failed"
                header = response.headers.get("retry-after", "")
                if header.replace(".", "", 1).isdigit():
                    retry_after = min(float(header), MAX_RETRY_AFTER_SECONDS)
            if attempt == self._max_attempts:
                raise ComponentError(code, retryable=True)
            self._sleep(
                retry_after if retry_after is not None else self._backoff * 2 ** (attempt - 1)
            )
        raise ComponentError("gateway_failed", retryable=True)  # pragma: no cover

    def models(self) -> dict[str, ModelInfo]:
        response = self._request("GET", "/v1/models")
        try:
            items = response.json()["data"]
            return {
                str(item["id"]): ModelInfo(
                    alias=str(item["id"]),
                    purpose=item.get("purpose"),
                    revision=item.get("revision"),
                    dimensions=item.get("dimensions"),
                )
                for item in items
            }
        except (ValueError, KeyError, TypeError):
            raise ComponentError("gateway_bad_response") from None

    def revision(self, alias: str) -> str:
        """An alias's current revision, as ``/v1/models`` lists it."""
        info = self.models().get(alias)
        if info is None:
            raise ComponentError("unknown_alias")
        if not _valid_revision(info.revision):
            raise ComponentError("missing_revision")
        return str(info.revision)

    def embed(
        self,
        texts: Sequence[str],
        *,
        alias: str,
        batch_size: int = 32,
        slot: Slot = lambda tokens: nullcontext(),
        checkpoint: Callable[[], None] = lambda: None,
    ) -> EmbedResult:
        if not texts:
            raise ComponentError("empty_input")
        rows: list[list[float]] = []
        revision: str | None = None
        served: str | None = None
        tokens = 0
        for start in range(0, len(texts), batch_size):
            checkpoint()
            batch = list(texts[start : start + batch_size])
            estimate = sum(math.ceil(len(text) / 4) for text in batch)
            with slot(estimate):
                response = self._request(
                    "POST",
                    "/v1/embeddings",
                    {"model": alias, "input": batch, "encoding_format": "float"},
                )
            batch_revision = response.headers.get(REVISION_HEADER)
            if not _valid_revision(batch_revision):
                raise ComponentError("missing_revision")
            if revision is not None and batch_revision != revision:
                raise ComponentError("revision_changed")
            revision = batch_revision
            body = self._json(response)
            rows += _parse_embeddings(body, len(batch))
            served = served or (str(body["model"]) if isinstance(body.get("model"), str) else None)
            usage = body.get("usage") or {}
            tokens += int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        if revision is None or len({len(row) for row in rows}) != 1:
            raise ComponentError("gateway_bad_response")
        return EmbedResult(np.asarray(rows, dtype=np.float32), alias, revision, served, tokens)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        alias: str,
        top_n: int | None = None,
        slot: Slot = lambda tokens: nullcontext(),
    ) -> RerankResult:
        """``(index, relevance_score)`` pairs, best first, ties by index. The pairs name every
        document unless ``top_n`` cuts them."""
        if not documents:
            return RerankResult([], alias, "", 0)
        body: dict[str, Any] = {"model": alias, "query": query, "documents": list(documents)}
        if top_n is not None:
            body["top_n"] = top_n
        estimate = math.ceil((len(query) + sum(len(text) for text in documents)) / 4)
        with slot(estimate):
            response = self._request("POST", "/v1/rerank", body)
        revision = response.headers.get(REVISION_HEADER)
        if not _valid_revision(revision):
            raise ComponentError("missing_revision")
        try:
            results = [
                (int(item["index"]), float(item["relevance_score"]))
                for item in self._json(response)["results"]
            ]
        except (ValueError, KeyError, TypeError):
            raise ComponentError("gateway_bad_response") from None
        if any(
            not 0 <= index < len(documents) or not math.isfinite(score) for index, score in results
        ):
            raise ComponentError("gateway_bad_response")
        ranked = sorted(results, key=lambda pair: (-pair[1], pair[0]))
        return RerankResult(ranked, alias, str(revision), estimate)

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        alias: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        seed: int | None = None,
        slot: Slot = lambda tokens: nullcontext(),
    ) -> ChatResult:
        """``POST /v1/chat/completions``: one completion (OpenAI's shape), the first choice's
        text with the alias's revision and the tokens used."""
        estimate = math.ceil(sum(len(m.get("content", "")) for m in messages) / 4) + max_tokens
        body: dict[str, Any] = {
            "model": alias,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        with slot(estimate):
            response = self._request("POST", "/v1/chat/completions", body)
        revision = response.headers.get(REVISION_HEADER)
        if not _valid_revision(revision):
            raise ComponentError("missing_revision")
        data = self._json(response)
        try:
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
        except (KeyError, IndexError, TypeError, ValueError):
            raise ComponentError("gateway_bad_response") from None
        if not isinstance(text, str):
            raise ComponentError("gateway_bad_response")
        return ChatResult(text, alias, str(revision), prompt_tokens, completion_tokens)

    @staticmethod
    def _json(response: httpx2.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            raise ComponentError("gateway_bad_response") from None
        if not isinstance(body, dict):
            raise ComponentError("gateway_bad_response")
        return body


def _parse_embeddings(body: dict[str, Any], expected: int) -> list[list[float]]:
    try:
        data = sorted(body["data"], key=lambda item: item["index"])
        vectors = [[float(value) for value in item["embedding"]] for item in data]
    except (ValueError, KeyError, TypeError):
        raise ComponentError("gateway_bad_response") from None
    if len(vectors) != expected or any(
        not vector or not all(math.isfinite(value) for value in vector) for vector in vectors
    ):
        raise ComponentError("gateway_bad_response")
    return vectors
