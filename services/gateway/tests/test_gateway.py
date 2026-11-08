"""The local gateway against a fake Ollama (``httpx2.MockTransport``), and StrataFi's own
``GatewayProvider`` against the gateway: the contract the pipeline components rely on."""

import json
from collections.abc import Iterator
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from stratafi_gateway import REVISION_HEADER
from stratafi_gateway.app import create_app
from stratafi_gateway.rerank import RerankerPool
from stratafi_gateway.settings import GatewaySettings, parse_pairs

from stratafi_components.models import GatewayProvider

EMBED_DIGEST = "sha256:0a109f422b47" + "0" * 52
CHAT_DIGEST = "sha256:357c53fb659c" + "1" * 52

TAGS = {
    "models": [
        {
            "name": "nomic-embed-text:latest",
            "digest": EMBED_DIGEST,
            "details": {"family": "nomic-bert", "parameter_size": "137M"},
        },
        {
            "name": "qwen2.5:3b",
            "digest": CHAT_DIGEST,
            "details": {
                "family": "qwen2",
                "parameter_size": "3.1B",
                "quantization_level": "Q4_K_M",
            },
        },
        {"name": "vision-only:1b", "digest": "sha256:" + "2" * 64},
    ]
}
SHOW = {
    "nomic-embed-text:latest": {
        "capabilities": ["embedding"],
        "model_info": {"nomic-bert.embedding_length": 768, "nomic-bert.context_length": 2048},
    },
    "qwen2.5:3b": {
        "capabilities": ["completion", "tools"],
        "model_info": {"qwen2.embedding_length": 2048, "qwen2.context_length": 32768},
    },
    "vision-only:1b": {"capabilities": ["vision"], "model_info": {}},
}


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[tuple[str, Any]] = []
        self.done_reason = "stop"
        self.down = False

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if self.down:
            raise httpx2.ConnectError("refused", request=request)
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, body))
        path = request.url.path
        if path == "/api/version":
            return httpx2.Response(200, json={"version": "0.40.15"})
        if path == "/api/tags":
            return httpx2.Response(200, json=TAGS)
        if path == "/api/show":
            return httpx2.Response(200, json=SHOW[body["model"]])
        if path == "/api/embed":
            if body["model"] not in SHOW:
                return httpx2.Response(404, json={"error": "model not found"})
            vectors = [[float(len(text)), 0.5, 0.25] for text in body["input"]]
            return httpx2.Response(
                200, json={"embeddings": vectors, "prompt_eval_count": 3 * len(vectors)}
            )
        if path == "/api/chat":
            final = {
                "done": True,
                "done_reason": self.done_reason,
                "prompt_eval_count": 12,
                "eval_count": 4,
                "eval_duration": 100_000_000,
                "load_duration": 5_000_000,
            }
            if not body.get("stream"):
                return httpx2.Response(
                    200, json={"message": {"role": "assistant", "content": "Hello world"}, **final}
                )
            lines = [
                {"message": {"role": "assistant", "content": "Hello"}, "done": False},
                {"message": {"role": "assistant", "content": " world"}, "done": False},
                {"message": {"role": "assistant", "content": ""}, **final},
            ]
            return httpx2.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
        return httpx2.Response(404, json={"error": "no route"})


def overlap_loader(_: str):  # a "cross-encoder" scoring by shared words
    def score(query: str, documents: list[str]) -> list[float]:
        words = set(query.lower().split())
        return [float(len(words & set(text.lower().split()))) for text in documents]

    return score


@pytest.fixture
def ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def client(ollama: FakeOllama) -> Iterator[TestClient]:
    app = create_app(
        GatewaySettings(),
        transport=httpx2.MockTransport(ollama.handler),
        rerankers=RerankerPool({"rerank-default": "fake/cross-encoder"}, loader=overlap_loader),
    )
    with TestClient(app) as test_client:
        yield test_client


def test_models_lists_embedders_chat_models_rerankers_and_aliases(client: TestClient) -> None:
    data = {item["id"]: item for item in client.get("/v1/models").json()["data"]}
    assert set(data) == {
        "nomic-embed-text",
        "qwen2.5:3b",
        "embed-default",
        "chat-default",
        "rerank-default",
    }
    embed = data["embed-default"]
    assert embed["purpose"] == "embed"
    assert embed["dimensions"] == 768
    assert embed["revision"] == "0a109f422b47" + "0" * 4
    assert embed["served_model"] == "nomic-embed-text:latest"
    chat = data["chat-default"]
    assert chat["purpose"] == "chat"
    assert "dimensions" not in chat
    assert chat["context_length"] == 32768
    assert chat["quantization"] == "Q4_K_M"
    assert data["rerank-default"]["purpose"] == "rerank"


def test_embeddings_return_vectors_with_the_weights_revision(
    client: TestClient, ollama: FakeOllama
) -> None:
    response = client.post(
        "/v1/embeddings", json={"model": "embed-default", "input": ["one", "three"]}
    )
    assert response.status_code == 200
    assert response.headers[REVISION_HEADER] == "0a109f422b470000"
    body = response.json()
    assert [item["embedding"][0] for item in body["data"]] == [3.0, 5.0]
    assert body["usage"]["total_tokens"] == 6
    sent = next(b for path, b in ollama.requests if path == "/api/embed")
    assert sent == {"model": "nomic-embed-text:latest", "input": ["one", "three"], "truncate": True}


def test_embeddings_refuse_chat_models_and_unknown_names(client: TestClient) -> None:
    for name in ("qwen2.5:3b", "no-such-model"):
        response = client.post("/v1/embeddings", json={"model": name, "input": "x"})
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "unknown_alias"


def test_chat_sets_the_context_window_and_reports_usage_and_speed(
    client: TestClient, ollama: FakeOllama
) -> None:
    ollama.done_reason = "length"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "chat-default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 64,
            "temperature": 0,
            "seed": 7,
        },
    )
    assert response.status_code == 200
    assert response.headers[REVISION_HEADER] == "357c53fb659c1111"
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
    assert body["timings"]["completion_tokens_per_second"] == 40.0
    sent = next(b for path, b in ollama.requests if path == "/api/chat")
    assert sent["model"] == "qwen2.5:3b"
    assert sent["stream"] is False
    assert sent["options"] == {"num_ctx": 8192, "temperature": 0, "num_predict": 64, "seed": 7}


def test_chat_streams_server_sent_events_with_usage_last(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 4


def test_rerank_orders_by_score_and_cuts_top_n(client: TestClient) -> None:
    response = client.post(
        "/v1/rerank",
        json={
            "model": "rerank-default",
            "query": "espresso dose",
            "documents": ["milk", {"text": "espresso dose and yield"}, "espresso"],
            "top_n": 2,
        },
    )
    assert response.status_code == 200
    assert len(response.headers[REVISION_HEADER]) == 16
    assert response.json()["results"] == [
        {"index": 1, "relevance_score": 2.0},
        {"index": 2, "relevance_score": 1.0},
    ]
    missing = client.post("/v1/rerank", json={"model": "nope", "query": "q", "documents": ["d"]})
    assert missing.status_code == 404


def test_ollama_down_is_a_retryable_503(client: TestClient, ollama: FakeOllama) -> None:
    ollama.down = True
    response = client.post("/v1/embeddings", json={"model": "embed-default", "input": "x"})
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "ollama_unreachable"
    assert client.get("/health").status_code == 503


def test_invalid_requests_never_echo_their_text(client: TestClient) -> None:
    secret = "loyalty card 12345"
    response = client.post("/v1/embeddings", json={"model": 1, "input": {"text": secret}})
    assert response.status_code == 400
    assert secret not in response.text


def test_aliases_merge_with_the_defaults_and_bad_pairs_are_refused() -> None:
    settings = GatewaySettings.from_env(
        {"STRATAFI_GATEWAY_ALIASES": "chat-default=llama3.2:3b, judge=qwen2.5:3b"}
    )
    assert settings.aliases["chat-default"] == "llama3.2:3b"
    assert settings.aliases["judge"] == "qwen2.5:3b"
    assert settings.aliases["embed-default"] == "nomic-embed-text"
    with pytest.raises(ValueError, match="alias=target"):
        parse_pairs("no-equals-sign", name="X")


def test_stratafis_gateway_provider_speaks_the_contract(client: TestClient) -> None:
    """The pipeline's own client: revisions, embeddings, rerank and chat all round-trip."""
    provider = GatewayProvider("http://testserver", client=client)
    assert provider.revision("embed-default") == "0a109f422b470000"
    embedded = provider.embed(["a", "bb", "ccc"], alias="embed-default", batch_size=2)
    assert embedded.vectors.shape == (3, 3)
    assert embedded.revision == "0a109f422b470000"
    ranked = provider.rerank("espresso dose", ["milk", "espresso dose"], alias="rerank-default")
    assert ranked.ranked[0][0] == 1
    answer = provider.chat([{"role": "user", "content": "Hi"}], alias="chat-default")
    assert answer.text == "Hello world"
    assert answer.completion_tokens == 4
