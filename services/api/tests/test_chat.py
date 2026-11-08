"""``POST /chat`` and ``GET /chat/models`` in the unit tier: the model gateway is a mock
transport, and chats without a knowledge source need no database. Retrieval through a run's
indexes is the search route's, tested with it."""

import json
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stratafi_api.app import create_app
from stratafi_api.identity import DevIdentityProvider, StaticWorkspaces
from stratafi_api.routes import chat
from stratafi_core.chat import GROUNDED_SYSTEM, PLAIN_SYSTEM, build_messages, citations_used
from stratafi_core.chat import Passage as ChatPassage
from stratafi_core.settings import OfflineSettings
from stratafi_testing.identity import DevIdentity

MODELS = {
    "object": "list",
    "data": [
        {"id": "chat-default", "purpose": "chat", "revision": "357c53fb659c5076"},
        {"id": "qwen2.5:3b", "purpose": "chat", "revision": "357c53fb659c5076"},
        {"id": "nomic-embed-text", "purpose": "embed", "revision": "0a10", "dimensions": 768},
    ],
}


class FakeGateway:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.down = False

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if self.down:
            raise httpx2.ConnectError("refused", request=request)
        if request.url.path == "/v1/models":
            return httpx2.Response(200, json=MODELS)
        body = json.loads(request.content)
        self.sent.append(body)
        if body["model"] == "missing":
            return httpx2.Response(404, json={"error": {"type": "unknown_alias"}})
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            {"choices": [{"index": 0, "delta": {"content": " there."}}]},
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
                "timings": {"completion_tokens_per_second": 41.2, "load_ms": 3.0},
            },
        ]
        stream = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx2.Response(
            200,
            content=stream.encode(),
            headers={"content-type": "text/event-stream", "X-Model-Revision": "357c53fb659c5076"},
        )


def build(dev_identity: DevIdentity, gateway: FakeGateway | None, url: str | None) -> FastAPI:
    app = create_app(
        OfflineSettings(models={"gateway_url": url}),
        identity=DevIdentityProvider(dev_identity.verifier),
        workspaces=StaticWorkspaces(auto_provision=True),
        connect=False,
    )
    if not any(getattr(route, "path", "") == "/api/v1/chat" for route in app.routes):
        app.include_router(chat.router, prefix="/api/v1")
    if gateway is not None:
        app.state.gateway_transport = httpx2.MockTransport(gateway.handler)
    return app


def events_of(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def client(dev_identity: DevIdentity, gateway: FakeGateway) -> TestClient:
    return TestClient(build(dev_identity, gateway, "http://gateway.test"))


def test_a_chat_without_a_knowledge_source_streams_the_model_alone(
    client: TestClient, gateway: FakeGateway, dev_identity: DevIdentity
) -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
            "model": "qwen2.5:3b",
            "temperature": 0,
            "seed": 7,
        },
        headers=dev_identity.headers("tenant0", "viewer"),
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = events_of(response.text)
    assert [name for name, _ in events] == ["delta", "delta", "done"]
    assert "".join(data["text"] for name, data in events if name == "delta") == "Hello there."
    done = events[-1][1]
    assert done["model"] == "qwen2.5:3b"
    assert done["revision"] == "357c53fb659c5076"
    assert done["citations"] == []
    assert done["usage"]["total_tokens"] == 23
    assert done["timings"]["completion_tokens_per_second"] == 41.2
    assert done["timings"]["ttft_ms"] is not None
    sent = gateway.sent[0]
    assert sent["stream"] is True
    assert sent["seed"] == 7
    assert sent["messages"][0] == {"role": "system", "content": PLAIN_SYSTEM}
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "user"]


def test_the_last_message_must_be_the_question(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"messages": [{"role": "assistant", "content": "Hi"}]},
        headers=dev_identity.headers("tenant0", "viewer"),
    )
    assert response.status_code == 400


def test_an_unknown_model_and_a_gateway_that_is_down_end_in_error_events(
    client: TestClient, gateway: FakeGateway, dev_identity: DevIdentity
) -> None:
    headers = dev_identity.headers("tenant0", "viewer")
    question = {"messages": [{"role": "user", "content": "Hi"}]}
    missing = events_of(
        client.post("/api/v1/chat", json={**question, "model": "missing"}, headers=headers).text
    )
    assert missing == [
        ("error", {"code": "model_not_found", "detail": "The model gateway refused."})
    ]
    gateway.down = True
    down = events_of(client.post("/api/v1/chat", json=question, headers=headers).text)
    assert down[-1][0] == "error"
    assert down[-1][1]["code"] == "gateway_unreachable"


def test_chat_models_lists_only_chat_models_with_the_default(
    client: TestClient, dev_identity: DevIdentity
) -> None:
    response = client.get("/api/v1/chat/models", headers=dev_identity.headers("tenant0", "viewer"))
    assert response.status_code == 200
    body = response.json()
    assert [model["id"] for model in body["models"]] == ["chat-default", "qwen2.5:3b"]
    assert body["default"] == "chat-default"


def test_without_a_gateway_chat_is_unavailable(dev_identity: DevIdentity) -> None:
    client = TestClient(build(dev_identity, None, None))
    headers = dev_identity.headers("tenant0", "viewer")
    for response in (
        client.get("/api/v1/chat/models", headers=headers),
        client.post(
            "/api/v1/chat", json={"messages": [{"role": "user", "content": "Hi"}]}, headers=headers
        ),
    ):
        assert response.status_code == 503
        assert response.json()["code"] == "models_not_configured"


def test_grounded_prompts_number_their_sources_and_keep_recent_history() -> None:
    passages = [
        ChatPassage("Dose 18 g.", "espresso.md", ("3 Recipes",)),
        ChatPassage("Steam to 60 C.", None, ()),
    ]
    history = [("user", f"q{n}") if n % 2 == 0 else ("assistant", f"a{n}") for n in range(20)]
    messages = build_messages([*history, ("user", "What dose?")], passages)
    assert messages[0]["content"] == GROUNDED_SYSTEM
    assert len(messages) == 1 + 8 + 1
    question = messages[-1]["content"]
    assert "[1] espresso.md — 3 Recipes\nDose 18 g." in question
    assert "[2]\nSteam to 60 C." in question
    assert question.endswith("Question: What dose?")
    assert build_messages([("user", "x")], [])[-1]["content"].startswith(
        "Sources:\n\n(no sources were found)"
    )
    with pytest.raises(ValueError, match="question"):
        build_messages([("assistant", "x")], None)


def test_citations_are_read_in_order_of_first_use() -> None:
    assert citations_used("Yes [2], and [1][2]; not [9] or [0].", 3) == [2, 1]
