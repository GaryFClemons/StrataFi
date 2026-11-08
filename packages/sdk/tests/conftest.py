"""A fake Stratafi API for the SDK's unit tests: canned answers by method and path, served
through ``httpx2.MockTransport``, and records shaped as the API's models."""

import json
import uuid
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

NOW = "2026-10-30T12:00:00Z"
Answer = httpx2.Response | Callable[[httpx2.Request], httpx2.Response]


class FakeApi:
    def __init__(self) -> None:
        self.answers: dict[tuple[str, str], list[Answer]] = {}
        self.requests: list[httpx2.Request] = []

    def on(self, method: str, path: str, *answers: Any, status: int = 200) -> None:
        """Answer ``method path`` (below ``/api/v1``) with each of ``answers`` in turn, then
        the last again. An answer is a JSON body, a response or a function of the request."""
        self.answers[(method, path)] = [
            answer
            if isinstance(answer, httpx2.Response) or callable(answer)
            else httpx2.Response(status, json=answer)
            for answer in answers
        ]

    def problem(self, method: str, path: str, status: int, code: str = "x") -> None:
        body = {
            "type": "urn:stratafi:problem:x",
            "title": "Problem",
            "status": status,
            "detail": "It went wrong.",
            "code": code,
        }
        self.on(method, path, httpx2.Response(status, json=body))

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        key = (request.method, request.url.path.removeprefix("/api/v1"))
        answers = self.answers.get(key)
        if not answers:
            body = {"title": "Not found", "status": 404, "detail": "Not found."}
            return httpx2.Response(404, json=body)
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        return answer(request) if callable(answer) else answer

    @property
    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)

    def sent(self, method: str, path: str) -> list[httpx2.Request]:
        return [
            request
            for request in self.requests
            if request.method == method and request.url.path == f"/api/v1{path}"
        ]

    def body(self, method: str, path: str) -> Any:
        return json.loads(self.sent(method, path)[-1].content)

    # --- records ---------------------------------------------------------------------------

    @staticmethod
    def page(*items: Any) -> dict[str, Any]:
        return {"items": list(items), "next_cursor": None}

    @staticmethod
    def run(status: str = "queued", **changes: Any) -> dict[str, Any]:
        return {
            "id": str(uuid.UUID(int=1)),
            "kind": "pipeline",
            "mode": "batch",
            "status": status,
            "pipeline_spec_id": str(uuid.UUID(int=2)),
            "corpus_version_id": str(uuid.UUID(int=3)),
            "until": None,
            "sample": None,
            "rerun_of": None,
            "rerun_from": None,
            "spec_hash": "a" * 64,
            "manifest_hash": None,
            "counts": {"tasks": {status: 4}},
            "error_code": None,
            "created_by": "u-tenant0-editor",
            "created_at": NOW,
            "started_at": None,
            "finished_at": None,
            **changes,
        }

    @staticmethod
    def eval_run(status: str = "queued", **changes: Any) -> dict[str, Any]:
        estimate = {"mean": 0.75, "low": 0.6, "high": 0.9, "n": 63}
        metrics = {
            "k_eval": 5,
            "metrics": {
                "knowledge-hit-v1": {
                    "final": {
                        "hit@5": estimate,
                        "recall@5": estimate,
                        "mrr@10": {**estimate, "mean": 0.5},
                        "ndcg@10": {**estimate, "mean": 0.55},
                    },
                    "fused": {"recall@5": {**estimate, "mean": 0.7}},
                }
            },
        }
        return {
            "id": str(uuid.UUID(int=11)),
            "status": status,
            "run_id": str(uuid.UUID(int=1)),
            "retrieval_config_id": str(uuid.UUID(int=12)),
            "eval_set_version_id": str(uuid.UUID(int=13)),
            "split": "dev",
            "options": {},
            "retrieval_hash": "b" * 64,
            "test_read": False,
            "results_cached": False,
            "retrieval_ref": {"name": "hybrid-v1", "version": 1},
            "eval_set": {"name": "halcyon-retrieval", "version": 1},
            "k_eval": 5,
            "counts": {},
            "metrics": metrics if status == "succeeded" else None,
            "manifest_hash": None,
            "error_code": None,
            "created_by": "u-tenant0-editor",
            "created_at": NOW,
            "started_at": None,
            "finished_at": None,
            **changes,
        }

    @staticmethod
    def spec(name: str, version: int = 1, document: dict[str, Any] | None = None) -> dict:
        return {
            "id": str(uuid.UUID(int=100 + version)),
            "name": name,
            "version": version,
            "spec_hash": "c" * 64,
            "document_hash": "d" * 64,
            "created_by": "u-tenant0-editor",
            "created_at": NOW,
            "document": document or pipeline_document(name),
        }

    @staticmethod
    def corpus(name: str = "halcyon", latest: dict[str, Any] | None = None) -> dict:
        return {
            "id": str(uuid.UUID(int=20)),
            "name": name,
            "description": "",
            "created_by": "u-tenant0-editor",
            "created_at": NOW,
            "updated_at": NOW,
            "entries": 25,
            "total_bytes": 1000,
            "latest_version": latest,
            "duplicate_groups": [],
        }

    @staticmethod
    def corpus_version(version: int = 1) -> dict[str, Any]:
        return {
            "id": str(uuid.UUID(int=30 + version)),
            "corpus_id": str(uuid.UUID(int=20)),
            "version": version,
            "manifest_hash": "e" * 64,
            "document_count": 25,
            "total_bytes": 1000,
            "created_by": "u-tenant0-editor",
            "frozen_at": NOW,
        }

    @staticmethod
    def eval_set_version(version: int = 1) -> dict[str, Any]:
        return {
            "id": str(uuid.UUID(int=40 + version)),
            "eval_set_id": str(uuid.UUID(int=40)),
            "version": version,
            "document": {},
            "document_hash": "f" * 64,
            "item_count": 90,
            "split_counts": {"dev": 63, "test": 27},
            "split_params": None,
            "source": {},
            "frozen": True,
            "frozen_at": NOW,
            "test_locked": False,
            "test_reads": 0,
            "last_test_read_at": None,
            "created_by": "u-tenant0-editor",
            "created_at": NOW,
        }

    @classmethod
    def eval_set(cls, name: str = "halcyon-retrieval", latest: int = 2) -> dict[str, Any]:
        return {
            "id": str(uuid.UUID(int=40)),
            "name": name,
            "description": "",
            "created_by": "u-tenant0-editor",
            "created_at": NOW,
            "updated_at": NOW,
            "latest_version": cls.eval_set_version(latest),
        }


def pipeline_document(name: str = "baseline-v1") -> dict[str, Any]:
    return {
        "schema": "stratafi.pipeline.v1",
        "name": name,
        "description": "",
        "stages": {
            "scan": {"checks": [{"component": "scan/content-safety", "params": {}}]},
            "parse": {"route": {"default": {"component": "parse/markdown"}}},
            "clean": [
                {"component": "clean/whitespace", "params": {}},
                {"component": "clean/boilerplate", "params": {}},
            ],
            "chunk": {
                "component": "chunk/sections",
                "params": {"max_tokens": 800, "min_tokens": 500, "overlap": 0.1},
            },
            "embed": {"component": "embed/gateway", "params": {"alias": "embed-default"}},
        },
    }


def sse(*events: tuple[str, dict[str, Any]]) -> httpx2.Response:
    """A ``/events`` answer: a keepalive comment, then each event."""
    lines = ["retry: 3000", ": connected", ""]
    for number, (kind, data) in enumerate(events, start=1):
        lines += [f"id: {number}", f"event: {kind}", f"data: {json.dumps(data)}", ""]
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content="\n".join(lines) + "\n"
    )


@pytest.fixture
def api() -> FakeApi:
    return FakeApi()


@pytest.fixture
def sse_response() -> Callable[..., httpx2.Response]:
    return sse


@pytest.fixture
def pipeline() -> Callable[..., dict[str, Any]]:
    return pipeline_document


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries and reconnections don't sleep in unit tests."""
    monkeypatch.setattr("stratafi.client._pause", lambda seconds: None)
