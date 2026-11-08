"""The training job's endings: a run asked to cancel ends ``cancelled``; a run
whose worker shuts down or loses its lease is handed back untouched, so it trains again."""

import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

from stratafi_core.components import Cancelled, ComponentError
from stratafi_core.settings import OfflineSettings
from stratafi_store import training as training_store
from stratafi_worker.training import TrainingRunner
from stratafi_worker.worker import JobCancelled


@dataclass(frozen=True)
class Run:
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    workspace_id: uuid.UUID = field(default_factory=uuid.uuid4)


class Engine:
    @contextmanager
    def begin(self):
        yield object()

    connect = begin


@dataclass
class Attempt:
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    attempts: int = 1
    max_attempts: int = 3


@dataclass
class Job:
    cancel_requested: bool = False
    stop: threading.Event = field(default_factory=threading.Event)
    job: Attempt = field(default_factory=Attempt)

    @property
    def payload(self) -> dict[str, Any]:
        return {"training_run_id": str(uuid.uuid4())}

    @contextmanager
    def provider(self, name: str, *, tokens: int = 0):
        yield 0.0


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> tuple[TrainingRunner, list[str]]:
    ended: list[str] = []
    run = Run()
    monkeypatch.setattr(training_store, "begin_training_run", lambda conn, run_id: run)
    monkeypatch.setattr(
        training_store,
        "finish_training_run",
        lambda conn, run, **values: ended.append(values["status"]),
    )
    found = TrainingRunner(
        Engine(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        lambda workspace_id: {},
        None,  # type: ignore[arg-type]
        settings=OfflineSettings(),
    )
    return found, ended


def stop(*args: Any) -> Any:
    raise Cancelled()


def test_a_shutdown_hands_the_run_back(runner, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer, ended = runner
    monkeypatch.setattr(trainer, "_train", stop)
    monkeypatch.setattr(trainer, "_cancelling", lambda run: False)
    with pytest.raises(JobCancelled):
        trainer(Job())  # type: ignore[arg-type]
    assert ended == []


def test_a_cancel_ends_the_run(runner, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer, ended = runner
    monkeypatch.setattr(trainer, "_train", stop)
    monkeypatch.setattr(trainer, "_cancelling", lambda run: True)
    with pytest.raises(JobCancelled):
        trainer(Job())  # type: ignore[arg-type]
    monkeypatch.setattr(trainer, "_cancelling", lambda run: False)
    with pytest.raises(JobCancelled):
        trainer(Job(cancel_requested=True))  # type: ignore[arg-type]
    assert ended == ["cancelled", "cancelled"]


def test_a_failure_ends_the_run_unless_a_retry_could_help(
    runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, ended = runner

    def fail(*args: Any) -> Any:
        raise ComponentError("gateway_timeout", retryable=True)

    monkeypatch.setattr(trainer, "_train", fail)
    with pytest.raises(ComponentError):
        trainer(Job())  # type: ignore[arg-type]
    assert ended == []
    last = Job(job=Attempt(attempts=3, max_attempts=3))
    assert trainer(last) == {"failed": "gateway_timeout"}  # type: ignore[arg-type]
    assert ended == ["failed"]
