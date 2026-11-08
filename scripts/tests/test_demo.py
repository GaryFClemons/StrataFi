"""scripts/demo_seed.py: a seed that needs no model gateway and no ClamAV, and says why a run
indexed nothing."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from stratafi_components import default_registry
from stratafi_core.specs import resolve_pipeline, resolve_retrieval

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def seed_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("script_demo_seed", ROOT / "scripts/demo_seed.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_seed_needs_no_model_gateway_and_no_clamav(seed_module: ModuleType) -> None:
    registry = default_registry()
    pipeline = resolve_pipeline(seed_module.PIPELINE, registry)
    assert pipeline.stages.embed.component == "embed/fake-hash"
    assert "scan/clamav" not in {check.component for check in pipeline.stages.scan.checks}
    rerankers = set()
    for document in seed_module.RETRIEVAL:
        config = resolve_retrieval(document, registry)
        rerankers.add(config.rerank.component)
    assert rerankers == {"rerank/fake-overlap", "rerank/none"}
    # Two configs differ only in the reranker: a controlled pair for the Models screen.
    first, second = (dict(document) for document in seed_module.RETRIEVAL[:2])
    assert {key for key in first if first[key] != second.get(key)} == {"name", "rerank"}


class _Client:
    """The SDK calls the seed makes, answering as a lab whose ClamAV signatures are stale."""

    def __init__(self) -> None:
        version = SimpleNamespace(id="v1", version=1, document_count=24)
        counts = SimpleNamespace(entries={"excluded": 24})
        run = SimpleNamespace(id="run-1", status="succeeded", record=SimpleNamespace(counts=counts))
        self.corpora = SimpleNamespace(
            create=lambda name, description: None,
            upload=lambda name, manifest: None,
            freeze=lambda name: version,
        )
        self.pipelines = SimpleNamespace(
            push=lambda document: SimpleNamespace(name=document["name"], version=1)
        )
        self.runs = SimpleNamespace(start=lambda *args, **kwargs: run)
        self.eval_sets = SimpleNamespace(import_file=self._unexpected)

    @staticmethod
    def _unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a run that indexed nothing isn't evaluated")

    def get(self, path: str, **params: Any) -> Any:
        if path == "/runs/run-1/tasks" and params.get("status") == "failed":
            return {"items": []}
        if path == "/runs/run-1/tasks":
            return {"items": [{"stage": "scan", "cache_key": "k1"}] * 24}
        assert path == "/artifacts/k1/content"
        return {
            "checks": [
                {"component": "scan/content-safety", "outcome": "clean", "reason": None},
                {"component": "scan/clamav", "outcome": "error", "reason": "signatures_stale"},
            ]
        }


def test_the_seed_says_why_a_run_indexed_nothing(
    seed_module: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    code = seed_module.seed(_Client(), pipeline=seed_module.PIPELINE, web="http://web")
    out = capsys.readouterr().out
    assert code == 1
    assert "0 of 24 documents indexed" in out
    assert "scan/clamav: signatures_stale" in out
    assert "freshclam must reach its" in out
