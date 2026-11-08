"""The search playground's API against Postgres: unsaved retrieval configs,
evidence highlights, rerankers through the API, and gated rerankers refusing."""

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from stratafi_testing.identity import DevIdentity
from stratafi_testing.lab import FIXTURES, running_lab

pytestmark = pytest.mark.integration

QUERY = "Who counts the tip jar at close, and where does the cash go?"


def config(**changes: Any) -> dict[str, Any]:
    return {
        "schema": "stratafi.retrieval.v1",
        "name": "playground",
        "retrievers": {"dense": {"top_n": 20}, "keyword": {"top_n": 20}},
        "fusion": {"method": "rrf", "k": 60},
        "candidates": 20,
        "rerank": "rerank/none",
        "top_k": 5,
        **changes,
    }


def test_unsaved_configs_rerankers_and_evidence(
    migrated_engine: sa.Engine, dev_identity: DevIdentity, tmp_path: Path, serve_app: Any
) -> None:
    names = ["tips.pdf", "wholesale.txt", "food-safety.docx"]
    files = {name: (FIXTURES / "formats" / name).read_bytes() for name in names}
    spec = {
        "schema": "stratafi.pipeline.v1",
        "name": "playground",
        "stages": {"embed": {"component": "embed/fake-hash", "params": {"dimensions": 64}}},
    }
    with running_lab(migrated_engine, dev_identity, tmp_path, serve_app) as lab:
        version = lab.upload_corpus("playground", files)
        run = lab.wait_run(lab.start_run(lab.save_spec(spec), version)["id"])

        def search(**body: Any):
            return lab.client.post(
                "/api/v1/search",
                json={"run_id": run["id"], "query": QUERY, **body},
                headers=lab.editor,
            )

        plain = search(retrieval_config=config(), evidence=["count the tip jar together"])
        assert plain.status_code == 200, plain.text
        found = plain.json()
        assert set(found["checkpoints"]) == {"dense", "keyword", "fused", "candidates", "reranked"}
        first = found["results"][0]
        assert first["entry"] == "tips.pdf"
        start, end = first["highlights"][0]
        assert "tip" in first["text"][start : end + 20]

        reranked = search(retrieval_config=config(rerank="rerank/fake-overlap", top_k=3))
        assert reranked.status_code == 200
        assert len(reranked.json()["results"]) == 3

        gated = search(retrieval_config=config(rerank="rerank/cohere"))
        assert (gated.status_code, gated.json()["code"]) == (409, "provider_not_allowed")
        # On by default: the gateway's chat model orders the candidates.
        listwise = search(retrieval_config=config(rerank="rerank/llm-listwise"))
        assert listwise.status_code == 200, listwise.text
        assert len(listwise.json()["results"]) == 5
        invalid = search(retrieval_config=config(top_k=50, candidates=10))
        assert (invalid.status_code, invalid.json()["code"]) == (400, "invalid_spec")
