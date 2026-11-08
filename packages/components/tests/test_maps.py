"""Pipeline maps (unit tier): stages, datastores and services from the real
component registry."""

from pathlib import Path

import pytest
import yaml

from stratafi_components import default_registry
from stratafi_core.components import SERVICES
from stratafi_core.maps import (
    Deployment,
    attach_overlay,
    build_map,
    eval_overlay,
    overlay_stages,
    to_mermaid,
)
from stratafi_core.specs import dump_document, resolve_pipeline, resolve_retrieval

FIXTURES = Path(__file__).resolve().parents[3] / "contracts" / "fixtures"
REGISTRY = default_registry()
DEPLOYMENT = Deployment(
    database=True,
    objects="s3",
    gateway=True,
    clamd=True,
    providers=frozenset({"tika"}),
)


def load(path: str) -> dict:
    return yaml.safe_load((FIXTURES / path).read_text(encoding="utf-8"))


def baseline() -> tuple[dict, dict]:
    pipeline = dump_document(resolve_pipeline(load("pipeline.v1/baseline-v1.yaml"), REGISTRY))
    retrieval = dump_document(resolve_retrieval(load("retrieval.v1/hybrid-v1.yaml"), REGISTRY))
    return pipeline, retrieval


def nodes_of(found: dict) -> dict[str, dict]:
    return {node["id"]: node for node in found["nodes"]}


def edges_of(found: dict) -> set[tuple[str, str, str]]:
    return {(edge["source"], edge["target"], edge["kind"]) for edge in found["edges"]}


def test_every_declared_service_is_one_the_map_knows() -> None:
    for component in REGISTRY:
        assert set(component.info.services) <= set(SERVICES), component.info.id
    declared = {c.info.id: c.info.services for c in REGISTRY if c.info.services}
    assert declared["scan/clamav"] == ("clamd",)
    assert declared["index/pgvector-hnsw"] == ("lab-index",)
    assert declared["extract/s3"] == ("s3-source",)
    assert declared["embed/sentence-transformers"] == ("model-files",)


def test_the_baseline_maps_its_stages_stores_and_services() -> None:
    pipeline, retrieval = baseline()
    found = build_map(pipeline, retrieval, REGISTRY, DEPLOYMENT)
    nodes = nodes_of(found)
    scans = [node for node in nodes if node.startswith("scan:")]
    assert len(scans) == len(pipeline["stages"]["scan"]["checks"])
    assert {"parse:default", "parse:md", "parse:pdf"} <= set(nodes)
    assert "clean:0" in nodes
    assert {"chunk", "enrich", "embed", "index:dense", "index:keyword"} <= set(nodes)
    assert nodes["enrich"]["noop"] is True
    assert {"query:input", "query:embed", "query:dense", "query:keyword", "query:fuse",
            "query:rerank", "query:final"} <= set(nodes)  # fmt: skip
    edges = edges_of(found)
    # Artifacts flow stage to stage; every stage writes its artifact and its records.
    assert ("extract", "scan:0", "flow") in edges
    assert ("embed", "index:dense", "flow") in edges
    assert ("chunk", "store:objects", "write") in edges
    assert ("chunk", "store:database", "write") in edges
    assert ("store:objects", "chunk", "read") in edges
    # In-memory indexes are read from the artifact store; filters read the corpus entries.
    assert ("store:objects", "query:dense", "read") in edges
    assert ("store:database", "query:fuse", "read") in edges
    assert ("index:dense", "query:dense", "flow") in edges
    # The gateway gets chunk text at ingest, the query and passages at query time.
    assert ("embed", "service:model-gateway", "call") in edges
    assert ("query:embed", "service:model-gateway", "call") in edges
    assert ("query:rerank", "service:model-gateway", "call") in edges
    clamav = next(node for node in nodes.values() if node.get("component") == "scan/clamav")
    assert (clamav["id"], "service:clamd", "call") in edges
    summary = found["summary"]
    assert summary["datastores"] == ["database", "objects"]
    assert "model-gateway" in summary["sends_text_to"]
    assert "clamd" in summary["sends_text_to"]
    assert nodes["store:objects"]["backend"] == "s3"
    assert nodes["service:model-gateway"]["configured"] is True
    assert found["schema"] == "stratafi.pipeline-map.v1"


def spec(**stages) -> dict:
    document = {
        "schema": "stratafi.pipeline.v1",
        "name": "mapped",
        "stages": {
            "scan": {
                "checks": [{"component": "scan/content-safety"}, {"component": "scan/clamav"}]
            },
            "parse": {
                "route": {
                    "default": {"component": "parse/markdown"},
                    "application/pdf": {"component": "parse/tika"},
                }
            },
            "clean": [{"component": "clean/whitespace"}, {"component": "clean/redact"}],
            "embed": {"component": "embed/sentence-transformers"},
            "index": {"dense": {"component": "index/pgvector-hnsw"}, "keyword": None},
            **stages,
        },
    }
    return dump_document(resolve_pipeline(document, REGISTRY))


def test_routes_rules_lab_indexes_and_in_process_models_have_their_edges() -> None:
    retrieval = dump_document(
        resolve_retrieval(
            {
                "schema": "stratafi.retrieval.v1",
                "name": "dense-cross",
                "retrievers": {"dense": {"top_n": 30}, "keyword": None},
                "fusion": {"method": "none"},
                "rerank": {"component": "rerank/cross-encoder"},
                "top_k": 5,
            },
            REGISTRY,
        )
    )
    found = build_map(spec(), retrieval, REGISTRY, Deployment(), evaluation=True)
    nodes = nodes_of(found)
    edges = edges_of(found)
    assert nodes["parse:default"]["row"] == 0
    assert nodes["parse:application/pdf"]["row"] == 1
    assert ("scan:1", "parse:application/pdf", "flow") in edges
    assert ("parse:application/pdf", "service:provider:tika", "call") in edges
    assert nodes["service:provider:tika"]["label"] == "Apache Tika"
    assert ("clean:0", "clean:1", "flow") in edges
    # Redaction runs in process: the rule calls no service.
    assert not any(src == "clean:1" and dst.startswith("service:") for src, dst, _ in edges)
    assert ("index:dense", "store:lab-index", "write") in edges
    assert ("index:dense", "store:objects", "write") not in edges
    assert ("store:lab-index", "query:dense", "read") in edges
    assert ("store:model-files", "embed", "read") in edges
    assert ("store:model-files", "query:rerank", "read") in edges
    assert "query:keyword" not in nodes
    assert "index:keyword" not in nodes
    assert ("query:results", "store:database", "write") in edges
    assert nodes["query:final"]["label"] == "Top 5"
    assert set(found["summary"]["lab_only"]) == {
        "embed/sentence-transformers",
        "rerank/cross-encoder",
    }
    # This deployment configures no Tika; the map still shows the call.
    assert nodes["service:provider:tika"]["configured"] is False
    assert nodes["store:objects"]["backend"] == "filesystem"


def test_a_retrieval_config_alone_reads_its_index_from_the_store() -> None:
    _, retrieval = baseline()
    found = build_map(None, retrieval, REGISTRY, DEPLOYMENT)
    nodes = nodes_of(found)
    assert not any(node["lane"] == "ingest" for node in nodes.values())
    assert ("store:objects", "query:keyword", "read") in edges_of(found)
    with pytest.raises(ValueError, match="needs a pipeline spec or a retrieval config"):
        build_map(None, None, REGISTRY, DEPLOYMENT)


def test_maps_hold_no_hosts_urls_or_credentials() -> None:
    pipeline, retrieval = baseline()
    text = str(build_map(pipeline, retrieval, REGISTRY, DEPLOYMENT))
    for marker in ("http://", "https://", "://", "password", "secret", "127.0.0.1", "localhost"):
        assert marker not in text


def test_overlays_put_a_runs_numbers_on_its_nodes() -> None:
    pipeline, retrieval = baseline()
    found = build_map(pipeline, retrieval, REGISTRY, DEPLOYMENT, evaluation=True)
    stages = overlay_stages(
        [
            {
                "stage": "chunk",
                "status": "succeeded",
                "cache_hit": True,
                "duration_ms": 2.0,
                "size_bytes": 10,
            },
            {
                "stage": "chunk",
                "status": "succeeded",
                "cache_hit": False,
                "duration_ms": 3.0,
                "size_bytes": 5,
            },
            {"stage": "scan", "status": "succeeded", "duration_ms": 1.0},
            {"stage": "parse", "route": "default", "status": "failed", "duration_ms": 4.0},
            {"stage": "index", "index_kind": "keyword", "status": "succeeded", "size_bytes": 7},
        ]
    )
    assert stages["chunk"] == {
        "tasks": 2,
        "statuses": {"succeeded": 2},
        "cache_hits": 1,
        "duration_ms": 5.0,
        "bytes": 15,
    }
    latency = {"rerank_ms": {"p50": 3.0, "p95": 9.0}, "total_ms": {"p50": 5.0, "p95": 12.0}}
    reaching = {"dense": {"hit@all": {"mean": 0.9}}, "final": {"hit@all": {"mean": 0.8}}}
    queries = eval_overlay({"latency_ms": latency, "metrics": {"knowledge-hit-v1": reaching}})
    attach_overlay(found, stages=stages, queries=queries)
    nodes = nodes_of(found)
    assert nodes["chunk"]["overlay"]["tasks"] == 2
    assert nodes["scan:0"]["overlay"] == nodes["scan:1"]["overlay"]
    assert nodes["parse:default"]["overlay"]["statuses"] == {"failed": 1}
    assert nodes["index:keyword"]["overlay"]["bytes"] == 7
    assert "overlay" not in nodes["index:dense"]
    assert nodes["query:rerank"]["overlay"] == {"latency_ms": {"p50": 3.0, "p95": 9.0}}
    assert nodes["query:dense"]["overlay"] == {"evidence_reaches": {"dense": 0.9}}
    assert nodes["query:final"]["overlay"]["evidence_reaches"] == {"final": 0.8}


def test_mermaid_draws_lanes_shapes_and_text_leaving_the_lab() -> None:
    pipeline, retrieval = baseline()
    found = build_map(pipeline, retrieval, REGISTRY, DEPLOYMENT)
    text = to_mermaid(found)
    assert text.startswith("flowchart LR\n")
    assert '  subgraph ingest["Ingest"]' in text
    assert '  subgraph query["Query time"]' in text
    assert 'n_store_objects[("Artifact store")]' in text
    assert 'n_service_model_gateway{{"Model gateway"}}' in text
    assert '-.->|"chunk text (text leaves the lab)"| n_service_model_gateway' in text
    assert "n_extract -->" in text
    lean = to_mermaid(found, stores=False)
    assert "store_objects" not in lean
    assert "n_service_model_gateway" in lean
