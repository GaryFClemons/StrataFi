"""Stratafi's versioned contracts.

Every spec and artifact has a JSON Schema named ``stratafi.<name>.vN``, exported from its
Pydantic model to ``contracts/schemas/<name>.schema.json`` (``scripts/export_contracts.py``),
with example documents in ``contracts/fixtures/<name>/``. A change that breaks a schema makes a
new version; the old one stays.

The API's bodies have their contract too: ``contracts/openapi.json``.
"""

from typing import Any

from pydantic import BaseModel

from stratafi_core.artifacts import (
    Chunk,
    IndexManifest,
    ParsedDocument,
    RawDocument,
    ScanReport,
    VectorsSidecar,
)
from stratafi_core.evalruns import EvalRunManifest
from stratafi_core.evalsets import EvalSet
from stratafi_core.experiments import ExperimentSpec
from stratafi_core.manifest import RunManifest
from stratafi_core.profiles import KnowledgeProfile
from stratafi_core.specs import PipelineSpec, RetrievalConfig
from stratafi_core.suites import TestRunRecord, TestSuite
from stratafi_core.training import ModelCard, TrainingRecord, TrainingRunManifest

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID_PREFIX = "urn:stratafi:schema:"

CONTRACTS: dict[str, tuple[type[BaseModel], str]] = {
    "stratafi.pipeline.v1": (PipelineSpec, "How documents become an index (a pipeline spec)"),
    "stratafi.retrieval.v1": (RetrievalConfig, "How a query searches an index"),
    "stratafi.raw-document.v1": (
        RawDocument,
        "A document's stored bytes and what they are: the extract stage's output",
    ),
    "stratafi.scan-report.v1": (
        ScanReport,
        "What each scan check found and what the policy decided",
    ),
    "stratafi.parsed-document.v1": (
        ParsedDocument,
        "A parsed document, or a cleaned one with its edits",
    ),
    "stratafi.chunk.v1": (Chunk, "One chunk: a line of a chunks artifact (JSONL)"),
    "stratafi.vectors.v1": (VectorsSidecar, "The sidecar of a vectors array: chunks and model"),
    "stratafi.index.v1": (IndexManifest, "A corpus-level index, as built"),
    "stratafi.run-manifest.v1": (RunManifest, "Everything a run recorded"),
    "stratafi.eval-set.v1": (
        EvalSet,
        "Questions with known evidence; also reads the retrieval YAML format",
    ),
    "stratafi.eval-run.v1": (EvalRunManifest, "Everything an eval run recorded"),
    "stratafi.experiment.v1": (
        ExperimentSpec,
        "A space of configurations against one eval set: factors, objective and budget",
    ),
    "stratafi.knowledge-profile.v1": (
        KnowledgeProfile,
        "A winning configuration with its evidence, compatibility and review",
    ),
    "stratafi.test-suite.v1": (
        TestSuite,
        "Automated checks on a target: thresholds, baselines, golden queries and triggers",
    ),
    "stratafi.test-run.v1": (
        TestRunRecord,
        "Everything a test run recorded: its resolved inputs, runs and each check's result",
    ),
    "stratafi.training-set.v1": (
        TrainingRecord,
        "One eval item as training data: a line of a training set (JSONL)",
    ),
    "stratafi.training-run.v1": (
        TrainingRunManifest,
        "Everything a training run recorded: its trainer, inputs, epochs and model",
    ),
    "stratafi.model-card.v1": (
        ModelCard,
        "A trained model: its base, trainer, training set, use and limits",
    ),
}


def register_contract(name: str, model: type[BaseModel], description: str) -> None:
    """Add a contract. Steps that add artifacts (scan reports, vectors) register theirs here."""
    if name in CONTRACTS and CONTRACTS[name][0] is not model:
        raise ValueError(f"{name} is registered to another model")
    CONTRACTS[name] = (model, description)


def json_schema(name: str) -> dict[str, Any]:
    """The JSON Schema of contract ``name``, as ``contracts/schemas`` holds it."""
    model, description = CONTRACTS[name]
    schema = model.model_json_schema(by_alias=True, mode="validation")
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": f"{SCHEMA_ID_PREFIX}{name}",
        "title": name,
        "description": description,
        **{key: value for key, value in schema.items() if key not in ("title", "description")},
    }


def schema_filename(name: str) -> str:
    return f"{name.removeprefix('stratafi.')}.schema.json"


def validate_contract(name: str, data: Any) -> BaseModel:
    """Validate ``data`` against contract ``name``'s model."""
    model, _ = CONTRACTS[name]
    return model.model_validate(data)
