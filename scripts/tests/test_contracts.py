"""The contracts: every fixture validates against its model and against its
exported JSON Schema, and the exported schemas are current."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from stratafi_core.contracts import CONTRACTS, schema_filename, validate_contract
from stratafi_core.specs import (
    PIPELINE_SCHEMA,
    RETRIEVAL_SCHEMA,
    resolve_pipeline,
    resolve_retrieval,
)
from stratafi_testing.components import stub_registry

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = ROOT / "contracts"


def load_script():
    path = ROOT / "scripts" / "export_contracts.py"
    spec = importlib.util.spec_from_file_location("script_export_contracts", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def documents(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if path.suffix in (".yaml", ".yml"):
        return [yaml.safe_load(text)]
    return [json.loads(text)]


FIXTURES = [
    (name, path)
    for name in sorted(CONTRACTS)
    for path in sorted((CONTRACTS_DIR / "fixtures" / name.removeprefix("stratafi.")).glob("*"))
]


def test_every_contract_has_fixtures() -> None:
    for name in CONTRACTS:
        assert any(contract == name for contract, _ in FIXTURES), name


@pytest.mark.parametrize(("name", "path"), FIXTURES, ids=[p.name for _, p in FIXTURES])
def test_the_schema_fixtures_validate(name: str, path: Path) -> None:
    schema = json.loads((CONTRACTS_DIR / "schemas" / schema_filename(name)).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for document in documents(path):
        errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
        assert errors == [], [f"{list(e.absolute_path)}: {e.message}" for e in errors]
        validate_contract(name, document)


def test_the_spec_fixtures_also_pass_the_validator() -> None:
    registry = stub_registry()
    resolvers = {PIPELINE_SCHEMA: resolve_pipeline, RETRIEVAL_SCHEMA: resolve_retrieval}
    for name, path in FIXTURES:
        if name in resolvers:
            for document in documents(path):
                resolvers[name](document, registry)


def test_the_schemas_refuse_what_the_models_refuse() -> None:
    schema = json.loads((CONTRACTS_DIR / "schemas" / schema_filename(PIPELINE_SCHEMA)).read_text())
    bad = {"schema": PIPELINE_SCHEMA, "name": "x", "stages": {"chunk": {"bogus": 1}}}
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(bad))


def test_the_exported_schemas_are_current() -> None:
    assert load_script().main(["--check"]) == 0


def test_a_stale_schema_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = load_script()
    root = tmp_path / "contracts"
    (root / "fixtures").mkdir(parents=True)
    assert script.main([], root=root) == 0
    (root / "schemas" / "chunk.v1.schema.json").write_text("{}\n")
    assert script.main(["--check"], root=root) == 1
