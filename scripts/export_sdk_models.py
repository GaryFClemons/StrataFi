#!/usr/bin/env python3
"""Generate the SDK's models from the OpenAPI contract, or check they're current.

    uv run python scripts/export_sdk_models.py            # write them
    uv run python scripts/export_sdk_models.py --check    # exit 1 if stale (CI runs this)

Every schema in ``contracts/openapi.json`` (``components.schemas``) becomes a Pydantic model in
``packages/sdk/src/stratafi/models.py``, so the Python SDK and the web client
(``npm run api:generate``) share one contract. Models ignore fields they don't know, so an
older SDK still reads a newer API's answers. The output is formatted with ruff, so the check
compares exactly what is committed.

The generator covers what FastAPI and Pydantic put in the contract: objects, arrays (and
tuples), maps, unions with null, ``$ref``\\ s, ``enum`` and ``const`` as ``Literal``, and the
``uuid``, ``date`` and ``date-time`` formats.
"""

import argparse
import json
import keyword
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "contracts" / "openapi.json"
OUTPUT = ROOT / "packages" / "sdk" / "src" / "stratafi" / "models.py"
RESERVED = {"schema", "json", "dict", "copy", "construct", "validate", "fields"}

HEADER = '''"""The Stratafi API's bodies as Pydantic models.

Generated from contracts/openapi.json by scripts/export_sdk_models.py: don't edit, regenerate.
Models ignore fields they don't know, so an older SDK reads a newer API's answers.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
'''


def class_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", name)


def literal(values: list[Any]) -> str:
    # JSON's true and false are Python's True and False; strings and numbers read the same.
    parts = [repr(value) if isinstance(value, bool) else json.dumps(value) for value in values]
    return f"Literal[{', '.join(parts)}]"


def annotation(schema: dict[str, Any]) -> str:
    """A schema as a type annotation."""
    if "$ref" in schema:
        return class_name(schema["$ref"].rsplit("/", 1)[-1])
    if "anyOf" in schema or "oneOf" in schema:
        parts: list[str] = []
        for option in schema.get("anyOf") or schema.get("oneOf") or []:
            part = annotation(option)
            if part not in parts:
                parts.append(part)
        if "None" in parts:  # keep None last, as people write it
            parts = [part for part in parts if part != "None"] + ["None"]
        return " | ".join(parts) if parts else "Any"
    if "const" in schema:
        return literal([schema["const"]])
    if "enum" in schema:
        return literal(schema["enum"])
    kind = schema.get("type")
    if kind == "string":
        return {"uuid": "uuid.UUID", "date-time": "datetime.datetime", "date": "datetime.date"}.get(
            schema.get("format", ""), "str"
        )
    if kind == "integer":
        return "int"
    if kind == "number":
        return "float"
    if kind == "boolean":
        return "bool"
    if kind == "null":
        return "None"
    if kind == "array":
        if "prefixItems" in schema:
            return f"tuple[{', '.join(annotation(item) for item in schema['prefixItems'])}]"
        items = schema.get("items")
        return f"list[{annotation(items)}]" if items else "list[Any]"
    if kind == "object" or "additionalProperties" in schema:
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict) and extra:
            return f"dict[str, {annotation(extra)}]"
        return "dict[str, Any]"
    return "Any"


def default_of(schema: dict[str, Any]) -> tuple[str, bool]:
    """A field's default as source, and whether it needs a factory."""
    if "default" not in schema:
        return "None", False
    value = schema["default"]
    if isinstance(value, list | dict):
        return f"lambda: {value!r}", True
    return repr(value), False


def field_source(name: str, schema: dict[str, Any], required: bool) -> str:
    python = name
    arguments: list[str] = []
    if keyword.iskeyword(name) or name in RESERVED or not name.isidentifier():
        python = re.sub(r"\W", "_", name) + "_"
        arguments.append(f"alias={name!r}")
    if not required:
        default, factory = default_of(schema)
        arguments.insert(0, f"default_factory={default}" if factory else f"default={default}")
    description = schema.get("description")
    if description:
        arguments.append(f"description={description!r}")
    kind = annotation(schema)
    if not required and "default" not in schema and "None" not in kind.split(" | "):
        kind = f"{kind} | None"
    if arguments:
        return f"    {python}: {kind} = Field({', '.join(arguments)})"
    return f"    {python}: {kind}"


def model_source(name: str, schema: dict[str, Any]) -> str:
    lines = [f"class {class_name(name)}(ApiModel):"]
    if schema.get("description"):
        lines.append(f"    {json.dumps(schema['description'])}")
        lines.append("")
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    fields = [field_source(field, sub, field in required) for field, sub in properties.items()]
    if not fields:
        lines.append("    pass")
    lines += fields
    return "\n".join(lines)


def render(contract: dict[str, Any]) -> str:
    schemas = contract.get("components", {}).get("schemas", {})
    parts = [HEADER]
    names = sorted(schemas)
    for name in names:
        parts.append("\n\n" + model_source(name, schemas[name]))
    parts.append(
        "\n\nfor _model in ("
        + ", ".join(class_name(name) for name in names)
        + ",):\n    _model.model_rebuild()\n"
    )
    source = "".join(parts)
    formatted = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", str(OUTPUT), "-"],
        input=source,
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return formatted.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the SDK's models from the contract.")
    parser.add_argument("--check", action="store_true", help="fail if the models are stale")
    args = parser.parse_args(argv)
    text = render(json.loads(CONTRACT.read_text(encoding="utf-8")))
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{OUTPUT.relative_to(ROOT).as_posix()} is stale", file=sys.stderr)
            print("run `uv run python scripts/export_sdk_models.py`", file=sys.stderr)
            return 1
        print("the SDK's models are current")
        return 0
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
