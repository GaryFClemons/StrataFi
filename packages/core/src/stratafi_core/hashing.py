"""Canonical JSON and the hashes built on it.

Two documents that mean the same thing must hash the same, so hashing goes through one canonical
form:

- keys sorted, no insignificant whitespace, UTF-8 (``ensure_ascii=False``)
- Pydantic models are dumped in JSON mode first, so a validated model hashes with every default
  filled in (``resolve`` before hashing is the caller's job: specs and parameters are validated,
  then dumped)
- tuples and lists are both arrays; sets and frozensets are sorted arrays
- dates and datetimes are ISO 8601 strings, enums their values
- NaN and infinities are refused; ``-0.0`` is written as ``0.0``

Numbers keep their JSON type: ``1`` and ``1.0`` differ. Parameters reach hashing through their
Pydantic models, which fix each field's type, so a value can't flip between the two.

``cache_key`` is the address of a stage output: a hash of the stage, the component and its
version, its resolved parameters, its inputs' keys, and the environment that changes output (a
model revision, a tokenizer version). The same work always gets the same key, so a retried job
or a later run finds the artifact already written.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

HEX64_LENGTH = 64


class CanonicalJsonError(ValueError):
    """A value has no canonical JSON form. The message names the path, never the value."""


def to_canonical(value: Any, path: str = "$") -> Any:
    """``value`` as plain JSON data: dicts with string keys, lists, strings, numbers, booleans
    and None."""
    if isinstance(value, BaseModel):
        return to_canonical(value.model_dump(mode="json", by_alias=True), path)
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError(f"{path}: NaN and infinities have no JSON form")
        return 0.0 if value == 0 else value
    if isinstance(value, Enum):
        return to_canonical(value.value, path)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"{path}: JSON object keys must be strings")
            result[key] = to_canonical(item, f"{path}.{key}")
        return result
    if isinstance(value, set | frozenset):
        items = [to_canonical(item, f"{path}[]") for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [to_canonical(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise CanonicalJsonError(f"{path}: {type(value).__name__} has no canonical JSON form")


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(value: Any) -> str:
    """The SHA-256 of ``value``'s canonical JSON."""
    return sha256_hex(canonical_bytes(value))


def is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == HEX64_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def cache_key(
    *,
    stage: str,
    component_id: str,
    component_version: str,
    params: Mapping[str, Any] | BaseModel,
    inputs: Sequence[str],
    env: Mapping[str, Any] | None = None,
    nonce: str | None = None,
) -> str:
    """The cache key of one stage output.

    ``params`` must already be resolved (validated, with every default filled in), and
    ``inputs`` are the input artifacts' cache keys or content hashes, in the order the stage
    reads them. ``nonce`` makes a deliberate re-run of a non-deterministic component a new
    artifact.
    """
    for key in inputs:
        if not is_hex64(key):
            raise CanonicalJsonError("inputs must be cache keys or SHA-256 content hashes")
    material: dict[str, Any] = {
        "stage": stage,
        "component_id": component_id,
        "component_version": component_version,
        "params": params,
        "inputs": list(inputs),
        "env": dict(env or {}),
    }
    if nonce is not None:
        material["nonce"] = nonce
    return content_hash(material)
