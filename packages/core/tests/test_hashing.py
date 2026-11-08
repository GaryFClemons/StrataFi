"""Canonical JSON and cache keys: stable across key order and default filling."""

import math
from datetime import UTC, date, datetime

import pytest
from pydantic import BaseModel, ConfigDict

from stratafi_core.hashing import (
    CanonicalJsonError,
    cache_key,
    canonical_json,
    content_hash,
    is_hex64,
)


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 800
    overlap: float = 0.1
    estimator: str = "chars/4"


def test_key_order_never_changes_the_hash() -> None:
    a = {"b": 1, "a": {"y": [1, 2, {"q": 1, "p": 2}], "x": None}}
    b = {"a": {"x": None, "y": [1, 2, {"p": 2, "q": 1}]}, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert content_hash(a) == content_hash(b)


def test_list_order_does_change_the_hash() -> None:
    assert content_hash([1, 2]) != content_hash([2, 1])


def test_a_model_hashes_with_its_defaults_filled_in() -> None:
    assert content_hash(Params()) == content_hash(Params(max_tokens=800))
    assert content_hash(Params()) == content_hash(
        {"max_tokens": 800, "overlap": 0.1, "estimator": "chars/4"}
    )
    assert content_hash(Params()) != content_hash(Params(max_tokens=600))


def test_the_canonical_form_is_compact_sorted_utf8() -> None:
    assert canonical_json({"é": "ü", "a": [1.5, True, None]}) == '{"a":[1.5,true,null],"é":"ü"}'


def test_tuples_sets_dates_and_negative_zero_have_one_form() -> None:
    assert canonical_json((1, 2)) == canonical_json([1, 2])
    assert canonical_json({3, 1, 2}) == canonical_json([1, 2, 3])
    assert canonical_json(date(2026, 10, 30)) == '"2026-10-30"'
    assert canonical_json(datetime(2026, 10, 30, tzinfo=UTC)) == '"2026-10-30T00:00:00+00:00"'
    assert canonical_json(-0.0) == canonical_json(0.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, {1: "a"}, b"bytes", object()])
def test_values_without_a_json_form_are_refused(value: object) -> None:
    with pytest.raises(CanonicalJsonError):
        canonical_json(value)


def test_the_error_names_the_path_not_the_value() -> None:
    class Secret:
        def __repr__(self) -> str:
            return "a private sentence"

        __str__ = __repr__

    with pytest.raises(CanonicalJsonError) as raised:
        canonical_json({"params": {"field": Secret()}})
    assert "$.params.field" in str(raised.value)
    assert "private" not in str(raised.value)


def test_cache_keys_cover_every_part() -> None:
    inputs = [content_hash("document")]
    base = {
        "stage": "parse",
        "component_id": "parse/markdown",
        "component_version": "1.0.0",
        "params": Params(),
        "inputs": inputs,
        "env": {"pypdf": "6.21.2"},
    }
    key = cache_key(**base)
    assert is_hex64(key)
    assert (
        cache_key(**{**base, "params": {"estimator": "chars/4", "overlap": 0.1, "max_tokens": 800}})
        == key
    )
    for change in (
        {"stage": "chunk"},
        {"component_id": "parse/other"},
        {"component_version": "1.0.1"},
        {"params": Params(max_tokens=1)},
        {"inputs": [content_hash("other")]},
        {"env": {"pypdf": "6.22.0"}},
    ):
        assert cache_key(**{**base, **change}) != key, change
    assert cache_key(**base, nonce="rerun-1") != key


def test_cache_key_inputs_must_be_hashes() -> None:
    with pytest.raises(CanonicalJsonError):
        cache_key(
            stage="parse",
            component_id="parse/x",
            component_version="1.0.0",
            params={},
            inputs=["not-a-hash"],
        )
