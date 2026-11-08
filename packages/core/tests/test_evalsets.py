"""Eval sets: the Halcyon demo set's retrieval YAML imports unchanged, other
formats load to the same items, and splits are seeded and stratified."""

import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from stratafi_core.evalsets import (
    NO_EVIDENCE_TAG,
    Asker,
    EvalItem,
    asker_of,
    assign_splits,
    dump_eval_set,
    eval_set_hash,
    items_to_jsonl,
    load_eval_set,
    resolve_eval_set,
    stratum,
)
from stratafi_core.specs import SpecInvalid
from stratafi_testing.canary import CanaryCheck

ROOT = Path(__file__).resolve().parents[3]
DEMO = ROOT / "fixtures" / "demo-corpus"
RETRIEVAL = (DEMO / "retrieval.yaml").read_text(encoding="utf-8")


def raw_queries() -> list[dict[str, Any]]:
    return yaml.safe_load(RETRIEVAL)["queries"]


def without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [without_none(item) for item in value]
    return value


def demo_set():
    return resolve_eval_set(load_eval_set(RETRIEVAL, "yaml", name="halcyon-retrieval"))


# --- the Halcyon demo set -------------------------------------------------------------------


def test_the_demo_set_imports_88_items_with_identical_evidence() -> None:
    """The demo set's retrieval YAML imports 88 items with identical expected, avoid,
    must_not_return and near misses."""
    resolved, report = demo_set()
    raw = raw_queries()
    assert len(resolved.items) == len(raw) == 88
    assert report.items == 88
    for item, source in zip(resolved.items, raw, strict=True):
        dumped = item.model_dump(mode="json", exclude_none=True)
        assert item.id == source["id"]
        assert item.query == source["query"]
        assert asker_of(item).name == source["asker"]
        for field in ("expected", "avoid", "must_not_return", "near_misses"):
            assert dumped.get(field, []) == without_none(source.get(field, [])), (item.id, field)
    assert sum(1 for item in resolved.items if item.must_not_return) == 9
    assert sum(1 for item in resolved.items if item.avoid) == 13
    assert sum(1 for item in resolved.items if not item.expected) == 21


def test_the_set_level_fields_carry_over() -> None:
    resolved, _ = demo_set()
    source = yaml.safe_load(RETRIEVAL)
    assert resolved.name == "halcyon-retrieval"
    assert resolved.tenant == source["tenant"] == "halcyon"
    assert resolved.as_of is not None
    assert resolved.as_of.isoformat() == source["as_of"]
    assert resolved.defaults.k == source["k"] == 5


def test_named_askers_resolve_through_the_sets_own_table() -> None:
    resolved, _ = demo_set()
    table = yaml.safe_load(RETRIEVAL)["askers"]
    used = {asker_of(item).name for item in resolved.items}
    assert used == {"staff", "manager", "exec"}
    assert set(resolved.askers) == used
    for name, asker in resolved.askers.items():
        assert (asker.roles, asker.groups) == (table[name]["roles"], table[name]["groups"])
        assert asker.user is None
    managers = next(item for item in resolved.items if item.id == "acl-safe-cash-limit")
    assert "managers" in asker_of(managers).groups
    staff_copy = next(i for i in resolved.items if i.id == "acl-safe-cash-limit--as-staff")
    assert "managers" not in asker_of(staff_copy).groups


def test_a_name_the_askers_table_lacks_is_an_issue() -> None:
    """Eval sets name their own askers: no built-in users stand behind a name."""
    document = yaml.safe_load(RETRIEVAL)
    del document["askers"]["exec"]
    with pytest.raises(SpecInvalid) as raised:
        resolve_eval_set(load_eval_set(yaml.safe_dump(document), "yaml", name="x"))
    found = {(issue.path, issue.code) for issue in raised.value.issues}
    assert {code for _, code in found} == {"unknown_asker"}
    assert len(found) == sum(1 for item in document["queries"] if item["asker"] == "exec")


def test_items_without_tags_get_their_documents_keys() -> None:
    resolved, report = demo_set()
    assert report.derived_tags == 88
    for item in resolved.items:
        keys = sorted({ref.doc_key for ref in item.expected})
        assert item.tags == (keys or [NO_EVIDENCE_TAG])


def test_the_askers_table_says_who_each_name_is() -> None:
    document = yaml.safe_load(RETRIEVAL)
    document["askers"]["staff"] = {"roles": ["teller"], "groups": ["branch"]}
    resolved, _ = resolve_eval_set(load_eval_set(yaml.safe_dump(document), "yaml", name="x"))
    staff = next(item for item in resolved.items if asker_of(item).name == "staff")
    assert asker_of(staff) == Asker(name="staff", roles=["teller"], groups=["branch"])


def test_resolving_twice_changes_nothing_and_the_hash_is_stable() -> None:
    resolved, _ = demo_set()
    again, report = resolve_eval_set(resolved)
    assert report.assigned_splits == 0
    assert report.derived_tags == 0
    assert eval_set_hash(again) == eval_set_hash(resolved)
    reloaded = load_eval_set(yaml.safe_dump(dump_eval_set(resolved)), "yaml")
    assert eval_set_hash(reloaded) == eval_set_hash(resolved)


# --- other formats ------------------------------------------------------------------------


def as_csv(items: list[EvalItem]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["id", "query", "asker", "doc_key", "version", "section", "evidence",
                    "source_uri", "avoid", "must_not_return", "tags", "split"],
    )  # fmt: skip
    writer.writeheader()
    for item in items:
        refs = lambda refs: ";".join(  # noqa: E731
            ref.doc_key + (f"@{ref.version}" if ref.version else "") for ref in refs
        )
        base = {
            "id": item.id,
            "query": item.query,
            "asker": asker_of(item).name or "",
            "avoid": refs(item.avoid),
            "must_not_return": refs(item.must_not_return),
            "tags": ";".join(item.tags),
            "split": item.split or "",
        }
        evidence = item.expected or [None]
        for ref in evidence:
            row = dict(base)
            if ref is not None:
                row.update({key: value or "" for key, value in ref.model_dump().items()})
            writer.writerow(row)
    return buffer.getvalue()


def comparable(items: list[EvalItem]) -> list[dict[str, Any]]:
    return [
        item.model_dump(mode="json", exclude_none=True, exclude={"near_misses", "asker"})
        | {"asker": asker_of(item).name}
        for item in items
    ]


def test_jsonl_and_csv_load_to_the_same_items_as_yaml() -> None:
    resolved, _ = demo_set()
    from_jsonl, _ = resolve_eval_set(
        load_eval_set(items_to_jsonl(resolved.items), "jsonl", name="halcyon-retrieval")
    )
    assert from_jsonl.items == resolved.items
    askers = {name: asker.model_dump(exclude_none=True) for name, asker in resolved.askers.items()}
    from_csv, _ = resolve_eval_set(
        load_eval_set(as_csv(resolved.items), "csv", name="demo-csv", overrides={"askers": askers})
    )
    assert comparable(from_csv.items) == comparable(resolved.items)
    assert from_csv.askers == resolved.askers


def test_a_jsonl_header_line_holds_the_sets_own_fields() -> None:
    text = "\n".join(
        [
            json.dumps({"schema": "stratafi.eval-set.v1", "as_of": "2026-11-02"}),
            json.dumps({"id": "q1", "query": "Who approves wires?", "asker": "staff"}),
        ]
    )
    loaded = load_eval_set(text, "jsonl", name="wires")
    assert loaded.as_of is not None
    assert loaded.as_of.isoformat() == "2026-11-02"
    assert [item.id for item in loaded.items] == ["q1"]


def test_csv_rows_with_the_same_id_merge_their_evidence() -> None:
    text = (
        "id,query,doc_key,version,section,evidence,tags,split\n"
        "q1,How long is the hold?,funds-availability,3,5.1 New,first 30 days,holds,dev\n"
        "q1,How long is the hold?,funds-availability,3,5.2 Availability,ninth day,,\n"
        "q2,Is there a crypto policy?,,,,,,\n"
    )
    loaded = load_eval_set(text, "csv", name="holds")
    first, second = loaded.items
    assert [ref.section for ref in first.expected] == ["5.1 New", "5.2 Availability"]
    assert (first.tags, first.split) == (["holds"], "dev")
    assert second.expected == []


@pytest.mark.parametrize(
    ("text", "fmt", "path", "code"),
    [
        ("items: [", "yaml", "(root)", "invalid_yaml"),
        ("- just a list", "yaml", "(root)", "not_an_object"),
        ("a: &a [1]\nb: *a\n", "yaml", "(root)", "invalid_yaml"),
        ('{"id": "q"}\nnot json\n', "jsonl", "lines.2", "invalid_json"),
        ("id,query,colour\nq,Hi?,red\n", "csv", "columns.2", "unknown_column"),
        ("id,doc_key\nq,x\n", "csv", "columns", "missing_column"),
        ("id,query\nq,One?\nq,Two?\n", "csv", "rows.3.query", "conflicting_query"),
    ],
)
def test_unreadable_sources_fail_with_a_path_and_a_code(
    text: str, fmt: Any, path: str, code: str
) -> None:
    with pytest.raises(SpecInvalid) as error:
        load_eval_set(text, fmt, name="x")
    assert [(issue.path, issue.code) for issue in error.value.issues] == [(path, code)]


def test_invalid_items_fail_with_precise_paths() -> None:
    document = {
        "name": "x",
        "items": [
            {"id": "q1", "query": "One?"},
            {"id": "q1", "query": "Two?", "expected": [{"doc_key": "Bad Key"}]},
            {"id": "q3", "query": "   "},
        ],
    }
    with pytest.raises(SpecInvalid) as error:
        load_eval_set(json.dumps(document), "yaml")
    paths = error.value.paths
    assert "items.1.expected.0.doc_key" in paths
    assert "items.2.query" in paths
    with pytest.raises(SpecInvalid) as unknown:
        resolve_eval_set(load_eval_set('{"name": "x", "items": [{"id": "q", "query": "Q?", '
                                       '"asker": "nobody"}]}', "yaml"))  # fmt: skip
    assert [(i.path, i.code) for i in unknown.value.issues] == [("items.0.asker", "unknown_asker")]


def test_issues_never_carry_the_documents_text(canary: CanaryCheck) -> None:
    document = {"name": "x", "items": [{"id": "q", "query": canary.value, "split": canary.value}]}
    with pytest.raises(SpecInvalid) as error:
        load_eval_set(json.dumps(document), "yaml")
    canary.assert_absent(str(error.value), repr(error.value.issues))
    canary.assert_not_logged()


# --- splits -------------------------------------------------------------------------------


def splits_of(items: list[EvalItem]) -> dict[str, str | None]:
    return {item.id: item.split for item in items}


def test_splits_are_deterministic_for_a_seed() -> None:
    """Splits are deterministic for a seed (and stratified by tag, below)."""
    first, _ = resolve_eval_set(load_eval_set(RETRIEVAL, "yaml", name="a"), seed=7)
    second, _ = resolve_eval_set(load_eval_set(RETRIEVAL, "yaml", name="b"), seed=7)
    other, _ = resolve_eval_set(load_eval_set(RETRIEVAL, "yaml", name="c"), seed=8)
    assert splits_of(first.items) == splits_of(second.items)
    assert splits_of(first.items) != splits_of(other.items)


def test_splits_are_stratified_by_tag() -> None:
    """Each tag stratum sends its share to the test split, within one item."""
    for seed in range(5):
        resolved, report = resolve_eval_set(
            load_eval_set(RETRIEVAL, "yaml", name="x"), test_fraction=0.3, seed=seed
        )
        assert report.splits["test"] == math.floor(88 * 0.3 + 0.5)
        strata: dict[Any, Counter[str]] = defaultdict(Counter)
        for item in resolved.items:
            strata[stratum(item)][item.split] += 1
        for counts in strata.values():
            size = sum(counts.values())
            assert math.floor(size * 0.3) <= counts["test"] <= math.ceil(size * 0.3)


def test_splits_stratify_by_the_gold_documents_file_types_when_known() -> None:
    items = [
        EvalItem(id=f"q{n}", query="Q?", expected=[{"doc_key": "pdf-doc" if n % 2 else "md-doc"}],
                 tags=["same"])
        for n in range(20)
    ]  # fmt: skip
    types = {"pdf-doc": "application/pdf", "md-doc": "text/markdown"}
    split = assign_splits(items, test_fraction=0.5, seed=1, file_types=types)
    tests = Counter(types[item.expected[0].doc_key] for item in split if item.split == "test")
    assert tests == {"application/pdf": 5, "text/markdown": 5}


def test_items_that_name_a_split_keep_it_unless_resplit() -> None:
    items = [EvalItem(id=f"q{n}", query="Q?", split="test") for n in range(10)]
    assert {item.split for item in assign_splits(items, test_fraction=0.2)} == {"test"}
    resplit = assign_splits(items, test_fraction=0.2, resplit=True)
    assert Counter(item.split for item in resplit) == {"dev": 8, "test": 2}


@settings(max_examples=60, deadline=None)
@given(
    tags=st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=1, max_size=60),
    fraction=st.floats(min_value=0, max_value=0.9),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_split_quotas_hold_for_any_strata(tags: list[str], fraction: float, seed: int) -> None:
    items = [EvalItem(id=f"q{n}", query="Q?", tags=[tag]) for n, tag in enumerate(tags)]
    split = assign_splits(items, test_fraction=fraction, seed=seed)
    assert split == assign_splits(items, test_fraction=fraction, seed=seed)
    assert sum(1 for item in split if item.split == "test") == math.floor(
        len(items) * fraction + 0.5
    )
    for tag in set(tags):
        size = tags.count(tag)
        test = sum(1 for item in split if item.tags == [tag] and item.split == "test")
        assert math.floor(size * fraction) <= test <= math.ceil(size * fraction)


def test_the_test_fraction_is_bounded() -> None:
    with pytest.raises(ValueError, match="fraction"):
        assign_splits([EvalItem(id="q", query="Q?")], test_fraction=0.95)
