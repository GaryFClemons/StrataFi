"""Eval sets: questions with known evidence.

``stratafi.eval-set.v1`` also reads the retrieval YAML format (``fixtures/demo-corpus/
retrieval.yaml``) unchanged:

- its top level (``tenant``, ``as_of``, ``k``, ``queries``) maps onto the eval set's (``tenant``,
  ``as_of``, ``defaults.k``, ``items``)
- a named asker (``asker: staff``) resolves through the set's own ``askers`` table: the user,
  roles and groups that document ACLs are checked against
- abstention items' ``near_misses`` and each evidence item's ``source_uri`` are kept

Gold labels point at documents and evidence text, never at chunks: an evidence item is a
``doc_key``, a ``version``, a ``section`` and an ``evidence`` phrase, which each run maps onto its
own chunks (``stratafi_core.relevance``).

``load_eval_set`` reads YAML (either shape), JSONL (one item per line, with an optional header
line) and CSV (one row per evidence item; rows with the same ``id`` merge) into one ``EvalSet``.
``resolve_eval_set`` then fills in what the source leaves open, so a stored version is complete
without it:

- every asker becomes an object with its name, user, roles and groups
- an item without tags gets derived ones: its expected documents' keys, or ``no-evidence``

- items without a split get one from ``assign_splits``: seeded, and stratified by tags and by the
  file types of the gold documents when those are known

Errors are ``SpecIssue``\\ s, a path and a code, never the offending text.
"""

import csv
import hashlib
import io
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from stratafi_core.hashing import canonical_json, content_hash
from stratafi_core.specs import NAME_PATTERN, SpecInvalid, SpecIssue, validation_issues

EVAL_SET_SCHEMA = "stratafi.eval-set.v1"
SPLITS = ("dev", "test")
MAX_ITEMS = 10_000
NO_EVIDENCE_TAG = "no-evidence"
DEFAULT_TEST_FRACTION = 0.3
MAX_TEST_FRACTION = 0.9

Format = Literal["yaml", "jsonl", "csv"]
Split = Literal["dev", "test"]
Name = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
ItemId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]
DocKey = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")]
DocVersion = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")]
Tag = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._:-]{0,63}$")]
AskerName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
Principal = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[^\x00-\x1f\x7f]+$")
]


def _as_text(value: Any) -> Any:
    """A version written as a YAML number (``version: 3``) is the string ``"3"``."""
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Asker(_Model):
    """Who asks: what ACL filtering sees of a person."""

    name: AskerName | None = Field(default=None, description="A name in the set's askers table")
    user: Principal | None = Field(
        default=None, description="Their user ID, for ACLs that list users"
    )
    roles: list[Principal] = Field(default_factory=list, max_length=64)
    groups: list[Principal] = Field(default_factory=list, max_length=256)


class EvidenceRef(_Model):
    """Gold evidence: a document version, a section, and a phrase from it."""

    doc_key: DocKey
    version: DocVersion | None = Field(default=None, description="None: any version")
    section: str | None = Field(default=None, min_length=1, max_length=500)
    evidence: str | None = Field(default=None, min_length=1, max_length=2000)
    source_uri: str | None = Field(default=None, min_length=1, max_length=512)

    _version = field_validator("version", mode="before")(_as_text)


class DocRef(_Model):
    """A document, or one version of it: what ``avoid`` and ``must_not_return`` name."""

    doc_key: DocKey
    version: DocVersion | None = Field(default=None, description="None: every version")

    _version = field_validator("version", mode="before")(_as_text)


class EvalItem(_Model):
    id: ItemId
    query: str = Field(min_length=1, max_length=4000)
    asker: Asker | AskerName | None = Field(
        default=None, description="A name from askers, or an asker; the set's default if none"
    )
    expected: list[EvidenceRef] = Field(
        default_factory=list,
        max_length=100,
        description="Gold evidence; empty means the answer is not in the corpus",
    )
    avoid: list[DocRef] = Field(
        default_factory=list, max_length=100, description="Must not rank above the first hit"
    )
    must_not_return: list[DocRef] = Field(
        default_factory=list, max_length=100, description="Must never be returned"
    )
    near_misses: list[EvidenceRef] = Field(
        default_factory=list,
        max_length=100,
        description="An abstention item's closest passages, which a good answer cites",
    )
    reference_answer: str | None = Field(default=None, max_length=20_000)
    tags: list[Tag] = Field(default_factory=list, max_length=32)
    split: Split | None = None

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the query is empty")
        return value


class EvalDefaults(_Model):
    k: int = Field(default=5, ge=1, le=100, description="The cutoff the set's own hit rule uses")
    asker: Asker | AskerName | None = Field(
        default=None, description="The asker of items that name none"
    )


class EvalSet(_Model):
    """``stratafi.eval-set.v1``."""

    schema_: Literal["stratafi.eval-set.v1"] = Field(default=EVAL_SET_SCHEMA, alias="schema")
    name: Name
    description: str = Field(default="", max_length=2000)
    tenant: str | None = Field(default=None, max_length=64, description="The source's tenant")
    as_of: date | None = Field(default=None, description='"Today" for effective-date filters')
    defaults: EvalDefaults = Field(default_factory=EvalDefaults)
    askers: dict[AskerName, Asker] = Field(default_factory=dict, max_length=256)
    items: list[EvalItem] = Field(min_length=1, max_length=MAX_ITEMS)

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvalSet":
        seen: set[str] = set()
        for index, item in enumerate(self.items):
            if item.id in seen:
                raise ValueError(f"items.{index}.id repeats an earlier item's")
            seen.add(item.id)
        return self


# --- loading ------------------------------------------------------------------------------


class _Loader(yaml.SafeLoader):
    """``yaml.safe_load``, refusing aliases: an eval set never needs them, and they can expand a
    small file into a huge document."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.events.AliasEvent):
            raise yaml.YAMLError("aliases are not allowed")
        return super().compose_node(parent, index)


def load_yaml(text: str) -> Any:
    """YAML without aliases. Raises ``SpecInvalid``."""
    try:
        return yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass
    except yaml.YAMLError:
        raise SpecInvalid([SpecIssue("(root)", "invalid_yaml")]) from None


def _from_retrieval_yaml(data: Mapping[str, Any]) -> dict[str, Any]:
    """The retrieval YAML shape (``k``, ``queries``) as an eval set's."""
    result = {key: value for key, value in data.items() if key not in ("k", "queries")}
    result["items"] = data["queries"]
    if "k" in data:
        result["defaults"] = {**dict(data.get("defaults") or {}), "k": data["k"]}
    return result


def _yaml_document(text: str) -> dict[str, Any]:
    data = load_yaml(text)
    if not isinstance(data, dict):
        raise SpecInvalid([SpecIssue("(root)", "not_an_object")])
    if "queries" in data and "items" not in data:
        return _from_retrieval_yaml(data)
    return dict(data)


def _jsonl_document(text: str) -> dict[str, Any]:
    """One item per line. A first line without a ``query`` holds the set's own fields."""
    header: dict[str, Any] = {}
    items: list[Any] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise SpecInvalid([SpecIssue(f"lines.{number}", "invalid_json")]) from None
        if not isinstance(value, dict):
            raise SpecInvalid([SpecIssue(f"lines.{number}", "not_an_object")])
        if "query" not in value and not items and not header:
            header = value
            continue
        items.append(value)
    return {**header, "items": items}


CSV_COLUMNS = (
    "id",
    "query",
    "asker",
    "doc_key",
    "version",
    "section",
    "evidence",
    "source_uri",
    "avoid",
    "must_not_return",
    "tags",
    "split",
    "reference_answer",
)
_EVIDENCE_COLUMNS = ("doc_key", "version", "section", "evidence", "source_uri")
_TEXT_COLUMNS = frozenset({"query", "section", "evidence", "reference_answer"})


def _refs(cell: str) -> list[dict[str, str]]:
    """``key@version;key`` as document references."""
    refs = []
    for part in cell.split(";"):
        part = part.strip()
        if part:
            key, _, version = part.partition("@")
            refs.append(
                {"doc_key": key.strip(), **({"version": version.strip()} if version else {})}
            )
    return refs


def _csv_document(text: str) -> dict[str, Any]:
    """One row per evidence item; rows with the same ``id`` are one item. ``avoid`` and
    ``must_not_return`` hold ``key@version`` references and ``tags`` names, separated by ``;``.
    A row with no ``doc_key`` adds no evidence (an abstention item)."""
    reader = csv.DictReader(io.StringIO(text.removeprefix("﻿")))
    columns = list(reader.fieldnames or [])
    issues = [
        SpecIssue(f"columns.{index}", "unknown_column")
        for index, column in enumerate(columns)
        if column not in CSV_COLUMNS
    ]
    if "id" not in columns or "query" not in columns:
        issues.append(SpecIssue("columns", "missing_column"))
    if issues:
        raise SpecInvalid(issues)
    items: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(reader, start=2):
        # Text is kept as written; names, references and splits are trimmed.
        cells = {
            key: (value or "") if key in _TEXT_COLUMNS else (value or "").strip()
            for key, value in row.items()
            if key is not None
        }
        item_id = cells.get("id", "")
        item = items.get(item_id)
        if item is None:
            item = {"id": item_id, "query": cells.get("query", ""), "expected": []}
            if cells.get("asker"):
                item["asker"] = cells["asker"]
            for column in ("avoid", "must_not_return"):
                if cells.get(column):
                    item[column] = _refs(cells[column])
            if cells.get("tags"):
                item["tags"] = [tag.strip() for tag in cells["tags"].split(";") if tag.strip()]
            for column in ("split", "reference_answer"):
                if cells.get(column):
                    item[column] = cells[column]
            items[item_id] = item
        elif cells.get("query") and cells["query"] != item["query"]:
            raise SpecInvalid([SpecIssue(f"rows.{number}.query", "conflicting_query")])
        if cells.get("doc_key"):
            item["expected"].append(
                {column: cells[column] for column in _EVIDENCE_COLUMNS if cells.get(column)}
            )
    return {"items": list(items.values())}


def load_eval_set(
    text: str,
    fmt: Format,
    *,
    name: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> EvalSet:
    """Parse an eval set. ``name`` names it whatever the source says; ``overrides`` replace
    set-level fields (``description``, ``as_of``, ``tenant``, ``defaults``) when not None,
    which JSONL and CSV sources need. Raises ``SpecInvalid``."""
    if fmt == "yaml":
        data = _yaml_document(text)
    elif fmt == "jsonl":
        data = _jsonl_document(text)
    elif fmt == "csv":
        data = _csv_document(text)
    else:
        raise SpecInvalid([SpecIssue("format", "unknown_format")])
    if name is not None:
        data["name"] = name
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value
    try:
        return EvalSet.model_validate(data)
    except ValidationError as exc:
        raise SpecInvalid(validation_issues(exc, branch_tags=frozenset({"Asker"}))) from None


def dump_eval_set(eval_set: EvalSet) -> dict[str, Any]:
    return eval_set.model_dump(mode="json", by_alias=True, exclude_none=True)


def eval_set_hash(eval_set: EvalSet) -> str:
    """The hash of the whole set, items included: identical imports are one version."""
    return content_hash(dump_eval_set(eval_set))


def items_to_jsonl(items: Sequence[EvalItem]) -> str:
    return "".join(
        json.dumps(item.model_dump(mode="json", exclude_none=True), ensure_ascii=False) + "\n"
        for item in items
    )


# --- resolving ----------------------------------------------------------------------------


def asker_of(item: EvalItem) -> Asker:
    """A resolved item's asker (``resolve_eval_set`` makes every asker an object)."""
    return item.asker if isinstance(item.asker, Asker) else Asker()


def derived_tags(item: EvalItem) -> list[str]:
    """Tags for an item that has none: its expected documents' keys, or ``no-evidence``."""
    return sorted({ref.doc_key for ref in item.expected}) or [NO_EVIDENCE_TAG]


def _seed_for(seed: int, key: Any) -> int:
    digest = hashlib.sha256(canonical_json([seed, key]).encode()).digest()
    return int.from_bytes(digest[:8], "big")


Stratum = tuple[tuple[str, ...], tuple[str, ...]]


def stratum(item: EvalItem, file_types: Mapping[str, str] | None = None) -> Stratum:
    """What a split keeps in proportion: an item's tags, and its gold documents' file types."""
    types = {
        file_types[ref.doc_key] for ref in item.expected if file_types and ref.doc_key in file_types
    }
    return tuple(sorted(item.tags)), tuple(sorted(types))


def _quotas(sizes: Mapping[Stratum, int], fraction: float, seed: int) -> dict[Stratum, int]:
    """Test items per stratum: each its floor or ceiling of ``size * fraction``, summing to the
    whole's rounded share. Largest remainders get the ceilings; ties go by the seed."""
    exact = {key: size * fraction for key, size in sizes.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    target = math.floor(sum(sizes.values()) * fraction + 0.5)
    spare = target - sum(quotas.values())
    order = sorted(sizes, key=lambda key: (-(exact[key] - quotas[key]), _seed_for(seed, key)))
    for key in order[:spare]:
        quotas[key] += 1
    return quotas


def assign_splits(
    items: Sequence[EvalItem],
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = 0,
    resplit: bool = False,
    file_types: Mapping[str, str] | None = None,
) -> list[EvalItem]:
    """Give each item without a split (every item with ``resplit``) ``dev`` or ``test``.

    Items are grouped by ``stratum``; each stratum sends its quota (``_quotas``) to ``test``,
    chosen by a generator seeded from ``seed`` and the stratum, so the same seed gives the same
    split, and a stratum's choice doesn't move when another stratum changes."""
    if not 0 <= test_fraction <= MAX_TEST_FRACTION:
        raise ValueError("the test fraction is between 0 and 0.9")
    strata: dict[Stratum, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        if resplit or item.split is None:
            strata[stratum(item, file_types)].append(index)
    quotas = _quotas({key: len(members) for key, members in strata.items()}, test_fraction, seed)
    splits: dict[int, Split] = {}
    for key in sorted(strata, key=canonical_json):
        members = sorted(strata[key], key=lambda index: items[index].id)
        chosen = random.Random(_seed_for(seed, key)).sample(members, quotas[key])  # noqa: S311 - a reproducible split, not a secret
        for index in members:
            splits[index] = "test" if index in chosen else "dev"
    return [
        item.model_copy(update={"split": splits[index]}) if index in splits else item
        for index, item in enumerate(items)
    ]


@dataclass(frozen=True)
class ResolveReport:
    """What resolving filled in: counts only."""

    items: int
    derived_tags: int
    assigned_splits: int
    splits: dict[str, int]


def resolve_eval_set(
    eval_set: EvalSet,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = 0,
    resplit: bool = False,
    file_types: Mapping[str, str] | None = None,
) -> tuple[EvalSet, ResolveReport]:
    """Fill in askers, tags and splits, as the module docstring says. Raises ``SpecInvalid``
    for an asker name that isn't in the set's ``askers`` table."""
    table: dict[str, Asker] = {
        name: asker.model_copy(update={"name": name}) for name, asker in eval_set.askers.items()
    }
    issues: list[SpecIssue] = []

    def resolve(asker: Asker | str | None, path: str) -> Asker | None:
        if asker is None or isinstance(asker, Asker):
            return asker
        if asker not in table:
            issues.append(SpecIssue(path, "unknown_asker"))
            return None
        return table[asker]

    default = resolve(eval_set.defaults.asker, "defaults.asker")
    items = []
    derived = 0
    for index, item in enumerate(eval_set.items):
        asker = resolve(item.asker, f"items.{index}.asker")
        update: dict[str, Any] = {"asker": asker if asker is not None else default or Asker()}
        if not item.tags:
            update["tags"] = derived_tags(item)
            derived += 1
        items.append(item.model_copy(update=update))
    if issues:
        raise SpecInvalid(issues)
    unsplit = sum(1 for item in items if resplit or item.split is None)
    items = assign_splits(
        items, test_fraction=test_fraction, seed=seed, resplit=resplit, file_types=file_types
    )
    resolved = eval_set.model_copy(
        update={
            "defaults": eval_set.defaults.model_copy(update={"asker": default}),
            "askers": dict(sorted(table.items())),
            "items": items,
        }
    )
    counts = {split: sum(1 for item in items if item.split == split) for split in SPLITS}
    return resolved, ResolveReport(len(items), derived, unsplit, counts)
