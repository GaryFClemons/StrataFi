"""Custom checks: a Python function's verdict on a run.

    # checks/tables.py
    from stratafi.checks import CheckContext, CheckResult

    def rate_tables_not_split(ctx: CheckContext) -> CheckResult:
        split = [
            c.chunk_id
            for c in ctx.chunks()
            if c.metadata.get("table_part") and "rate" in (c.heading_path[-1:] or [""])[0].lower()
        ]
        return CheckResult.assert_true(not split, value=len(split), detail={"chunks": split[:20]})

A suite names it as ``python: checks/tables.py:rate_tables_not_split``; ``stratafi test run``
sends the file with the suite. The lab runs it only in its sandbox, and only where the owner
turned custom checks on after the sandbox's security review: a container with no network, no
credentials, the run's data mounted read-only, a small writable ``scratch``, and limits on
time, memory, processes and output.

``CheckContext`` reads the run's data from a bundle directory: ``context.json`` (the suite, the
check and the run's IDs, and the eval run's aggregates), ``chunks.jsonl``, ``documents.jsonl``
and ``queries.jsonl``. ``CheckContext.from_records`` makes one from your own data, to test a
check before the lab runs it. A check returns a ``CheckResult``; what it says goes into the test
run's record and reports, so ``detail`` holds IDs, codes and numbers, never a document's text,
as the lab's logs and records do: a string with a space (or anything but ``A-Z a-z 0-9
._:/@+=,~-``) becomes null, a key such as ``text`` or ``query`` is dropped, and a detail over
4 KB becomes ``{"truncated": true}``. ``to_dict`` shows what the lab will keep.

This module imports nothing outside the standard library, so the sandbox carries only it.
"""

import json
import math
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONTEXT_FILE = "context.json"
CHUNKS_FILE = "chunks.jsonl"
DOCUMENTS_FILE = "documents.jsonl"
QUERIES_FILE = "queries.jsonl"
RESULT_MARKER = "STRATAFI-CHECK-RESULT "
MAX_DETAIL_BYTES = 4096
# The lab's rule for what may sit beside IDs in a record (stratafi_core.logs, kept in step by a
# test): token-shaped strings, and no field named for content.
SAFE_STRING_RE = re.compile(r"[A-Za-z0-9._:/@+=,~-]{0,200}")
KEY_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,63}")
CODE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
TEXT_FIELD_NAMES = frozenset(
    {
        "answer",
        "body",
        "chunk",
        "content",
        "context",
        "document",
        "excerpt",
        "message",
        "passage",
        "prompt",
        "query",
        "question",
        "snippet",
        "text",
        "title",
    }
)
METADATA_SUFFIXES = (
    "_id",
    "_ids",
    "_hash",
    "_sha256",
    "_key",
    "_count",
    "_chars",
    "_bytes",
    "_tokens",
    "_len",
    "_type",
    "_kind",
    "_ms",
    "_seconds",
)


@dataclass(frozen=True)
class ChunkView:
    """One chunk of the run: its text, where it sits, and what the chunker recorded."""

    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    page_start: int | None = None
    page_end: int | None = None
    index_text: str = ""
    context: str | None = None


@dataclass(frozen=True)
class DocumentView:
    """One corpus entry of the run: its name, detected type and where it stopped."""

    document_id: str
    name: str
    media_type: str | None = None
    state: str | None = None
    scan_outcome: str | None = None
    parse_status: str | None = None


@dataclass(frozen=True)
class QueryView:
    """One query of the run's eval run: its tags, gold file types and scores."""

    item_id: str
    has_evidence: bool
    tags: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def value(
        self, name: str, rule: str = "knowledge-hit-v1", checkpoint: str = "final"
    ) -> float | None:
        """A metric such as ``recall@5`` for this query, or None."""
        found = ((self.metrics.get(rule) or {}).get(checkpoint) or {}).get(name)
        return float(found) if isinstance(found, int | float) else None


def _read_lines(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _fields(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    known = cls.__dataclass_fields__  # type: ignore[attr-defined]
    return {key: value for key, value in data.items() if key in known}


class CheckContext:
    """The run a custom check judges, read from a bundle directory."""

    def __init__(self, root: str | Path, *, scratch: str | Path | None = None) -> None:
        self.root = Path(root)
        context = self.root / CONTEXT_FILE
        self._context: dict[str, Any] = (
            json.loads(context.read_text(encoding="utf-8")) if context.is_file() else {}
        )
        self.scratch = Path(scratch) if scratch is not None else Path(tempfile.mkdtemp())

    @classmethod
    def from_records(
        cls,
        directory: str | Path,
        *,
        chunks: Iterable[Mapping[str, Any]] = (),
        documents: Iterable[Mapping[str, Any]] = (),
        queries: Iterable[Mapping[str, Any]] = (),
        context: Mapping[str, Any] | None = None,
    ) -> "CheckContext":
        """A context over your own records, written as a bundle into ``directory``: to test a
        check on your machine before the lab runs it."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / CONTEXT_FILE).write_text(json.dumps(dict(context or {})), encoding="utf-8")
        for name, records in (
            (CHUNKS_FILE, chunks),
            (DOCUMENTS_FILE, documents),
            (QUERIES_FILE, queries),
        ):
            with (root / name).open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(dict(record)) + "\n")
        return cls(root)

    @property
    def check_id(self) -> str | None:
        return self._context.get("check_id")

    @property
    def suite(self) -> dict[str, Any]:
        """The suite's name and version."""
        return dict(self._context.get("suite") or {})

    @property
    def runs(self) -> dict[str, Any]:
        """The IDs of the pipeline run and the eval run the check judges."""
        return dict(self._context.get("runs") or {})

    @property
    def metrics(self) -> dict[str, Any]:
        """The eval run's aggregates: metrics with their intervals, latency, violations."""
        return dict(self._context.get("metrics") or {})

    def metric(
        self, name: str, rule: str = "knowledge-hit-v1", checkpoint: str = "final"
    ) -> float | None:
        """An aggregate metric's mean, such as ``recall@5``, or None."""
        found = ((self.metrics.get("metrics") or {}).get(rule) or {}).get(checkpoint) or {}
        estimate = found.get(name)
        if isinstance(estimate, Mapping) and isinstance(estimate.get("mean"), int | float):
            return float(estimate["mean"])
        return None

    def chunks(self) -> Iterator[ChunkView]:
        for record in _read_lines(self.root / CHUNKS_FILE):
            yield ChunkView(**_fields(ChunkView, record))

    def documents(self) -> Iterator[DocumentView]:
        for record in _read_lines(self.root / DOCUMENTS_FILE):
            yield DocumentView(**_fields(DocumentView, record))

    def queries(self) -> Iterator[QueryView]:
        for record in _read_lines(self.root / QUERIES_FILE):
            yield QueryView(**_fields(QueryView, record))


def is_text_field(name: str) -> bool:
    """Whether a key such as ``text`` or ``query_text`` names content; ``chunk_id`` doesn't."""
    lowered = name.lower()
    if lowered.endswith(METADATA_SUFFIXES):
        return False
    return lowered in TEXT_FIELD_NAMES or any(
        part in TEXT_FIELD_NAMES for part in lowered.split("_")
    )


def _plain(value: Any, depth: int = 0) -> Any:
    """``value`` as a record may keep it: numbers, booleans, token-shaped strings, and lists
    and mappings of at most 50 of them, 5 levels deep."""
    if depth > 4:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) < 2**53 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if SAFE_STRING_RE.fullmatch(value) else None
    if isinstance(value, Mapping):
        return {
            key: _plain(item, depth + 1)
            for key, item in list(value.items())[:50]
            if isinstance(key, str) and KEY_RE.fullmatch(key) and not is_text_field(key)
        }
    if isinstance(value, list | tuple):
        return [_plain(item, depth + 1) for item in list(value)[:50]]
    return None


@dataclass
class CheckResult:
    """A custom check's verdict: ``passed`` (None skips it), a value, a short code, and detail
    of IDs and numbers."""

    passed: bool | None
    value: float | None = None
    code: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def assert_true(
        cls,
        condition: bool,
        *,
        value: float | None = None,
        detail: Mapping[str, Any] | None = None,
        code: str = "assertion_failed",
    ) -> "CheckResult":
        return cls(bool(condition), value, None if condition else code, dict(detail or {}))

    @classmethod
    def passing(
        cls, *, value: float | None = None, detail: Mapping[str, Any] | None = None
    ) -> "CheckResult":
        return cls(True, value, None, dict(detail or {}))

    @classmethod
    def failing(
        cls,
        code: str = "failed",
        *,
        value: float | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> "CheckResult":
        return cls(False, value, code, dict(detail or {}))

    @classmethod
    def skip(cls, code: str = "skipped") -> "CheckResult":
        return cls(None, None, code, {})

    def to_dict(self) -> dict[str, Any]:
        """The verdict as the lab records it: plain JSON within its limits. A code is lower
        case, digits and ``_.-``."""
        detail = _plain(dict(self.detail)) if isinstance(self.detail, Mapping) else {}
        if len(json.dumps(detail)) > MAX_DETAIL_BYTES:
            detail = {"truncated": True}
        value = None
        if isinstance(self.value, int | float) and not isinstance(self.value, bool):
            value = float(self.value) if math.isfinite(float(self.value)) else None
        code = self.code if isinstance(self.code, str) and CODE_RE.fullmatch(self.code) else None
        if self.code and code is None:
            code = "check_code_invalid"
        passed = self.passed if self.passed is None else bool(self.passed)
        return {"passed": passed, "value": value, "code": code, "detail": detail}
