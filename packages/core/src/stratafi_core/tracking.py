"""Tracking: annotations, flattened runs and the search language.

**Annotations.** Pipeline runs, eval runs and experiments carry a title, tags, a Markdown note and
an archive mark. They are people's text: callers store and return them, and never log them.
``check_title``, ``check_note`` and ``merge_tags`` enforce the rules and raise
``AnnotationError`` with a code and the field, never the value.

**Flattening.** An eval run becomes a ``TrackedRun``: its attributes, its parameters (dotted
paths into its resolved pipeline spec's stages and its resolved retrieval config, which share no
top-level key, plus the models it recorded under ``models.``), its metrics, and its tags.
Metric keys have one canonical, shortest form:

- ``recall@5``: the ``knowledge-hit-v1`` rule at the ``final`` checkpoint
- ``fused.recall@5``: that rule at another checkpoint
- ``evidence-coverage-v1.final.recall@5``: another rule
- ``latency.total_ms.p95``, ``uplift.recall@5``, ``findability``, ``violations.avoid``,
  ``violations.must_not_return``, ``lost.retrieval``, ``queries``, ``with_evidence``,
  ``abstention``

**The search language** is MLflow's ``search_runs`` filter: clauses joined by ``and``, each
``<entity>.<key> <operator> <value>``. ``parse_filter`` and ``parse_order`` raise
``SearchError`` with a code and the character position, never the text. ``matches`` evaluates a
filter as SQL would: a clause on a key the run doesn't have is false, but for ``IS NULL``.
``sort_runs`` orders by keys, missing values last either way.
"""

import functools
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from stratafi_core.metrics import CHECKPOINTS
from stratafi_core.relevance import RULES

Scalar = str | int | float | bool | None

# --- annotations --------------------------------------------------------------------------

TITLE_MAX = 120
NOTE_MAX = 20_000
TAGS_MAX = 50
TAG_VALUE_MAX = 500
TAG_KEY_RE = re.compile(r"[A-Za-z0-9_.\-/:]{1,64}")
RESERVED_PREFIX = "stratafi."


class AnnotationError(ValueError):
    """A title, note or tag broke a rule. ``code`` and ``field`` never hold the value."""

    def __init__(self, code: str, field_name: str) -> None:
        super().__init__(code)
        self.code = code
        self.field = field_name


def check_title(value: str | None) -> str | None:
    """A title, trimmed; an empty one clears it."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) > TITLE_MAX:
        raise AnnotationError("title_too_long", "title")
    if any(ord(char) < 32 for char in text):
        raise AnnotationError("title_has_control_characters", "title")
    return text


def check_note(value: str | None) -> str | None:
    """A Markdown note; an empty one clears it."""
    if value is None or not value.strip():
        return None
    if len(value) > NOTE_MAX:
        raise AnnotationError("note_too_long", "note")
    return value


def check_tag_key(key: str) -> str:
    if not TAG_KEY_RE.fullmatch(key):
        raise AnnotationError("tag_key_invalid", "tags")
    if key.lower().startswith(RESERVED_PREFIX):
        raise AnnotationError("tag_key_reserved", "tags")
    return key


def merge_tags(
    current: Mapping[str, str],
    *,
    set_tags: Mapping[str, str] | None = None,
    delete: Iterable[str] = (),
) -> dict[str, str]:
    """``current`` without ``delete``'s keys, with ``set_tags`` added or replaced."""
    merged = {key: value for key, value in current.items() if key not in set(delete)}
    for key, value in (set_tags or {}).items():
        check_tag_key(key)
        if not isinstance(value, str):
            raise AnnotationError("tag_value_invalid", "tags")
        if len(value) > TAG_VALUE_MAX:
            raise AnnotationError("tag_value_too_long", "tags")
        merged[key] = value
    if len(merged) > TAGS_MAX:
        raise AnnotationError("too_many_tags", "tags")
    return dict(sorted(merged.items()))


# --- flattening ---------------------------------------------------------------------------

DEFAULT_RULE = "knowledge-hit-v1"
DEFAULT_CHECKPOINT = "final"
METRIC_NAME_RE = re.compile(r"(hit|recall|precision|mrr|ndcg)@(\d{1,4}|all)")
LATENCY_STATS = ("p50", "p95", "mean")
LOST_BUCKETS = ("found", "parse", "retrieval", "rerank", "filtered")


def flatten(value: Any, prefix: str = "") -> dict[str, Scalar]:
    """Dotted paths to every scalar in ``value``: mappings by key, lists by index."""
    flat: dict[str, Scalar] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            flat.update(flatten(item, f"{prefix}.{index}" if prefix else str(index)))
    elif value is None or isinstance(value, str | int | float | bool):
        if prefix:
            flat[prefix] = value
    else:
        flat[prefix] = str(value)
    return flat


RETRIEVAL_NAMES = ("schema", "name", "description")


def flatten_params(
    stages: Mapping[str, Any] | None,
    retrieval: Mapping[str, Any] | None,
    models: Mapping[str, Any] | None = None,
) -> dict[str, Scalar]:
    """An eval run's parameters: the pipeline's stages and the retrieval config (without their
    names), and the models it recorded under ``models.``."""
    params = flatten(dict(stages or {}))
    params.update(
        flatten({key: v for key, v in (retrieval or {}).items() if key not in RETRIEVAL_NAMES})
    )
    if models:
        params.update(flatten(dict(models), "models"))
    return dict(sorted(params.items()))


def metric_key(key: str) -> str:
    """The canonical, shortest form of a metric key (the module docstring lists them)."""
    parts = key.split(".")
    if len(parts) == 3 and parts[0] in RULES and parts[1] in CHECKPOINTS:
        rule, checkpoint, name = parts
    elif len(parts) == 2 and parts[0] in CHECKPOINTS:
        rule, checkpoint, name = DEFAULT_RULE, parts[0], parts[1]
    elif len(parts) == 1 and METRIC_NAME_RE.fullmatch(parts[0]):
        rule, checkpoint, name = DEFAULT_RULE, DEFAULT_CHECKPOINT, parts[0]
    elif parts[0] == "uplift" and len(parts) in (2, 3):
        rule = parts[1] if len(parts) == 3 else DEFAULT_RULE
        return f"uplift.{parts[-1]}" if rule == DEFAULT_RULE else f"uplift.{rule}.{parts[-1]}"
    else:
        return key
    if rule == DEFAULT_RULE:
        return name if checkpoint == DEFAULT_CHECKPOINT else f"{checkpoint}.{name}"
    return f"{rule}.{checkpoint}.{name}"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _mean(estimate: Any) -> float | None:
    return _number(estimate.get("mean")) if isinstance(estimate, Mapping) else None


def flatten_metrics(summary: Mapping[str, Any] | None) -> dict[str, float]:
    """An eval run's summary (``stratafi_core.metrics.summarize``) as canonical metric keys."""
    if not summary:
        return {}
    flat: dict[str, float] = {}
    for rule, checkpoints in (summary.get("metrics") or {}).items():
        for checkpoint, names in (checkpoints or {}).items():
            for name, estimate in (names or {}).items():
                if (value := _mean(estimate)) is not None:
                    flat[metric_key(f"{rule}.{checkpoint}.{name}")] = value
    for rule, names in (summary.get("uplift") or {}).items():
        for name, estimate in (names or {}).items():
            if (value := _mean(estimate)) is not None:
                flat[metric_key(f"uplift.{rule}.{name}")] = value
    for step, stats in (summary.get("latency_ms") or {}).items():
        for stat in LATENCY_STATS:
            if (value := _number((stats or {}).get(stat))) is not None:
                flat[f"latency.{step}.{stat}"] = value
    findability = (summary.get("findability") or {}).get("rate")
    if (value := _number(findability)) is not None:
        flat["findability"] = value
    for kind, counts in (summary.get("violations") or {}).items():
        if (value := _number((counts or {}).get("count"))) is not None:
            flat[f"violations.{kind}"] = value
    for bucket in LOST_BUCKETS:
        if (value := _number((summary.get("where_lost") or {}).get(bucket))) is not None:
            flat[f"lost.{bucket}"] = value
    for name in ("queries", "with_evidence", "abstention"):
        if (value := _number(summary.get(name))) is not None:
            flat[name] = value
    return dict(sorted(flat.items()))


def headline_names(k_eval: int) -> list[str]:
    """The headline metrics' keys: hit and recall at the eval set's k, MRR and nDCG at 10."""
    return [f"hit@{k_eval}", f"recall@{k_eval}", "mrr@10", "ndcg@10"]


Entity = Literal["metrics", "params", "tags", "attributes"]
ENTITIES: dict[str, Entity] = {
    "metrics": "metrics",
    "metric": "metrics",
    "params": "params",
    "param": "params",
    "parameters": "params",
    "tags": "tags",
    "tag": "tags",
    "attributes": "attributes",
    "attribute": "attributes",
    "attr": "attributes",
}


@dataclass(frozen=True)
class TrackedRun:
    """One eval run, flattened for searching and comparing."""

    id: str
    attributes: dict[str, Scalar] = field(default_factory=dict)
    params: dict[str, Scalar] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def lookup(self, entity: Entity, key: str) -> tuple[bool, Scalar]:
        """Whether the run has ``key``, and its value."""
        if entity == "metrics":
            key = metric_key(key)
        values: Mapping[str, Any] = getattr(self, entity)
        if key in values:
            return True, values[key]
        return False, None

    def value(self, column: str) -> Scalar:
        """A column such as ``metrics.recall@5``: its value, or None."""
        entity, key = split_column(column)
        return self.lookup(entity, key)[1]


def split_column(column: str) -> tuple[Entity, str]:
    """``metrics.recall@5`` → ("metrics", "recall@5"); a column with no entity is refused."""
    head, _, key = column.partition(".")
    entity = ENTITIES.get(head.lower())
    if entity is None or not key:
        raise SearchError("unknown_entity", 0)
    return entity, key.strip("`")


# --- the search language --------------------------------------------------------------------

MAX_FILTER = 4000
MAX_CLAUSES = 50
MAX_LIST = 100
MAX_ORDER = 5


class SearchError(ValueError):
    """A filter or ordering that doesn't parse: a code and the character position it failed at,
    never the text."""

    def __init__(self, code: str, position: int) -> None:
        super().__init__(f"{code} at {position}")
        self.code = code
        self.position = position


Op = Literal["=", "!=", "<", "<=", ">", ">=", "like", "ilike", "not like", "not ilike", "in",
             "not in", "is null", "is not null"]  # fmt: skip


@dataclass(frozen=True)
class Clause:
    entity: Entity
    key: str
    op: Op
    value: Any = None  # a scalar, or a tuple for in and not in


@dataclass(frozen=True)
class Order:
    entity: Entity
    key: str
    descending: bool = False


@dataclass(frozen=True)
class _Token:
    kind: str  # key, string, number, op, word, lparen, rparen, comma, backtick
    text: str
    value: Any
    position: int


_KEY_CHARS = re.compile(r"[A-Za-z_][A-Za-z0-9_.@/:\-\[\]]*")
_NUMBER = re.compile(r"-?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
_WORDS = {"and", "or", "like", "ilike", "in", "not", "is", "null", "true", "false", "asc", "desc"}


def _quoted(text: str, start: int, quote: str) -> tuple[str, int]:
    """The string from ``start`` (after its opening quote) to its closing quote, unescaped."""
    out: list[str] = []
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            out.append(text[index + 1])
            index += 2
            continue
        if char == quote:
            return "".join(out), index + 1
        out.append(char)
        index += 1
    raise SearchError("unterminated_string", start - 1)


def _tokens(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "'\"":
            value, end = _quoted(text, index + 1, char)
            tokens.append(_Token("string", text[index:end], value, index))
            index = end
            continue
        if char == "`":
            value, end = _quoted(text, index + 1, "`")
            tokens.append(_Token("backtick", text[index:end], value, index))
            index = end
            continue
        if char in "(),":
            kind = {"(": "lparen", ")": "rparen", ",": "comma"}[char]
            tokens.append(_Token(kind, char, char, index))
            index += 1
            continue
        if char in "=!<>":
            for op in ("!=", "<>", "<=", ">=", "==", "=", "<", ">"):
                if text.startswith(op, index):
                    normal = {"<>": "!=", "==": "="}.get(op, op)
                    tokens.append(_Token("op", op, normal, index))
                    index += len(op)
                    break
            else:
                raise SearchError("unexpected_character", index)
            continue
        number = _NUMBER.match(text, index)
        if number and (char.isdigit() or char in "-."):
            literal = number.group()
            tokens.append(_Token("number", literal, float(literal), index))
            index = number.end()
            continue
        key = _KEY_CHARS.match(text, index)
        if key:
            word = key.group()
            kind = "word" if word.lower() in _WORDS else "key"
            tokens.append(_Token(kind, word, word.lower() if kind == "word" else word, index))
            index = key.end()
            continue
        raise SearchError("unexpected_character", index)
    return tokens


class _Reader:
    def __init__(self, tokens: list[_Token], length: int) -> None:
        self.tokens = tokens
        self.index = 0
        self.length = length

    def peek(self) -> _Token | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def next(self) -> _Token | None:
        token = self.peek()
        if token is not None:
            self.index += 1
        return token

    def position(self) -> int:
        token = self.peek()
        return token.position if token is not None else self.length

    def word(self, *words: str) -> bool:
        token = self.peek()
        if token is not None and token.kind == "word" and token.value in words:
            self.index += 1
            return True
        return False

    def key(self) -> tuple[Entity, str]:
        token = self.next()
        if token is None or token.kind != "key":
            raise SearchError("expected_key", token.position if token else self.length)
        head, dot, rest = token.text.partition(".")
        entity = ENTITIES.get(head.lower())
        if entity is None or not dot:
            raise SearchError("unknown_entity", token.position)
        if not rest:
            quoted = self.next()
            if quoted is None or quoted.kind != "backtick" or not quoted.value:
                raise SearchError("expected_key", quoted.position if quoted else self.length)
            return entity, quoted.value
        return entity, rest

    def value(self) -> Scalar:
        token = self.next()
        if token is None:
            raise SearchError("expected_value", self.length)
        if token.kind == "string":
            return token.value
        if token.kind == "number":
            number = float(token.value)
            return int(number) if number.is_integer() and "." not in token.text else number
        if token.kind == "word" and token.value in ("true", "false"):
            return token.value == "true"
        raise SearchError("expected_value", token.position)


def parse_filter(text: str | None) -> list[Clause]:
    """A filter's clauses; an empty filter matches every run."""
    text = text or ""
    if len(text) > MAX_FILTER:
        raise SearchError("filter_too_long", MAX_FILTER)
    reader = _Reader(_tokens(text), len(text))
    clauses: list[Clause] = []
    while reader.peek() is not None:
        if clauses:
            if reader.word("or"):
                raise SearchError("or_unsupported", reader.tokens[reader.index - 1].position)
            if not reader.word("and"):
                raise SearchError("expected_and", reader.position())
        entity, key = reader.key()
        clauses.append(_comparison(reader, entity, key))
        if len(clauses) > MAX_CLAUSES:
            raise SearchError("too_many_clauses", reader.position())
    return clauses


def _comparison(reader: _Reader, entity: Entity, key: str) -> Clause:
    token = reader.peek()
    if token is None:
        raise SearchError("expected_operator", reader.length)
    if token.kind == "op":
        reader.next()
        return Clause(entity, key, token.value, reader.value())
    negated = reader.word("not")
    if reader.word("like", "ilike"):
        kind = reader.tokens[reader.index - 1].value
        pattern = reader.next()
        if pattern is None or pattern.kind != "string":
            raise SearchError("expected_string", pattern.position if pattern else reader.length)
        op = f"not {kind}" if negated else kind
        return Clause(entity, key, op, pattern.value)  # type: ignore[arg-type]
    if reader.word("in"):
        opening = reader.next()
        if opening is None or opening.kind != "lparen":
            raise SearchError("expected_list", opening.position if opening else reader.length)
        values: list[Scalar] = []
        while True:
            values.append(reader.value())
            if len(values) > MAX_LIST:
                raise SearchError("list_too_long", reader.position())
            separator = reader.next()
            if separator is not None and separator.kind == "rparen":
                break
            if separator is None or separator.kind != "comma":
                raise SearchError(
                    "expected_list", separator.position if separator else reader.length
                )
        return Clause(entity, key, "not in" if negated else "in", tuple(values))
    if negated:
        raise SearchError("expected_operator", reader.position())
    if reader.word("is"):
        negated = reader.word("not")
        if not reader.word("null"):
            raise SearchError("expected_null", reader.position())
        return Clause(entity, key, "is not null" if negated else "is null")
    raise SearchError("expected_operator", token.position)


def parse_order(items: Sequence[str] | None) -> list[Order]:
    """``order_by`` items: a key, then ``ASC`` (the default) or ``DESC``."""
    orders: list[Order] = []
    for item in items or ():
        if len(item) > 300:
            raise SearchError("order_too_long", 300)
        reader = _Reader(_tokens(item), len(item))
        entity, key = reader.key()
        descending = False
        if reader.word("desc"):
            descending = True
        else:
            reader.word("asc")
        if reader.peek() is not None:
            raise SearchError("expected_direction", reader.position())
        orders.append(Order(entity, key, descending))
    if len(orders) > MAX_ORDER:
        raise SearchError("too_many_orders", 0)
    return orders


# --- evaluating ---------------------------------------------------------------------------


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _as_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return _as_text(left).lower() == _as_text(right).lower()
    first, second = _as_number(left), _as_number(right)
    if first is not None and second is not None:
        return first == second
    return _as_text(left) == _as_text(right)


def _order(left: Any, right: Any) -> int | None:
    """-1, 0 or 1 comparing ``left`` with ``right``; None if they don't compare."""
    first, second = _as_number(left), _as_number(right)
    if first is not None and second is not None:
        return (first > second) - (first < second)
    if isinstance(left, str) and isinstance(right, str):
        return (left > right) - (left < right)
    return None


@functools.lru_cache(maxsize=256)
def _like(pattern: str, insensitive: bool) -> re.Pattern[str]:
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            parts.append(re.escape(pattern[index + 1]))
            index += 2
            continue
        parts.append(".*" if char == "%" else "." if char == "_" else re.escape(char))
        index += 1
    return re.compile("".join(parts), re.DOTALL | (re.IGNORECASE if insensitive else 0))


def _holds(clause: Clause, run: TrackedRun) -> bool:
    found, value = run.lookup(clause.entity, clause.key)
    if clause.op == "is null":
        return not found or value is None
    if clause.op == "is not null":
        return found and value is not None
    if not found or value is None:
        return False
    match clause.op:
        case "=":
            return _equal(value, clause.value)
        case "!=":
            return not _equal(value, clause.value)
        case "<" | "<=" | ">" | ">=":
            compared = _order(value, clause.value)
            if compared is None:
                return False
            return {"<": compared < 0, "<=": compared <= 0, ">": compared > 0,
                    ">=": compared >= 0}[clause.op]  # fmt: skip
        case "like" | "ilike" | "not like" | "not ilike":
            pattern = _like(str(clause.value), clause.op.endswith("ilike"))
            hit = pattern.fullmatch(_as_text(value)) is not None
            return not hit if clause.op.startswith("not") else hit
        case "in":
            return any(_equal(value, option) for option in clause.value)
        case "not in":
            return not any(_equal(value, option) for option in clause.value)
    return False  # pragma: no cover - every operator is handled


def matches(run: TrackedRun, clauses: Sequence[Clause]) -> bool:
    return all(_holds(clause, run) for clause in clauses)


def _compare_values(left: Any, right: Any) -> int:
    """A total order for sorting: numbers, then text; within each, ascending."""
    first, second = _as_number(left), _as_number(right)
    if first is not None and second is not None:
        return (first > second) - (first < second)
    if (first is None) != (second is None):
        return -1 if first is not None else 1
    a, b = _as_text(left), _as_text(right)
    return (a > b) - (a < b)


def sort_runs(runs: Sequence[TrackedRun], orders: Sequence[Order]) -> list[TrackedRun]:
    """``runs`` by ``orders``, missing values last whichever way; ties by creation, newest
    first, then ID."""

    def compare(a: TrackedRun, b: TrackedRun) -> int:
        for order in orders:
            found_a, value_a = a.lookup(order.entity, order.key)
            found_b, value_b = b.lookup(order.entity, order.key)
            missing_a, missing_b = not found_a or value_a is None, not found_b or value_b is None
            if missing_a or missing_b:
                if missing_a and missing_b:
                    continue
                return 1 if missing_a else -1
            result = _compare_values(value_a, value_b)
            if result:
                return -result if order.descending else result
        newer = _compare_values(
            b.attributes.get("created_at") or "", a.attributes.get("created_at") or ""
        )
        return newer or ((a.id > b.id) - (a.id < b.id))

    return sorted(runs, key=functools.cmp_to_key(compare))


def search(
    runs: Sequence[TrackedRun], *, filter: str | None = None, order_by: Sequence[str] = ()
) -> list[TrackedRun]:
    """The runs a filter matches, in order."""
    clauses = parse_filter(filter)
    orders = parse_order(order_by)
    return sort_runs([run for run in runs if matches(run, clauses)], orders)


def available_keys(runs: Iterable[TrackedRun]) -> dict[str, list[str]]:
    """Every metric, parameter and tag key the runs have, sorted, for pickers."""
    metrics: set[str] = set()
    params: set[str] = set()
    tags: set[str] = set()
    for run in runs:
        metrics.update(run.metrics)
        params.update(run.params)
        tags.update(run.tags)
    return {"metrics": sorted(metrics), "params": sorted(params), "tags": sorted(tags)}


def differing_params(runs: Sequence[TrackedRun]) -> list[str]:
    """The parameter keys whose values aren't the same in every run (missing counts as a
    value)."""
    keys = sorted({key for run in runs for key in run.params})
    return [
        key
        for key in keys
        if len({_as_text(run.params[key]) if key in run.params else "\0" for run in runs}) > 1
    ]
