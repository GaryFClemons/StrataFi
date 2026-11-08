"""Results as data: lists of records, an eval run's metrics and a comparison, each as plain dicts
(``.to_dicts()``) or a pandas DataFrame (``.to_frame()``), and Markdown tables for reports.

pandas is optional (``pip install 'stratafi[frames]'``); only ``.to_frame()`` needs it.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, overload

from pydantic import BaseModel

from stratafi.errors import ConfigError
from stratafi.models import CompareOut

DEFAULT_RULE = "knowledge-hit-v1"
DEFAULT_CHECKPOINT = "final"


def plain(value: Any) -> Any:
    """A record as JSON-ready data: models dumped by their API names, wrappers and lists by
    ``to_dict`` and ``to_dicts``."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "to_dicts"):
        return value.to_dicts()
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    return value


def frame(rows: Sequence[Mapping[str, Any]], *, flatten: bool = False) -> Any:
    """``rows`` as a pandas DataFrame; nested dicts become dotted columns when ``flatten``."""
    try:
        import pandas
    except ImportError:
        raise ConfigError(".to_frame() needs pandas: pip install 'stratafi[frames]'") from None
    if flatten:
        return pandas.json_normalize(list(rows), sep=".")
    return pandas.DataFrame.from_records(list(rows))


class Records[T](Sequence[T]):
    """A list of API records (Pydantic models) that converts to dicts or a DataFrame."""

    def __init__(self, items: Iterable[T] = ()) -> None:
        self._items = list(items)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> "Records[T]": ...

    def __getitem__(self, index: int | slice) -> "T | Records[T]":
        if isinstance(index, slice):
            return Records(self._items[index])
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"Records({len(self._items)} items)"

    def to_dicts(self) -> list[dict[str, Any]]:
        return [plain(item) for item in self._items]

    def to_frame(self) -> Any:
        """One row per record; nested fields become dotted columns (``counts.tasks.succeeded``)."""
        return frame(self.to_dicts(), flatten=True)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


class Metrics:
    """An eval run's metrics: means with 95% bootstrap intervals by match rule, checkpoint and
    metric (``recall@5``), as the eval run record stores them (``summary``)."""

    def __init__(self, summary: Mapping[str, Any] | None) -> None:
        self.summary: dict[str, Any] = dict(summary or {})

    def __bool__(self) -> bool:
        return bool(self.summary.get("metrics"))

    def __repr__(self) -> str:
        return f"Metrics({len(self.to_dicts())} estimates)"

    @property
    def k_eval(self) -> int | None:
        return self.summary.get("k_eval")

    def estimate(
        self, name: str, *, rule: str = DEFAULT_RULE, checkpoint: str = DEFAULT_CHECKPOINT
    ) -> dict[str, Any]:
        """One metric's estimate: ``mean``, ``low``, ``high`` (the 95% interval) and ``n``.
        Raises ``KeyError`` for a metric the eval run didn't compute."""
        found = self.summary.get("metrics", {}).get(rule, {}).get(checkpoint, {})
        if name not in found:
            raise KeyError(f"no {name} for {rule} at {checkpoint}")
        return dict(found[name] or {"mean": None, "low": None, "high": None, "n": 0})

    def select(
        self, *names: str, rule: str = DEFAULT_RULE, checkpoint: str = DEFAULT_CHECKPOINT
    ) -> dict[str, float | None]:
        """The means of ``names`` (``"recall@5", "mrr@10"``), for one rule and checkpoint."""
        return {
            name: _number(self.estimate(name, rule=rule, checkpoint=checkpoint).get("mean"))
            for name in names
        }

    def headline(self, *, rule: str = DEFAULT_RULE) -> dict[str, float | None]:
        """The final list's hit and recall at the eval set's k, and MRR and nDCG at 10."""
        k = self.k_eval or 5
        names = [f"hit@{k}", f"recall@{k}", "mrr@10", "ndcg@10"]
        found = self.summary.get("metrics", {}).get(rule, {}).get(DEFAULT_CHECKPOINT, {})
        return self.select(*(name for name in names if name in found), rule=rule)

    def to_dicts(self) -> list[dict[str, Any]]:
        """One row per estimate: rule, checkpoint, metric, mean, low, high, n."""
        rows = []
        for rule, checkpoints in self.summary.get("metrics", {}).items():
            for checkpoint, metrics in checkpoints.items():
                for name, estimate in metrics.items():
                    estimate = estimate or {}
                    rows.append(
                        {
                            "rule": rule,
                            "checkpoint": checkpoint,
                            "metric": name,
                            "mean": _number(estimate.get("mean")),
                            "low": _number(estimate.get("low")),
                            "high": _number(estimate.get("high")),
                            "n": estimate.get("n"),
                        }
                    )
        return rows

    def to_frame(self) -> Any:
        return frame(self.to_dicts())

    def to_markdown(self, *, rule: str = DEFAULT_RULE, digits: int = 3) -> str:
        """The headline metrics with their intervals, as a Markdown table."""
        rows = []
        for name in self.headline(rule=rule):
            estimate = self.estimate(name, rule=rule)
            rows.append(
                [
                    name,
                    _format(estimate.get("mean"), digits),
                    _interval(estimate.get("low"), estimate.get("high"), digits),
                    _format(estimate.get("n"), digits),
                ]
            )
        return markdown_table(["metric", "mean", "95% interval", "queries"], rows)


class Comparison:
    """Eval run B against eval run A (``GET /eval-comparisons``): per metric, the means, B minus
    A with its 95% interval, the paired test's p-values, and wins, losses and ties."""

    def __init__(self, record: CompareOut) -> None:
        self.record = record

    def __repr__(self) -> str:
        return f"Comparison({self.record.a.id} -> {self.record.b.id}, {self.record.paired} paired)"

    def to_dict(self) -> dict[str, Any]:
        return plain(self.record)

    def to_dicts(self) -> list[dict[str, Any]]:
        """One row per metric."""
        return [plain(row) for row in self.record.metrics]

    def to_frame(self) -> Any:
        return frame(self.to_dicts())

    def queries(self) -> Records[Any]:
        """Per query: the first metric's values in A and B, B's worst losses first."""
        return Records(self.record.queries)

    def to_markdown(self, *, digits: int = 3) -> str:
        record = self.record
        lines = [
            f"Eval run B `{record.b.id}` against A `{record.a.id}`: {record.rule} at "
            f"{record.checkpoint}, {record.paired} queries paired.",
            "",
        ]
        rows = [
            [
                row.metric,
                _format(row.mean_a, digits),
                _format(row.mean_b, digits),
                _format(row.delta, digits, signed=True),
                _interval(row.low, row.high, digits),
                _format(row.p_value, digits),
                _format(row.p_holm, digits),
                "/".join(str(value) for value in (row.wins, row.losses, row.ties))
                if row.wins is not None
                else "-",
            ]
            for row in record.metrics
        ]
        lines.append(
            markdown_table(
                ["metric", "A", "B", "B - A", "95% interval", "p", "p (Holm)", "wins/losses/ties"],
                rows,
            )
        )
        for title, changes in (
            ("Retrieval config", record.retrieval_diff),
            ("Pipeline", record.pipeline_diff),
        ):
            if changes:
                lines += ["", f"{title} changes:", ""]
                lines += [f"- `{change.path}` {change.op}" for change in changes]
        if record.warnings:
            lines += ["", f"Warnings: {', '.join(record.warnings)}"]
        return "\n".join(lines)


def _format(value: Any, digits: int, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return str(value)


def _interval(low: Any, high: Any, digits: int) -> str:
    if low is None or high is None:
        return "-"
    return f"[{_format(low, digits)}, {_format(high, digits)}]"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


class Reports:
    """Report helpers: ``client.reports.table(rows)``."""

    def table(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        columns: Sequence[str] | None = None,
        digits: int = 3,
    ) -> str:
        """``rows`` as a Markdown table; floats get ``digits`` decimals. The columns are
        ``columns``, or every key in order of first appearance."""
        rows = list(rows)
        if columns is None:
            columns = list(dict.fromkeys(key for row in rows for key in row))
        return markdown_table(
            columns, ([_format(row.get(column), digits) for column in columns] for row in rows)
        )
