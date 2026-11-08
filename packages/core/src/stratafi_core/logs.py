"""JSON logging that holds metadata only: the "no text fields" policy.

Logs carry IDs, hashes, codes and counts, never document, chunk or query text. Code is
responsible for that, and every package has canary tests for it. This formatter is the backstop:

- Each record becomes one JSON object on stderr.
- **Text-bearing fields are dropped.** An ``extra=`` field named like content (``text``,
  ``query``, ``chunk``, ``content``, ``title``...) is written as ``"[redacted]"``, whatever it
  holds.
- **Free text is redacted.** A string argument or ``extra=`` value that isn't token-shaped
  (letters, digits and ``._:/@+=,~-``, at most 200 characters) is written as
  ``"[text redacted]"``, so a sentence never reaches a log. A single word still passes, which is
  why the rule stays the code's and the canary tests check it.
- **Other values become their type name,** except numbers, booleans and None.
- **Exceptions are written as their type and frames,** never their message: messages can carry
  content (a pydantic ``ValidationError`` prints its input).
- Request, workspace, run and job IDs come from ``log_context``.

Ruff's ``G`` rules forbid f-strings and ``%``-formatting in logging calls, so a message is always
a fixed format string and only its arguments vary.
"""

import json
import logging
import os
import re
import sys
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, TextIO

REDACTED_FIELD = "[redacted]"
REDACTED_TEXT = "[text redacted]"
# Field names that hold content. Matched on the lowercased name, and on its parts split at "_".
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
_SAFE_STRING_RE = re.compile(r"[A-Za-z0-9._:/@+=,~-]{0,200}")
_CONTEXT_FIELDS = ("request_id", "workspace_id", "run_id", "job_id")
_context: ContextVar[Mapping[str, str]] = ContextVar(
    "stratafi_log_context", default=MappingProxyType({})
)

# Attributes every LogRecord has; anything else on a record came from ``extra=``. Uvicorn adds
# ``color_message``, an ANSI-coloured copy of the message, which is noise in JSON.
_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}
_MAX_FRAMES = 12
# Capped at WARNING: their DEBUG output carries request and response content.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "httpx2",
    "httpcore2",
    "hpack",
    "multipart",
    "asyncio",
    "botocore",
    "boto3",
    "s3transfer",
    "urllib3",
    "psycopg",
    "sqlalchemy",
    "alembic",
)


# Suffixes that make a name metadata about content, not content: ``document_id``,
# ``chunk_count``, ``query_hash``.
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


def is_text_field(name: str) -> bool:
    """Whether a field named ``name`` would hold content: ``text``, ``query_text``,
    ``chunk_preview``. ``document_id`` and ``chunk_count`` are metadata, so they pass."""
    lowered = name.lower()
    if lowered.endswith(METADATA_SUFFIXES):
        return False
    return lowered in TEXT_FIELD_NAMES or any(
        part in TEXT_FIELD_NAMES for part in lowered.split("_")
    )


def safe_value(value: Any) -> Any:
    """``value`` if the policy lets it into a log, else a marker or its type name."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if _SAFE_STRING_RE.fullmatch(value) else REDACTED_TEXT
    return type(value).__name__


def safe_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the policy to a mapping of fields: logs, audit details, event payloads."""
    return {
        key: REDACTED_FIELD if is_text_field(key) else safe_value(value)
        for key, value in fields.items()
    }


class TextFieldError(ValueError):
    """A record meant for metadata holds text. The message names the field, never the value."""


def check_metadata(fields: Mapping[str, Any], *, where: str = "fields") -> None:
    """Refuse ``fields`` unless they hold only metadata: numbers, booleans, None, token-shaped
    strings, and lists or mappings of them, under names that don't hold content. For audit
    details, artifact summaries and event payloads, which are stored, not just logged."""

    def check(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str) or is_text_field(key):
                    raise TextFieldError(f"{where}: {path}{key!s:.40} holds content")
                check(item, f"{path}{key}.")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                check(item, f"{path}{index}.")
        elif safe_value(value) is not value:
            raise TextFieldError(f"{where}: {path.rstrip('.') or '(root)'} isn't metadata")

    check(fields, "")


@contextmanager
def log_context(**ids: str | None) -> Iterator[None]:
    """Add IDs (``request_id``, ``workspace_id``, ``run_id``, ``job_id``) to every record logged
    inside the block."""
    unknown = set(ids) - set(_CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"unknown log context fields: {', '.join(sorted(unknown))}")
    current = dict(_context.get())
    current.update({key: value for key, value in ids.items() if value is not None})
    token = _context.set(current)
    try:
        yield
    finally:
        _context.reset(token)


def current_log_context() -> Mapping[str, str]:
    return _context.get()


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "service": self.service,
            "logger": record.name,
            "msg": self._message(record),
        }
        for key, value in _context.get().items():
            payload[key] = safe_value(value)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RECORD_ATTRS and not key.startswith("_")
        }
        payload.update(safe_fields(extras))
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, _, tb = record.exc_info
            payload["exc_type"] = f"{exc_type.__module__}.{exc_type.__qualname__}"
            payload["exc_frames"] = [
                f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"
                for frame in traceback.extract_tb(tb)[-_MAX_FRAMES:]
            ]
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _message(record: logging.LogRecord) -> str:
        msg = str(record.msg)
        args = record.args
        if not args:
            return msg
        if isinstance(args, Mapping):
            safe_args: Any = {key: safe_value(value) for key, value in args.items()}
        else:
            safe_args = tuple(safe_value(value) for value in args)
        try:
            return msg % safe_args
        except (TypeError, ValueError):
            return msg


_HANDLER_FLAG = "_stratafi_json_handler"


def configure_logging(
    service: str, *, level: str | int = "INFO", stream: TextIO | None = None
) -> logging.Handler:
    """Send every log record to ``stream`` (default stderr) as JSON. Safe to call twice."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter(service))
    setattr(handler, _HANDLER_FLAG, True)
    root.addHandler(handler)
    root.setLevel(level)
    # Uvicorn logs through the root handler; its access log would record raw paths and queries.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    quiet_noisy_loggers()
    return handler


def quiet_noisy_loggers() -> None:
    """Cap the HTTP, S3 and database clients' loggers at WARNING: below that they log request
    URLs, queries and bodies. ``configure_logging`` does this; so do the canary tests, so they
    check what production would write."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
