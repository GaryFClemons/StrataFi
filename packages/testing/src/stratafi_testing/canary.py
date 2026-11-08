"""Canaries for the rule that document, chunk and query text never reaches a log.

A test plants a canary, a random marker no real text contains, wherever document, chunk or
query text would go, runs the code under test, and then checks that the marker turns up in none
of these:

- a captured log record: its rendered message, each attribute a caller attached with ``extra=``
  (nested values included, by their ``repr``), and the JSON line Stratafi's formatter writes
  for it
- anything the test wrote to stdout or stderr
- any text the test hands over itself: response bodies, events, error messages

Exception messages are left out on purpose. The JSON formatter writes an exception's type and
frames, never its message, because a message can carry content (a pydantic ``ValidationError``
repeats its input).
"""

import base64
import logging
import secrets
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from stratafi_core.logs import JsonFormatter

if TYPE_CHECKING:
    import pytest

CANARY_PREFIX = "STRATAFI-CANARY-"

# What every log record carries by itself; any other attribute was attached with ``extra=``.
STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

RecordFormatter = Callable[[logging.LogRecord], str]


def new_canary() -> str:
    """A fresh marker: the prefix and 16 random base32 characters, all capitals and digits, so
    it never looks like a word or a field name."""
    return CANARY_PREFIX + base64.b32encode(secrets.token_bytes(10)).decode("ascii")


def production_format(record: logging.LogRecord) -> str:
    """The line Stratafi's services write for ``record`` (``stratafi_core.logs.JsonFormatter``)."""
    return JsonFormatter("canary-check").format(record)


def record_extras(record: logging.LogRecord) -> dict[str, object]:
    """The attributes a caller attached to ``record`` with ``extra=``."""
    attached = record.__dict__.keys() - STANDARD_RECORD_ATTRS
    return {name: value for name, value in record.__dict__.items() if name in attached}


class CanaryLeak(AssertionError):
    """The canary showed up where text must not go. The message says where, never the
    canary."""


class CanaryCheck:
    """What a test logged and printed, searched for one canary."""

    def __init__(
        self,
        value: str,
        caplog: "pytest.LogCaptureFixture",
        capfd: "pytest.CaptureFixture[str]",
        *,
        format_record: RecordFormatter | None = None,
    ) -> None:
        self.value = value
        self._caplog = caplog
        self._capfd = capfd
        self._format = format_record or production_format
        # Output is read from capfd once and kept, since reading it empties the capture.
        self._output: list[str] = []

    def _record_views(self, record: logging.LogRecord) -> Iterator[str]:
        yield record.getMessage()
        for name, value in record_extras(record).items():
            yield f"{name}={value!r}"
        yield self._format(record)

    def captured_text(self) -> str:
        """Every captured record, as the code built it and as the formatter writes it, then
        everything written to stdout and stderr so far."""
        out, err = self._capfd.readouterr()
        self._output += [out, err]
        views = [view for record in self._caplog.records for view in self._record_views(record)]
        return "\n".join([*views, *self._output])

    def assert_absent(self, *texts: str | bytes) -> None:
        """Fail if the canary is in any of ``texts``; the failure names the text's position."""
        for position, text in enumerate(texts):
            searched = text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text
            if self.value in searched:
                raise CanaryLeak(f"the canary was found in checked text #{position}")

    def assert_not_logged(self) -> None:
        """Fail if the canary is in a captured log record or in the test's output."""
        if self.value in self.captured_text():
            raise CanaryLeak("the canary was found in logs or in the test's output streams")
