"""The SDK's errors, each with the exit code the ``stratafi`` command gives it.

Exit codes: ``0`` passed, ``1`` a check failed, ``2`` a usage or configuration error, ``3`` a
platform error (unreachable, out of quota). Messages carry the API's problem details, which
never echo what was sent, so they hold no document, chunk or query text.
"""

from typing import Any

PASSED = 0
CHECK_FAILED = 1
USAGE = 2
PLATFORM = 3


class StratafiError(Exception):
    """Something the SDK couldn't do. ``exit_code`` is the command line's for it."""

    exit_code = USAGE


class ConfigError(StratafiError):
    """The SDK isn't configured (no ``STRATAFI_URL``), or a reference, file or argument is
    wrong: a corpus, spec or eval set that doesn't exist, say."""


class UnreachableError(StratafiError):
    """The API didn't answer."""

    exit_code = PLATFORM


class WaitTimeout(StratafiError):
    """A wait ran out of time. The run goes on."""

    exit_code = PLATFORM


class ApiError(StratafiError):
    """The API answered with a problem (RFC 9457). Client errors (4xx) are usage errors, except
    429; 429 and server errors are platform errors."""

    def __init__(self, status: int, problem: dict[str, Any]) -> None:
        self.status = status
        self.problem = problem
        self.code: str | None = problem.get("code")
        self.title = str(problem.get("title") or "Error")
        self.detail = str(problem.get("detail") or "")
        self.errors: list[str] = list(problem.get("errors") or [])
        self.issues: list[dict[str, str]] = list(problem.get("issues") or [])
        self.request_id: str | None = problem.get("request_id")
        super().__init__(self._message())

    @property
    def exit_code(self) -> int:  # type: ignore[override]
        return PLATFORM if self.status == 429 or self.status >= 500 else USAGE

    def _message(self) -> str:
        message = f"{self.status} {self.title}"
        if self.detail and self.detail != self.title:
            message += f": {self.detail}"
        if self.code:
            message += f" [{self.code}]"
        if self.errors:
            message += f" (fields: {', '.join(self.errors)})"
        if self.issues:
            listed = ", ".join(f"{issue.get('path')}: {issue.get('code')}" for issue in self.issues)
            message += f" (issues: {listed})"
        if self.request_id:
            message += f" (request {self.request_id})"
        return message
