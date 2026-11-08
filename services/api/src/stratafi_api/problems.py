"""Errors as RFC 9457 problem details (``application/problem+json``), never echoing input.

Every error body looks like::

    {"type": "urn:stratafi:problem:unauthorized", "title": "Unauthorized", "status": 401,
     "detail": "A valid identity token is required.", "code": "expired",
     "instance": "urn:stratafi:request:r-8a2e...", "request_id": "r-8a2e..."}

``type`` is a URN, so it identifies the problem without pointing at a site Stratafi doesn't own;
``code`` is a finer reason (an identity error, a field rule). FastAPI's default validation
response echoes the request's input, and exception messages can carry content, so the handlers
replace both: validation failures name only the failing fields, and unexpected exceptions become
a fixed 500. A field path part that came from the client (a dict key, an unexpected field name)
is masked as ``*``, so the path doesn't echo input either.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from stratafi_core.logs import current_log_context

logger = logging.getLogger("stratafi.api.problems")

PROBLEM_MEDIA_TYPE = "application/problem+json"
TYPE_PREFIX = "urn:stratafi:problem:"
INSTANCE_PREFIX = "urn:stratafi:request:"

# kind → (status, title, default detail)
KINDS: dict[str, tuple[int, str, str]] = {
    "invalid_request": (400, "Invalid request", "The request is invalid."),
    "unauthorized": (401, "Unauthorized", "A valid identity token is required."),
    "forbidden": (403, "Forbidden", "You don't have access to this."),
    "not_found": (404, "Not found", "Not found."),
    "method_not_allowed": (405, "Method not allowed", "This method isn't allowed here."),
    "conflict": (409, "Conflict", "The request conflicts with the current state."),
    "payload_too_large": (413, "Payload too large", "The request is too large."),
    "unsupported_media_type": (
        415,
        "Unsupported media type",
        "This content type isn't supported here.",
    ),
    "too_many_requests": (429, "Too many requests", "Too many requests. Try again later."),
    "internal": (500, "Internal error", "Something went wrong."),
    "unavailable": (503, "Service unavailable", "A service behind this is unavailable."),
}
_KIND_BY_STATUS = {status: kind for kind, (status, _, _) in KINDS.items()}
_FIELD_PART_RE = re.compile(r"[a-z_][a-z0-9_]{0,63}")
_MAX_FIELDS = 10


class ProblemIssue(BaseModel):
    """One problem inside a document the client sent: a dotted path and a code, never a value."""

    path: str
    code: str


class ProblemDetails(BaseModel):
    """An RFC 9457 problem. ``code``, ``errors`` and ``issues`` are Stratafi's extension
    members."""

    type: str = Field(examples=["urn:stratafi:problem:forbidden"])
    title: str
    status: int
    detail: str
    instance: str | None = None
    request_id: str | None = None
    code: str | None = None
    errors: list[str] | None = Field(default=None, description="Paths of the invalid fields")
    issues: list[ProblemIssue] | None = Field(
        default=None, description="Problems in a submitted document: paths and codes"
    )


class Problem(Exception):
    """Raise anywhere in the API to answer with a problem. ``detail`` must never include user
    content; the default is safe."""

    def __init__(
        self,
        kind: str,
        detail: str | None = None,
        *,
        code: str | None = None,
        errors: Sequence[str] | None = None,
        issues: Sequence[ProblemIssue] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if kind not in KINDS:
            raise ValueError("unknown problem kind")
        super().__init__(kind)
        self.kind = kind
        self.detail = detail
        self.code = code
        self.errors = list(errors) if errors is not None else None
        self.issues = list(issues) if issues is not None else None
        self.headers = dict(headers or {})


def problem_body(
    kind: str,
    detail: str | None = None,
    *,
    code: str | None = None,
    errors: Sequence[str] | None = None,
    issues: Sequence[ProblemIssue] | None = None,
    status: int | None = None,
) -> dict[str, Any]:
    default_status, title, default_detail = KINDS[kind]
    request_id = current_log_context().get("request_id")
    body = ProblemDetails(
        type=TYPE_PREFIX + kind,
        title=title,
        status=status or default_status,
        detail=detail or default_detail,
        instance=f"{INSTANCE_PREFIX}{request_id}" if request_id else None,
        request_id=request_id,
        code=code,
        errors=list(errors) if errors is not None else None,
        issues=list(issues) if issues is not None else None,
    )
    return body.model_dump(mode="json", exclude_none=True)


def problem_response(
    kind: str,
    detail: str | None = None,
    *,
    code: str | None = None,
    errors: Sequence[str] | None = None,
    issues: Sequence[ProblemIssue] | None = None,
    status: int | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = problem_body(kind, detail, code=code, errors=errors, issues=issues, status=status)
    return JSONResponse(
        body, status_code=body["status"], headers=dict(headers or {}), media_type=PROBLEM_MEDIA_TYPE
    )


def safe_field_names(errors: Sequence[Mapping[str, Any]]) -> list[str]:
    """Dotted paths of the failing fields, with anything the client controls masked as ``*``."""
    names: list[str] = []
    for error in errors:
        loc = list(error.get("loc", ()))
        if error.get("type") == "extra_forbidden" and loc:
            loc[-1] = "*"
        parts = [
            str(part)
            if isinstance(part, int) or (isinstance(part, str) and _FIELD_PART_RE.fullmatch(part))
            else "*"
            for part in loc
        ]
        name = ".".join(parts) or "*"
        if name not in names:
            names.append(name)
    return names[:_MAX_FIELDS]


async def _problem_handler(request: Request, exc: Problem) -> JSONResponse:
    return problem_response(
        exc.kind,
        exc.detail,
        code=exc.code,
        errors=exc.errors,
        issues=exc.issues,
        headers=exc.headers,
    )


async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    fields = safe_field_names(errors)
    logger.warning(
        "request validation failed",
        extra={
            "fields": ",".join(fields),
            "error_types": ",".join(sorted({str(e.get("type")) for e in errors}))[:200],
        },
    )
    if any(error.get("type") == "json_invalid" for error in errors):
        return problem_response(
            "invalid_request", "The request body isn't valid JSON.", code="json_invalid"
        )
    return problem_response(
        "invalid_request", "Some fields are invalid.", code="validation", errors=fields
    )


async def _http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # exc.detail can hold anything a library chose to put there, so it's never returned.
    kind = _KIND_BY_STATUS.get(exc.status_code)
    if kind is None:
        kind = "internal" if exc.status_code >= 500 else "invalid_request"
    return problem_response(kind, status=exc.status_code, headers=exc.headers)


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    # The JSON formatter records the exception's type and frames, never its message.
    logger.error("unhandled exception", exc_info=exc)
    return problem_response("internal")


def install_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(Problem, _problem_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_handler)
    app.add_exception_handler(Exception, _unhandled_handler)


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for routes that can answer with these problems."""
    return {
        status: {
            "model": ProblemDetails,
            "description": KINDS[_KIND_BY_STATUS[status]][1],
            "content": {
                PROBLEM_MEDIA_TYPE: {"schema": {"$ref": "#/components/schemas/ProblemDetails"}}
            },
        }
        for status in statuses
    }
