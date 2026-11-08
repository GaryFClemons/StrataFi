"""Cursor pagination: ``?limit=`` and an opaque ``?cursor=``, answered with ``Page``.

A cursor is the position after the last item of the previous page (a keyset, never an offset),
encoded as base64url JSON with a version. Clients pass it back unchanged. Cursors aren't secret
or signed: every query is scoped to the caller's workspace, so a forged cursor can only skip
around inside data the caller may already read. A malformed one is a 400 ``invalid_cursor``.
"""

import base64
import binascii
import json
from typing import Annotated, Any

from fastapi import Query
from pydantic import BaseModel, Field

from stratafi_api.problems import Problem

CURSOR_VERSION = 1
MAX_CURSOR_LENGTH = 512
DEFAULT_LIMIT = 50
MAX_LIMIT = 200


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None = Field(
        description="Pass as `cursor` to get the next page; null on the last page."
    )


def encode_cursor(position: list[Any]) -> str:
    payload = json.dumps({"v": CURSOR_VERSION, "k": position}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def decode_cursor(cursor: str, *, size: int) -> list[Any]:
    """The keyset in ``cursor``, which must hold ``size`` values."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (UnicodeError, binascii.Error, ValueError):
        raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor") from None
    if not isinstance(data, dict) or data.get("v") != CURSOR_VERSION:
        raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    position = data.get("k")
    if not isinstance(position, list) or len(position) != size:
        raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
    return position


class PageParams(BaseModel):
    limit: int
    cursor: str | None


def page_params(
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT, description="Items per page")] = DEFAULT_LIMIT,
    cursor: Annotated[
        str | None,
        Query(
            max_length=MAX_CURSOR_LENGTH,
            pattern=r"^[A-Za-z0-9_-]+$",
            description="`next_cursor` from the previous page",
        ),
    ] = None,
) -> PageParams:
    return PageParams(limit=limit, cursor=cursor)
