"""Identifier rules shared by every contract.

A workspace maps one-to-one to a tenant, named by a short lowercase tenant ID. Request IDs
travel in headers and logs. Validators use ``re.fullmatch`` so a trailing newline never slips
past, and never echo the value they reject.
"""

import re
import secrets
from typing import Annotated

from pydantic import AfterValidator

TENANT_ID_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
SERVICE_NAME_RE = re.compile(r"[a-z][a-z0-9-]{0,62}")


def _checker(pattern: re.Pattern[str], what: str):
    def check(value: str) -> str:
        if not isinstance(value, str) or not pattern.fullmatch(value):
            # Never echo the value: identifiers can arrive from untrusted input.
            raise ValueError(f"invalid {what}")
        return value

    return check


validate_tenant_id = _checker(TENANT_ID_RE, "tenant ID")
validate_request_id = _checker(REQUEST_ID_RE, "request ID")
validate_service_name = _checker(SERVICE_NAME_RE, "service name")

TenantId = Annotated[str, AfterValidator(validate_tenant_id)]
RequestId = Annotated[str, AfterValidator(validate_request_id)]


def is_valid_request_id(value: object) -> bool:
    return isinstance(value, str) and REQUEST_ID_RE.fullmatch(value) is not None


def new_request_id() -> str:
    """A fresh request ID: ``r-`` plus 16 hex characters."""
    return f"r-{secrets.token_hex(8)}"
