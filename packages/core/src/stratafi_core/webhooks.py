"""Signed webhook deliveries.

A webhook receives a JSON body and three headers:

- ``Stratafi-Event``: what happened, such as ``test_run.finished``
- ``Stratafi-Delivery``: the delivery's ID, the same on every retry, so a receiver can ignore one
  it has already handled
- ``Stratafi-Signature``: ``t=<unix seconds>,v1=<hex>``, where the hex is the HMAC-SHA256 of
  ``<t>.<body>`` keyed with the webhook's secret

A receiver checks the signature over the raw body, in constant time, and refuses a timestamp
more than ``TOLERANCE_SECONDS`` away from its clock, so a captured delivery can't be replayed
later. ``verify`` does both; the SDK has the same function (``stratafi.webhooks``), for receivers
that don't install Stratafi's packages.
"""

import base64
import hashlib
import hmac
import secrets
import time
import uuid

SIGNATURE_HEADER = "Stratafi-Signature"
EVENT_HEADER = "Stratafi-Event"
DELIVERY_HEADER = "Stratafi-Delivery"
SECRET_PREFIX = "whsec_"  # noqa: S105 - a prefix that marks a secret, not one
TOLERANCE_SECONDS = 300
EVENTS = ("test_run.finished",)


def new_secret() -> str:
    """A webhook's signing secret: shown once when it's created."""
    return SECRET_PREFIX + secrets.token_urlsafe(32)


def derive_secret(key: str, webhook_id: uuid.UUID) -> str:
    """A webhook's signing secret derived from the deployment's key: Postgres
    keeps none, and the worker derives it again for each delivery. Rotating the key rotates
    every webhook's secret."""
    digest = hmac.new(
        key.encode("utf-8"), b"stratafi-webhook-v1:" + webhook_id.bytes, hashlib.sha256
    ).digest()
    return SECRET_PREFIX + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def signature(secret: str, timestamp: int, body: bytes) -> str:
    message = str(timestamp).encode("ascii") + b"." + body
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def signature_header(secret: str, body: bytes, *, timestamp: int | None = None) -> str:
    """The ``Stratafi-Signature`` value for ``body``, signed now (or at ``timestamp``)."""
    moment = int(time.time()) if timestamp is None else timestamp
    return f"t={moment},v1={signature(secret, moment, body)}"


def verify(
    secret: str,
    body: bytes,
    header: str | None,
    *,
    tolerance: int = TOLERANCE_SECONDS,
    now: float | None = None,
) -> bool:
    """Whether ``header`` signs ``body`` with ``secret``, within ``tolerance`` seconds."""
    if not header:
        return False
    timestamp: int | None = None
    candidates: list[str] = []
    for part in header.split(","):
        key, _, value = part.strip().partition("=")
        if key == "t" and value.isdigit():
            timestamp = int(value)
        elif key == "v1" and value:
            candidates.append(value)
    if timestamp is None or not candidates:
        return False
    if abs((time.time() if now is None else now) - timestamp) > tolerance:
        return False
    expected = signature(secret, timestamp, body)
    return any(hmac.compare_digest(expected, candidate) for candidate in candidates)
