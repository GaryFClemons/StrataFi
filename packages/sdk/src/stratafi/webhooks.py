"""Checking a Stratafi webhook delivery in a receiver.

    from stratafi.webhooks import verify

    if not verify(secret, request.body, request.headers.get("Stratafi-Signature")):
        return 400

A delivery's ``Stratafi-Signature`` is ``t=<unix seconds>,v1=<hex>``: the HMAC-SHA256 of
``<t>.<raw body>`` keyed with the webhook's secret, which its creator got once. Check the raw
bytes as they arrived, before parsing them, and refuse old timestamps so a captured delivery
can't be replayed. ``Stratafi-Delivery`` is the same on every retry of one delivery: handle each
ID once. The same scheme as the server's ``stratafi_core.webhooks``; the SDK doesn't import it.
"""

import hashlib
import hmac
import time

SIGNATURE_HEADER = "Stratafi-Signature"
EVENT_HEADER = "Stratafi-Event"
DELIVERY_HEADER = "Stratafi-Delivery"
TOLERANCE_SECONDS = 300


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
    message = str(timestamp).encode("ascii") + b"." + body
    expected = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(expected, candidate) for candidate in candidates)
