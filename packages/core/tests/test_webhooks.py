"""Webhook signatures: the server's signing and both verifiers agree, and a
changed body, another secret, or an old timestamp fails."""

import pytest

from stratafi import webhooks as sdk
from stratafi_core.webhooks import SECRET_PREFIX, new_secret, signature_header, verify

BODY = b'{"type":"test_run.finished","data":{"test_run_id":"t1"}}'


@pytest.mark.parametrize("check", [verify, sdk.verify])
def test_a_signed_body_verifies_and_anything_else_fails(check) -> None:
    secret = new_secret()
    header = signature_header(secret, BODY, timestamp=1_800_000_000)
    assert check(secret, BODY, header, now=1_800_000_010)
    assert not check(secret, BODY + b" ", header, now=1_800_000_010)
    assert not check(new_secret(), BODY, header, now=1_800_000_010)
    assert not check(secret, BODY, header, now=1_800_000_000 + 301)
    assert not check(secret, BODY, None)
    assert not check(secret, BODY, "t=abc,v1=00")
    rotated = f"{header},v1={'0' * 64}"
    assert check(secret, BODY, rotated, now=1_800_000_000)


def test_secrets_are_long_and_marked() -> None:
    secret = new_secret()
    assert secret.startswith(SECRET_PREFIX)
    assert len(secret) > 40
    assert secret != new_secret()


def test_derived_secrets_are_stable_per_webhook_and_key() -> None:
    import uuid

    from stratafi_core.webhooks import derive_secret

    first, second = uuid.uuid4(), uuid.uuid4()
    secret = derive_secret("key-1", first)
    assert secret.startswith(SECRET_PREFIX)
    assert secret == derive_secret("key-1", first)
    assert secret != derive_secret("key-1", second)
    assert secret != derive_secret("key-2", first)  # rotating the key rotates every secret
    body = b'{"status":"passed"}'
    assert verify(secret, body, signature_header(secret, body))
