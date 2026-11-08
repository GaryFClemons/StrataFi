"""Identity tokens: minting, verification, key rotation, the worker's service tokens, and every
reason a token is refused."""

import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stratafi_core.identity import (
    ALGORITHM,
    AUDIENCE,
    ISSUER,
    Caller,
    IdentityConfigError,
    IdentityError,
    SigningServiceTokens,
    TokenSigner,
    TokenVerifier,
    generate_keypair,
    key_id,
    load_public_keys,
    public_key_pem,
    service_tokens,
)


@pytest.fixture
def signer() -> TokenSigner:
    return TokenSigner(Ed25519PrivateKey.generate())


@pytest.fixture
def verifier(signer: TokenSigner) -> TokenVerifier:
    return TokenVerifier.for_public_key(signer.public_key)


def claims(**overrides: object) -> dict[str, object]:
    now = int(time.time())
    base: dict[str, object] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "u-tenant0-owner",
        "tid": "tenant0",
        "roles": ["owner"],
        "groups": ["all-staff"],
        "rid": "r-0123456789abcdef",
        "iat": now,
        "exp": now + 60,
    }
    base.update(overrides)
    return {key: value for key, value in base.items() if value is not None}


def sign(signer: TokenSigner, payload: dict[str, object], **headers: object) -> str:
    return jwt.encode(
        payload,
        signer._key,
        algorithm=ALGORITHM,
        headers={"kid": signer.kid, **headers},
    )


def reason(verifier: TokenVerifier, token: str) -> str:
    with pytest.raises(IdentityError) as error:
        verifier.verify(token)
    return error.value.reason


def test_a_minted_token_verifies_with_every_claim(
    signer: TokenSigner, verifier: TokenVerifier
) -> None:
    token = signer.mint(
        sub="u-tenant0-editor", tenant="tenant0", roles=["editor"], groups=["lending"], ttl=120
    )
    identity = verifier.verify(token)
    assert (identity.sub, identity.tid) == ("u-tenant0-editor", "tenant0")
    assert (identity.roles, identity.groups) == (("editor",), ("lending",))
    assert identity.exp - identity.iat == 120
    assert identity.rid.startswith("r-")
    assert jwt.get_unverified_header(token)["kid"] == signer.kid


def test_expired_tokens_are_refused(signer: TokenSigner, verifier: TokenVerifier) -> None:
    old = int(time.time()) - 600
    assert reason(verifier, sign(signer, claims(iat=old, exp=old + 60))) == "expired"


def test_a_token_from_another_key_is_refused(verifier: TokenVerifier) -> None:
    stranger = TokenSigner(Ed25519PrivateKey.generate())
    token = stranger.mint(sub="u-tenant0-owner", tenant="tenant0")
    assert reason(verifier, token) == "unknown_key"


def test_a_forged_kid_does_not_help(signer: TokenSigner, verifier: TokenVerifier) -> None:
    stranger = TokenSigner(Ed25519PrivateKey.generate())
    forged = jwt.encode(claims(), stranger._key, algorithm=ALGORITHM, headers={"kid": signer.kid})
    assert reason(verifier, forged) == "bad_signature"


def test_a_missing_tenant_is_refused(signer: TokenSigner, verifier: TokenVerifier) -> None:
    assert reason(verifier, sign(signer, claims(tid=None))) == "missing_tenant"


def test_an_invalid_tenant_is_refused(signer: TokenSigner, verifier: TokenVerifier) -> None:
    assert reason(verifier, sign(signer, claims(tid="Not A Tenant"))) == "invalid_claims"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"aud": "someone-else"}, "wrong_audience"),
        ({"iss": "someone-else"}, "wrong_issuer"),
        ({"rid": None}, "invalid_claims"),
        ({"exp": int(time.time()) + 3600}, "invalid_claims"),
        ({"roles": ["service:kb-ingest"]}, "invalid_claims"),
        ({"extra": "claim"}, "invalid_claims"),
    ],
)
def test_bad_claims_are_refused(
    signer: TokenSigner,
    verifier: TokenVerifier,
    overrides: dict[str, object],
    expected: str,
) -> None:
    assert reason(verifier, sign(signer, claims(**overrides))) == expected


@pytest.mark.parametrize("token", ["", "not-a-jwt", "a.b.c", "x" * 9000])
def test_malformed_tokens_are_refused(verifier: TokenVerifier, token: str) -> None:
    assert reason(verifier, token) == "malformed"


def test_an_unsigned_token_is_refused(verifier: TokenVerifier) -> None:
    unsigned = jwt.encode(claims(), key=None, algorithm="none")
    assert reason(verifier, unsigned) == "bad_algorithm"


def test_minting_refuses_bad_claims(signer: TokenSigner) -> None:
    with pytest.raises(IdentityError, match="invalid_ttl"):
        signer.mint(sub="u-tenant0-owner", tenant="tenant0", ttl=301)
    with pytest.raises(IdentityError, match="invalid_claims"):
        signer.mint(sub="u-x", tenant="Bad Tenant")
    with pytest.raises(IdentityError, match="invalid_claims"):
        signer.mint(sub="svc:worker", tenant="tenant0")


def test_key_files_round_trip_and_hold_several_keys(tmp_path: Path) -> None:
    first = generate_keypair(tmp_path / "a.pem", tmp_path / "a.pub.pem")
    second_signer = TokenSigner(Ed25519PrivateKey.generate())
    bundle = tmp_path / "bundle.pub.pem"
    bundle.write_bytes(
        (tmp_path / "a.pub.pem").read_bytes() + public_key_pem(second_signer.public_key)
    )
    keys = load_public_keys(bundle)
    assert set(keys) == {first, key_id(second_signer.public_key)}
    verifier = TokenVerifier.from_file(bundle)
    signer = TokenSigner.from_file(tmp_path / "a.pem")
    assert verifier.verify(signer.mint(sub="u-1", tenant="tenant0")).sub == "u-1"
    assert verifier.verify(second_signer.mint(sub="u-2", tenant="tenant0")).sub == "u-2"
    with pytest.raises(FileExistsError):
        generate_keypair(tmp_path / "a.pem", tmp_path / "a.pub.pem")


def test_unusable_key_files_are_config_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pem"
    empty.write_text("nothing here")
    with pytest.raises(IdentityConfigError):
        load_public_keys(empty)
    with pytest.raises(IdentityConfigError):
        load_public_keys(tmp_path / "missing.pem")
    with pytest.raises(IdentityConfigError):
        TokenSigner.from_file(empty)


def _write_keys(path: Path, *keys: Ed25519PrivateKey) -> Path:
    path.write_bytes(b"".join(public_key_pem(key.public_key()) for key in keys))
    return path


def test_a_rotated_key_is_read_again_for_an_unknown_kid(tmp_path: Path) -> None:
    old, new = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    path = _write_keys(tmp_path / "keys.pub.pem", old)
    verifier = TokenVerifier.from_file(path, reload_seconds=0)
    token = TokenSigner(new).mint(sub="u-1", tenant="local")
    with pytest.raises(IdentityError) as refused:
        verifier.verify(token)
    assert refused.value.reason == "unknown_key"
    _write_keys(path, old, new)  # the new key is published beside the old one
    assert verifier.verify(token).sub == "u-1"


def test_unknown_kids_read_the_file_at_most_once_an_interval(tmp_path: Path) -> None:
    old, new = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    path = _write_keys(tmp_path / "keys.pub.pem", old)
    verifier = TokenVerifier.from_file(path)  # 30 seconds between reads
    token = TokenSigner(new).mint(sub="u-1", tenant="local")
    _write_keys(path, old, new)
    with pytest.raises(IdentityError):
        verifier.verify(token)  # read moments ago, at construction: not again yet


def test_a_broken_key_file_keeps_the_keys_in_hand(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    path = _write_keys(tmp_path / "keys.pub.pem", key)
    verifier = TokenVerifier.from_file(path, reload_seconds=0)
    path.write_text("not a key")
    with pytest.raises(IdentityError):
        verifier.verify(TokenSigner(Ed25519PrivateKey.generate()).mint(sub="u", tenant="t"))
    assert verifier.verify(TokenSigner(key).mint(sub="u", tenant="t")).sub == "u"


def test_a_recorded_token_verifies_at_its_recorded_time() -> None:
    signer = TokenSigner(Ed25519PrivateKey.generate(), clock=lambda: 1_700_000_000)
    token = signer.mint(sub="u-1", tenant="tenant0", ttl=300)
    verifier = TokenVerifier.for_public_key(signer.public_key)
    with pytest.raises(IdentityError) as late:
        verifier.verify(token)
    assert late.value.reason == "expired"
    assert verifier.verify(token, now=1_700_000_100).sub == "u-1"
    for now, reason in ((1_700_000_400, "expired"), (1_699_999_000, "not_yet_valid")):
        with pytest.raises(IdentityError) as refused:
            verifier.verify(token, now=now)
        assert refused.value.reason == reason


def test_an_identity_is_a_caller_with_its_claims() -> None:
    signer = TokenSigner(Ed25519PrivateKey.generate())
    token = signer.mint(sub="u-1", tenant="tenant0", roles=("editor",), groups=("g",))
    caller = TokenVerifier.for_public_key(signer.public_key).verify(token).caller()
    assert caller == Caller("u-1", "tenant0", ("editor",), ("g",), None)
    assert not caller.is_service


def test_the_worker_signs_its_service_tokens_with_the_development_key(tmp_path: Path) -> None:
    key = tmp_path / "dev.pem"
    generate_keypair(key, tmp_path / "dev.pub.pem")
    signing = service_tokens(signing_key=key, name="stratafi")
    assert isinstance(signing, SigningServiceTokens)
    minted = signing.token("tenant0")
    assert minted is not None
    claims = jwt.decode(minted, options={"verify_signature": False})
    assert (claims["sub"], claims["tid"], claims["roles"]) == (
        "svc:stratafi",
        "tenant0",
        ["service:stratafi"],
    )
    assert service_tokens(signing_key=None, name="stratafi") is None
