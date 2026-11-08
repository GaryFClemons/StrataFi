"""Identity tokens.

A person's calls to the API carry a short-lived EdDSA JWT in ``X-Stratafi-Identity`` that names
who is calling (scripts and CI may send an API token instead). Stratafi runs on one machine, so
the tokens are signed with a local development key that ``scripts/dev_setup.py`` writes, and the
API's development sign-in mints them for the standalone UI and the SDK. The rest of Stratafi sees
a token's claims as a ``Caller``.

- Claims: ``iss``, ``aud``, ``sub``, ``tid`` (the tenant, which names the workspace), ``roles``,
  ``groups``, ``rid`` (request ID), ``iat`` and ``exp``; optionally ``agent`` and ``sid``.
  Lifetimes are at most five minutes.
- Tokens name their signing key in the ``kid`` header: the first 16 characters of the base64url
  SHA-256 of the raw public key. A public key file may hold several PEM blocks, so keys can
  rotate with an overlap.
- The worker's own service token comes from ``ServiceTokens``: minted with the development key
  for each call to the model gateway.
"""

import base64
import contextlib
import hashlib
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from stratafi_core.files import write_private_file
from stratafi_core.ids import RequestId, TenantId, new_request_id, validate_service_name

ISSUER = "stratafi-dev"
AUDIENCE = "stratafi"
ALGORITHM = "EdDSA"
HEADER = "X-Stratafi-Identity"
MAX_TTL_SECONDS = 300
MAX_TOKEN_LENGTH = 8192
SERVICE_SUBJECT_PREFIX = "svc:"
SERVICE_ROLE_PREFIX = "service:"
REQUIRED_CLAIMS = ("iss", "aud", "sub", "tid", "rid", "iat", "exp")

ClaimItem = Annotated[str, StringConstraints(min_length=1, max_length=128)]


class IdentityError(Exception):
    """A token can't be minted or verified. ``reason`` is a short code, safe to log and return."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class IdentityConfigError(Exception):
    """Signing or verification keys are missing or unusable. Services should refuse to start."""


class Identity(BaseModel):
    """The verified claims of an identity token.

    Service tokens (``sub`` = ``svc:<name>``) carry exactly the role ``service:<name>``, no
    groups and no agent, so they never hold user rights. User tokens can't carry service roles.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    iss: Literal["stratafi-dev"]
    aud: Literal["stratafi"]
    sub: str = Field(min_length=1, max_length=128)
    tid: TenantId
    roles: tuple[ClaimItem, ...] = Field(default=(), max_length=64)
    groups: tuple[ClaimItem, ...] = Field(default=(), max_length=256)
    agent: str | None = Field(default=None, min_length=1, max_length=63)
    sid: str | None = Field(default=None, min_length=1, max_length=128)
    rid: RequestId
    iat: int
    exp: int

    @model_validator(mode="after")
    def _check_shape(self) -> "Identity":
        if self.exp <= self.iat:
            raise ValueError("exp must be after iat")
        if self.exp - self.iat > MAX_TTL_SECONDS:
            raise ValueError("token lifetime is over 5 minutes")
        if self.sub.startswith(SERVICE_SUBJECT_PREFIX):
            name = validate_service_name(self.sub.removeprefix(SERVICE_SUBJECT_PREFIX))
            if self.roles != (SERVICE_ROLE_PREFIX + name,) or self.groups or self.agent is not None:
                raise ValueError("a service token carries only its service role")
        elif any(role.startswith(SERVICE_ROLE_PREFIX) for role in self.roles):
            raise ValueError("a user token can't carry a service role")
        return self

    @property
    def is_service(self) -> bool:
        return self.sub.startswith(SERVICE_SUBJECT_PREFIX)

    def caller(self) -> "Caller":
        return Caller(
            sub=self.sub,
            tid=self.tid,
            roles=self.roles,
            groups=self.groups,
            agent=self.agent,
        )


@dataclass(frozen=True)
class Caller:
    """Who a verified identity token names: its subject, tenant, roles, groups and agent."""

    sub: str
    tid: str
    roles: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    agent: str | None = None

    @property
    def is_service(self) -> bool:
        return self.sub.startswith(SERVICE_SUBJECT_PREFIX)


# --- keys ---------------------------------------------------------------------------------


def key_id(public_key: Ed25519PublicKey) -> str:
    """The ``kid`` for a public key: the first 16 characters of its base64url SHA-256."""
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")[:16]


def public_key_pem(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def generate_keypair(private_path: Path, public_path: Path, *, overwrite: bool = False) -> str:
    """Write a new Ed25519 signing key (mode 0600) and its public key. Returns the ``kid``."""
    if not overwrite and private_path.exists():
        raise FileExistsError(private_path)
    key = Ed25519PrivateKey.generate()
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    write_private_file(private_path, private_pem)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(public_key_pem(key.public_key()))
    return key_id(key.public_key())


def load_private_key(path: Path | str) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise IdentityConfigError("can't read the identity signing key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise IdentityConfigError("the identity signing key isn't Ed25519")
    return key


_PUBLIC_PEM_RE = re.compile(rb"-----BEGIN PUBLIC KEY-----.+?-----END PUBLIC KEY-----", re.S)


def parse_public_keys(data: bytes) -> dict[str, Ed25519PublicKey]:
    """Every Ed25519 public key in PEM ``data``, keyed by ``kid``."""
    keys: dict[str, Ed25519PublicKey] = {}
    for block in _PUBLIC_PEM_RE.findall(data):
        try:
            key = serialization.load_pem_public_key(block)
        except ValueError as exc:
            raise IdentityConfigError("an identity public key can't be parsed") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise IdentityConfigError("an identity public key isn't Ed25519")
        keys[key_id(key)] = key
    if not keys:
        raise IdentityConfigError("no identity public keys found")
    return keys


def load_public_keys(path: Path | str) -> dict[str, Ed25519PublicKey]:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise IdentityConfigError("can't read the identity public keys") from exc
    return parse_public_keys(data)


# --- minting ------------------------------------------------------------------------------


class TokenSigner:
    """Mints identity tokens: the development sign-in, dev tooling and tests."""

    def __init__(
        self, private_key: Ed25519PrivateKey, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._key = private_key
        self._clock = clock
        self.kid = key_id(private_key.public_key())

    @classmethod
    def from_file(cls, path: Path | str) -> "TokenSigner":
        return cls(load_private_key(path))

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._key.public_key()

    def mint(
        self,
        *,
        sub: str,
        tenant: str,
        roles: Iterable[str] = (),
        groups: Iterable[str] = (),
        agent: str | None = None,
        sid: str | None = None,
        request_id: str | None = None,
        ttl: int = MAX_TTL_SECONDS,
    ) -> str:
        """A user token. Raises ``IdentityError`` if the claims don't make a valid identity."""
        if sub.startswith(SERVICE_SUBJECT_PREFIX):
            raise IdentityError("invalid_claims")
        if not 0 < ttl <= MAX_TTL_SECONDS:
            raise IdentityError("invalid_ttl")
        now = int(self._clock())
        try:
            identity = Identity(
                iss=ISSUER,
                aud=AUDIENCE,
                sub=sub,
                tid=tenant,
                roles=tuple(roles),
                groups=tuple(groups),
                agent=agent,
                sid=sid,
                rid=request_id or new_request_id(),
                iat=now,
                exp=now + ttl,
            )
        except ValidationError:
            raise IdentityError("invalid_claims") from None
        payload = identity.model_dump(mode="json", exclude_none=True)
        return jwt.encode(payload, self._key, algorithm=ALGORITHM, headers={"kid": self.kid})

    def mint_service(
        self, *, name: str, tenant: str, request_id: str | None = None, ttl: int = MAX_TTL_SECONDS
    ) -> str:
        """A service token for ``svc:<name>`` in ``tenant``: exactly its service role, no groups
        (what the worker sends with embeddings, rerank and scans)."""
        validate_service_name(name)
        if not 0 < ttl <= MAX_TTL_SECONDS:
            raise IdentityError("invalid_ttl")
        now = int(self._clock())
        try:
            identity = Identity(
                iss=ISSUER,
                aud=AUDIENCE,
                sub=SERVICE_SUBJECT_PREFIX + name,
                tid=tenant,
                roles=(SERVICE_ROLE_PREFIX + name,),
                rid=request_id or new_request_id(),
                iat=now,
                exp=now + ttl,
            )
        except ValidationError:
            raise IdentityError("invalid_claims") from None
        payload = identity.model_dump(mode="json", exclude_none=True)
        return jwt.encode(payload, self._key, algorithm=ALGORITHM, headers={"kid": self.kid})


# --- verification -------------------------------------------------------------------------

_JWT_ERRORS: tuple[tuple[type[jwt.PyJWTError], str], ...] = (
    (jwt.ExpiredSignatureError, "expired"),
    (jwt.ImmatureSignatureError, "not_yet_valid"),
    (jwt.InvalidAudienceError, "wrong_audience"),
    (jwt.InvalidIssuerError, "wrong_issuer"),
    (jwt.InvalidSignatureError, "bad_signature"),
    (jwt.MissingRequiredClaimError, "invalid_claims"),
)
# How often a verifier may read its key file again for a token naming an unknown key: often
# enough for a rotation, too rarely for random kids to cost.
KEY_RELOAD_SECONDS = 30.0


class _Keys:
    """Trusted public keys by ``kid``; with a file, read again (at most every ``reload_seconds``)
    when a token names a key it doesn't hold. A failed read keeps the keys it has."""

    def __init__(
        self,
        keys: Mapping[str, Ed25519PublicKey],
        *,
        path: Path | None = None,
        reload_seconds: float = KEY_RELOAD_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not keys:
            raise IdentityConfigError("no identity public keys")
        self._keys = dict(keys)
        self._path = path
        self._reload_seconds = reload_seconds
        self._clock = clock
        self._read_at = clock()
        self._lock = threading.Lock()

    def get(self, kid: object) -> Ed25519PublicKey | None:
        if not isinstance(kid, str) or not kid:
            return None
        key = self._keys.get(kid)
        if key is not None or self._path is None:
            return key
        with self._lock:
            if self._clock() - self._read_at < self._reload_seconds:
                return self._keys.get(kid)
            self._read_at = self._clock()
            with contextlib.suppress(IdentityConfigError):
                self._keys = load_public_keys(self._path)
            return self._keys.get(kid)


def _decode(
    token: str,
    keys: _Keys,
    *,
    issuer: str,
    audience: str,
    required: Iterable[str],
    leeway: int,
    now: float | None,
) -> dict[str, Any]:
    """The verified claims of an EdDSA token from ``issuer`` for ``audience``. ``now`` checks a
    recorded token's times against the time it was recorded instead of the clock."""
    if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_LENGTH:
        raise IdentityError("malformed")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise IdentityError("malformed") from None
    if header.get("alg") != ALGORITHM:
        raise IdentityError("bad_algorithm")
    key = keys.get(header.get("kid"))
    if key is None:
        raise IdentityError("unknown_key")
    options: dict[str, Any] = {"require": list(required)}
    if now is not None:
        options.update(verify_exp=False, verify_iat=False, verify_nbf=False)
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[ALGORITHM],
            audience=audience,
            issuer=issuer,
            leeway=leeway,
            options=options,
        )
    except jwt.PyJWTError as exc:
        if isinstance(exc, jwt.MissingRequiredClaimError) and exc.claim == "tid":
            raise IdentityError("missing_tenant") from None
        for error_type, reason in _JWT_ERRORS:
            if isinstance(exc, error_type):
                raise IdentityError(reason) from None
        raise IdentityError("malformed") from None
    if now is not None:
        exp, iat = claims.get("exp"), claims.get("iat")
        if not isinstance(exp, int) or not isinstance(iat, int):
            raise IdentityError("invalid_claims")
        if exp + leeway <= now:
            raise IdentityError("expired")
        if iat - leeway > now:
            raise IdentityError("not_yet_valid")
    return claims


class TokenVerifier:
    """Verifies identity tokens against one or more trusted public keys. ``from_file`` reads the
    file again when a token names an unknown key (rotation)."""

    def __init__(
        self,
        keys: Mapping[str, Ed25519PublicKey],
        *,
        leeway_seconds: int = 5,
        path: Path | None = None,
        reload_seconds: float = KEY_RELOAD_SECONDS,
    ) -> None:
        self._keys = _Keys(keys, path=path, reload_seconds=reload_seconds)
        self._leeway = leeway_seconds

    @classmethod
    def for_public_key(cls, public_key: Ed25519PublicKey, **kwargs: Any) -> "TokenVerifier":
        return cls({key_id(public_key): public_key}, **kwargs)

    @classmethod
    def from_file(cls, path: Path | str, **kwargs: Any) -> "TokenVerifier":
        return cls(load_public_keys(path), path=Path(path), **kwargs)

    def verify(self, token: str, *, now: float | None = None) -> Identity:
        """Return the token's identity, or raise ``IdentityError`` with a reason code. ``now``
        checks a recorded token's times against when it was recorded."""
        claims = _decode(
            token,
            self._keys,
            issuer=ISSUER,
            audience=AUDIENCE,
            required=REQUIRED_CLAIMS,
            leeway=self._leeway,
            now=now,
        )
        try:
            return Identity.model_validate(claims)
        except ValidationError:
            raise IdentityError("invalid_claims") from None


# --- the worker's own service identity ----------------------------------------------------


class ServiceTokens(Protocol):
    def token(self, tenant: str) -> str | None:
        """A service token for ``tenant`` to send now, or None: the call then carries no
        identity."""
        ...


class SigningServiceTokens:
    """A fresh token from the development key for every call."""

    def __init__(self, signer: TokenSigner, name: str) -> None:
        self.signer = signer
        self.name = name

    def token(self, tenant: str) -> str | None:
        return self.signer.mint_service(name=self.name, tenant=tenant)


def service_tokens(*, signing_key: Path | None, name: str) -> ServiceTokens | None:
    """The worker's service identity from settings: tokens signed with ``signing_key``, or none
    without one."""
    if signing_key is not None:
        return SigningServiceTokens(TokenSigner.from_file(signing_key), name)
    return None
