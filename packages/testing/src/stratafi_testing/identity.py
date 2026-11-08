"""Development identities for tests: a throwaway key pair and identity tokens for the synthetic
users."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stratafi_core.identity import (
    ALGORITHM,
    HEADER,
    TokenSigner,
    TokenVerifier,
    generate_keypair,
    load_private_key,
)
from stratafi_testing.users import synthetic_user


@dataclass
class DevIdentity:
    signer: TokenSigner
    verifier: TokenVerifier
    private_key: Ed25519PrivateKey
    private_key_path: Path
    public_key_path: Path

    @classmethod
    def create(cls, directory: Path) -> "DevIdentity":
        private_path = directory / "identity-signing.pem"
        public_path = directory / "identity-signing.pub.pem"
        generate_keypair(private_path, public_path)
        private_key = load_private_key(private_path)
        return cls(
            signer=TokenSigner(private_key),
            verifier=TokenVerifier.from_file(public_path),
            private_key=private_key,
            private_key_path=private_path,
            public_key_path=public_path,
        )

    def token(
        self,
        tenant: str = "tenant0",
        role: str = "owner",
        *,
        roles: tuple[str, ...] | None = None,
        groups: tuple[str, ...] | None = None,
        ttl: int = 300,
    ) -> str:
        """A token for the synthetic ``role`` user of ``tenant``."""
        user = synthetic_user(tenant, role)
        return self.signer.mint(
            sub=user.user_id,
            tenant=tenant,
            roles=user.roles if roles is None else roles,
            groups=user.groups if groups is None else groups,
            ttl=ttl,
        )

    def headers(
        self, tenant: str = "tenant0", role: str = "owner", **kwargs: Any
    ) -> dict[str, str]:
        return {HEADER: self.token(tenant, role, **kwargs)}

    def sign_claims(self, claims: dict[str, Any], *, kid: str | None = None) -> str:
        """Sign arbitrary claims with the trusted key, skipping validation: for negative tests."""
        return jwt.encode(
            claims, self.private_key, algorithm=ALGORITHM, headers={"kid": kid or self.signer.kid}
        )
