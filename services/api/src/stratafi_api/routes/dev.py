"""Development sign-in: short-lived tokens for the synthetic users, for the standalone UI
and the SDK.

Identity tokens live at most 300 seconds, so a browser session can't run on a pasted token; the
standalone host asks here for a fresh one as the old one ages. With no body it signs in the local
owner (tenant ``local``, role ``owner``), the one person a laptop's lab has. The route exists
only outside ``prod``, with ``identity.dev_sign_in_key`` set (the development signing key), and
it is left out of the contract.

The identities match ``stratafi_testing.users``: ``u-<tenant>-<role>`` with the role as its
workspace role (none for a guest), in the group ``all-staff``.
"""

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from stratafi_core.identity import MAX_TTL_SECONDS, TokenSigner
from stratafi_core.settings import Settings

logger = logging.getLogger("stratafi.api.dev")


class DevTokenIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant: str = Field(default="local", pattern=r"^[a-z0-9_]{1,64}$")
    role: Literal["owner", "editor", "viewer", "guest"] = "owner"


class DevTokenOut(BaseModel):
    token: str
    expires_in: int = Field(description="Seconds")
    subject: str


def enabled(settings: Settings) -> bool:
    identity = settings.identity
    return settings.env != "prod" and bool(identity.dev_sign_in_key)


def build_router(key_path: Path) -> APIRouter:
    signer = TokenSigner.from_file(key_path)
    router = APIRouter(tags=["development"], include_in_schema=False)

    @router.post("/dev/token", response_model=DevTokenOut)
    def dev_token(body: DevTokenIn) -> DevTokenOut:
        subject = f"u-{body.tenant}-{body.role}"
        token = signer.mint(
            sub=subject,
            tenant=body.tenant,
            roles=() if body.role == "guest" else (body.role,),
            groups=("all-staff",),
            ttl=MAX_TTL_SECONDS,
        )
        logger.info("development token minted", extra={"tenant": body.tenant, "role": body.role})
        return DevTokenOut(token=token, expires_in=MAX_TTL_SECONDS, subject=subject)

    return router
