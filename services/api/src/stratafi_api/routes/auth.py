"""How the UI signs in: the one route that needs no identity.

The standalone UI signs in with the development sign-in when it's on: as the local
owner by itself, or as another synthetic user. This route tells it whether it can; it holds
nothing secret.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from stratafi_api.problems import problem_responses
from stratafi_api.routes import dev

router = APIRouter(tags=["identity"])


class SignInOut(BaseModel):
    dev_sign_in: bool = Field(
        default=False, description="Whether the development sign-in route is on"
    )


@router.get(
    "/auth/config",
    response_model=SignInOut,
    summary="How the UI signs in: whether the development sign-in is on",
    responses=problem_responses(503),
)
def sign_in_config(request: Request) -> SignInOut:
    settings = request.app.state.settings
    on = dev.enabled(settings) and getattr(request.app.state, "dev_sign_in", False)
    return SignInOut(dev_sign_in=on)
