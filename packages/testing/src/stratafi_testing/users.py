"""Synthetic users: one per workspace role in each tenant, plus a guest with no role.

``local`` is the lab's one person on a laptop, whom the UI and the SDK sign in as its owner;
``tenant0`` and ``halcyon`` (the demo corpus's) are the tests' tenants. A token's ``roles``
claim carries Stratafi's workspace roles directly. Every name is invented.
"""

from dataclasses import dataclass

TENANTS = ("local", "tenant0", "halcyon")
ROLES = ("owner", "editor", "viewer", "guest")


@dataclass(frozen=True)
class SyntheticUser:
    tenant: str
    username: str
    name: str
    roles: tuple[str, ...]
    groups: tuple[str, ...]

    @property
    def user_id(self) -> str:
        """The ``sub`` claim that dev tokens use for this user."""
        return f"u-{self.tenant}-{self.username}"


_TEMPLATES: dict[str, tuple[str, tuple[str, ...]]] = {
    "owner": ("Olive Owner (synthetic)", ("owner",)),
    "editor": ("Eddie Editor (synthetic)", ("editor",)),
    "viewer": ("Vic Viewer (synthetic)", ("viewer",)),
    "guest": ("Gale Guest (synthetic)", ()),
}

SYNTHETIC_USERS: dict[str, dict[str, SyntheticUser]] = {
    tenant: {
        role: SyntheticUser(tenant, role, name, roles, ("all-staff",))
        for role, (name, roles) in _TEMPLATES.items()
    }
    for tenant in TENANTS
}


def synthetic_user(tenant: str = "tenant0", role: str = "owner") -> SyntheticUser:
    try:
        return SYNTHETIC_USERS[tenant][role]
    except KeyError:
        raise KeyError(f"no synthetic {role} user in tenant {tenant}") from None
