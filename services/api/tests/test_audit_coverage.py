"""Audit coverage: every route that changes something records an audit event, or
says here why it needn't. Run as a script for the report:

    uv run python services/api/tests/test_audit_coverage.py
"""

import importlib
import inspect
import pkgutil
import re
import sys
from collections.abc import Iterator

import stratafi_api.routes as route_package

MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
# A call that records an audit event: directly, through a route module's helper, or through
# intake (``ingest``), a suite's save, or an event that triggers runs (``fire``).
AUDITING = re.compile(r"record_audit_event|_audit\(|\bfire\(|_ingest\(|\bingest\(|_save\(")

# Routes that use a write verb but change nothing, so there's nothing to audit.
READ_ONLY = {
    "/experiments/plan": "plans an experiment without saving it",
    "/search": "searches a run's index",
    "/chat": "retrieves and streams an answer; nothing is stored",
    "/eval-runs/search": "searches eval runs with the search language",
    "/pipeline-specs/validate": "validates without saving",
    "/retrieval-configs/validate": "validates without saving",
    "/test-suites/validate": "validates without saving",
    "/dev/sign-in": "development sign-in mints a token and keeps nothing",
}


def mutating_routes() -> Iterator[tuple[str, str, str, bool]]:
    """(methods, path, module, audited) for every write route."""
    for info in pkgutil.iter_modules(route_package.__path__):
        module = importlib.import_module(f"{route_package.__name__}.{info.name}")
        routers = [value for value in vars(module).values() if type(value).__name__ == "APIRouter"]
        for router in routers:
            for route in router.routes:
                methods = set(getattr(route, "methods", None) or ()) & MUTATING
                if not methods:
                    continue
                source = inspect.getsource(route.endpoint)
                yield (
                    ",".join(sorted(methods)),
                    route.path,
                    info.name,
                    bool(AUDITING.search(source)),
                )


def test_every_write_route_is_audited_or_read_only() -> None:
    routes = list(mutating_routes())
    assert len(routes) > 40
    missing = [
        f"{methods} {path}"
        for methods, path, _, audited in routes
        if not audited and path not in READ_ONLY
    ]
    assert missing == []


def test_read_only_routes_still_exist() -> None:
    paths = {path for _, path, _, _ in mutating_routes()}
    assert set(READ_ONLY) - {"/dev/sign-in"} <= paths


if __name__ == "__main__":
    rows = sorted(mutating_routes(), key=lambda row: row[1])
    audited = sum(1 for row in rows if row[3])
    for methods, path, _module, ok in rows:
        state = (
            "audited" if ok else f"read-only: {READ_ONLY[path]}" if path in READ_ONLY else "MISSING"
        )
        print(f"{methods:7} {path:60} {state}")
    print(f"{audited} of {len(rows)} write routes audited; {len(rows) - audited} read-only")
    sys.exit(0)
