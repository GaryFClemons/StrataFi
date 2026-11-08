"""Pytest fixtures for every package, loaded through the ``pytest11`` entry point.

Tier gating, by marker:

- ``integration`` tests get their services from ``compose_services``, which skips the test,
  saying why, when neither Docker nor local services are available
- ``model`` tests skip unless ``STRATAFI_TEST_MODEL_URL`` names an OpenAI-compatible endpoint
- ``system`` tests skip unless ``STRATAFI_TEST_SYSTEM=1`` says the full stack is up
- ``browser`` tests (also ``integration``: they start their own stack) skip unless
  ``STRATAFI_TEST_BROWSER=1`` says the web app is built and Playwright's Chromium is installed

Fixtures:

- ``repo_root``: the repo root (pytest's rootdir)
- ``canary``: a ``CanaryCheck`` with a fresh canary value
- ``dev_identity``: a throwaway key pair and tokens for the synthetic users
- ``compose_services``: start Postgres, SeaweedFS, ClamAV or Tika (integration tests only)
- ``fresh_database``: ``url = fresh_database()``, an empty database with pgvector, dropped after
  the test (integration tests only)
- ``migrated_engine``: an engine on a fresh database with every migration applied
- ``serve_app``: ``url = serve_app(app)`` runs an ASGI app on 127.0.0.1 with a free port
- ``model_endpoint``: ``STRATAFI_TEST_MODEL_URL`` (model tests only)
"""

import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from stratafi_core.logs import quiet_noisy_loggers
from stratafi_testing.canary import CanaryCheck, new_canary
from stratafi_testing.compose import ServiceEndpoints, ServiceManager, check_integration_marker
from stratafi_testing.databases import DatabaseFactory
from stratafi_testing.identity import DevIdentity
from stratafi_testing.servers import ServerThread

MODEL_URL_ENV = "STRATAFI_TEST_MODEL_URL"
SYSTEM_ENV = "STRATAFI_TEST_SYSTEM"
BROWSER_ENV = "STRATAFI_TEST_BROWSER"


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("system") and os.environ.get(SYSTEM_ENV) != "1":
        pytest.skip(f"system tier: start the full stack and set {SYSTEM_ENV}=1")
    if item.get_closest_marker("model") and not os.environ.get(MODEL_URL_ENV):
        pytest.skip(f"model tier: set {MODEL_URL_ENV} to an OpenAI-compatible endpoint")
    if item.get_closest_marker("browser") and os.environ.get(BROWSER_ENV) != "1":
        pytest.skip(f"browser tier: build web/ (npm run build) and set {BROWSER_ENV}=1")


@pytest.fixture(scope="session")
def repo_root(pytestconfig: pytest.Config) -> Path:
    return Path(pytestconfig.rootpath)


@pytest.fixture
def canary(caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]) -> CanaryCheck:
    """Uses ``capfd``, so a test can't also request ``capsys``. Client libraries' loggers are
    capped as in production (``quiet_noisy_loggers``), so a test client's own request log isn't
    mistaken for the code's."""
    quiet_noisy_loggers()
    caplog.set_level(logging.DEBUG)
    return CanaryCheck(new_canary(), caplog, capfd)


@pytest.fixture
def dev_identity(tmp_path: Path) -> DevIdentity:
    return DevIdentity.create(tmp_path / "keys")


@pytest.fixture(scope="session")
def _service_manager(repo_root: Path) -> Iterator[ServiceManager]:
    manager = ServiceManager(repo_root)
    yield manager
    manager.close()


@pytest.fixture
def compose_services(
    request: pytest.FixtureRequest, _service_manager: ServiceManager
) -> Callable[..., ServiceEndpoints]:
    """``compose_services("postgres", "seaweedfs")`` starts them and returns their addresses."""
    check_integration_marker(request.node)
    return _service_manager.start


@pytest.fixture
def fresh_database(
    compose_services: Callable[..., ServiceEndpoints],
) -> Iterator[Callable[[], str]]:
    factory = DatabaseFactory(compose_services("postgres").pg_admin_dsn())
    yield factory.create
    factory.drop_all()


@pytest.fixture
def migrated_engine(fresh_database: Callable[[], str]) -> Iterator[Any]:
    from stratafi_store.db import create_engine
    from stratafi_store.migrations import upgrade

    engine = create_engine(fresh_database(), pool_size=10, max_overflow=10)
    upgrade(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def serve_app() -> Iterator[Callable[[Any], str]]:
    servers: list[ServerThread] = []

    def start(app: Any) -> str:
        server = ServerThread(app).start()
        servers.append(server)
        return server.url

    yield start
    for server in servers:
        server.stop()


@pytest.fixture
def model_endpoint() -> str:
    url = os.environ.get(MODEL_URL_ENV)
    if not url:
        pytest.skip(f"model tier: set {MODEL_URL_ENV}")
    return url.rstrip("/")
