"""Services for the integration tier: Postgres with pgvector, SeaweedFS, ClamAV and Tika.

Tests call the ``compose_services`` fixture with service names and get their addresses back::

    services = compose_services("postgres", "seaweedfs")
    services.pg_admin_dsn(), services.s3_endpoint(), services.clamd_address()

By default the services run in Docker, from this repo's ``compose.yaml``, as their own compose
project (``stratafi-test``) with fixed test-only credentials and ports Docker picks on
127.0.0.1, so tests never touch the development project's data. The project stops and its
volumes are removed at the end of the session, unless ``STRATAFI_KEEP_SERVICES=1``.

Tests never pull or build an image: they run offline. Pull them once beforehand::

    docker compose --profile seaweedfs --profile clamav --profile tika pull

Without Docker, point the tests at local services instead, one variable per service
(docs/guides/local-development.md):

- ``STRATAFI_TEST_PG_ADMIN_DSN``: a superuser URL (``postgresql://...``) for a Postgres 16 with
  pgvector installed
- ``STRATAFI_TEST_S3_ENDPOINT``, ``STRATAFI_TEST_S3_ACCESS_KEY``, ``STRATAFI_TEST_S3_SECRET_KEY``,
  ``STRATAFI_TEST_S3_BUCKET``: an S3 API such as a local SeaweedFS; the bucket must exist
- ``STRATAFI_TEST_CLAMD``: ``host:port`` of a clamd, or ``fake`` for the in-process fake
- ``STRATAFI_TEST_TIKA_URL``: a Tika 3.2 server's URL, such as ``http://127.0.0.1:9998``

A service with neither Docker nor its variables makes the test skip, saying why.
"""

import os
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stratafi_testing.clamd import fake_clamd

PROJECT = "stratafi-test"
KEEP_ENV = "STRATAFI_KEEP_SERVICES"
SERVICES = ("postgres", "seaweedfs", "clamav", "tika")
CONTAINER_PORTS = {"postgres": 5432, "seaweedfs": 9000, "clamav": 3310, "tika": 9998}
# compose.yaml reads these. Empty ports make Docker pick free ones on 127.0.0.1. Compose checks
# every required variable in the file, the app profile's too, so each one has a test value: a
# checkout without .env (CI, a fresh clone) starts the services as well. The optional services'
# profiles are on, so every compose command of the test project knows them.
TEST_ENV = {
    "COMPOSE_PROFILES": "seaweedfs,clamav,tika",
    "STRATAFI_PG_ADMIN_USER": "stratafi_admin",
    "STRATAFI_PG_ADMIN_PASSWORD": "stratafi-test-admin-password",
    "STRATAFI_PG_PASSWORD": "stratafi-test-password",
    "STRATAFI_S3_ACCESS_KEY": "stratafi-test",
    "STRATAFI_S3_SECRET_KEY": "stratafi-test-secret-key",
    "STRATAFI_S3_BUCKET": "stratafi-test",
    "STRATAFI_S3_ADMIN_ACCESS_KEY": "stratafi-test-admin",
    "STRATAFI_S3_ADMIN_SECRET_KEY": "stratafi-test-admin-secret-key",
    "STRATAFI_PG_PORT": "",
    "STRATAFI_S3_PORT": "",
    "STRATAFI_CLAMAV_PORT": "",
    "STRATAFI_TIKA_PORT": "",
}
LOCAL_ENV = {
    "postgres": ("STRATAFI_TEST_PG_ADMIN_DSN",),
    "seaweedfs": (
        "STRATAFI_TEST_S3_ENDPOINT",
        "STRATAFI_TEST_S3_ACCESS_KEY",
        "STRATAFI_TEST_S3_SECRET_KEY",
        "STRATAFI_TEST_S3_BUCKET",
    ),
    "clamav": ("STRATAFI_TEST_CLAMD",),
    "tika": ("STRATAFI_TEST_TIKA_URL",),
}
SETUP_HINT = (
    "Images are pulled before tests, never during them. Run:\n"
    "  docker compose --profile seaweedfs --profile clamav --profile tika pull"
)


class ComposeError(RuntimeError):
    pass


def find_docker() -> str | None:
    # Python 3.12.0's shutil.which can return Docker Desktop's extensionless shim on Windows.
    if os.name == "nt":
        return shutil.which("docker.exe") or shutil.which("docker")
    return shutil.which("docker")


def docker_available(timeout: float = 20.0) -> tuple[bool, str]:
    """Whether the Docker daemon answers, and why not if it doesn't."""
    docker = find_docker()
    if docker is None:
        return False, "the docker CLI isn't installed"
    try:
        result = subprocess.run(  # noqa: S603 - fixed arguments, no user input
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "docker info timed out"
    if result.returncode != 0 or not result.stdout.strip():
        return False, "the Docker daemon isn't reachable"
    return True, f"Docker {result.stdout.strip()}"


class ComposeProject:
    """``docker compose`` for this repo's ``compose.yaml``, as the ``stratafi-test`` project."""

    def __init__(self, repo_root: Path, docker: str, project: str = PROJECT) -> None:
        self.repo_root = repo_root
        self.docker = docker
        self.project = project

    def run(self, *args: str, timeout: float = 600) -> subprocess.CompletedProcess[str]:
        command = [
            self.docker,
            "compose",
            "--project-directory",
            str(self.repo_root),
            "--file",
            str(self.repo_root / "compose.yaml"),
            "--project-name",
            self.project,
            *args,
        ]
        result = subprocess.run(  # noqa: S603 - arguments come from test code, not user input
            command,
            cwd=self.repo_root,
            env={**os.environ, **TEST_ENV},
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise ComposeError(f"compose {' '.join(args)} failed:\n{result.stderr[-3000:]}")
        return result

    def up(self, services: Iterable[str], *, wait_timeout: int = 300) -> None:
        names = list(services)
        if "seaweedfs" in names:
            names.append("seaweedfs-init")  # creates the bucket
        try:
            self.run(
                "up",
                "--detach",
                "--wait",
                "--wait-timeout",
                str(wait_timeout),
                "--pull",
                "never",
                "--no-build",
                *names,
                timeout=wait_timeout + 60,
            )
        except ComposeError as exc:
            raise ComposeError(f"{exc}\n{SETUP_HINT}") from None

    def port(self, service: str, container_port: int) -> tuple[str, int]:
        output = self.run("port", service, str(container_port)).stdout.strip().splitlines()
        if not output:
            raise ComposeError(f"{service} doesn't publish port {container_port}")
        host, _, port = output[0].rpartition(":")
        return host.strip("[]") or "127.0.0.1", int(port)

    def down(self) -> None:
        self.run("down", "--volumes", "--remove-orphans")


@dataclass
class ServiceEndpoints:
    """Where the requested services answer, as seen from the test process."""

    addresses: dict[str, tuple[str, int]] = field(default_factory=dict)
    pg_admin_dsn_value: str | None = None
    s3: dict[str, str] = field(default_factory=dict)

    def address(self, service: str) -> tuple[str, int]:
        try:
            return self.addresses[service]
        except KeyError:
            raise KeyError(f"{service} wasn't requested from compose_services") from None

    def pg_admin_dsn(self, database: str = "postgres") -> str:
        """A superuser URL (``postgresql://...``) for ``database``."""
        if self.pg_admin_dsn_value is None:
            raise KeyError("postgres wasn't requested from compose_services")
        from stratafi_testing.databases import with_database

        return with_database(self.pg_admin_dsn_value, database)

    def s3_endpoint(self) -> str:
        return self._s3("endpoint")

    def s3_access_key(self) -> str:
        return self._s3("access_key")

    def s3_secret_key(self) -> str:
        return self._s3("secret_key")

    def s3_bucket(self) -> str:
        return self._s3("bucket")

    def clamd_address(self) -> tuple[str, int]:
        return self.address("clamav")

    def tika_url(self) -> str:
        host, port = self.address("tika")
        return f"http://{host}:{port}"

    def _s3(self, key: str) -> str:
        if not self.s3:
            raise KeyError("seaweedfs wasn't requested from compose_services")
        return self.s3[key]


def check_integration_marker(node: Any) -> None:
    """compose_services belongs to the integration tier only; fail loudly anywhere else."""
    if node.get_closest_marker("integration") is None:
        import pytest

        pytest.fail("compose_services is for integration tests: add @pytest.mark.integration")


def _local(service: str, environ: Mapping[str, str]) -> bool:
    return all(environ.get(name) for name in LOCAL_ENV[service])


class ServiceManager:
    """Session-wide: starts each service once, and stops the project at the end."""

    def __init__(
        self,
        repo_root: Path,
        *,
        environ: Mapping[str, str] | None = None,
        docker_check: Callable[[], tuple[bool, str]] = docker_available,
    ) -> None:
        self.repo_root = repo_root
        self.environ = os.environ if environ is None else environ
        self._docker_check = docker_check
        self._docker: tuple[bool, str] | None = None
        self._project: ComposeProject | None = None
        self._started: set[str] = set()
        self._fakes = ExitStack()
        self._fake_clamd: tuple[str, int] | None = None
        self.endpoints = ServiceEndpoints()

    def start(self, *services: str) -> ServiceEndpoints:
        import pytest

        unknown = sorted(set(services) - set(SERVICES))
        if unknown:
            raise ValueError(f"unknown services: {', '.join(unknown)}")
        local = [service for service in services if _local(service, self.environ)]
        in_docker = [service for service in services if service not in local]
        for service in local:
            self._use_local(service)
        if in_docker:
            ok, reason = self._docker_status()
            if not ok:
                variables = ", ".join(name for s in in_docker for name in LOCAL_ENV[s])
                pytest.skip(f"integration tier needs Docker ({reason}), or set {variables}")
            self._start_in_docker(in_docker)
        return self.endpoints

    def close(self) -> None:
        self._fakes.close()
        if self._project is not None and self._started and self.environ.get(KEEP_ENV) != "1":
            self._project.down()

    def _docker_status(self) -> tuple[bool, str]:
        if self._docker is None:
            self._docker = self._docker_check()
        return self._docker

    def _use_local(self, service: str) -> None:
        env = self.environ
        if service == "postgres":
            self.endpoints.pg_admin_dsn_value = env["STRATAFI_TEST_PG_ADMIN_DSN"]
        elif service == "seaweedfs":
            self.endpoints.s3 = {
                "endpoint": env["STRATAFI_TEST_S3_ENDPOINT"],
                "access_key": env["STRATAFI_TEST_S3_ACCESS_KEY"],
                "secret_key": env["STRATAFI_TEST_S3_SECRET_KEY"],
                "bucket": env["STRATAFI_TEST_S3_BUCKET"],
            }
        elif service == "tika":
            from urllib.parse import urlsplit

            url = urlsplit(env["STRATAFI_TEST_TIKA_URL"])
            self.endpoints.addresses["tika"] = (url.hostname or "127.0.0.1", url.port or 9998)
        elif service == "clamav":
            value = env["STRATAFI_TEST_CLAMD"]
            if value == "fake":
                if self._fake_clamd is None:
                    self._fake_clamd = self._fakes.enter_context(fake_clamd())
                self.endpoints.addresses["clamav"] = self._fake_clamd
            else:
                host, _, port = value.rpartition(":")
                self.endpoints.addresses["clamav"] = (host or "127.0.0.1", int(port))

    def _start_in_docker(self, services: list[str]) -> None:
        if self._project is None:
            docker = find_docker()
            if docker is None:
                raise ComposeError("the docker CLI disappeared after the daemon check")
            self._project = ComposeProject(self.repo_root, docker)
        new = [service for service in services if service not in self._started]
        if new:
            self._project.up(new)
            self._started.update(new)
        for service in services:
            self.endpoints.addresses[service] = self._project.port(
                service, CONTAINER_PORTS[service]
            )
        if "postgres" in services:
            host, port = self.endpoints.addresses["postgres"]
            user = TEST_ENV["STRATAFI_PG_ADMIN_USER"]
            password = TEST_ENV["STRATAFI_PG_ADMIN_PASSWORD"]
            self.endpoints.pg_admin_dsn_value = (
                f"postgresql://{user}:{password}@{host}:{port}/postgres"
            )
        if "seaweedfs" in services:
            host, port = self.endpoints.addresses["seaweedfs"]
            self.endpoints.s3 = {
                "endpoint": f"http://{host}:{port}",
                "access_key": TEST_ENV["STRATAFI_S3_ACCESS_KEY"],
                "secret_key": TEST_ENV["STRATAFI_S3_SECRET_KEY"],
                "bucket": TEST_ENV["STRATAFI_S3_BUCKET"],
            }
