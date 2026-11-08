"""Stratafi's settings: one pydantic-settings model for the API and the worker.

Values come from, highest priority first:

1. arguments passed to ``Settings(...)`` (tests, scripts)
2. environment variables: ``STRATAFI_ENV``, ``STRATAFI_LOG_LEVEL``, and any nested value as
   ``STRATAFI_<SECTION>__<KEY>``, for example ``STRATAFI_WORKER__LEASE_SECONDS=30``
3. ``config/stratafi.yaml`` with its overlay and ``${VAR}`` expansion (``stratafi_core.config``)

Settings never read ``.env`` themselves; on the host run ``uv run --env-file .env ...``.
Secrets are ``SecretStr``, so they never print. Component defaults live in each component's
parameter schema, not here.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from stratafi_core.config import ConfigError, load_yaml_config

ResourceClass = Literal["cpu", "gpu", "io"]
WorkspaceRole = Literal["owner", "editor", "viewer"]
SETTINGS_FILE = "stratafi.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiSettings(_Section):
    # Containers set STRATAFI_API__HOST=0.0.0.0; on the host the API listens on loopback only.
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=0, le=65535)


class DatabaseSettings(_Section):
    # A libpq URL, ``postgresql://user:password@host:port/database``.
    url: SecretStr | None = None
    pool_size: PositiveInt = 5
    max_overflow: int = Field(default=5, ge=0)


class StorageSettings(_Section):
    # Files under local_root by default; "s3" for an S3 API such as compose's SeaweedFS.
    driver: Literal["s3", "local"] = "local"
    endpoint: str | None = None
    region: str = "us-east-1"
    access_key: str | None = None
    secret_key: SecretStr | None = None
    bucket: str = "stratafi"
    # The local driver's root directory, relative to the working directory.
    local_root: Path = Path("data/artifacts")
    presign_seconds: int = Field(default=900, ge=1, le=7 * 24 * 3600)


DEV_ROLE_MAP: dict[str, WorkspaceRole] = {"owner": "owner", "editor": "editor", "viewer": "viewer"}


class IdentitySettings(_Section):
    # The PEM file of trusted Ed25519 public keys for identity tokens: the development key's,
    # read again for a token naming an unknown key.
    public_key: Path | None = None
    # Token role → workspace role, and token group → workspace role. Unset, a token's roles are
    # its workspace roles, as the development sign-in mints them.
    role_map: dict[str, WorkspaceRole] | None = None
    group_map: dict[str, WorkspaceRole] = Field(default_factory=dict)
    # Create a workspace the first time a tenant's token arrives. Off: unknown tenants are
    # refused.
    auto_provision_workspaces: bool = True
    # The worker's service identity (svc:<service_name>): the key that signs its service tokens
    # for the model gateway, the development key. With none, calls carry no identity (the fakes
    # don't check).
    service_name: str = "stratafi"
    service_signing_key: Path | None = None
    # Development sign-in for the standalone UI and the SDK: the key that signs short-lived
    # tokens for the synthetic users. Never in prod; unset turns it off.
    dev_sign_in_key: Path | None = None

    def roles_for(self) -> dict[str, WorkspaceRole]:
        """The role map in force: the settings', or the development sign-in's own roles."""
        if self.role_map is not None:
            return dict(self.role_map)
        return dict(DEV_ROLE_MAP)


class ModelSettings(_Section):
    # The model gateway, OpenAI-compatible: services/gateway on this machine. Empty: no gateway.
    gateway_url: str | None = "http://127.0.0.1:8091"
    # Logical name → gateway alias, for example ``embedding: kb-embed``.
    aliases: dict[str, str] = Field(default_factory=dict)
    # In-process models (sentence-transformers and the like); off refuses them.
    in_process: bool = True
    # Price per 1,000 tokens by provider, recorded with each run for its cost estimate.
    prices: dict[str, float] = Field(default_factory=dict)
    # enrich/llm-context's chat calls through the gateway; off refuses the component.
    enrichment_enabled: bool = True
    # rerank/llm-listwise's chat calls through the gateway; off refuses the component.
    llm_rerank_enabled: bool = True


class ServiceSettings(_Section):
    # Where scan/clamav reaches clamd, as host:port.
    clamd: str | None = None
    # Apache Tika's server, for parse/tika.
    tika_url: str | None = None


class LicenseSettings(_Section):
    # Licenses that need review, such as AGPL-3.0, allowed for components that declare
    # them. None until the owner reviews one.
    allowed: tuple[str, ...] = ()


class CleanSettings(_Section):
    # The key clean/redact's hash mode keys its HMAC with (STRATAFI_CLEAN__REDACTION_KEY). Unset:
    # the hash mode fails with redaction_key_not_configured; the other modes need no key.
    redaction_key: SecretStr | None = None


class ProviderLimit(_Section):
    requests_per_minute: PositiveInt | None = None
    tokens_per_minute: PositiveInt | None = None
    concurrency: PositiveInt | None = None


class ProviderEndpoint(_Section):
    """Where a hosted provider answers, and the environment variable that holds its key (never
    the key itself)."""

    url: str | None = None
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    region: str | None = None


class ProviderSettings(_Section):
    # Hosted providers that may receive document text. None until the owner approves one.
    allowed: tuple[str, ...] = ()
    # Their endpoints, by provider name (the part after provider: in a component's egress).
    endpoints: dict[str, ProviderEndpoint] = Field(default_factory=dict)
    # Rate and concurrency limits per provider: shared by every worker process through Postgres,
    # or per process with shared_limits off.
    limits: dict[str, ProviderLimit] = Field(default_factory=dict)
    shared_limits: bool = True


class WorkerSettings(_Section):
    # Jobs one worker runs at once, by resource class. The gpu slot runs in-process embedders and
    # rerankers, and fine-tunes them.
    slots: dict[ResourceClass, int] = Field(default_factory=lambda: {"cpu": 2, "gpu": 1, "io": 4})
    poll_seconds: float = Field(default=1.0, gt=0)
    lease_seconds: PositiveInt = 60
    heartbeat_seconds: float = Field(default=15.0, gt=0)
    shutdown_grace_seconds: float = Field(default=30.0, ge=0)
    max_attempts: PositiveInt = 3
    backoff_base_seconds: float = Field(default=5.0, gt=0)
    backoff_max_seconds: float = Field(default=600.0, gt=0)

    @field_validator("slots")
    @classmethod
    def _non_negative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("slot counts can't be negative")
        return value


class ObservabilitySettings(_Section):
    # Prometheus metrics (stratafi_core.observability). The API serves them at
    # /api/v1/metrics, with the queue read from Postgres when scraped; a bearer token, when set,
    # guards them, and a prod deployment must set one. The worker serves its own on
    # worker_metrics_port (none by default), at worker_metrics_host.
    metrics: bool = True
    metrics_token: SecretStr | None = None
    worker_metrics_port: int | None = Field(default=None, ge=1, le=65535)
    worker_metrics_host: str = "127.0.0.1"


class RetentionSettings(_Section):
    # Days to keep each kind of record; None keeps it until deleted (
    # stratafi_store.retention). A document's expiry purges it by lineage, leaving a tombstone;
    # a derived artifact's takes the artifacts derived from it too.
    document_days: PositiveInt | None = None
    artifact_days: PositiveInt | None = None
    audit_days: PositiveInt | None = None
    # How often one worker sweeps, and the most documents and artifact roots one sweep purges.
    sweep_minutes: PositiveInt = 60
    batch: PositiveInt = 200


class LimitSettings(_Section):
    # One uploaded file: 256 MiB.
    upload_bytes: PositiveInt = 256 * 1024 * 1024
    upload_files: PositiveInt = Field(default=1000, description="Files in one upload request")
    pages: PositiveInt = 2000
    # Archives: entries, expanded size and compression ratio (stratafi_components.archives).
    archive_entries: PositiveInt = 10_000
    archive_expanded_bytes: PositiveInt = 512 * 1024 * 1024
    archive_ratio: PositiveInt = 100
    parse_timeout_seconds: PositiveInt = 120
    # Each parse's writable memory and address space (Linux and macOS; Windows has the time
    # limit only). A parser may ask for more address space, never more writable memory.
    parse_memory_bytes: PositiveInt = 2 * 1024 * 1024 * 1024
    # How a parse is confined (stratafi_components.confine): "required" fails a parse that
    # can't have seccomp (no network, nothing that reaches the worker); "best_effort" applies
    # what the kernel allows; "off" applies none. A prod deployment must require it.
    parse_confinement: Literal["required", "best_effort", "off"] = "best_effort"


class S3Source(_Section):
    """A bucket the API may import from (extract/s3). Credentials stay in configuration; the
    API never accepts them from a caller."""

    bucket: str = Field(min_length=3, max_length=63)
    # Only keys under this prefix can be imported; a caller's prefix is relative to it.
    prefix: str = ""
    endpoint: str | None = None
    region: str = "us-east-1"
    access_key: str | None = None
    secret_key: SecretStr | None = None


class IntakeSettings(_Section):
    # Server folders the API may import from, by name (extract/folder). None by default: name
    # one, such as fixtures, in a development overlay.
    folder_roots: dict[str, Path] = Field(default_factory=dict)
    # Buckets the API may import from, by name (extract/s3). None by default.
    s3_sources: dict[str, S3Source] = Field(default_factory=dict)
    # Where uploads are spooled while they stream in; the system's temporary directory if unset.
    spool_dir: Path | None = None
    # Unpack ZIP archives (unpack/zip) and email messages (unpack/email) when they arrive.
    unpack_archives: bool = True
    unpack_email: bool = True
    # Fetch documents from URLs (extract/url), through the SSRF guard, on these ports only.
    url_import: bool = True
    url_ports: tuple[int, ...] = (80, 443)


class SandboxSettings(_Section):
    # Where custom checks run: "none" runs none, "docker" runs each check in its
    # own container from ``image`` (docker/sandbox/Dockerfile) with no network, a read-only
    # root, no capabilities, a non-root user and these limits. ``runtime`` "auto" picks gVisor
    # (runsc) when Docker lists it.
    backend: Literal["none", "docker"] = "none"
    docker: str = "docker"
    image: str = "stratafi/check-sandbox:1"
    runtime: Literal["auto", "runsc", "runc"] = "auto"
    memory_mb: int = Field(default=512, ge=64, le=8192)
    cpus: float = Field(default=1.0, gt=0, le=8)
    pids: int = Field(default=64, ge=8, le=1024)
    scratch_mb: int = Field(default=64, ge=1, le=1024)
    output_bytes: int = Field(default=1_048_576, ge=4096, le=16_777_216)
    # Seconds a container may take to start, on top of the check's own timeout_s.
    start_seconds: float = Field(default=20.0, gt=0, le=300)
    # Where the worker writes each check's bundle and code, mounted read-only into the
    # container. A worker that runs in a container needs it at the same path on the host.
    workdir: Path | None = None
    # The sandbox's security review: the sign-off's reference (a ticket or document ID).
    # Without it no workspace can turn custom checks on ("Needs a human").
    review: str | None = Field(default=None, max_length=200)


class SuiteSettings(_Section):
    # Webhooks reach public addresses on these ports only, through the SSRF guard that
    # URL import uses. Development and tests may allow private addresses and any port (None),
    # for receivers on the same machine.
    webhook_allow_private: bool = False
    webhook_ports: tuple[int, ...] | None = (80, 443)
    webhook_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    webhook_attempts: PositiveInt = 5
    # Derives each webhook's signing secret, so Postgres keeps none; from a secret
    # store, as STRATAFI_SUITES__WEBHOOK_SECRET_KEY. Without it, secrets are stored as before.
    # Rotating it rotates every derived secret.
    webhook_secret_key: SecretStr | None = None
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)

    def custom_checks_blocked(self) -> str | None:
        """Why no workspace can run custom checks here, or None: the sandbox isn't configured,
        or its security review isn't signed off."""
        if self.sandbox.backend == "none":
            return "sandbox_not_configured"
        if not (self.sandbox.review or "").strip():
            return "sandbox_not_reviewed"
        return None


class _YamlSource(PydanticBaseSettingsSource):
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        relpath = getattr(self.settings_cls, "yaml_config", None)
        if not relpath:
            return {}
        data = load_yaml_config(relpath, required=False)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(f"{relpath}: expected a mapping")
        return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STRATAFI_",
        env_nested_delimiter="__",
        extra="ignore",
        env_ignore_empty=True,
        frozen=True,
    )

    yaml_config: ClassVar[str | None] = SETTINGS_FILE

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    api: ApiSettings = ApiSettings()
    database: DatabaseSettings = DatabaseSettings()
    storage: StorageSettings = StorageSettings()
    identity: IdentitySettings = IdentitySettings()
    models: ModelSettings = ModelSettings()
    services: ServiceSettings = ServiceSettings()
    clean: CleanSettings = CleanSettings()
    licenses: LicenseSettings = LicenseSettings()
    providers: ProviderSettings = ProviderSettings()
    worker: WorkerSettings = WorkerSettings()
    retention: RetentionSettings = RetentionSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    limits: LimitSettings = LimitSettings()
    intake: IntakeSettings = IntakeSettings()
    suites: SuiteSettings = SuiteSettings()

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError("unknown log level")
        return level

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    def deployment_problems(self) -> list[str]:
        """Why this configuration can't run: the API and the worker refuse to start while there
        is any. Only prod asks for more than the defaults."""
        problems = []
        if self.is_prod and self.observability.metrics and not self.observability.metrics_token:
            problems.append(
                "observability.metrics_token must be set in prod: the metrics endpoint shows the "
                "queue and every route's traffic"
            )
        if self.is_prod and self.limits.parse_confinement != "required":
            problems.append(
                "limits.parse_confinement must be required in prod: parsers read untrusted "
                "documents and must not reach the network or the worker"
            )
        return problems

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, _YamlSource(settings_cls))


class OfflineSettings(Settings):
    """Settings with no YAML or environment: defaults and arguments only. For tests and for
    exporting contracts, which must not depend on the machine."""

    yaml_config: ClassVar[str | None] = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)
