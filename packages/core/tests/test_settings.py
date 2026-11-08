"""Settings: YAML with overlays, environment overrides, secrets that never print, and defaults
for one machine."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from stratafi_core.config import ConfigError
from stratafi_core.envfile import parse_env
from stratafi_core.settings import OfflineSettings, Settings


@pytest.fixture
def config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("STRATAFI_CONFIG_DIR", "STRATAFI_CONFIG_OVERLAY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "config/overlays/dev").mkdir(parents=True)
    (tmp_path / "config/stratafi.yaml").write_text(
        "database: {url: '${DB_URL}'}\n"
        "storage: {bucket: from-yaml, secret_key: '${S3_SECRET}'}\n"
        "worker: {lease_seconds: 60, slots: {cpu: 2, io: 4}}\n",
        encoding="utf-8",
    )
    (tmp_path / "config/overlays/dev/stratafi.yaml").write_text(
        "worker: {slots: {cpu: 8}}\nidentity: {auto_provision_workspaces: true}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STRATAFI_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("STRATAFI_CONFIG_OVERLAY", str(tmp_path / "config/overlays/dev"))
    monkeypatch.setenv("DB_URL", "postgresql://stratafi:pw-123@127.0.0.1:5442/stratafi")
    monkeypatch.setenv("S3_SECRET", "s3-secret-value")
    return tmp_path


def test_yaml_overlay_and_environment_combine(
    config_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRATAFI_WORKER__LEASE_SECONDS", "30")
    monkeypatch.setenv("STRATAFI_ENV", "test")
    settings = Settings()
    assert settings.env == "test"
    assert settings.worker.lease_seconds == 30, "the environment overrides YAML"
    assert settings.worker.slots == {"cpu": 8, "io": 4}, "the overlay merges into the base"
    assert settings.identity.auto_provision_workspaces is True
    assert settings.storage.bucket == "from-yaml"
    assert settings.database.url is not None
    assert settings.database.url.get_secret_value().endswith("/stratafi")


def test_secrets_never_print(config_env: Path) -> None:
    settings = Settings()
    shown = f"{settings!r} {settings} {settings.model_dump()}"
    assert "pw-123" not in shown
    assert "s3-secret-value" not in shown


def test_a_missing_variable_stops_loading(
    config_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DB_URL")
    with pytest.raises(ConfigError, match="DB_URL"):
        Settings()


def test_invalid_values_are_refused(config_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRATAFI_LOG_LEVEL", "chatty")
    with pytest.raises(ValidationError):
        Settings()


def test_negative_slots_are_refused() -> None:
    with pytest.raises(ValidationError):
        OfflineSettings(worker={"slots": {"cpu": -1}})


def test_offline_settings_ignore_the_machine(
    config_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRATAFI_ENV", "prod")
    settings = OfflineSettings()
    assert settings.env == "dev"
    assert settings.database.url is None
    assert OfflineSettings(env="test").env == "test"


def test_the_shipped_config_loads_with_the_example_environment(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = dict(parse_env((repo_root / ".env.example").read_text(encoding="utf-8")))
    for key, value in example.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STRATAFI_CONFIG_DIR", str(repo_root / "config"))
    for overlay in ("dev", "test"):
        monkeypatch.setenv("STRATAFI_CONFIG_OVERLAY", str(repo_root / "config/overlays" / overlay))
        settings = Settings()
        assert settings.identity.auto_provision_workspaces is True
        assert settings.providers.allowed == ()
        assert settings.models.in_process is True
    monkeypatch.delenv("STRATAFI_CONFIG_OVERLAY")
    # One person on one machine: the shipped config makes their workspace on first sight.
    assert Settings().identity.auto_provision_workspaces is True


def _one_machine(settings: Settings) -> None:
    assert settings.storage.driver == "local"
    assert settings.models.gateway_url == "http://127.0.0.1:8091"
    assert settings.worker.slots == {"cpu": 2, "gpu": 1, "io": 4}
    assert settings.models.enrichment_enabled is True
    assert settings.models.llm_rerank_enabled is True


def test_the_defaults_suit_one_machine_without_a_launcher(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in [name for name in os.environ if name.startswith("STRATAFI_")]:
        monkeypatch.delenv(name)
    _one_machine(OfflineSettings())
    example = dict(parse_env((repo_root / ".env.example").read_text(encoding="utf-8")))
    for key, value in example.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STRATAFI_CONFIG_DIR", str(repo_root / "config"))
    monkeypatch.setenv("STRATAFI_CONFIG_OVERLAY", str(repo_root / "config/overlays/dev"))
    _one_machine(Settings())
    # The shipped YAML's own defaults, with neither variable in the environment.
    monkeypatch.delenv("STRATAFI_STORAGE__DRIVER")
    monkeypatch.delenv("STRATAFI_MODEL_GATEWAY_URL")
    _one_machine(Settings())
    # An empty gateway URL turns the gateway off.
    monkeypatch.setenv("STRATAFI_MODEL_GATEWAY_URL", "")
    assert not Settings().models.gateway_url
