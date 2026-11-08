"""compose.yaml for one machine: Postgres by default; ClamAV, Tika and SeaweedFS each behind a
profile; the app profile's API, worker, web app and model gateway; ports on loopback only; and a
key copier with only the capabilities it needs."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SECRETS = {
    "STRATAFI_PG_ADMIN_PASSWORD": "x",
    "STRATAFI_PG_PASSWORD": "x",
    "STRATAFI_S3_SECRET_KEY": "x",
    "STRATAFI_S3_ADMIN_SECRET_KEY": "x",
}
# What a fresh .env leaves the optional services at, whatever this machine's environment says.
OPTIONAL_OFF = {
    "STRATAFI_CLAMAV_ADDRESS": "",
    "STRATAFI_TIKA_URL": "",
    "STRATAFI_STORAGE__DRIVER": "",
}


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))


def test_postgres_runs_by_default_and_the_rest_behind_profiles(compose: dict[str, Any]) -> None:
    profiles = {
        name: tuple(service.get("profiles", ())) for name, service in compose["services"].items()
    }
    assert profiles == {
        "postgres": (),
        "seaweedfs": ("seaweedfs",),
        "seaweedfs-init": ("seaweedfs",),
        "clamav": ("clamav",),
        "tika": ("tika",),
        "api": ("app",),
        "worker": ("app",),
        "gateway": ("app",),
        "keys": ("app",),
        "web": ("app",),
    }


def test_every_port_is_on_loopback_and_every_image_is_pinned(compose: dict[str, Any]) -> None:
    for name, service in compose["services"].items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), name
        image = service["image"]
        assert not image.endswith(":latest"), name
        assert ":" in image, name


def test_the_gateway_serves_ollama_from_the_host(compose: dict[str, Any]) -> None:
    gateway = compose["services"]["gateway"]
    assert gateway["command"] == ["stratafi-gateway"]
    assert gateway["environment"]["STRATAFI_GATEWAY_OLLAMA_URL"].startswith(
        "http://host.docker.internal:"
    )
    assert "host.docker.internal:host-gateway" in gateway["extra_hosts"]
    app = compose["x-app-environment"]
    assert app["STRATAFI_MODEL_GATEWAY_URL"] == "http://gateway:8091"


def test_the_key_copier_has_only_what_it_needs(compose: dict[str, Any]) -> None:
    copier = compose["services"]["keys"]
    assert copier["cap_drop"] == ["ALL"]
    assert set(copier["cap_add"]) == {"CHOWN", "DAC_OVERRIDE"}
    assert copier["security_opt"] == ["no-new-privileges:true"]
    assert copier["read_only"] is True
    assert copier["pull_policy"] == "never"
    script = copier["command"][0]
    assert script.startswith("set -eu\n")
    assert "-m 0400" in script
    assert "chown 10001 " in script
    # The app's containers read the copies; nothing else mounts the keys' folder.
    for name in ("api", "worker"):
        assert "keys:/run/stratafi/keys:ro" in compose["services"][name]["volumes"]
    mounts = [
        volume
        for name, service in compose["services"].items()
        for volume in service.get("volumes", [])
        if name != "keys" and "data/keys" in str(volume)
    ]
    assert mounts == []


def test_health_checks_survive_podmans_docker_api(compose: dict[str, Any]) -> None:
    """Podman's Docker-compatible API splits an exec-form check's arguments at their spaces
    (Podman 4.9), so a check with spaced arguments uses the shell form."""
    for name, service in compose["services"].items():
        test = service.get("healthcheck", {}).get("test")
        if isinstance(test, list) and test[0] == "CMD":
            assert not any(" " in argument for argument in test[1:]), name


def _config(*profiles: str, **env: str) -> dict[str, Any]:
    environment = {**os.environ, **SECRETS, **OPTIONAL_OFF, **env}
    for name in ("COMPOSE_PROJECT_NAME", "COMPOSE_FILE", "COMPOSE_PROFILES"):
        environment.pop(name, None)
    arguments = [part for profile in profiles for part in ("--profile", profile)]
    result = subprocess.run(
        [
            *("docker", "compose", "--env-file", os.devnull, "-f", "compose.yaml"),
            *arguments,
            *("config", "--format", "json"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and "docker: 'compose' is not a docker command" in result.stderr:
        pytest.skip("needs the compose plugin")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("docker") is None, reason="needs the docker CLI (no daemon)")
def test_compose_resolves_each_profile() -> None:
    """`docker compose config` interpolates and checks the file without a daemon."""
    assert set(_config()["services"]) == {"postgres"}

    app = _config("app")
    assert set(app["services"]) == {"postgres", "api", "worker", "gateway", "keys", "web"}
    worker = app["services"]["worker"]["environment"]
    assert worker["STRATAFI_MODEL_GATEWAY_URL"] == "http://gateway:8091"
    assert worker["STRATAFI_STORAGE__DRIVER"] == "local"
    # The optional services are off until .env names them.
    assert worker["STRATAFI_CLAMAV_ADDRESS"] == ""
    assert worker["STRATAFI_TIKA_URL"] == ""
    published = {
        name: [(port["host_ip"], port["target"]) for port in service["ports"]]
        for name, service in app["services"].items()
        if service.get("ports")
    }
    assert published == {
        "postgres": [("127.0.0.1", 5432)],
        "api": [("127.0.0.1", 8000)],
        "gateway": [("127.0.0.1", 8091)],
        "web": [("127.0.0.1", 4180)],
    }

    everything = _config(
        "app",
        "clamav",
        "tika",
        "seaweedfs",
        STRATAFI_CLAMAV_ADDRESS="127.0.0.1:3320",
        STRATAFI_TIKA_URL="http://127.0.0.1:9998",
        STRATAFI_STORAGE__DRIVER="s3",
    )
    api = everything["services"]["api"]
    assert api["environment"]["STRATAFI_CLAMAV_ADDRESS"] == "clamav:3310"
    assert api["environment"]["STRATAFI_TIKA_URL"] == "http://tika:9998"
    assert api["environment"]["STRATAFI_STORAGE__DRIVER"] == "s3"
    assert api["environment"]["STRATAFI_S3_ENDPOINT"] == "http://seaweedfs:9000"
    assert api["depends_on"]["seaweedfs-init"]["required"] is False
