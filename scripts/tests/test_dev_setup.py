"""scripts/dev_setup.py: .env from .env.example, the signing key, and dev token minting."""

import importlib.util
import os
import stat
import sys
import time
from pathlib import Path
from types import ModuleType

import jwt
import pytest

from stratafi_core.envfile import read_env_file
from stratafi_core.identity import TokenVerifier


@pytest.fixture
def dev_setup(repo_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "script_dev_setup", repo_root / "scripts" / "dev_setup.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    (tmp_path / ".env.example").write_text(
        "# shared\n"
        "A=1\n"
        "SECRET=__GENERATE_SECRET__\n"
        "PORT=5442\n"
        "DSN=postgresql://u:${SECRET}@127.0.0.1:${PORT}/x\n"
        "A=duplicate\n",
        encoding="utf-8",
    )
    return tmp_path


def test_setup_writes_env_and_the_signing_key(
    dev_setup: ModuleType, fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert dev_setup.main([], root=fake_repo) == 0
    env = read_env_file(fake_repo / ".env")
    assert env["A"] == "1"
    assert len(env["SECRET"]) >= 40
    assert env["DSN"] == f"postgresql://u:{env['SECRET']}@127.0.0.1:5442/x"
    assert (fake_repo / "data/keys/identity-signing.pem").is_file()
    assert (fake_repo / "data/keys/identity-signing.pub.pem").is_file()
    if sys.platform != "win32":
        mode = stat.S_IMODE((fake_repo / ".env").stat().st_mode)
        assert mode == 0o600
    assert env["SECRET"] not in capsys.readouterr().out


def test_setup_keeps_existing_values_and_keys(dev_setup: ModuleType, fake_repo: Path) -> None:
    dev_setup.main([], root=fake_repo)
    first = read_env_file(fake_repo / ".env")
    key = (fake_repo / "data/keys/identity-signing.pem").read_bytes()
    with (fake_repo / ".env").open("a", encoding="utf-8") as handle:
        handle.write("MINE=kept\n")
    dev_setup.main([], root=fake_repo)
    second = read_env_file(fake_repo / ".env")
    assert second["SECRET"] == first["SECRET"]
    assert second["MINE"] == "kept"
    assert (fake_repo / "data/keys/identity-signing.pem").read_bytes() == key


def test_an_undefined_reference_stops_setup(dev_setup: ModuleType, tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("A=${MISSING}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="MISSING"):
        dev_setup.main([], root=tmp_path)


def test_a_worktree_gets_its_own_compose_project(dev_setup: ModuleType, tmp_path: Path) -> None:
    worktree = tmp_path / "Stratafi-feature"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    assert dev_setup.compose_project_name(worktree) == "stratafi-feature"
    (tmp_path / "main").mkdir()
    (tmp_path / "main" / ".git").mkdir()
    assert dev_setup.compose_project_name(tmp_path / "main") is None


def mint(dev_setup: ModuleType, root: Path, capsys: pytest.CaptureFixture[str], *args: str) -> str:
    assert dev_setup.main(["token", *args], root=root) == 0
    return capsys.readouterr().out.strip()


def test_a_minted_token_has_claims_that_parse(
    dev_setup: ModuleType, fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dev_setup.main([], root=fake_repo)
    capsys.readouterr()
    token = mint(dev_setup, fake_repo, capsys, "--tenant", "tenant0", "--user", "editor")
    verifier = TokenVerifier.from_file(fake_repo / "data/keys/identity-signing.pub.pem")
    identity = verifier.verify(token)
    assert identity.tid == "tenant0"
    assert identity.sub == "u-tenant0-editor"
    assert identity.roles == ("editor",)
    assert identity.groups == ("all-staff",)
    assert time.time() < identity.exp <= time.time() + 300
    raw = jwt.decode(token, options={"verify_signature": False})
    assert {"tid", "sub", "roles", "groups", "exp"} <= set(raw)


def test_any_subject_with_chosen_roles(
    dev_setup: ModuleType, fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dev_setup.main([], root=fake_repo)
    capsys.readouterr()
    token = mint(
        dev_setup,
        fake_repo,
        capsys,
        *("--tenant", "halcyon", "--sub", "u-me", "--roles", "owner,viewer", "--ttl", "60"),
    )
    identity = TokenVerifier.from_file(fake_repo / "data/keys/identity-signing.pub.pem").verify(
        token
    )
    assert (identity.tid, identity.sub, identity.roles) == ("halcyon", "u-me", ("owner", "viewer"))
    assert identity.exp - identity.iat == 60


@pytest.mark.parametrize(
    "args",
    [
        ("--tenant", "tenant0", "--user", "nobody"),
        ("--tenant", "Not A Tenant", "--user", "owner"),
        ("--tenant", "tenant0", "--user", "owner", "--ttl", "301"),
    ],
)
def test_bad_token_requests_fail_cleanly(
    dev_setup: ModuleType, fake_repo: Path, args: tuple[str, ...]
) -> None:
    dev_setup.main([], root=fake_repo)
    assert dev_setup.main(["token", *args], root=fake_repo) == 2


def test_no_key_no_token(dev_setup: ModuleType, tmp_path: Path) -> None:
    assert dev_setup.main(["token", "--tenant", "tenant0", "--user", "owner"], root=tmp_path) == 2


def test_tokens_are_refused_in_prod(
    dev_setup: ModuleType, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_setup.main([], root=fake_repo)
    monkeypatch.setenv("STRATAFI_ENV", "prod")
    assert dev_setup.main(["token", "--tenant", "tenant0", "--user", "owner"], root=fake_repo) == 2


def test_the_real_example_builds(dev_setup: ModuleType, repo_root: Path, tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_bytes((repo_root / ".env.example").read_bytes())
    sections, counts = dev_setup.build_env(tmp_path, {})
    values = dict(entry for _, entries in sections for entry in entries)
    assert counts["generated"] == 4  # the S3 administrator's too
    assert values["STRATAFI_DATABASE_URL"].startswith("postgresql://stratafi:")
    assert "${" not in "".join(values.values())
    assert os.sep not in values["STRATAFI_S3_BUCKET"]
