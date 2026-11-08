"""YAML config: overlay merging, ${VAR} expansion and the errors."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from stratafi_core.config import (
    ConfigError,
    deep_merge,
    expand_env,
    load_config_file,
    load_yaml_config,
    safe_relative_path,
)


def test_mappings_merge_and_everything_else_is_replaced() -> None:
    base = {"a": {"x": 1, "y": [1, 2]}, "b": "keep", "c": {"deep": {"one": 1}}}
    overlay = {"a": {"y": [3], "z": True}, "c": {"deep": {"two": 2}}, "d": None}
    assert deep_merge(base, overlay) == {
        "a": {"x": 1, "y": [3], "z": True},
        "b": "keep",
        "c": {"deep": {"one": 1, "two": 2}},
        "d": None,
    }
    assert base["a"] == {"x": 1, "y": [1, 2]}, "the base must not change"


@pytest.mark.parametrize(
    ("base", "overlay", "expected"),
    [
        ({"a": 1}, [1], [1]),
        ([1, 2], {"a": 1}, {"a": 1}),
        ("text", None, None),
        ({"a": {"b": 1}}, {"a": "scalar"}, {"a": "scalar"}),
    ],
)
def test_a_non_mapping_replaces_whatever_was_there(base, overlay, expected) -> None:
    assert deep_merge(base, overlay) == expected


def test_variables_expand_everywhere() -> None:
    env = {"HOST": "db", "PORT": "5432"}
    value = {"url": "pg://${HOST}:${PORT}/x", "list": ["${HOST}", 3, {"n": "${PORT}"}], "n": 7}
    assert expand_env(value, env) == {
        "url": "pg://db:5432/x",
        "list": ["db", 3, {"n": "5432"}],
        "n": 7,
    }


def test_defaults_and_escapes() -> None:
    env = {"SET": "yes", "EMPTY": ""}
    assert expand_env("${MISSING:-fallback}", env) == "fallback"
    assert expand_env("${MISSING:-}", env) == ""
    assert expand_env("${SET:-fallback}", env) == "yes"
    assert expand_env("${EMPTY:-fallback}", env) == "", "a set but empty variable is used"
    assert expand_env("$${SET} costs $5", env) == "${SET} costs $5"


def test_a_missing_variable_is_an_error_naming_it() -> None:
    with pytest.raises(ConfigError, match="STRATAFI_SECRET_THING is not set"):
        expand_env({"a": ["${STRATAFI_SECRET_THING}"]}, {})


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_an_overlay_file_merges_over_the_base(tmp_path: Path) -> None:
    write(tmp_path / "config/app.yaml", "db: {url: '${DB}', pool: 5}\nnames: [a, b]\n")
    write(tmp_path / "overlay/app.yaml", "db: {pool: 9}\nnames: [c]\n")
    env = {
        "STRATAFI_CONFIG_DIR": str(tmp_path / "config"),
        "STRATAFI_CONFIG_OVERLAY": str(tmp_path / "overlay"),
        "DB": "pg://x",
    }
    assert load_yaml_config("app.yaml", env=env) == {
        "db": {"url": "pg://x", "pool": 9},
        "names": ["c"],
    }


def test_an_overlay_alone_is_enough(tmp_path: Path) -> None:
    write(tmp_path / "overlay/only.yaml", "a: 1\n")
    env = {
        "STRATAFI_CONFIG_DIR": str(tmp_path / "config"),
        "STRATAFI_CONFIG_OVERLAY": str(tmp_path / "overlay"),
    }
    assert load_yaml_config("only.yaml", env=env) == {"a": 1}


def test_missing_files(tmp_path: Path) -> None:
    env = {"STRATAFI_CONFIG_DIR": str(tmp_path)}
    with pytest.raises(ConfigError, match="not found"):
        load_yaml_config("absent.yaml", env=env)
    assert load_yaml_config("absent.yaml", env=env, required=False) is None


def test_invalid_yaml_names_the_line_but_not_the_content(tmp_path: Path) -> None:
    write(tmp_path / "bad.yaml", "ok: 1\nsecret: [unclosed hunter2\n")
    with pytest.raises(ConfigError, match=r"invalid YAML \(line") as error:
        load_yaml_config("bad.yaml", env={"STRATAFI_CONFIG_DIR": str(tmp_path)})
    assert "hunter2" not in str(error.value)


@pytest.mark.parametrize("path", ["../etc/passwd", "/etc/passwd", "a/../../b", "C:/x", "a\\b", ""])
def test_config_paths_cannot_escape_the_root(path: str) -> None:
    with pytest.raises(ConfigError):
        safe_relative_path(path)


class Model(BaseModel):
    port: int
    password: str


def test_validation_errors_name_fields_but_never_values(tmp_path: Path) -> None:
    write(tmp_path / "m.yaml", "port: not-a-number\npassword: ${PW}\n")
    env = {"STRATAFI_CONFIG_DIR": str(tmp_path), "PW": "hunter2"}
    with pytest.raises(ConfigError, match=r"port \(int_parsing\)") as error:
        load_config_file("m.yaml", Model, env=env)
    assert "hunter2" not in str(error.value)
    assert "not-a-number" not in str(error.value)
