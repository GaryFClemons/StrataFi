"""YAML configuration with overlays and ``${VAR}`` expansion.

How a config file is read:

- Config loads from ``STRATAFI_CONFIG_DIR`` (default ``config``, relative to the working
  directory).
- If ``STRATAFI_CONFIG_OVERLAY`` names a directory, a file there with the same relative path is
  deep-merged on top: mappings merge key by key, and everything else (lists included) is
  replaced.
- Finally ``${VAR}`` in string values is expanded from the environment. ``${VAR:-default}``
  gives a default, and ``$${`` writes a literal ``${``. A missing variable without a default is
  an error.

Secrets never live in config files: a file refers to them by variable name. So errors name
files, fields and variables, never values.
"""

import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

CONFIG_DIR_ENV = "STRATAFI_CONFIG_DIR"
CONFIG_OVERLAY_ENV = "STRATAFI_CONFIG_OVERLAY"

_VAR_RE = re.compile(r"\$\$\{|\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Config is missing or invalid. Messages name files, fields and variables, never values."""


def _environ(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def config_root(env: Mapping[str, str] | None = None) -> Path:
    return Path(_environ(env).get(CONFIG_DIR_ENV) or "config")


def overlay_root(env: Mapping[str, str] | None = None) -> Path | None:
    value = _environ(env).get(CONFIG_OVERLAY_ENV)
    return Path(value) if value else None


def safe_relative_path(path: str | PurePosixPath) -> PurePosixPath:
    """A config-relative POSIX path that can't escape its root."""
    text = str(path)
    if not text or "\\" in text or ":" in text:
        raise ConfigError("config paths are relative POSIX paths")
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError("config paths can't be absolute or contain '..'")
    return relative


def deep_merge(base: Any, overlay: Any) -> Any:
    """Merge ``overlay`` onto ``base``: mappings merge recursively, anything else is replaced."""
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = deep_merge(base[key], value) if key in base else value
        return merged
    return overlay


def expand_env(value: Any, env: Mapping[str, str] | None = None) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in every string inside ``value``."""
    environ = _environ(env)
    if isinstance(value, str):

        def substitute(match: re.Match[str]) -> str:
            if match.group(0) == "$${":
                return "${"
            name, default = match.group(1), match.group(2)
            if name in environ:
                return environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is not set")

        return _VAR_RE.sub(substitute, value)
    if isinstance(value, Mapping):
        return {key: expand_env(item, environ) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item, environ) for item in value]
    return value


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise ConfigError(f"{path.as_posix()}: invalid YAML{where}") from None


def load_yaml_config(
    relpath: str | PurePosixPath,
    *,
    env: Mapping[str, str] | None = None,
    required: bool = True,
) -> Any:
    """Load one config file with its overlay merged in and ``${VAR}`` expanded."""
    relative = safe_relative_path(relpath)
    base_file = config_root(env) / relative
    overlay_dir = overlay_root(env)
    overlay_file = overlay_dir / relative if overlay_dir is not None else None
    have_base = base_file.is_file()
    have_overlay = overlay_file is not None and overlay_file.is_file()
    if not have_base and not have_overlay:
        if required:
            raise ConfigError(f"config file not found: {relative}")
        return None
    data = _read_yaml(base_file) if have_base else None
    if have_overlay and overlay_file is not None:
        overlay = _read_yaml(overlay_file)
        data = deep_merge(data, overlay) if data is not None else overlay
    return expand_env(data, env)


def describe_validation_error(exc: ValidationError) -> str:
    """Field paths and error types only: config values can hold expanded secrets."""
    parts = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ())) or "(root)"
        parts.append(f"{loc} ({error.get('type')})")
    return ", ".join(parts[:10])


def load_config_file[M: BaseModel](
    relpath: str | PurePosixPath, model: type[M], *, env: Mapping[str, str] | None = None
) -> M:
    data = load_yaml_config(relpath, env=env)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{relpath}: {describe_validation_error(exc)}") from None
