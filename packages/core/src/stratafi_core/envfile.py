"""``.env`` files in the plain form Docker Compose and uv both read the same way.

A file is lines of ``KEY=value``. Blank lines and lines starting with ``#`` are skipped, a line
may start with ``export``, and a value may sit in single or double quotes, which are dropped.
Nothing else is interpreted: no escapes, no inline comments, no ``${VAR}`` expansion. That
subset reads the same in both tools, which is why ``scripts/dev_setup.py`` writes only it.
"""

import re
import string
from pathlib import Path

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Characters a value can hold without quotes: no spaces, quotes, $, # or backslashes.
_BARE = frozenset(string.ascii_letters + string.digits + "_./:@%+=,~-")


def _unquoted(value: str) -> str:
    if len(value) > 1 and value[0] in "\"'" and value.endswith(value[0]):
        return value[1:-1]
    return value


def parse_env(text: str) -> list[tuple[str, str]]:
    """The file's ``(key, value)`` pairs, in order. A line that isn't ``KEY=value`` raises
    ``ValueError`` naming its line number, never its content."""
    pairs: list[tuple[str, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, equals, value = line.partition("=")
        key = key.strip()
        if key.startswith("export") and key[6:7].isspace():
            key = key[6:].strip()
        if not equals or not _KEY_RE.fullmatch(key):
            raise ValueError(f"line {number} of the env file isn't KEY=value")
        pairs.append((key, _unquoted(value.strip())))
    return pairs


def read_env_file(path: Path) -> dict[str, str]:
    """A file's values by key; a key set twice keeps its last value."""
    return dict(parse_env(path.read_text(encoding="utf-8")))


def format_env_value(value: str) -> str:
    """``value`` as it goes after ``KEY=``: bare when that's safe, else in single quotes, which
    Compose doesn't interpolate. A value needing quotes can't also hold a single quote."""
    if all(char in _BARE for char in value):
        return value
    if "'" in value:
        raise ValueError("an env value can't hold both special characters and a single quote")
    return f"'{value}'"
