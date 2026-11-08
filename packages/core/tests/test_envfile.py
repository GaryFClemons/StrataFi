"""Reading and writing .env files in the subset Docker Compose and uv agree on."""

from pathlib import Path

import pytest

from stratafi_core.envfile import format_env_value, parse_env, read_env_file


def test_lines_comments_quotes_and_export() -> None:
    text = (
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        "  SPACED = padded  \n"
        "export EXPORTED=yes\n"
        "export\tTABBED=1\n"
        "export=a key named export\n"
        'DOUBLE="two words"\n'
        "SINGLE='$not_expanded'\n"
        "EMPTY=\n"
        "URL=postgresql://user:pw@127.0.0.1:5432/db?x=1\n"
        "HASH=keeps # everything\n"
    )
    assert parse_env(text) == [
        ("PLAIN", "value"),
        ("SPACED", "padded"),
        ("EXPORTED", "yes"),
        ("TABBED", "1"),
        ("export", "a key named export"),
        ("DOUBLE", "two words"),
        ("SINGLE", "$not_expanded"),
        ("EMPTY", ""),
        ("URL", "postgresql://user:pw@127.0.0.1:5432/db?x=1"),
        ("HASH", "keeps # everything"),
    ]


@pytest.mark.parametrize("line", ["no equals sign", "1BAD=key", "BAD KEY=secret", "=orphan"])
def test_a_line_that_isnt_key_value_names_its_number_not_its_text(line: str) -> None:
    with pytest.raises(ValueError, match="line 2 of the env file") as error:
        parse_env(f"OK=1\n{line}\n")
    assert line not in str(error.value)


def test_a_file_reads_as_a_dict(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("A=1\nB='two'\nA=3\n", encoding="utf-8")
    assert read_env_file(path) == {"A": "3", "B": "two"}


def test_values_are_quoted_only_when_they_need_it() -> None:
    bare = "plain-value_1.2:3@host/path%2B=,~"
    assert format_env_value(bare) == bare
    assert format_env_value("") == ""
    assert format_env_value("two words") == "'two words'"
    assert format_env_value("$HOME") == "'$HOME'"
    with pytest.raises(ValueError, match="single quote"):
        format_env_value("it's spaced")
    # A written value reads back as itself.
    for value in ("two words", "$HOME", "plain"):
        assert parse_env(f"K={format_env_value(value)}") == [("K", value)]
