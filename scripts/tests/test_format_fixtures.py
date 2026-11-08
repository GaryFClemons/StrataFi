"""The format fixtures (``fixtures/formats``): what ``scripts/format_fixtures.py`` writes, byte for
byte, and each of their documents in every format."""

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load() -> Any:
    path = ROOT / "scripts" / "format_fixtures.py"
    spec = importlib.util.spec_from_file_location("script_format_fixtures", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_format_fixtures_are_reproducible() -> None:
    assert load().main(["--check"]) == 0


def test_each_document_comes_in_every_format() -> None:
    script = load()
    names = sorted(path.name for path in script.OUTPUT.iterdir() if path.is_file())
    assert names == sorted(
        f"{name}{extension}" for name in script.DOCUMENTS for extension in script.EXTENSIONS
    )
    for name, source in script.DOCUMENTS.items():
        assert (script.OUTPUT / f"{name}.md").read_bytes() == (script.SOURCE / source).read_bytes()
        text = (script.OUTPUT / f"{name}.txt").read_bytes()
        assert text.startswith(b"\xef\xbb\xbf")
        assert b"\r\n" in text
        assert b"\n" not in text.replace(b"\r\n", b"")
