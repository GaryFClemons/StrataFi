"""The parse stage: the default parsers on the format fixtures, and parses that
run in a sandbox with a time and memory limit."""

import hashlib
import time
from pathlib import Path

import pytest

from stratafi_components import default_registry
from stratafi_components.detect import detect
from stratafi_components.extract import raw_document
from stratafi_components.parsers.markdown import MarkdownParser
from stratafi_components.parsers.pdf import PypdfParser
from stratafi_components.sandbox import run_isolated
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.settings import LimitSettings
from stratafi_testing.canary import CanaryCheck

ROOT = Path(__file__).resolve().parents[3]
FORMATS = ROOT / "fixtures" / "formats"


def raw_for(path: Path):
    data = path.read_bytes()
    return raw_document("d-1", hashlib.sha256(data).hexdigest(), len(data), detect(path, path.name))


def test_markdown_keeps_the_documents_structure() -> None:
    path = FORMATS / "tips.md"
    component = MarkdownParser()
    document = component.parse_local(path, raw_for(path), component.resolve_params({}))
    assert document.title == "Tip Policy"
    assert document.pages == []
    kinds = [(e.type, e.level) for e in document.elements]
    assert kinds[:4] == [("title", None), ("paragraph", None), ("heading", 2), ("paragraph", None)]
    assert ("heading", 3) in kinds
    headings = [e.text for e in document.elements if e.type == "heading"]
    assert headings[:3] == ["1 Who shares in tips", "2 Card and app tips", "3 Cash tips"]
    assert "3.1 Counting the tip jar" in headings
    for element in document.elements:
        assert document.text[element.char_start : element.char_end] == element.text


def test_pdfs_carry_their_page_sizes() -> None:
    path = FORMATS / "food-safety.pdf"
    component = PypdfParser()
    document = component.parse_local(path, raw_for(path), component.resolve_params({}))
    assert [page.number for page in document.pages] == [1, 2]
    assert (document.pages[0].width, document.pages[0].height) == (612.0, 792.0)
    assert {element.page for element in document.elements} == {1, 2}


def test_the_default_parsers_record_themselves() -> None:
    import pypdf

    assert PypdfParser.static_env(PypdfParser.resolve_params({})) == {"pypdf": pypdf.__version__}
    registry = default_registry()
    assert registry.get("parse/markdown").info.accepts == ("text/markdown", "text/plain")
    assert registry.get("parse/markitdown").unavailable() is None  # a regular dependency now


def test_a_parse_runs_in_the_sandbox(tmp_path: Path) -> None:
    path = FORMATS / "tips.pdf"
    component = PypdfParser()
    context = StageContext()
    sandboxed = component.parse(path, raw_for(path), component.resolve_params({}), context)
    local = component.parse_local(path, raw_for(path), component.resolve_params({}))
    assert sandboxed == local
    assert context.metrics["pages"] == 1
    markdown = MarkdownParser()
    path = FORMATS / "tips.txt"
    context = StageContext()
    sandboxed = markdown.parse(path, raw_for(path), markdown.resolve_params({}), context)
    assert sandboxed == markdown.parse_local(path, raw_for(path), markdown.resolve_params({}))
    assert context.metrics["elements"] == len(sandboxed.elements)


def test_a_parser_refusal_keeps_its_reason(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nnot really")
    component = PypdfParser()
    with pytest.raises(ComponentError) as raised:
        component.parse(path, raw_for(path), component.resolve_params({}), StageContext())
    assert raised.value.code == "malformed"


def test_a_hanging_parse_is_killed_at_its_limit() -> None:
    started = time.monotonic()
    with pytest.raises(ComponentError) as raised:
        run_isolated("stratafi_testing.sandbox_targets:hang", {"seconds": 60}, timeout=2)
    assert raised.value.code == "sandbox_timeout"
    assert time.monotonic() - started < 10


def test_the_parse_limits_come_from_the_settings(tmp_path: Path) -> None:
    limits = LimitSettings(parse_timeout_seconds=1)
    assert StageContext(limits=limits).limits.parse_timeout_seconds == 1


def test_a_parse_over_its_memory_limit_fails_alone() -> None:
    with pytest.raises(ComponentError) as raised:
        run_isolated(
            "stratafi_testing.sandbox_targets:allocate",
            {"bytes": 1024 * 1024 * 1024},
            timeout=60,
            memory_bytes=512 * 1024 * 1024,
        )
    assert raised.value.code == "sandbox_memory"
    assert (
        run_isolated(
            "stratafi_testing.sandbox_targets:allocate",
            {"bytes": 10 * 1024 * 1024},
            timeout=60,
            memory_bytes=512 * 1024 * 1024,
        )
        == 10 * 1024 * 1024
    )


def test_a_crash_and_a_refusal_are_codes() -> None:
    with pytest.raises(ComponentError) as raised:
        run_isolated("stratafi_testing.sandbox_targets:crash", {}, timeout=30)
    assert raised.value.code == "sandbox_crashed"
    with pytest.raises(ComponentError) as raised:
        run_isolated("stratafi_testing.sandbox_targets:refuse", {}, timeout=30)
    assert raised.value.code == "encrypted"


def test_the_sandbox_gets_a_scratch_directory_and_a_small_environment(tmp_path: Path) -> None:
    result = run_isolated(
        "stratafi_testing.sandbox_targets:echo", {"a": 1}, timeout=30, scratch_parent=tmp_path
    )
    assert result["payload"] == {"a": 1}
    assert Path(result["cwd"]).parent == tmp_path
    assert not Path(result["cwd"]).exists()  # deleted afterwards
    assert set(result["env"]) <= {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TZ",
        "TEMP",
        "TMP",
        "TMPDIR",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "LC_CTYPE",
    }
    # Temporary files go to the scratch directory, which goes afterwards.
    assert result["tmpdir"] == result["cwd"]


def test_exception_messages_never_leave_the_sandbox(canary: CanaryCheck) -> None:
    with pytest.raises(ComponentError) as raised:
        run_isolated("stratafi_testing.sandbox_targets:leak", {"secret": canary.value}, timeout=30)
    assert raised.value.code == "sandbox_failed"
    canary.assert_absent(str(raised.value), canary.captured_text())
