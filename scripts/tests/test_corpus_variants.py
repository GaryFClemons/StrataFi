"""Multi-format corpora: a Markdown corpus rendered as DOCX, digital PDF, scanned
PDF, HTML and PDF with tables as images reproduces for a seed, keeps its documents' keys, and in
the digital variants keeps every evidence phrase findable by a reference parser."""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pypdf
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "fixtures" / "demo-corpus"
FIXTURE = ROOT / "fixtures" / "multi-format"


def load() -> Any:
    path = ROOT / "scripts" / "corpus_variants.py"
    spec = importlib.util.spec_from_file_location("script_corpus_variants", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


variants = load()


def documents(folder: Path) -> dict[str, bytes]:
    """A variant's rendered documents, by their paths."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name not in ("manifest.yaml", "variant.json")
    }


def test_markdown_blocks() -> None:
    document = variants.parse_markdown(
        "# Title\n\nIntro, line one\nand two.\n\n## 1 Section\n\n- first\n  continued\n- second\n"
        "\n3. third\n4. fourth\n\n| A | B \\| C |\n| --- | :---: |\n| 1 | 2 |\n| 3 |\nAfter.\n"
    )
    assert document.title == "Title"
    assert document.blocks == (
        variants.Text("Intro, line one and two."),
        variants.Heading(2, "1 Section"),
        variants.Items(False, ("first continued", "second")),
        variants.Items(True, ("third", "fourth"), 3),
        variants.Table(("A", "B | C"), (("1", "2"), ("3",))),
        variants.Text("After."),
    )
    assert document.blocks[4].padded()[-1] == ("3", "")


def test_variants_are_reproducible_for_a_seed(tmp_path: Path) -> None:
    """Variants are reproducible for a seed."""
    source = FIXTURE / "corpus"
    first = variants.render_corpus(source, tmp_path / "a", seed=3)
    again = variants.render_corpus(source, tmp_path / "b", seed=3)
    for variant in variants.VARIANTS:
        assert documents(first[variant]) == documents(again[variant]), variant
        assert (first[variant] / "manifest.yaml").read_bytes() == (
            again[variant] / "manifest.yaml"
        ).read_bytes()
    other = variants.render_corpus(source, tmp_path / "c", seed=4, variants=["pdf", "scanned-pdf"])
    assert documents(other["pdf"]) == documents(first["pdf"])  # nothing in it is random
    scanned, before = documents(other["scanned-pdf"]), documents(first["scanned-pdf"])
    assert scanned.keys() == before.keys()
    assert all(scanned[name] != before[name] for name in scanned)  # its own noise and skew


def _in_tables_only(eval_set: dict[str, Any]) -> set[tuple[str, int]]:
    """The evidence items (query, position) that only a table of their document holds."""
    from stratafi_core.evalsets import EvidenceRef
    from stratafi_core.relevance import findability

    manifest = variants.load_manifest(DEMO / "corpus")
    prose: dict[tuple[str, str], str] = {}
    for entry in manifest["documents"]:
        parsed = variants.parse_markdown((DEMO / "corpus" / entry["path"]).read_text("utf-8"))
        parts = [parsed.title or ""]
        for block in parsed.blocks:
            if isinstance(block, variants.Items):
                parts += block.items
            elif not isinstance(block, variants.Table):
                parts.append(block.text)
        prose[(entry["doc_key"], str(entry["version"]))] = "\n".join(parts)
    only = set()
    for item in eval_set["queries"]:
        for position, expected in enumerate(item.get("expected") or []):
            ref = EvidenceRef.model_validate(expected)
            if not findability(ref, prose[(ref.doc_key, ref.version or "")]).findable:
                only.add((item["id"], position))
    return only


def test_evidence_stays_findable_in_the_digital_variants(tmp_path: Path) -> None:
    """In the digital variants (DOCX, PDF, HTML), every evidence phrase of the Halcyon demo
    set stays findable by a reference parser; with tables as images, exactly the evidence only a
    table holds is lost."""
    eval_set = yaml.safe_load((DEMO / "retrieval.yaml").read_text(encoding="utf-8"))
    rendered = variants.render_corpus(
        DEMO / "corpus", tmp_path, variants=[*variants.DIGITAL, "pdf-table-images"]
    )
    for variant in variants.DIGITAL:
        found = variants.evidence_findability(rendered[variant], eval_set)
        assert len(found) == 81
        assert [query for query, _, flag in found if not flag] == [], variant
    lost = {
        (query, position)
        for query, position, flag in variants.evidence_findability(
            rendered["pdf-table-images"], eval_set
        )
        if not flag
    }
    assert len(lost) == 14
    assert lost == _in_tables_only(eval_set)


def parsed(parser_id: str, path: Path) -> Any:
    import hashlib

    from stratafi_components import default_registry
    from stratafi_components.detect import detect
    from stratafi_components.extract import raw_document

    data = path.read_bytes()
    raw = raw_document("d", hashlib.sha256(data).hexdigest(), len(data), detect(path, path.name))
    component = default_registry().get(parser_id)()
    return component.parse_local(path, raw, component.resolve_params({}))


def test_the_parsers_see_the_same_structure(tmp_path: Path) -> None:
    source = FIXTURE / "corpus" / "operations" / "brew-standards-v6.md"
    markdown = parsed("parse/markdown", source)
    rendered = variants.render_corpus(FIXTURE / "corpus", tmp_path, variants=["docx", "pdf"])
    docx = parsed("parse/markitdown", rendered["docx"] / "operations" / "brew-standards-v6.docx")
    pdf = parsed("parse/pdfplumber", rendered["pdf"] / "operations" / "brew-standards-v6.pdf")

    def headings(document: Any) -> list[tuple[str, int | None]]:
        return [(e.text, e.level) for e in document.elements if e.type == "heading"]

    assert markdown.title == pdf.title == "Espresso and Brew Standards"
    assert headings(docx) == headings(markdown)  # Word heading styles keep their levels
    assert [text for text, _ in headings(pdf)][1:2] == ["1 Dialing in"]  # by font size
    row = "Decaf espresso, double | 18 g | 38 g | 28\u201333 seconds | 201°F (94°C)"
    assert row in docx.text
    assert row in markdown.text


def test_the_committed_variants_keep_their_documents_keys() -> None:
    from stratafi_components.manifest import load_corpus_manifest

    source_text = (FIXTURE / "corpus" / "manifest.yaml").read_bytes()
    source = variants.load_manifest(FIXTURE / "corpus")
    for variant in variants.VARIANTS:
        folder = FIXTURE / "variants" / variant
        manifest = variants.load_manifest(folder)
        assert manifest == variants.variant_manifest(source, variant)
        loaded = load_corpus_manifest((folder / "manifest.yaml").read_text(encoding="utf-8"))
        assert [(d.doc_key, d.version) for d in loaded.documents] == [
            ("brew-standards", "6"),
            ("wholesale", "4"),
        ]
        assert all((folder / entry["path"]).is_file() for entry in manifest["documents"])
        record = json.loads((folder / "variant.json").read_text(encoding="utf-8"))
        assert (record["variant"], record["seed"], record["documents"]) == (variant, 0, 2)
        assert record["source_manifest"] == hashlib.sha256(source_text).hexdigest()


def test_the_committed_variants_hold_what_they_should() -> None:
    eval_set = yaml.safe_load((FIXTURE / "retrieval.yaml").read_text(encoding="utf-8"))
    folder = FIXTURE / "variants"
    for variant in variants.DIGITAL:
        found = variants.evidence_findability(folder / variant, eval_set)
        assert [flag for *_, flag in found] == [True] * 5, variant
    pictured = variants.evidence_findability(folder / "pdf-table-images", eval_set)
    lost = {query for query, _, flag in pictured if not flag}
    assert lost == {"mf-decaf-shot-time", "mf-wholesale-200-lb"}  # the two in tables
    for name in ("brew-standards-v6", "wholesale-v4"):
        scanned = pypdf.PdfReader(folder / "scanned-pdf" / "operations" / f"{name}.pdf")
        digital = pypdf.PdfReader(folder / "pdf" / "operations" / f"{name}.pdf")
        assert len(scanned.pages) == len(digital.pages)
        assert all(not page.extract_text().strip() for page in scanned.pages)  # pictures only
        assert scanned.metadata is not None
        assert scanned.metadata.title == digital.metadata.title  # type: ignore[union-attr]


def test_rendering_replaces_only_its_own_folders(tmp_path: Path) -> None:
    source = FIXTURE / "corpus"
    mine = tmp_path / "html"
    mine.mkdir()
    (mine / "notes.txt").write_text("not the script's", encoding="utf-8")
    with pytest.raises(ValueError, match="didn't write it"):
        variants.render_corpus(source, tmp_path, variants=["html"])
    assert (mine / "notes.txt").is_file()
    out = tmp_path / "out"
    variants.render_corpus(source, out, variants=["html"])
    (out / "html" / "stale.html").write_text("left over", encoding="utf-8")
    variants.render_corpus(source, out, variants=["html"])
    assert not (out / "html" / "stale.html").exists()


def test_the_command_checks_its_arguments(tmp_path: Path) -> None:
    source = str(FIXTURE / "corpus")
    for argv in (["--variants", "rtf"], ["--dpi", "10"]):
        with pytest.raises(SystemExit) as exit_info:
            variants.main([source, str(tmp_path), *argv])
        assert exit_info.value.code == 2
    not_markdown = tmp_path / "source"
    not_markdown.mkdir()
    (not_markdown / "a.txt").write_text("text", encoding="utf-8")
    (not_markdown / "manifest.yaml").write_text(
        "documents:\n  - {path: a.txt, collection: c, doc_key: a, version: '1', "
        "effective_from: 2026-01-01, acl: {groups: [g]}}\n",
        encoding="utf-8",
    )
    assert variants.main([str(not_markdown), str(tmp_path / "out")]) == 2
