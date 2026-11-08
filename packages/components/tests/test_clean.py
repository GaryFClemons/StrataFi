"""Cleaning rules on parsed-document fixtures: edits are positions only, the
result's offsets stay exact, and every rule's before and after text can be rebuilt."""

import json
from itertools import pairwise
from pathlib import Path

import pytest

from stratafi_components import default_registry
from stratafi_components.clean import (
    Dehyphenate,
    UnicodeNormalize,
    Whitespace,
    apply_edits,
    collapse_edits,
    replay,
    run_clean,
    trim_edits,
)
from stratafi_core.artifacts import ParsedDocument, build_text
from stratafi_core.hashing import content_hash

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "contracts" / "fixtures" / "parsed-document.v1" / "policy.json"
ZERO_WIDTH_SPACE, SOFT_HYPHEN, BOM = chr(0x200B), chr(0x00AD), chr(0xFEFF)
FULLWIDTH_ACH = chr(0xFF21) + chr(0xFF23) + chr(0xFF28)


def parsed(*elements: tuple[str, str]) -> ParsedDocument:
    texts = [text for _, text in elements]
    text, offsets = build_text(texts)
    return ParsedDocument(
        document_id="d-1",
        parser={"id": "parse/test", "version": "1.0.0", "params_hash": content_hash({})},
        media_type="text/plain",
        elements=[
            {"id": f"e{index}", "type": kind, "text": body, "char_start": s, "char_end": e}
            for index, ((kind, body), (s, e)) in enumerate(zip(elements, offsets, strict=True))
        ],
        text=text,
    )


def rules(*specs: tuple[type, dict]) -> list:
    return [(cls.info.id, cls(), cls.resolve_params(params)) for cls, params in specs]


def test_whitespace_changes_nothing_on_pdfplumber_output() -> None:
    document = ParsedDocument.model_validate(json.loads(FIXTURE.read_text()))
    cleaned = run_clean(document, rules((Whitespace, {})))
    assert cleaned.text == document.text
    assert cleaned.edits == []
    assert cleaned.elements == document.elements


def test_whitespace_collapses_labels_and_keeps_paragraph_lines() -> None:
    document = parsed(
        ("heading", "  1   Purpose\t"),
        ("paragraph", "Line one  \n  line two"),
        ("code", "x  =  1"),
    )
    cleaned = run_clean(document, rules((Whitespace, {})))
    assert [e.text for e in cleaned.elements] == ["1 Purpose", "Line one  \n  line two", "x  =  1"]
    assert [(e.replacement_kind, document.text[e.start : e.end]) for e in cleaned.edits] == [
        ("delete", "  "),
        ("space", "   "),
        ("delete", "\t"),
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("trim", "Line one\n  line two"), ("collapse", "Line one line two")],
)
def test_whitespace_can_trim_or_collapse_paragraphs(mode: str, expected: str) -> None:
    document = parsed(("paragraph", " Line one  \n  line two  "))
    cleaned = run_clean(document, rules((Whitespace, {"paragraphs": mode})))
    assert cleaned.elements[0].text == expected.strip() if mode == "collapse" else expected


def test_unicode_normalizes_and_drops_invisible_characters() -> None:
    raw = f"The ﬁnal {FULLWIDTH_ACH} fee{ZERO_WIDTH_SPACE} is ﬁve{SOFT_HYPHEN} dollars{BOM}."
    document = parsed(("paragraph", raw), ("heading", "Ⅳ Fees"))
    cleaned = run_clean(document, rules((UnicodeNormalize, {})))
    assert cleaned.elements[0].text == "The final ACH fee is five dollars."
    assert cleaned.elements[1].text == "IV Fees"
    kinds = {edit.replacement_kind for edit in cleaned.edits}
    assert kinds == {"nfkc", "delete"}
    keep = run_clean(document, rules((UnicodeNormalize, {"remove_invisible": False})))
    assert ZERO_WIDTH_SPACE in keep.elements[0].text


def test_combining_marks_normalize_with_their_letter() -> None:
    decomposed = "Cafe" + chr(0x0301) + " policy"
    cleaned = run_clean(parsed(("paragraph", decomposed)), rules((UnicodeNormalize, {})))
    assert cleaned.elements[0].text == "Caf" + chr(0x00E9) + " policy"


def test_dehyphenate_joins_words_split_across_lines() -> None:
    document = parsed(
        ("paragraph", "Requests need ap-\nproval in writ-\n ing. Self-\nService stays.")
    )
    cleaned = run_clean(document, rules((Dehyphenate, {})))
    assert cleaned.elements[0].text == "Requests need approval in writing. Self-\nService stays."
    spaced = parsed(("paragraph", "Requests need ap- proval and well-known terms."))
    assert run_clean(spaced, rules((Dehyphenate, {}))).elements[0].text == spaced.text
    joined = run_clean(spaced, rules((Dehyphenate, {"across": "line_breaks_and_spaces"})))
    assert joined.elements[0].text == "Requests need approval and well-known terms."


def test_rules_chain_and_offsets_stay_exact() -> None:
    document = parsed(
        ("title", " Funds   Availability "),
        ("paragraph", f"ﬁrst  line —  ex-\nample{ZERO_WIDTH_SPACE}"),
        ("heading", "2   Holds"),
    )
    cleaned = run_clean(
        document,
        rules(
            (UnicodeNormalize, {}),
            (Dehyphenate, {}),
            (Whitespace, {"paragraphs": "collapse"}),
        ),
    )
    assert [e.text for e in cleaned.elements] == [
        "Funds Availability",
        "first line — example",
        "2 Holds",
    ]
    for element in cleaned.elements:
        assert cleaned.text[element.char_start : element.char_end] == element.text
    assert [e.id for e in cleaned.elements] == ["e0", "e1", "e2"]


def test_before_and_after_text_rebuild_from_the_edits_alone() -> None:
    document = parsed(
        ("heading", "  1   Purpose"),
        ("paragraph", f"ﬁne  print in-\nside{SOFT_HYPHEN}"),
    )
    cleaned = run_clean(
        document,
        rules((UnicodeNormalize, {}), (Dehyphenate, {}), (Whitespace, {"paragraphs": "collapse"})),
    )
    steps = replay(document, cleaned.edits or [])
    assert len(steps) == 3
    assert steps[0][0] == document.text
    assert steps[-1][1] == cleaned.text
    for (_, after), (before, _) in pairwise(steps):
        assert after == before


def test_a_cleaned_document_always_has_edits() -> None:
    cleaned = run_clean(parsed(("paragraph", "Already clean.")), [])
    assert cleaned.edits == []
    assert cleaned.text == "Already clean."


def test_edit_helpers() -> None:
    assert apply_edits("a  b", collapse_edits("a  b")) == "a b"
    assert apply_edits("  a \n b  ", collapse_edits("  a \n b  ")) == "a b"
    assert apply_edits(" a  \n b ", trim_edits(" a  \n b ")) == "a\n b"
    assert apply_edits("abc", []) == "abc"


def test_the_clean_components_are_registered() -> None:
    registry = default_registry()
    assert {"clean/whitespace", "clean/unicode", "clean/dehyphenate"} <= {
        c.info.id for c in registry.for_stage("clean")
    }
    assert registry.get("clean/whitespace").info.baseline is True
