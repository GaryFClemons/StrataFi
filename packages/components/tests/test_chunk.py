"""Chunkers: ``chunk/sections``, the default, on the format fixtures and on any
document, and ``chunk/fixed-tokens``.

What ``chunk/sections`` promises, checked below on generated documents: every chunk's text is
the document's own text across its span (but for a table's later row groups, which repeat the
header row), every word of the document is in a chunk, no chunk crosses a top-level section,
no chunk passes ``max_tokens`` (the generated headings and table rows are short, the one case
the chunker can't help), and an overlap only repeats the end of the chunk before. A fixed-tokens
chunk by ``chars/4`` passes ``size`` only when it is one word longer than the whole window.
"""

import hashlib
import re
from itertools import pairwise
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stratafi_components import default_registry
from stratafi_components.chunk import (
    EnrichNone,
    FixedTokens,
    body_elements,
    label_title,
    render_index_text,
    tiktoken_encoding,
)
from stratafi_components.chunkers import Sections
from stratafi_components.clean import Whitespace, run_clean
from stratafi_components.detect import detect
from stratafi_components.estimators import chars_estimate
from stratafi_components.extract import raw_document
from stratafi_core.artifacts import Chunk, ParsedDocument, build_text
from stratafi_core.components import ComponentError
from stratafi_core.hashing import content_hash

ROOT = Path(__file__).resolve().parents[3]
FORMATS = ROOT / "fixtures" / "formats"
KEY = "a" * 64
REGISTRY = default_registry()


def cleaned(path: Path, parser_id: str = "parse/markdown") -> ParsedDocument:
    data = path.read_bytes()
    raw = raw_document("d-1", hashlib.sha256(data).hexdigest(), len(data), detect(path, path.name))
    parser = REGISTRY.get(parser_id)()
    parsed = parser.parse_local(path, raw, parser.resolve_params({}))
    return run_clean(parsed, [("clean/whitespace", Whitespace(), Whitespace.resolve_params({}))])


def sections(document: ParsedDocument, label: str | None = None, **params) -> list[Chunk]:
    component = Sections()
    return component.chunk(document, component.resolve_params(params), cache_key=KEY, label=label)


def document_of(*elements: tuple) -> ParsedDocument:
    """A parsed document from (type, text[, level]) tuples."""
    text, offsets = build_text([element[1] for element in elements])
    records = []
    for index, (element, (start, end)) in enumerate(zip(elements, offsets, strict=True)):
        kind, body, *rest = element
        record = {"id": f"e{index}", "type": kind, "text": body, "char_start": start}
        record["char_end"] = end
        if rest:
            record["level"] = rest[0]
        records.append(record)
    return ParsedDocument(
        document_id="d-1",
        parser={"id": "parse/test", "version": "1.0.0", "params_hash": content_hash({})},
        media_type="text/markdown",
        elements=records,
        text=text,
    )


def top_section(document: ParsedDocument, position: int) -> str | None:
    """The top-level heading in force at ``position``."""
    headings = [e for e in document.elements if e.type == "heading"]
    top = min((e.level or 1 for e in headings), default=None)
    found = None
    for heading in headings:
        if heading.char_start > position:
            break
        if (heading.level or 1) == top:
            found = heading.id
    return found


# --- chunk/sections on the format fixtures -------------------------------------------------------


def test_sections_follow_the_documents_headings() -> None:
    document = cleaned(FORMATS / "tips.md")
    chunks = sections(document, label="tips")
    sections_of = [chunk.metadata.get("section") for chunk in chunks]
    # The version line comes before any section; then one chunk per top-level section, the two
    # short subsections of "3 Cash tips" sharing one.
    assert sections_of == [
        None,
        "1 Who shares in tips",
        "2 Card and app tips",
        "3.1 Counting the tip jar",
        "4 Dividing the pool",
        "5 Catering orders",
        "6 Questions and corrections",
    ]
    cash = chunks[3]
    assert cash.heading_path == ["3 Cash tips", "3.1 Counting the tip jar"]
    assert cash.text.startswith("3 Cash tips\n\n3.1 Counting the tip jar\n\n")
    assert "3.2 Paying out cash tips" in cash.text
    for chunk in chunks:
        assert chunk.text == document.text[chunk.char_start : chunk.char_end]
        assert chunk.tokens["chars/4"] <= 400
        assert chunk.index_text.startswith("Tip Policy\n")
    assert chunks[1].index_text.startswith("Tip Policy\n1 Who shares in tips\n1 Who shares")


def test_a_table_stays_whole_when_it_fits_and_splits_by_rows_under_its_header() -> None:
    document = cleaned(FORMATS / "wholesale.md")
    chunks = sections(document)
    [tiers] = [chunk for chunk in chunks if chunk.metadata.get("section") == "2 Price tiers"]
    assert "Tier 4: Flagship | 400 lb and up | $9.10 | $11.60 | 6" in tiers.text
    table = next(e for e in document.elements if e.type == "table")
    assert table.id in tiers.element_ids
    small = sections(document, max_tokens=64, min_tokens=0)
    groups = [chunk for chunk in small if table.id in chunk.element_ids]
    assert len(groups) > 1
    header = table.text.split("\n")[0]
    assert groups[1].metadata["header_repeated"] is True
    for group in groups[1:]:
        assert group.text.startswith(header + "\n")
        assert group.tokens["chars/4"] <= 64
    rows = [row for row in table.text.split("\n")[1:]]
    for row in rows:
        assert sum(row in group.text for group in groups) == 1, row


def test_a_long_paragraph_splits_at_sentences_and_overlap_repeats_whole_sentences() -> None:
    sentences = [f"Sentence {n} says the espresso is pulled for 28 seconds." for n in range(40)]
    document = document_of(
        ("heading", "1 Recipes", 2), ("paragraph", " ".join(sentences)), ("heading", "2 Milk", 2)
    )
    chunks = sections(document, max_tokens=100, min_tokens=0, overlap=30)
    body = [chunk for chunk in chunks if chunk.metadata.get("section") == "1 Recipes"]
    assert len(body) > 3
    for chunk in body:
        assert chunk.tokens["chars/4"] <= 100
        assert chunk.text == document.text[chunk.char_start : chunk.char_end]
        # Every piece ends at a sentence end.
        assert chunk.text.endswith(".")
    for previous, chunk in pairwise(body):
        shared = document.text[chunk.char_start : previous.char_end]
        assert shared
        assert previous.text.endswith(shared)
        assert chars_estimate(shared) <= 30
        assert re.match(r"Sentence \d+", shared)
    # "2 Milk" has no content: its own chunk, never part of section 1's.
    assert chunks[-1].text == "2 Milk"
    assert all("2 Milk" not in chunk.text for chunk in body)


def test_short_subsections_share_a_chunk_until_min_tokens() -> None:
    document = document_of(
        ("heading", "3 Cash", 2),
        ("heading", "3.1 Counting", 3),
        ("paragraph", "Two people count the tip jar at close."),
        ("heading", "3.2 Paying out", 3),
        ("paragraph", "Shares are paid in cash every Tuesday."),
        ("heading", "4 Pool", 2),
        ("paragraph", "Shares follow hours worked."),
    )
    merged = sections(document, min_tokens=200)
    assert [chunk.heading_path for chunk in merged] == [
        ["3 Cash", "3.1 Counting"],
        ["4 Pool"],
    ]
    split = sections(document, min_tokens=5)
    assert [chunk.heading_path for chunk in split] == [
        ["3 Cash", "3.1 Counting"],
        ["3 Cash", "3.2 Paying out"],
        ["4 Pool"],
    ]
    assert split[1].text.startswith("3.2 Paying out\n\n")


def test_pdf_chunks_carry_pages() -> None:
    if reason := REGISTRY.get("parse/pdfplumber").unavailable():
        pytest.skip(f"parse/pdfplumber {reason}")
    document = cleaned(FORMATS / "food-safety.pdf", "parse/pdfplumber")
    chunks = sections(document, label="food-safety")
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)
    by_id = {element.id: element for element in document.elements}
    for chunk in chunks:
        assert all(by_id[i].type != "title" for i in chunk.element_ids)
        pages = {by_id[i].page for i in chunk.element_ids}
        assert (min(pages), max(pages)) == (chunk.page_start, chunk.page_end)


def test_empty_and_oversized_documents_are_refused() -> None:
    title_only = document_of(("title", "Only a title"))
    with pytest.raises(ComponentError) as raised:
        sections(title_only)
    assert raised.value.code == "empty_document"
    with pytest.raises(ComponentError) as raised:
        sections(cleaned(FORMATS / "food-safety.md"), max_chunks=2)
    assert raised.value.code == "too_many_chunks"


def test_chunk_ids_hash_the_document_the_chunker_key_and_the_ordinal() -> None:
    document = cleaned(FORMATS / "food-safety.md")
    first = sections(document)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in sections(document)]
    other = Sections().chunk(document, Sections.resolve_params({}), cache_key="b" * 64)
    assert first[0].chunk_id != other[0].chunk_id
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_sections_params_are_checked() -> None:
    with pytest.raises(ValueError, match="min_tokens"):
        Sections.resolve_params({"max_tokens": 100, "min_tokens": 200})
    with pytest.raises(ValueError, match="overlap"):
        Sections.resolve_params({"max_tokens": 100, "overlap": 60})
    params = Sections.resolve_params({"token_estimator": "tiktoken:cl100k_base"})
    assert "tiktoken" in Sections.static_env(params)


# --- chunk/sections on any document ------------------------------------------------------------

WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789$%,'", min_size=1, max_size=12)
LONG_WORD = st.text(alphabet="abcdefghij", min_size=200, max_size=900)
SPACE = st.sampled_from([" ", " ", " ", "  ", "\n", " \n "])
END = st.sampled_from([".", ".", "!", "?", "", ";"])


@st.composite
def sentences(draw) -> str:
    words = draw(st.lists(st.one_of(WORD, WORD, WORD, WORD, LONG_WORD), min_size=1, max_size=40))
    parts = [words[0]]
    for word in words[1:]:
        parts += [draw(SPACE), word]
    return "".join(parts) + draw(END)


@st.composite
def paragraphs(draw) -> str:
    items = draw(st.lists(sentences(), min_size=1, max_size=12))
    parts = [items[0]]
    for item in items[1:]:
        parts += [draw(st.sampled_from([" ", "\n", "  "])), item]
    return "".join(parts)


@st.composite
def tables(draw) -> str:
    cell = st.lists(WORD, min_size=1, max_size=2).map(" ".join)
    columns = draw(st.integers(1, 3))
    rows = draw(
        st.lists(st.lists(cell, min_size=columns, max_size=columns), min_size=2, max_size=30)
    )
    return "\n".join(" | ".join(row) for row in rows)


@st.composite
def documents(draw) -> ParsedDocument:
    """Sections: each heading (levels 1 to 3) followed by paragraphs or tables."""
    elements: list[tuple] = []
    if draw(st.booleans()):
        elements.append(("paragraph", draw(paragraphs())))
    for _ in range(draw(st.integers(1, 8))):
        words = draw(st.lists(WORD, min_size=1, max_size=5))
        elements.append(("heading", " ".join(words), draw(st.integers(1, 3))))
        for _ in range(draw(st.integers(1, 3))):
            if draw(st.integers(0, 4)) == 0:
                elements.append(("table", draw(tables())))
            else:
                elements.append(("paragraph", draw(paragraphs())))
    return document_of(*elements)


PARAMS = st.builds(
    lambda max_tokens, min_share, overlap_share: (
        max_tokens,
        int(max_tokens * min_share),
        int(max_tokens * overlap_share),
    ),
    st.integers(96, 400),
    st.floats(0, 1),
    st.floats(0, 0.5),
)
PROPERTY_SETTINGS = settings(
    max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@PROPERTY_SETTINGS
@given(documents(), PARAMS)
def test_sections_cover_the_document_within_their_limits(document: ParsedDocument, params) -> None:
    max_tokens, min_tokens, overlap = params
    chunks = sections(document, max_tokens=max_tokens, min_tokens=min_tokens, overlap=overlap)
    ids = {element.id for element in document.elements}
    covered = [False] * len(document.text)
    for ordinal, chunk in enumerate(chunks):
        assert chunk.ordinal == ordinal
        assert 0 <= chunk.char_start < chunk.char_end <= len(document.text)
        assert set(chunk.element_ids) <= ids
        assert chunk.index_text.endswith(chunk.text)
        assert chunk.tokens["chars/4"] <= max_tokens
        if not chunk.metadata.get("header_repeated"):
            assert chunk.text == document.text[chunk.char_start : chunk.char_end]
        covered[chunk.char_start : chunk.char_end] = [True] * (chunk.char_end - chunk.char_start)
        # Never across a top-level section.
        starts = [e.char_start for e in document.elements if e.id in chunk.element_ids]
        assert len({top_section(document, start) for start in starts}) == 1
    for element in body_elements(document):
        for word in re.finditer(r"\S+", element.text):
            start = element.char_start + word.start()
            end = element.char_start + word.end()
            assert all(covered[start:end]), "a word is outside every chunk"


@PROPERTY_SETTINGS
@given(documents(), PARAMS)
def test_overlap_only_repeats_the_end_of_the_chunk_before(document: ParsedDocument, params) -> None:
    max_tokens, min_tokens, overlap = params
    chunks = sections(document, max_tokens=max_tokens, min_tokens=min_tokens, overlap=overlap)
    for previous, chunk in pairwise(chunks):
        assert chunk.char_start >= previous.char_start
        if chunk.char_start < previous.char_end:
            shared = document.text[chunk.char_start : previous.char_end]
            assert overlap > 0
            assert previous.text.endswith(shared)
            assert chars_estimate(shared) <= overlap
            assert chunk.metadata.get("section") == previous.metadata.get("section")


# --- index text --------------------------------------------------------------------------------


def test_index_text_drops_empty_lines() -> None:
    template = "{title}\n{heading_path}\n{text}"
    assert (
        render_index_text(template, {"title": "T", "heading_path": "", "text": "x\ny"}) == "T\nx\ny"
    )
    assert (
        render_index_text("Section: {section}\n{text}", {"section": None, "text": "b"})
        == "Section: \nb"
    )
    assert render_index_text("{context}\n{text}", {"context": "c", "text": "t"}) == "c\nt"
    assert label_title("food-safety") == "Food safety"


def test_an_untitled_document_is_titled_by_its_label() -> None:
    document = document_of(("paragraph", "Shares follow hours worked."))
    [chunk] = sections(document, label="tip-pool")
    assert chunk.index_text == "Tip pool\nShares follow hours worked."


# --- chunk/fixed-tokens -------------------------------------------------------------------------


@st.composite
def flat_documents(draw) -> ParsedDocument:
    elements: list[tuple] = []
    for _ in range(draw(st.integers(1, 12))):
        if draw(st.booleans()):
            words = draw(st.lists(WORD, min_size=1, max_size=8))
            elements.append(("heading", " ".join(words), draw(st.integers(1, 3))))
        else:
            elements.append(("paragraph", draw(paragraphs())))
    if not any(kind == "paragraph" for kind, *_ in elements):
        elements.append(("paragraph", draw(paragraphs())))
    return document_of(*elements)


@st.composite
def fixed_params(draw) -> tuple[int, int]:
    size = draw(st.integers(8, 300))
    return size, draw(st.integers(0, size - 1))


@PROPERTY_SETTINGS
@given(flat_documents(), fixed_params())
def test_fixed_windows_cover_the_text_and_never_cut_a_word(
    document: ParsedDocument, params
) -> None:
    size, overlap = params
    component = FixedTokens()
    chunks = component.chunk(
        document, component.resolve_params({"size": size, "overlap": overlap}), cache_key=KEY
    )
    body = body_elements(document)
    start = body[0].char_start
    words = list(re.finditer(r"\S+", document.text[start:]))
    covered = [False] * len(document.text)
    for chunk in chunks:
        assert document.text[chunk.char_start : chunk.char_end] == chunk.text
        covered[chunk.char_start : chunk.char_end] = [True] * len(chunk.text)
        single_word = re.fullmatch(r"\S+", chunk.text) is not None
        assert chunk.tokens["chars/4"] <= size or single_word
        assert not chunk.text[0].isspace()
        assert not chunk.text[-1].isspace()
    for word in words:
        assert all(covered[start + word.start() : start + word.end()])
    for previous, chunk in pairwise(chunks):
        assert chunk.char_start > previous.char_start, "windows always move on"
        shared = max(0, previous.char_end - chunk.char_start)
        assert shared <= max(overlap * 4, 0) or chunk.char_start == previous.char_end


def test_fixed_windows_by_tiktoken_count_exactly() -> None:
    try:
        encoding = tiktoken_encoding("cl100k_base")
    except ComponentError:
        pytest.skip("the cl100k_base encoding can't load here (offline)")
    document = cleaned(FORMATS / "food-safety.md")
    component = FixedTokens()
    params = component.resolve_params(
        {"size": 120, "overlap": 20, "token_estimator": "tiktoken:cl100k_base"}
    )
    chunks = component.chunk(document, params, cache_key=KEY)
    assert len(chunks) > 3
    for chunk in chunks[:-1]:
        assert chunk.tokens["tiktoken:cl100k_base"] == 120
        assert len(encoding.encode(chunk.text)) in range(118, 123)  # re-encoding a slice can shift
    assert chunks[-1].tokens["tiktoken:cl100k_base"] <= 120
    assert "tiktoken" in FixedTokens.static_env(params)


def test_fixed_windows_follow_headings() -> None:
    document = cleaned(FORMATS / "food-safety.md")
    chunks = FixedTokens().chunk(
        document, FixedTokens.resolve_params({"size": 64, "overlap": 8}), cache_key=KEY
    )
    assert chunks[0].heading_path == []
    assert any(chunk.heading_path == ["5 Allergens", "5.1 What we track"] for chunk in chunks)


# --- enrich/none and the registry --------------------------------------------------------------


def test_enrich_none_passes_chunks_through_and_everything_is_registered() -> None:
    chunks = sections(cleaned(FORMATS / "wholesale.md"))
    assert EnrichNone().enrich(chunks, EnrichNone.resolve_params({})) == chunks
    for component_id in ("chunk/sections", "chunk/fixed-tokens", "enrich/none"):
        assert component_id in REGISTRY
    schema = REGISTRY.describe("chunk/sections").params_schema["properties"]
    assert (schema["max_tokens"]["default"], schema["min_tokens"]["default"]) == (400, 200)
    assert schema["overlap"]["default"] == 0
