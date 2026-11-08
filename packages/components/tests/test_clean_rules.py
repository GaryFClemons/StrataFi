"""The cleaning rules on fixtures: redaction of what the PII detector finds, repeated
lines, boilerplate, near duplicates across a corpus, language, and minimum content; the diff;
and the redaction canary (a redacted value appears in no artifact, log or response). Every
value that looks like personal data here is synthetic."""

import json
import logging

import pytest
from pydantic import ValidationError

from stratafi_components.chunkers import Sections
from stratafi_components.clean import (
    MASK,
    NEAR_DUPLICATE,
    Boilerplate,
    Corpus,
    CorpusDocument,
    Language,
    MinContent,
    NearDuplicates,
    Redact,
    RepeatedLines,
    Whitespace,
    diff_hunks,
    replay,
    run_clean,
)
from stratafi_components.index import Bm25Index, IndexInput
from stratafi_core.artifacts import ParsedDocument, build_text
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.hashing import content_hash
from stratafi_testing.canary import CanaryCheck

SSN = "123-45-6789"
OTHER_SSN = "234-56-7890"
CARD = "4111 1111 1111 1111"


def parsed(*elements: tuple[str, str] | tuple[str, str, int]) -> ParsedDocument:
    texts = [element[1] for element in elements]
    text, offsets = build_text(texts)
    return ParsedDocument(
        document_id="d-1",
        parser={"id": "parse/test", "version": "1.0.0", "params_hash": content_hash({})},
        title="Policy",
        media_type="application/pdf",
        elements=[
            {
                "id": f"e{index}",
                "type": element[0],
                "text": element[1],
                "page": element[2] if len(element) > 2 else None,
                "char_start": start,
                "char_end": end,
            }
            for index, (element, (start, end)) in enumerate(zip(elements, offsets, strict=True))
        ],
        text=text,
    )


def rules(*specs: tuple[type, dict]) -> list:
    return [(cls.info.id, cls(), cls.resolve_params(params)) for cls, params in specs]


def context(**services) -> StageContext:
    return StageContext(workspace_id="ws-1", services=services)


# --- clean/redact ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("placeholder", "Test SSN [SSN] is on file; call [SSN] back."),
        ("mask", f"Test SSN {MASK * 11} is on file; call {MASK * 11} back."),
    ],
)
def test_redaction_replaces_every_span(mode: str, expected: str) -> None:
    document = parsed(
        ("heading", "Accounts"),
        ("paragraph", f"Test SSN {SSN} is on file; call {OTHER_SSN} back."),
    )
    cleaned = run_clean(document, rules((Redact, {"mode": mode})), context())
    assert cleaned.elements[1].text == expected
    assert all(edit.rule == "clean/redact" for edit in cleaned.edits)
    assert [edit.replacement_kind for edit in cleaned.edits] == [mode, mode]
    # Replaying the edits rebuilds the cleaned text without the values.
    assert replay(document, cleaned.edits)[-1][1] == cleaned.text


def test_keyed_hashes_link_equal_values_without_showing_them() -> None:
    document = parsed(("paragraph", f"A {SSN}, B {SSN}, C {OTHER_SSN}."))
    services = {"redaction_key": b"synthetic-test-key"}
    cleaned = run_clean(document, rules((Redact, {"mode": "hash"})), context(**services))
    labels = [edit.label for edit in cleaned.edits]
    assert labels[0] == labels[1] != labels[2]
    assert all(label.startswith("[SSN:") and len(label) == 14 for label in labels)
    other_key = run_clean(
        document, rules((Redact, {"mode": "hash"})), context(redaction_key=b"another-key")
    )
    assert other_key.edits[0].label != labels[0]
    with pytest.raises(ComponentError) as error:
        run_clean(document, rules((Redact, {"mode": "hash"})), context())
    assert error.value.code == "redaction_key_not_configured"


def test_redaction_redacts_only_its_categories() -> None:
    document = parsed(("paragraph", f"SSN {SSN}, card {CARD}, desk help@example.com."))
    cards = run_clean(document, rules((Redact, {"categories": ["card_number"]})), context())
    assert cards.text == f"SSN {SSN}, card [CARD_NUMBER], desk help@example.com."
    every = run_clean(document, rules((Redact, {"categories": ["email", "ssn"]})), context())
    assert every.text == "SSN [SSN], card " + CARD + ", desk [EMAIL]."
    # The defaults are the high-risk identifiers: the email stays.
    defaults = run_clean(document, rules((Redact, {})), StageContext())
    assert defaults.text == "SSN [SSN], card [CARD_NUMBER], desk help@example.com."
    with pytest.raises(ValidationError):
        Redact.resolve_params({"categories": ["passport"]})


def test_redaction_scans_each_element_on_its_own() -> None:
    # A value split across two elements is two halves, neither of them a value.
    document = parsed(("paragraph", "SSN 123-45-"), ("paragraph", "6789 and 234-56-7890"))
    cleaned = run_clean(document, rules((Redact, {})), context())
    assert [element.text for element in cleaned.elements] == [
        "SSN 123-45-",
        "6789 and [SSN]",
    ]


# --- the canary --------------------------------------------------------------------------------


def test_a_redacted_value_appears_in_no_artifact_log_or_response(canary: CanaryCheck) -> None:
    """The canary: once clean/redact has run, the value is in no artifact made from the
    cleaned document (cleaned JSON, chunks, the keyword index), no diff the API returns, and
    no log line."""
    value = SSN
    document = parsed(
        ("heading", "Test records"),
        ("paragraph", f"The test record lists SSN {value}. Keep it private."),
        ("paragraph", f"   Repeat: {value}   "),
    )
    logging.getLogger("stratafi").debug("cleaning %s", document.document_id)
    cleaned = run_clean(
        document, rules((Whitespace, {"paragraphs": "collapse"}), (Redact, {})), context()
    )
    cleaned_json = cleaned.model_dump_json(by_alias=True)
    chunks = Sections().chunk(cleaned, Sections.resolve_params({}), cache_key="k" * 64)
    chunks_jsonl = "\n".join(chunk.model_dump_json() for chunk in chunks)
    built = Bm25Index().build(
        [IndexInput("c" * 64, chunks)], Bm25Index.resolve_params({}), StageContext()
    )
    hunks, total = diff_hunks(document, list(cleaned.edits or []), context_chars=200)
    response = json.dumps([hunk.__dict__ for hunk in hunks])
    assert total == len(cleaned.edits or []) > 0
    assert any(hunk.redaction for hunk in hunks)
    for text in (cleaned_json, chunks_jsonl, built.payload.decode(), response):
        assert value not in text
        # Nor any long piece of it: a context window never cuts a value in half.
        assert value[:7] not in text
    canary.value = value
    canary.assert_not_logged()


def test_the_diff_masks_values_redacted_by_a_later_rule() -> None:
    document = parsed(("paragraph", f"Call  back {SSN}  today."))
    cleaned = run_clean(
        document, rules((Whitespace, {"paragraphs": "collapse"}), (Redact, {})), context()
    )
    hunks, _ = diff_hunks(document, list(cleaned.edits or []))
    whitespace = [hunk for hunk in hunks if hunk.rule == "clean/whitespace"]
    assert whitespace
    assert all(SSN not in hunk.before and SSN not in hunk.after for hunk in hunks)
    redaction = next(hunk for hunk in hunks if hunk.redaction)
    assert "[SSN]" in redaction.after
    assert MASK * len(SSN) in redaction.before


# --- clean/repeated-lines ------------------------------------------------------------------


def test_running_headers_and_page_numbers_are_dropped() -> None:
    topics = ["tills", "deposits", "refunds", "gift cards", "cafés"]
    pages = []
    for page, topic in enumerate(topics, start=1):
        pages += [
            ("paragraph", "Halcyon Coffee Co.\nCash Handling Standard", page),
            ("paragraph", f"This section covers {topic} and the rules staff follow.", page),
            ("paragraph", f"Page {page} of 5", page),
        ]
    pages.append(("page_footer", "Confidential", 5))
    document = parsed(*pages)
    cleaned = run_clean(document, rules((RepeatedLines, {})))
    texts = [element.text for element in cleaned.elements]
    assert texts[0] == ""
    assert texts[2] == ""
    assert texts[-1] == ""  # page furniture
    for index, topic in zip(range(1, 15, 3), topics, strict=True):
        assert texts[index] == f"This section covers {topic} and the rules staff follow."


def test_lines_repeating_on_few_pages_stay() -> None:
    document = parsed(
        ("paragraph", "Shared line", 1),
        ("paragraph", "Shared line", 2),
        ("paragraph", "Unique", 3),
        ("paragraph", "Other", 4),
    )
    assert run_clean(document, rules((RepeatedLines, {}))).text == document.text


# --- clean/boilerplate ---------------------------------------------------------------------


def test_navigation_and_footers_are_dropped() -> None:
    document = parsed(
        ("paragraph", "Home | Menu | Locations | Contact us"),
        ("paragraph", "Skip to main content"),
        ("heading", "Privacy policy"),
        ("paragraph", "Orders placed after 3 pm ship the next business day under this policy."),
        ("paragraph", "© 2026 Halcyon Coffee Co. All rights reserved."),
        ("paragraph", "We use cookies to improve this site."),
        ("page_footer", "Footer"),
    )
    cleaned = run_clean(document, rules((Boilerplate, {})))
    assert [bool(e.text) for e in cleaned.elements] == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    custom = run_clean(
        parsed(("paragraph", "Price sheet updated weekly")),
        rules((Boilerplate, {"phrases": ["updated weekly"]})),
    )
    assert custom.text == ""


# --- clean/near-duplicates -----------------------------------------------------------------

BASE = " ".join(
    f"Rule {i}: managers confirm refund requests by calling the number on file." for i in range(40)
)


def corpus_of(own_rank: list[str], others: dict[str, tuple[list[str], str]]) -> Corpus:
    return Corpus(
        key="own",
        rank=tuple(own_rank),
        others=[
            CorpusDocument(key, tuple(rank), lambda text=text: text)
            for key, (rank, text) in others.items()
        ],
    )


def test_an_older_near_duplicate_is_dropped_and_the_newest_kept() -> None:
    document = parsed(("paragraph", BASE))
    newer = BASE.replace("Rule 39", "Rule 39 (revised)")
    unrelated = " ".join(f"Café {i} opens at seven and closes at five." for i in range(40))
    older = corpus_of(
        ["2025-01-01", "", "a"],
        {"new": (["2026-01-01", "", "b"], newer), "x": (["", "", "c"], unrelated)},
    )
    dropped = run_clean(document, rules((NearDuplicates, {})), corpus=older)
    assert dropped.text.strip() == ""
    assert NEAR_DUPLICATE in dropped.warnings
    newest = corpus_of(["2026-06-01", "", "a"], {"old": (["2026-01-01", "", "b"], newer)})
    kept = run_clean(document, rules((NearDuplicates, {})), corpus=newest)
    assert kept.text == document.text
    assert NEAR_DUPLICATE not in kept.warnings


def test_without_a_corpus_near_duplicates_does_nothing() -> None:
    document = parsed(("paragraph", BASE))
    assert run_clean(document, rules((NearDuplicates, {}))).text == document.text


def test_minhash_estimates_similarity_and_lsh_finds_candidates() -> None:
    from stratafi_components import minhash

    first = minhash.signature(BASE)
    assert minhash.similarity(first, minhash.signature(BASE)) == 1.0
    edited = minhash.signature(BASE.replace("Rule 3:", "Rule three:"))
    assert 0.8 < minhash.similarity(first, edited) < 1.0
    different = minhash.signature("Completely different words about café hours and holidays.")
    assert minhash.similarity(first, different) < 0.2
    found = minhash.near_duplicates({"a": first, "b": edited, "c": different}, "a", threshold=0.8)
    assert [key for key, _ in found] == ["b"]
    bands, rows = minhash.bands_for(128, 0.85)
    assert bands * rows == 128
    assert abs((1 / bands) ** (1 / rows) - 0.85) < 0.05


# --- clean/language and clean/min-content ---------------------------------------------------

ENGLISH = (
    "The manager must confirm each refund by calling the number on file before any money is sent."
)
SPANISH = "El gerente debe confirmar la devolución llamando al número que consta en el expediente."


def test_language_tags_and_filters() -> None:
    document = parsed(
        ("heading", "Refunds"),
        ("paragraph", ENGLISH),
        ("paragraph", SPANISH),
        ("paragraph", ENGLISH),
    )
    tagged = run_clean(document, rules((Language, {})))
    assert tagged.language == "en"
    assert tagged.text == document.text
    kept = run_clean(document, rules((Language, {"keep": ["en"]})))
    assert [bool(e.text) for e in kept.elements] == [True, True, False, True]
    spanish = run_clean(parsed(("paragraph", SPANISH)), rules((Language, {})))
    assert spanish.language == "es"


def test_small_elements_are_dropped_except_kept_types() -> None:
    document = parsed(
        ("heading", "Prices"),
        ("paragraph", "3"),
        ("paragraph", "See below."),
        ("list_item", "Latte"),
        ("paragraph", "A large latte costs five dollars at every café."),
    )
    cleaned = run_clean(document, rules((MinContent, {})))
    assert [bool(e.text) for e in cleaned.elements] == [True, False, False, True, True]
    loose = run_clean(document, rules((MinContent, {"min_chars": 0, "min_words": 0})))
    assert loose.text == document.text
