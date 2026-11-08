"""Relevance mapping: normalization, the three match rules, coverage that is robust
to parser differences, and findability."""

import re
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from stratafi_core.evalsets import EvidenceRef
from stratafi_core.relevance import (
    DOC,
    EVIDENCE_COVERAGE,
    KNOWLEDGE_HIT,
    Matcher,
    Passage,
    coverage,
    findability,
    highlights,
    lcs_length,
    normalize,
    tokens,
)

DEMO = Path(__file__).resolve().parents[3] / "fixtures" / "demo-corpus"


def test_normalization() -> None:
    assert normalize("  The ﬁrst  ACCOUNT,\n\topened. ") == "the first account, opened"
    assert normalize("a new ac-\ncount is") == "a new account is"
    assert normalize("co­operative") == "cooperative"
    assert normalize("“While an account is new:”") == "while an account is new"
    quoted = "$7,000 (section 5) is \u2018new\u2019."
    assert tokens(quoted) == ["7,000", "section", "5", "is", "new"]
    assert tokens("rate-and-term refinance | 97%") == ["rate-and-term", "refinance", "97"]


def lcs_reference(a: list[str], b: list[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            table[i + 1][j + 1] = (
                table[i][j] + 1 if a[i] == b[j] else max(table[i][j + 1], table[i + 1][j])
            )
    return table[-1][-1]


@settings(max_examples=200, deadline=None)
@given(
    a=st.lists(st.sampled_from("abcde"), max_size=40),
    b=st.lists(st.sampled_from("abcdef"), max_size=80),
)
def test_the_bit_parallel_lcs_is_the_lcs(a: list[str], b: list[str]) -> None:
    assert lcs_length(a, b) == lcs_reference(a, b)


def test_coverage_counts_tokens_in_order_within_a_window() -> None:
    evidence = tokens("An account is a new account during the first 30 calendar days")
    exact = tokens("Rules. An account is a new account during the first 30 calendar days after.")
    assert coverage(evidence, exact) == 1.0
    reordered = tokens("During the first 30 calendar days an account is a new account")
    assert 0.4 < coverage(evidence, reordered) < 1.0
    # The same words, scattered across a long text, don't add up to a match.
    filler = ["lorem"] * 40
    scattered = [word for token in evidence for word in (token, *filler)]
    assert coverage(evidence, scattered) < 0.3
    assert coverage([], exact) == 0.0
    assert coverage(evidence, []) == 0.0


# --- robustness: whitespace, hyphenation, ligatures ---------------------------------------

LIGATURES = {"ffi": "ﬃ", "ffl": "ﬄ", "ff": "ﬀ", "fi": "ﬁ", "fl": "ﬂ"}
WORDS = st.text(alphabet="abcdefilmnorstu", min_size=2, max_size=12)


def ligate(text: str) -> str:
    return re.sub("|".join(LIGATURES), lambda match: LIGATURES[match.group()], text)


@settings(max_examples=150, deadline=None)
@given(
    words=st.lists(WORDS, min_size=3, max_size=25),
    before=st.lists(WORDS, max_size=30),
    after=st.lists(WORDS, max_size=30),
    gaps=st.lists(st.sampled_from([" ", "  ", "\n", "\t", " \n ", "\r\n"]), min_size=1),
    hyphenate=st.lists(st.integers(min_value=0, max_value=1000), max_size=5),
    ligatures=st.booleans(),
)
def test_coverage_is_robust_to_whitespace_hyphenation_and_ligatures(
    words: list[str],
    before: list[str],
    after: list[str],
    gaps: list[str],
    hyphenate: list[int],
    ligatures: bool,
) -> None:
    """Coverage stays whole whatever whitespace, hyphenated line breaks and ligatures a
    parser leaves in the chunk."""
    evidence = " ".join(words)
    parts = list(words)
    for position in hyphenate:
        index = position % len(parts)
        word = parts[index]
        if len(word) >= 4 and "-" not in word:
            cut = 1 + position % (len(word) - 2)
            parts[index] = f"{word[:cut]}-\n{word[cut:]}"
    body = "".join(part + gaps[n % len(gaps)] for n, part in enumerate(parts))
    if ligatures:
        body = ligate(body)
    chunk = " ".join([*before, body.strip(), *after])
    assert coverage(tokens(evidence), tokens(chunk)) == 1.0
    assert normalize(evidence) in normalize(chunk)


# --- the match rules ----------------------------------------------------------------------

EVIDENCE = EvidenceRef(
    doc_key="pto-leave",
    version="3",
    section="2.1 How PTO accrues",
    evidence="Hourly staff accrue 1 hour of PTO for every 30 hours worked, from their first day",
)


def stated_rule(passage: Passage, ref: EvidenceRef) -> bool:
    """The hit rule as the demo set's retrieval.yaml states it, written out plainly: a result
    from an expected doc_key and version whose section matches, or whose text contains the
    evidence phrase (compared without case, punctuation at the ends, or runs of whitespace)."""

    def plain(text: str) -> str:
        return " ".join(text.lower().split()).strip(".,;:!?\"'()")

    if passage.doc_key != ref.doc_key or passage.version != ref.version:
        return False
    section = passage.section is not None and plain(passage.section) == plain(ref.section or "")
    return section or (ref.evidence is not None and plain(ref.evidence) in plain(passage.text))


CRAFTED = [
    # (doc_key, version, section, text)
    ("pto-leave", "3", "2.1 How PTO accrues", "Unrelated words."),
    ("pto-leave", "3", "2.1 how pto ACCRUES", "Unrelated words."),
    ("pto-leave", "3", "2.2 Balance cap", EVIDENCE.evidence),
    ("pto-leave", "3", None, f"Intro.\n{EVIDENCE.evidence}. More rules."),
    ("pto-leave", "3", None, EVIDENCE.evidence.upper().replace(" ", "  ")),
    ("pto-leave", "3", "3 Paid sick time", "Hourly staff accrue PTO as they work."),
    ("pto-leave", "2", "2.1 How PTO accrues", EVIDENCE.evidence),
    ("scheduling", "3", "2.1 How PTO accrues", EVIDENCE.evidence),
    ("pto-leave", "3", "2.1 How PTO accrues.", "x"),
    ("pto-leave", None, None, EVIDENCE.evidence),
]


@pytest.mark.parametrize("case", CRAFTED, ids=[str(n) for n in range(len(CRAFTED))])
def test_knowledge_hit_agrees_with_the_stated_rule(case: tuple) -> None:
    """knowledge-hit-v1 agrees with the retrieval format's stated rule on crafted cases."""
    doc_key, version, section, text = case
    passage = Passage("c1", doc_key, version, section, text)
    judgement = Matcher([EVIDENCE]).judge(passage)
    assert judgement.relevant(KNOWLEDGE_HIT) == stated_rule(passage, EVIDENCE)


def test_the_three_rules_side_by_side() -> None:
    other = EvidenceRef(
        doc_key="pto-leave",
        version="3",
        section="2.2 Balance cap",
        evidence="An hourly employee's PTO balance can hold up to 80 hours",
    )
    matcher = Matcher([EVIDENCE, other], threshold=0.7)
    same_doc = matcher.judge(Passage("c1", "pto-leave", "3", "4 Holidays", "Cafés close."))
    assert same_doc.covered == {KNOWLEDGE_HIT: frozenset(), EVIDENCE_COVERAGE: frozenset(),
                                DOC: frozenset({0, 1})}  # fmt: skip
    partial = matcher.judge(
        Passage(
            "c2",
            "pto-leave",
            "3",
            "Other",
            "Hourly staff accrue 1 hour of PTO for every 30 hours they work, from the first day.",
        )
    )
    assert partial.covered[KNOWLEDGE_HIT] == frozenset()
    assert 0.7 <= partial.grades[0] < 1.0
    assert partial.covered[EVIDENCE_COVERAGE] == frozenset({0})
    both = matcher.judge(
        Passage("c3", "pto-leave", "3", "2.2 Balance cap", f"{EVIDENCE.evidence}.")
    )
    assert both.covered[KNOWLEDGE_HIT] == frozenset({0, 1})
    assert both.grades[0] == 1.0
    elsewhere = matcher.judge(Passage("c4", "pto-leave", "2", None, EVIDENCE.evidence))
    assert elsewhere.covered == {
        rule: frozenset() for rule in (KNOWLEDGE_HIT, EVIDENCE_COVERAGE, DOC)
    }


def test_evidence_without_a_phrase_falls_back_to_its_section() -> None:
    ref = EvidenceRef(doc_key="tips", version="3", section="3.1 Counting the tip jar")
    matcher = Matcher([ref])
    hit = matcher.judge(Passage("c1", "tips", "3", "3.1 counting the tip jar", "Count it."))
    assert hit.covered[EVIDENCE_COVERAGE] == frozenset({0})
    assert hit.covered[KNOWLEDGE_HIT] == frozenset({0})
    miss = matcher.judge(Passage("c2", "tips", "3", "4 Dividing the pool", "Counting."))
    assert miss.covered[EVIDENCE_COVERAGE] == frozenset()


# --- findability --------------------------------------------------------------------------


def test_findability_finds_evidence_a_parser_kept_and_not_what_it_lost() -> None:
    # A word broken across a line with a hyphen, and a ligature, as parsers leave them.
    text = (
        "2.1 How PTO accrues\n\nHourly staff ac-\ncrue 1 hour of PTO for every 30 hours "
        "worked, from their ﬁrst day."
    )
    assert findability(EVIDENCE, text).findable
    assert findability(EVIDENCE, text).grade == 1.0
    lost = "2.1 How PTO accrues\n\n[table omitted]"
    assert not findability(EVIDENCE, lost).findable
    section_only = EvidenceRef(doc_key="x", section="2.1 How PTO accrues")
    assert findability(section_only, lost).findable


def test_every_demo_set_evidence_item_is_findable_in_its_markdown() -> None:
    """The demo set's own evidence, against its own corpus: table rows read ``a | b | c`` in
    both, and coverage finds every phrase."""
    manifest = yaml.safe_load((DEMO / "corpus" / "manifest.yaml").read_text(encoding="utf-8"))
    texts = {
        (doc["doc_key"], str(doc["version"])): (DEMO / "corpus" / doc["path"]).read_text("utf-8")
        for doc in manifest["documents"]
    }
    queries = yaml.safe_load((DEMO / "retrieval.yaml").read_text(encoding="utf-8"))["queries"]
    refs = [
        EvidenceRef.model_validate(ref)
        for query in queries
        for ref in [*query.get("expected", []), *query.get("near_misses", [])]
    ]
    assert len(refs) > 100
    for ref in refs:
        found = findability(ref, texts[(ref.doc_key, ref.version)])
        assert found.findable, (ref.doc_key, ref.version, ref.section)


def test_highlights_mark_the_matched_words_in_the_raw_text() -> None:
    text = "Rules.\nHourly staff ac-\ncrue PTO from their ﬁrst 30 hours worked. Other."
    spans = highlights("staff accrue PTO from their first 30 hours", text)
    marked = [text[start:end] for start, end in spans]
    assert marked == ["staff", "ac-\ncrue", "PTO", "from", "their", "ﬁrst", "30", "hours"]
    assert highlights("nothing alike here", text) == []
