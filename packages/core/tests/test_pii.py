"""The in-process PII detector: each category's shape and validation, the defaults, overlaps,
and the Halcyon demo corpus. Every value here is synthetic: example SSNs, published test card
numbers, the standard example IBANs, 555-01xx phone numbers, example.com addresses and
documentation IP ranges."""

import itertools
from pathlib import Path

import pytest

from stratafi_core.pii import (
    DEFAULT_CATEGORIES,
    PII_CATEGORIES,
    canonical_categories,
    count_by_category,
    decode_text,
    find_pii,
    iban_valid,
    luhn_valid,
)

DEMO_CORPUS = Path(__file__).resolve().parents[3] / "fixtures" / "demo-corpus" / "corpus"


def found(text: str, *categories: str) -> list[tuple[str, str]]:
    spans = find_pii(text, categories or PII_CATEGORIES)
    return [(span.category, text[span.start : span.end]) for span in spans]


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("SSN 123-45-6789 on file.", ["123-45-6789"]),
        ("(234-56-7890)", ["234-56-7890"]),
        ("area 000-12-3456, 666-12-3456 and 900-12-3456", []),
        ("group 123-00-4567, serial 123-45-0000", []),
        ("longer runs 1123-45-6789, 123-45-67890, 123-45-6789-1, x123-45-6789", []),
        ("unformatted 123456789 or 123 45 6789", []),
    ],
)
def test_ssns_are_formatted_and_in_range(text: str, values: list[str]) -> None:
    assert [value for _, value in found(text, "ssn")] == values


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("Visa test 4111 1111 1111 1111.", ["4111 1111 1111 1111"]),
        ("dashed 5555-5555-5555-4444", ["5555-5555-5555-4444"]),
        ("unbroken 4012888888881881", ["4012888888881881"]),
        ("Amex test 3782 822463 10005", ["3782 822463 10005"]),
        ("after a year: 2026 4111 1111 1111 1111", ["4111 1111 1111 1111"]),
        ("fails Luhn 4111 1111 1111 1112", []),
        ("mixed separators 4111 1111-1111 1111", []),
        ("odd grouping 41 1111 1111 1111 11", []),
        ("too short 4111 1111 1111", []),
        ("leading zero 0000 0000 0000 0000", []),
        ("glued to letters 4111111111111111abc", []),
        ("a decimal 0.4111111111111111", []),
    ],
)
def test_card_numbers_pass_luhn_and_look_printed(text: str, values: list[str]) -> None:
    assert [value for _, value in found(text, "card_number")] == values


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("Pay GB82 WEST 1234 5698 7654 32 today.", ["GB82 WEST 1234 5698 7654 32"]),
        ("DE89370400440532013000", ["DE89370400440532013000"]),
        ("GB82 WEST 1234 5698 7654 32 EUR", ["GB82 WEST 1234 5698 7654 32"]),
        ("wrong check digits GB82 WEST 1234 5698 7654 33", []),
        ("lowercase gb82 west 1234 5698 7654 32", []),
        ("too short GB82 WEST 1234", []),
    ],
)
def test_ibans_pass_mod_97(text: str, values: list[str]) -> None:
    assert [value for _, value in found(text, "iban")] == values


def test_the_checksums() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("4111-1111")
    assert iban_valid("GB82WEST12345698765432")
    assert not iban_valid("GB01WEST12345698765432")  # check digits must be 02 to 98
    assert not iban_valid("GB82WEST1234569876543!")


def test_emails_phones_and_ip_addresses() -> None:
    text = (
        "Write to jane.doe@example.com or a..b@example.org, or help@desk at noon. "
        "Call (312) 555-0142, 312-555-0143, +1 312 555 0144, +13125550145 or 1-312-555-0146; "
        "not 911-555-0147, 312-911-0148 or 123-555-0149. "
        "Hosts 192.0.2.10 and 198.51.100.255; not 256.1.1.1, 1.2.3.4.5 or 01.2.3.4."
    )
    assert found(text, "email") == [("email", "jane.doe@example.com")]
    assert [value for _, value in found(text, "phone")] == [
        "(312) 555-0142",
        "312-555-0143",
        "+1 312 555 0144",
        "+13125550145",
        "1-312-555-0146",
    ]
    assert [value for _, value in found(text, "ip_address")] == ["192.0.2.10", "198.51.100.255"]


def test_the_defaults_are_the_high_risk_identifiers() -> None:
    assert DEFAULT_CATEGORIES == ("ssn", "card_number", "iban")
    text = "SSN 123-45-6789, email jane.doe@example.com, phone (312) 555-0142, host 192.0.2.10"
    assert [span.category for span in find_pii(text)] == ["ssn"]
    assert count_by_category(find_pii(text, PII_CATEGORIES)) == {
        "ssn": 1,
        "email": 1,
        "phone": 1,
        "ip_address": 1,
    }


def test_spans_are_offsets_in_text_order_and_never_overlap() -> None:
    text = "IBAN GB82 WEST 1234 5698 7654 32, SSN 123-45-6789, card 4111111111111111."
    spans = find_pii(text)
    assert [span.category for span in spans] == ["iban", "ssn", "card_number"]
    assert all(left.end <= right.start for left, right in itertools.pairwise(spans))
    assert text[spans[1].start : spans[1].end] == "123-45-6789"


def test_categories_are_checked_and_put_in_order() -> None:
    assert canonical_categories(["phone", "ssn", "phone"]) == ["ssn", "phone"]
    with pytest.raises(ValueError, match="unknown PII categories: passport"):
        find_pii("text", ["ssn", "passport"])


def test_text_decodes_by_its_byte_order_mark_then_utf8_then_windows_1252() -> None:
    assert decode_text("café".encode("utf-8-sig")) == "café"
    assert decode_text("café".encode("utf-16")) == "café"
    assert decode_text("café".encode()) == "café"
    assert decode_text("café".encode("cp1252")) == "café"


def test_the_halcyon_demo_corpus_has_nothing_the_defaults_flag() -> None:
    """The default categories don't trip on an ordinary company handbook: the demo corpus's 24
    documents have no SSN, card number or IBAN (nor, as it happens, any other category)."""
    documents = sorted(DEMO_CORPUS.rglob("*.md"))
    assert len(documents) == 24
    for path in documents:
        text = decode_text(path.read_bytes())
        assert find_pii(text) == [], path.name
