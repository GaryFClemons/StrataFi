"""Personal data (PII) in text, found in process: a pattern for each category's shape, then each
candidate's own validation, so a number that merely looks right isn't reported.

``find_pii(text, categories)`` returns spans: a category and character offsets into ``text``,
never the values. Whoever needs a value reads it from their own copy of the text; reports and
results carry counts by category only (``count_by_category``).

The categories:

- ``ssn``: a US Social Security number written ``###-##-####``. The area (the first three
  digits) is never 000, 666 or 900 to 999; the group (the middle two) never 00; the serial
  (the last four) never 0000.
- ``card_number``: a payment card number of 13 to 19 digits that passes the Luhn check and
  doesn't start with 0. Unbroken, or grouped by single spaces or by single dashes (one kind per
  number) the way cards are printed: fours with a shorter last group, or 4-6-5 and 4-6-4.
- ``iban``: an International Bank Account Number: a two-letter country code, check digits from
  02 to 98 and 11 to 30 more letters and digits, 15 to 34 characters in all, unbroken or in
  groups split by single spaces. The check digits must hold: the number is 1 modulo 97.
- ``email``: an address whose domain has at least two labels and ends in a top-level domain of
  letters.
- ``phone``: a North American (NANP) number: an area code and an exchange that start with 2 to 9
  and aren't N11 service codes, then four digits. Separated consistently by spaces, dots or
  dashes, or with the area code in parentheses, optionally after +1 or 1; or +1 and ten digits.
- ``ip_address``: a dotted IPv4 address, each of its four numbers 0 to 255 without leading
  zeros.

The first three are on by default (``DEFAULT_CATEGORIES``): each identifies a person or an
account on its own, and a business document rarely holds one by accident. Email addresses,
phone numbers and IP addresses fill any organization's handbook (a help desk, a store's line, a
printer's address), so they're opt-in.

When finds overlap, the one that starts first wins, then the longer one, then the earlier
category in ``PII_CATEGORIES``.

``scan/pii``, ``clean/redact`` and the suites' ``no_findings`` checks all use this detector.
A change to what it finds changes their output, so it comes with a new version of
``scan/pii`` and ``clean/redact``: cached scan reports and redactions are keyed by those.
"""

import codecs
import re
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from typing import Literal, get_args

PiiCategory = Literal["ssn", "card_number", "iban", "email", "phone", "ip_address"]
PII_CATEGORIES: tuple[str, ...] = get_args(PiiCategory)
DEFAULT_CATEGORIES: tuple[str, ...] = ("ssn", "card_number", "iban")


@dataclass(frozen=True)
class PiiSpan:
    """One find: its category and where it is. ``text[start:end]`` is the value."""

    category: str
    start: int
    end: int


def canonical_categories(categories: Iterable[str]) -> list[str]:
    """``categories`` without repeats, in ``PII_CATEGORIES`` order, so equal choices compare
    and hash alike. Raises ``ValueError`` for a category the detector doesn't know."""
    wanted = set(categories)
    unknown = wanted.difference(PII_CATEGORIES)
    if unknown:
        raise ValueError(f"unknown PII categories: {', '.join(sorted(unknown))}")
    return [category for category in PII_CATEGORIES if category in wanted]


# --- checksums -------------------------------------------------------------------------------


def luhn_valid(digits: str) -> bool:
    """The Luhn (mod 10) check that payment card numbers carry in their last digit."""
    if not digits.isdigit():
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def iban_valid(compact: str) -> bool:
    """ISO 13616: a country code, check digits 02 to 98, and the whole number 1 modulo 97 once
    its first four characters move to the end and letters become 10 to 35."""
    if not 15 <= len(compact) <= 34 or not compact.isascii() or not compact.isalnum():
        return False
    country, check, rest = compact[:2], compact[2:4], compact[4:]
    if not (country.isalpha() and country.isupper() and check.isdigit()):
        return False
    if not 2 <= int(check) <= 98 or rest != rest.upper():
        return False
    remainder = 0
    for char in rest + country + check:
        value = int(char, 36)
        remainder = (remainder * (100 if value > 9 else 10) + value) % 97
    return remainder == 1


# --- one finder per category -----------------------------------------------------------------

Found = Iterator[tuple[int, int]]

_SSN = re.compile(r"(?<![\w-])([0-9]{3})-([0-9]{2})-([0-9]{4})(?![\w-])")


def _ssn(text: str) -> Found:
    for match in _SSN.finditer(text):
        area, group, serial = match.groups()
        if area in ("000", "666") or area.startswith("9") or group == "00" or serial == "0000":
            continue
        yield match.span()


# Digit groups joined by one kind of separator (a space or a dash), or one unbroken run.
_DIGIT_GROUPS = re.compile(r"(?<![\w.,])[0-9]+(?:([ -])[0-9]+(?:\1[0-9]+)*)?(?![\w]|[.,][0-9])")
_CARD_LENGTHS = range(13, 20)


def _card_grouping(sizes: list[int]) -> bool:
    """How a card number is printed: one run, fours ending in a group of one to four, or the
    4-6-5 and 4-6-4 groupings of 15- and 14-digit cards."""
    if len(sizes) == 1:
        return True
    if sizes in ([4, 6, 5], [4, 6, 4]):
        return True
    return all(size == 4 for size in sizes[:-1]) and 1 <= sizes[-1] <= 4


def _card_number(text: str) -> Found:
    for match in _DIGIT_GROUPS.finditer(text):
        separator = match.group(1)
        groups = match.group().split(separator) if separator else [match.group()]
        offsets, position = [], match.start()
        for group in groups:
            offsets.append(position)
            position += len(group) + 1
        first = 0
        while first < len(groups):
            for last in range(len(groups) - 1, first - 1, -1):
                window = groups[first : last + 1]
                digits = "".join(window)
                if (
                    len(digits) in _CARD_LENGTHS
                    and digits[0] != "0"
                    and _card_grouping([len(group) for group in window])
                    and luhn_valid(digits)
                ):
                    yield offsets[first], offsets[last] + len(groups[last])
                    first = last
                    break
            first += 1


_IBAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]){11,40}(?![A-Za-z0-9])")


def _iban(text: str) -> Found:
    """The longest leading run of the candidate's space-separated groups that checks out, so
    capitals and digits that follow a number (a currency code) don't spoil it."""
    for match in _IBAN.finditer(text):
        groups = match.group().split(" ")
        for count in range(len(groups), 0, -1):
            kept = groups[:count]
            if iban_valid("".join(kept)):
                yield match.start(), match.start() + len(" ".join(kept))
                break


_EMAIL = re.compile(
    r"(?<![\w.%+-])"
    r"[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}"
    r"@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,24}"
    r"(?![\w-]|\.[A-Za-z0-9])"
)


def _email(text: str) -> Found:
    for match in _EMAIL.finditer(text):
        local = match.group().split("@", 1)[0]
        if ".." in local or local.endswith("."):
            continue
        yield match.span()


_PHONE = re.compile(
    r"(?<![\w+(])"
    r"(?:\+1 ?|1[ .-])?"
    r"(?:\((?P<area>[2-9][0-9]{2})\) ?(?P<exchange>[2-9][0-9]{2})[ .-]"
    r"|(?P<area2>[2-9][0-9]{2})(?P<sep>[ .-])(?P<exchange2>[2-9][0-9]{2})(?P=sep))"
    r"[0-9]{4}"
    r"(?![\w]|[.-][0-9])"
)
_PHONE_E164 = re.compile(
    r"(?<![\w+])\+1(?P<area>[2-9][0-9]{2})(?P<exchange>[2-9][0-9]{2})[0-9]{4}(?![0-9])"
)


def _service_code(code: str) -> bool:
    """N11 codes (211, 311 ... 911) are service numbers, never an area code or an exchange."""
    return code[1:] == "11"


def _phone(text: str) -> Found:
    for match in _PHONE.finditer(text):
        area = match.group("area") or match.group("area2")
        exchange = match.group("exchange") or match.group("exchange2")
        if not (_service_code(area) or _service_code(exchange)):
            yield match.span()
    for match in _PHONE_E164.finditer(text):
        if not (_service_code(match.group("area")) or _service_code(match.group("exchange"))):
            yield match.span()


_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
_IPV4 = re.compile(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\w]|\.[0-9])")


def _ip_address(text: str) -> Found:
    for match in _IPV4.finditer(text):
        yield match.span()


_FINDERS = {
    "ssn": _ssn,
    "card_number": _card_number,
    "iban": _iban,
    "email": _email,
    "phone": _phone,
    "ip_address": _ip_address,
}
_RANK = {category: rank for rank, category in enumerate(PII_CATEGORIES)}


def find_pii(text: str, categories: Collection[str] = DEFAULT_CATEGORIES) -> list[PiiSpan]:
    """The PII in ``text`` of the given categories: non-overlapping spans in text order."""
    found = [
        PiiSpan(category, start, end)
        for category in canonical_categories(categories)
        for start, end in _FINDERS[category](text)
    ]
    found.sort(key=lambda span: (span.start, -(span.end - span.start), _RANK[span.category]))
    kept: list[PiiSpan] = []
    for span in found:
        if kept and span.start < kept[-1].end:
            continue
        kept.append(span)
    return kept


def count_by_category(spans: Iterable[PiiSpan]) -> dict[str, int]:
    """How many finds of each category, in ``PII_CATEGORIES`` order: what reports carry."""
    counts: dict[str, int] = {}
    for span in spans:
        counts[span.category] = counts.get(span.category, 0) + 1
    return {category: counts[category] for category in PII_CATEGORIES if category in counts}


_BYTE_ORDER_MARKS = (
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


def decode_text(data: bytes) -> str:
    """Bytes of a text file as text: by its byte-order mark, else UTF-8, else Windows-1252.
    Never fails; undecodable bytes become U+FFFD."""
    for mark, codec in _BYTE_ORDER_MARKS:
        if data.startswith(mark):
            return data[len(mark) :].decode(codec, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")
