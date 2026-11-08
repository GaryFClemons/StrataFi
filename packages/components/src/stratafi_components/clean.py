"""The clean stage. Cleaning rules run in order on a parsed document, and the result is a
cleaned document: the same schema, plus ``edits``.

Each rule looks at one element at a time and returns edits: a span of the element's text and a
replacement kind (``delete``, ``space``, ``newline``, ``nfkc``, ``join``). An edit's ``start``
and ``end`` are offsets into the document's text as that rule saw it: the parsed text for the
first rule, the previous rule's output for the next. So the before and after text of every rule
can be rebuilt from the parsed document and the edits alone (``replay``), and no text is stored
twice.

- ``clean/whitespace``, the default cleaning step: headings, titles and other short elements
  collapse to single spaces. Paragraphs keep their line breaks unless ``paragraphs`` says
  otherwise. On ``parse/markdown``'s output, whose paragraphs are already single lines, it
  changes little.
- ``clean/unicode`` normalizes to NFKC and drops invisible characters (zero-width spaces, soft
  hyphens, controls other than line breaks and tabs).
- ``clean/dehyphenate`` joins words split across line breaks: ``exam-`` / ``ple``.

Other rules look at the whole document, or the corpus:

- ``clean/redact`` masks personal data: the in-process PII detector (``stratafi_core.pii``)
  finds the spans in each element, and each becomes a ``mask`` (one character per character), a
  typed ``placeholder`` (``[SSN]``) or a keyed ``hash`` (``[SSN:1a2b3c4d]``, an HMAC with the
  workspace's redaction key, so equal values stay linkable without being readable). Edits keep
  positions and tokens, never values, and nothing here logs text.
- ``clean/repeated-lines`` drops running headers, footers and page numbers: lines at the top or
  bottom of a page that repeat on many pages (digits count as one, so "Page 3 of 9" repeats).
- ``clean/boilerplate`` drops navigation and footer elements: menus of short items between
  separators, and short elements that are cookie banners, copyright lines and the like.
- ``clean/near-duplicates`` compares the document with every other parsed document of the run's
  corpus version by MinHash LSH; when a newer version is a near duplicate, this one is dropped
  whole and marked ``near_duplicate``, so the run stops it after Clean. It is the one rule that
  needs the corpus: the run waits for every parse before cleaning (``stratafi_core.planner``).
- ``clean/language`` tags the document's main language and, when ``keep`` names languages,
  drops paragraphs confidently in another one.
- ``clean/min-content`` drops elements below a size: stray page furniture and fragments.

Dropping an element deletes its text; the element stays, empty, so offsets and IDs still hold,
and chunkers skip empty elements.
"""

import hashlib
import hmac
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stratafi_components import language as languages
from stratafi_components import minhash
from stratafi_core.artifacts import Edit, Element, ParsedDocument, ReplacementKind, build_text
from stratafi_core.components import Component, ComponentInfo, StageContext
from stratafi_core.pii import DEFAULT_CATEGORIES, PiiCategory, canonical_categories, find_pii

_WHITESPACE_RUN = re.compile(r"\s+")
_TRAILING_ON_LINE = re.compile(r"[ \t]+(?=\n)")
INVISIBLE = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad")
# Elements whose text is a short label: always collapsed to one line.
LABEL_TYPES = frozenset(
    {"title", "heading", "caption", "page_header", "page_footer", "page_number"}
)


MASK = "\u2588"  # a full block: one per masked character
REDACTION_KEY_SERVICE = "redaction_key"
NEAR_DUPLICATE = "near_duplicate"


@dataclass(frozen=True)
class TextEdit:
    """An edit inside one element's text. ``label`` is a redaction's token."""

    start: int
    end: int
    kind: ReplacementKind
    replacement: str
    label: str | None = None


@dataclass(frozen=True)
class CorpusDocument:
    """Another parsed document of the run's corpus version, for corpus-wide rules. ``rank``
    orders versions: a higher rank is newer. ``text`` loads its parsed text when asked."""

    key: str
    rank: tuple[Any, ...]
    text: Callable[[], str]


@dataclass(frozen=True)
class Corpus:
    """What corpus-wide rules see: this document's key and rank, and every other document."""

    key: str
    rank: tuple[Any, ...]
    others: Sequence[CorpusDocument] = ()
    # Signatures by (document key, rule parameters), shared across documents of one process.
    cache: dict[tuple[str, str], Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleOutput:
    """A rule's edits, one list per element in order, and what it says of the document."""

    edits: list[list[TextEdit]]
    language: str | None = None
    warnings: tuple[str, ...] = ()


def apply_edits(text: str, edits: list[TextEdit]) -> str:
    """``text`` with ``edits`` (sorted, not overlapping) applied."""
    parts: list[str] = []
    position = 0
    for edit in edits:
        parts.append(text[position : edit.start])
        parts.append(edit.replacement)
        position = edit.end
    parts.append(text[position:])
    return "".join(parts)


def replacement_for(kind: ReplacementKind, original: str, label: str | None = None) -> str:
    """What an edit of ``kind`` puts in place of ``original``: how edits are replayed."""
    if kind in ("placeholder", "hash"):
        if label is None:
            raise ValueError("a redaction token needs its label")
        return label
    return {
        "delete": "",
        "join": "",
        "space": " ",
        "newline": "\n",
        "nfkc": unicodedata.normalize("NFKC", original),
        "mask": MASK * len(original),
    }[kind]


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CleanComponent(Component):
    """A cleaning rule. Most look at one element at a time (``edits``); rules that need the
    whole document, a service or the corpus override ``apply``."""

    # True for a rule that compares the document with the rest of the corpus version.
    corpus_scope: ClassVar[bool] = False

    def edits(self, element: Element, params: BaseModel) -> list[TextEdit]:
        raise NotImplementedError

    def apply(
        self,
        document: ParsedDocument,
        params: BaseModel,
        context: StageContext | None = None,
        corpus: Corpus | None = None,
    ) -> RuleOutput:
        return RuleOutput([self.edits(element, params) for element in document.elements])


def drop(element: Element) -> list[TextEdit]:
    """Edits that empty an element."""
    return [TextEdit(0, len(element.text), "delete", "")] if element.text else []


# --- clean/whitespace ----------------------------------------------------------------------------


def collapse_edits(text: str) -> list[TextEdit]:
    """Edits that turn ``text`` into ``" ".join(text.split())``."""
    edits = []
    for match in _WHITESPACE_RUN.finditer(text):
        start, end = match.span()
        if start == 0 or end == len(text):
            edits.append(TextEdit(start, end, "delete", ""))
        elif match.group() != " ":
            edits.append(TextEdit(start, end, "space", " "))
    return edits


def trim_edits(text: str) -> list[TextEdit]:
    """Edits that strip the text, and spaces and tabs at the end of each line."""
    edits = [TextEdit(*m.span(), "delete", "") for m in _TRAILING_ON_LINE.finditer(text)]
    leading = len(text) - len(text.lstrip())
    trailing = len(text.rstrip())
    if leading:
        edits = [e for e in edits if e.start >= leading]
        edits.insert(0, TextEdit(0, leading, "delete", ""))
    if trailing < len(text) and trailing >= leading:
        edits = [e for e in edits if e.end <= trailing]
        edits.append(TextEdit(trailing, len(text), "delete", ""))
    return edits


class WhitespaceParams(_Params):
    paragraphs: Literal["keep", "trim", "collapse"] = Field(
        default="keep",
        description="keep: line breaks as they are; trim: strip each line's end; collapse: "
        "one space between words",
    )


class Whitespace(CleanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="clean/whitespace",
        stage="clean",
        version="1.0.0",
        title="Whitespace",
        description="The default cleaning step: labels such as headings collapse to single "
        "spaces. Paragraphs can be trimmed or collapsed too.",
        license="MIT",
        baseline=True,
    )
    Params: ClassVar[type[BaseModel]] = WhitespaceParams

    def edits(self, element: Element, params: BaseModel) -> list[TextEdit]:
        params = cast(WhitespaceParams, params)
        if element.type in LABEL_TYPES:
            return collapse_edits(element.text)
        if element.type == "code" or params.paragraphs == "keep":
            return []
        if params.paragraphs == "collapse":
            return collapse_edits(element.text)
        return trim_edits(element.text)


# --- clean/unicode --------------------------------------------------------------------------------


def _segments(text: str) -> list[tuple[int, int]]:
    """Spans of a starter character and the combining marks after it."""
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(text) + 1):
        if index == len(text) or unicodedata.combining(text[index]) == 0:
            spans.append((start, index))
            start = index
    return spans if text else []


class UnicodeParams(_Params):
    remove_invisible: bool = Field(
        default=True, description="Drop zero-width spaces, soft hyphens and control characters"
    )


class UnicodeNormalize(CleanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="clean/unicode",
        stage="clean",
        version="1.0.0",
        title="Unicode",
        description="NFKC normalization: ligatures, full-width forms and compatibility "
        "characters become their plain forms. Drops invisible characters.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = UnicodeParams

    def edits(self, element: Element, params: BaseModel) -> list[TextEdit]:
        params = cast(UnicodeParams, params)
        edits = []
        for start, end in _segments(element.text):
            segment = element.text[start:end]
            first = segment[0]
            invisible = first in INVISIBLE or (
                unicodedata.category(first) == "Cc" and first not in "\n\t"
            )
            if params.remove_invisible and invisible:
                edits.append(TextEdit(start, end, "delete", ""))
                continue
            normalized = unicodedata.normalize("NFKC", segment)
            if normalized != segment:
                edits.append(TextEdit(start, end, "nfkc", normalized))
        return edits


# --- clean/dehyphenate ----------------------------------------------------------------------------


_BREAKS = {
    "line_breaks": re.compile(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[^\W\d_])"),
    "line_breaks_and_spaces": re.compile(r"(?<=[^\W\d_])-(?:[ \t]*\n[ \t]*| )(?=[^\W\d_])"),
}


class DehyphenateParams(_Params):
    across: Literal["line_breaks", "line_breaks_and_spaces"] = Field(
        default="line_breaks",
        description="Where a split word's halves meet: a line break, or also a single space "
        "(parsers that join lines with spaces)",
    )


class Dehyphenate(CleanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="clean/dehyphenate",
        stage="clean",
        version="1.0.0",
        title="Dehyphenate",
        description="Joins words split across line breaks, when the next half starts in "
        "lowercase: exam-/ple becomes example.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = DehyphenateParams

    def edits(self, element: Element, params: BaseModel) -> list[TextEdit]:
        params = cast(DehyphenateParams, params)
        if element.type == "code":
            return []
        return [
            TextEdit(*match.span(), "join", "")
            for match in _BREAKS[params.across].finditer(element.text)
            if element.text[match.end()].islower()
        ]


# --- clean/redact ----------------------------------------------------------------------------


class RedactParams(_Params):
    mode: Literal["mask", "placeholder", "hash"] = Field(
        default="placeholder",
        description="mask: one block per character; placeholder: [SSN]; hash: [SSN:1a2b3c4d], "
        "keyed with the workspace's redaction key",
    )
    categories: list[PiiCategory] = Field(
        default_factory=lambda: list(DEFAULT_CATEGORIES),
        min_length=1,
        description="What to redact: ssn, card_number and iban by default; email, phone and "
        "ip_address on request",
    )

    @field_validator("categories")
    @classmethod
    def _canonical(cls, value: list[str]) -> list[str]:
        return canonical_categories(value)


class Redact(CleanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="clean/redact",
        stage="clean",
        version="2.0.0",
        title="Redact personal data",
        description="Masks the personal data the in-process PII detector finds in each element "
        "(SSNs, card numbers and IBANs by default): a mask, a typed placeholder such as [SSN], "
        "or a keyed hash. Edits keep positions and tokens, never values.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = RedactParams

    def _label(self, mode: str, category: str, value: str, key: bytes, workspace: str) -> str:
        token = category.upper()
        if mode == "placeholder":
            return f"[{token}]"
        digest = hmac.new(key, f"{workspace}:{category}:{value}".encode(), hashlib.sha256)
        return f"[{token}:{digest.hexdigest()[:8]}]"

    def apply(
        self,
        document: ParsedDocument,
        params: BaseModel,
        context: StageContext | None = None,
        corpus: Corpus | None = None,
    ) -> RuleOutput:
        params = cast(RedactParams, params)
        context = context or StageContext()
        key = context.service(REDACTION_KEY_SERVICE) if params.mode == "hash" else b""
        per_element: list[list[TextEdit]] = []
        for element in document.elements:
            context.checkpoint()
            edits: list[TextEdit] = []
            for span in find_pii(element.text, params.categories):
                value = element.text[span.start : span.end]
                if params.mode == "mask":
                    edits.append(TextEdit(span.start, span.end, "mask", MASK * len(value)))
                    continue
                label = self._label(params.mode, span.category, value, key, context.workspace_id)
                kind = cast(ReplacementKind, params.mode)
                edits.append(TextEdit(span.start, span.end, kind, label, label))
            per_element.append(edits)
        return RuleOutput(per_element)


# --- clean/repeated-lines --------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
PAGE_FURNITURE = frozenset({"page_header", "page_footer", "page_number"})


def _normalized_line(line: str) -> str:
    return _DIGITS.sub("#", _SPACES.sub(" ", line.strip().lower()))


class RepeatedLinesParams(_Params):
    min_pages: int = Field(default=3, ge=2, le=1000, description="Pages a line must repeat on")
    min_share: float = Field(
        default=0.5, gt=0, le=1, description="Share of the pages a line must repeat on"
    )
    edge_lines: int = Field(
        default=3, ge=1, le=20, description="Lines from the top and bottom of a page to consider"
    )
    drop_furniture: bool = Field(
        default=True, description="Also drop elements typed page_header, page_footer, page_number"
    )


class RepeatedLines(CleanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="clean/repeated-lines",
        stage="clean",
        version="1.0.0",
        title="Repeated lines",
        description="Drops running headers, footers and page numbers: lines at the top or "
        "bottom of pages that repeat across many pages, digits counted as one.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = RepeatedLinesParams

    def apply(
        self,
        document: ParsedDocument,
        params: BaseModel,
        context: StageContext | None = None,
        corpus: Corpus | None = None,
    ) -> RuleOutput:
        params = cast(RepeatedLinesParams, params)
        per_element: list[list[TextEdit]] = [[] for _ in document.elements]
        # Each page's lines in order: (element index, start, end incl. newline, normalized).
        pages: dict[int, list[tuple[int, int, int, str]]] = {}
        for index, element in enumerate(document.elements):
            if params.drop_furniture and element.type in PAGE_FURNITURE:
                per_element[index] = drop(element)
                continue
            if element.page is None:
                continue
            position = 0
            for line in element.text.splitlines(keepends=True):
                normalized = _normalized_line(line)
                if normalized:
                    pages.setdefault(element.page, []).append(
                        (index, position, position + len(line), normalized)
                    )
                position += len(line)
        if len(pages) < 2:
            return RuleOutput(per_element)
        edges: dict[int, list[tuple[int, int, int, str]]] = {}
        counts: Counter[str] = Counter()
        for page, lines in pages.items():
            edge = lines[: params.edge_lines] + lines[-params.edge_lines :]
            edges[page] = list(dict.fromkeys(edge))
            counts.update({line[3] for line in edges[page]})
        needed = max(params.min_pages, int(params.min_share * len(pages) + 0.999))
        repeated = {line for line, count in counts.items() if count >= needed and len(line) <= 200}
        for lines in edges.values():
            for index, start, end, normalized in lines:
                if normalized in repeated and not any(
                    edit.start <= start < edit.end for edit in per_element[index]
                ):
                    per_element[index].append(TextEdit(start, end, "delete", ""))
        for edits in per_element:
            edits.sort(key=lambda edit: edit.start)
        return RuleOutput(per_element)


# --- clean/boilerplate -----------------------------------------------------------------------

BOILERPLATE = re.compile(
    r"cookie|all rights reserved|skip to (?:main )?content|privacy (?:policy|notice)"
    r"|terms (?:of|and) (?:use|service|conditions)|back to top|subscribe to our newsletter"
    r"|follow us|share (?:this|on)|copyright \d{4}|\u00a9 ?\d{4}|\(c\) ?\d{4}",
    re.IGNORECASE,
)
_MENU_SEPARATORS = re.compile(r"\s[|\u2022\u00b7\u203a\u00bb/]\s")


class BoilerplateParams(_Params):
    max_words: int = Field(
        default=30, ge=1, le=200, description="Only elements this short can be boilerplate"
    )
    menu_items: int = Field(
        default=3, ge=2, le=50, description="Separated short items that make a menu"
    )
    phrases: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="More phrases (matched case-insensitively) that mark an element as boilerplate",
    )
    drop_furniture: bool = Field(
        default=True, description="Also drop elements typed page_header and page_footer"
    )


class Boilerplate(CleanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="clean/boilerplate",
        stage="clean",
        version="1.0.0",
        title="Boilerplate",
        description="Drops navigation and footers: menus of short items between separators, "
        "and short elements such as cookie banners, copyright lines and share links.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = BoilerplateParams

    def edits(self, element: Element, params: BaseModel) -> list[TextEdit]:
        params = cast(BoilerplateParams, params)
        if params.drop_furniture and element.type in ("page_header", "page_footer"):
            return drop(element)
        if element.type in ("title", "heading", "table", "code"):
            return []
        text = element.text
        words = text.split()
        if not words or len(words) > params.max_words:
            return []
        lowered = text.lower()
        if BOILERPLATE.search(text) or any(p.lower() in lowered for p in params.phrases if p):
            return drop(element)
        items = [item for item in _MENU_SEPARATORS.split(f" {text} ") if item.strip()]
        if len(items) >= params.menu_items and all(len(item.split()) <= 4 for item in items):
            return drop(element)
        return []


# --- clean/near-duplicates -------------------------------------------------------------------


class NearDuplicatesParams(_Params):
    threshold: float = Field(
        default=0.85, ge=0.5, le=1.0, description="Estimated Jaccard similarity of shingles"
    )
    shingle_words: int = Field(default=5, ge=1, le=20)
    num_perm: int = Field(default=128, ge=16, le=1024, description="MinHash permutations")


class NearDuplicates(CleanComponent):
    corpus_scope: ClassVar[bool] = True
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="clean/near-duplicates",
        stage="clean",
        version="1.0.0",
        title="Near duplicates",
        description="MinHash LSH across the corpus version: a document whose newer version is "
        "a near duplicate is dropped, and the run stops it after Clean. Newer means a later "
        "effective date, then a later version, then a later name.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = NearDuplicatesParams

    def _signature(self, corpus: Corpus, key: str, text: Callable[[], str], params) -> Any:
        cache_key = (key, f"{params.shingle_words}:{params.num_perm}")
        if cache_key not in corpus.cache:
            corpus.cache[cache_key] = minhash.signature(
                text(), k=params.shingle_words, num_perm=params.num_perm
            )
        return corpus.cache[cache_key]

    def apply(
        self,
        document: ParsedDocument,
        params: BaseModel,
        context: StageContext | None = None,
        corpus: Corpus | None = None,
    ) -> RuleOutput:
        params = cast(NearDuplicatesParams, params)
        nothing = RuleOutput([[] for _ in document.elements])
        if corpus is None or not corpus.others:
            return nothing
        signatures = {
            corpus.key: self._signature(corpus, corpus.key, lambda: document.text, params)
        }
        ranks = {corpus.key: corpus.rank}
        for other in corpus.others:
            if context is not None:
                context.checkpoint()
            signatures[other.key] = self._signature(corpus, other.key, other.text, params)
            ranks[other.key] = other.rank
        found = minhash.near_duplicates(signatures, corpus.key, threshold=params.threshold)
        if context is not None:
            context.metric("near_duplicates", len(found))
        if any(ranks[other] > corpus.rank for other, _ in found):
            return RuleOutput(
                [drop(element) for element in document.elements], warnings=(NEAR_DUPLICATE,)
            )
        return nothing


# --- clean/language --------------------------------------------------------------------------


class LanguageParams(_Params):
    keep: list[Literal["en", "es", "fr", "de", "pt", "it", "nl"]] = Field(
        default_factory=list,
        description="Drop paragraphs confidently in other languages; empty only tags",
    )
    min_words: int = Field(default=8, ge=3, le=200, description="Shorter elements are never judged")


class Language(CleanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="clean/language",
        stage="clean",
        version="1.0.0",
        title="Language",
        description="Tags the document's main language by function words (English, Spanish, "
        "French, German, Portuguese, Italian, Dutch) and can drop paragraphs in others.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = LanguageParams

    def apply(
        self,
        document: ParsedDocument,
        params: BaseModel,
        context: StageContext | None = None,
        corpus: Corpus | None = None,
    ) -> RuleOutput:
        params = cast(LanguageParams, params)
        per_element: list[list[TextEdit]] = [[] for _ in document.elements]
        weight: Counter[str] = Counter()
        for index, element in enumerate(document.elements):
            if element.type in ("code", "table"):
                continue
            found = languages.identify(element.text, min_words=params.min_words)
            if found is None:
                continue
            weight[found] += len(element.text)
            if params.keep and found not in params.keep:
                per_element[index] = drop(element)
        main = min(weight, key=lambda language: (-weight[language], language)) if weight else None
        return RuleOutput(per_element, language=main)


# --- clean/min-content -----------------------------------------------------------------------


class MinContentParams(_Params):
    min_chars: int = Field(default=20, ge=0, le=10_000, description="Characters, spaces aside")
    min_words: int = Field(default=3, ge=0, le=1000)
    keep_types: list[str] = Field(
        default_factory=lambda: ["title", "heading", "list_item", "table", "code"],
        description="Element types that are never dropped",
    )


class MinContent(CleanComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="clean/min-content",
        stage="clean",
        version="1.0.0",
        title="Minimum content",
        description="Drops elements with fewer characters or words than the minimum: stray "
        "page furniture and fragments. Headings, list items, tables and code are kept.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = MinContentParams

    def edits(self, element: Element, params: BaseModel) -> list[TextEdit]:
        params = cast(MinContentParams, params)
        if element.type in params.keep_types or not element.text:
            return []
        chars = len("".join(element.text.split()))
        if chars < params.min_chars or len(element.text.split()) < params.min_words:
            return drop(element)
        return []


# --- the stage --------------------------------------------------------------------------------


def apply_rule(
    document: ParsedDocument,
    rule_id: str,
    rule: CleanComponent,
    params: BaseModel,
    context: StageContext | None = None,
    corpus: Corpus | None = None,
) -> tuple[ParsedDocument, list[Edit]]:
    """One rule over the document. Returns the new document and the rule's edits, as offsets
    into ``document``'s text."""
    output = rule.apply(document, params, context, corpus)
    texts: list[str] = []
    edits: list[Edit] = []
    for element, local in zip(document.elements, output.edits, strict=True):
        texts.append(apply_edits(element.text, local))
        edits += [
            Edit(
                rule=rule_id,
                element_id=element.id,
                start=element.char_start + edit.start,
                end=element.char_start + edit.end,
                replacement_kind=edit.kind,
                label=edit.label,
            )
            for edit in local
        ]
    update: dict[str, Any] = {}
    if output.language is not None:
        update["language"] = output.language
    if output.warnings:
        update["warnings"] = [*document.warnings, *output.warnings]
    if edits:
        text, offsets = build_text(texts)
        update["elements"] = [
            element.model_copy(update={"text": new_text, "char_start": start, "char_end": end})
            for element, new_text, (start, end) in zip(
                document.elements, texts, offsets, strict=True
            )
        ]
        update["text"] = text
    return (document.model_copy(update=update) if update else document), edits


def run_clean(
    document: ParsedDocument,
    rules: list[tuple[str, CleanComponent, BaseModel]],
    context: StageContext | None = None,
    corpus: Corpus | None = None,
) -> ParsedDocument:
    """Apply ``rules`` (ID, component, resolved params) in order. The result carries every
    rule's edits, in order. ``corpus`` is for corpus-wide rules (``clean/near-duplicates``)."""
    edits: list[Edit] = []
    current = document
    for rule_id, rule, params in rules:
        if context is not None:
            context.checkpoint()
        current, rule_edits = apply_rule(current, rule_id, rule, params, context, corpus)
        edits += rule_edits
    if context is not None:
        context.metric("edits", len(edits))
        context.metric("chars_removed", len(document.text) - len(current.text))
        context.metric(
            "redactions", sum(1 for e in edits if e.label or e.replacement_kind == "mask")
        )
    return ParsedDocument.model_validate(
        {
            **current.model_dump(by_alias=True),
            "edits": [e.model_dump(exclude_none=True) for e in edits],
        }
    )


def replay(document: ParsedDocument, edits: list[Edit]) -> list[tuple[str, str]]:
    """Rebuild each rule's before and after text from the parsed document and the edits, in
    order: what the UI shows as a diff."""
    steps: list[tuple[str, str]] = []
    text = document.text
    for _, run in groupby(edits, key=lambda edit: edit.rule):
        group = sorted(run, key=lambda edit: edit.start)
        replacements = [
            TextEdit(
                e.start,
                e.end,
                e.replacement_kind,
                replacement_for(e.replacement_kind, text[e.start : e.end], e.label),
            )
            for e in group
        ]
        after = apply_edits(text, replacements)
        steps.append((text, after))
        text = after
    return steps


REDACTION_KINDS = frozenset({"mask", "placeholder", "hash"})


def redacted_values(document: ParsedDocument, edits: list[Edit]) -> list[str]:
    """The text each redaction replaced, rebuilt from the input document: what a response must
    never show. Longest first, so masking one never leaves part of another."""
    values: set[str] = set()
    text = document.text
    for _, run in groupby(edits, key=lambda edit: edit.rule):
        group = sorted(run, key=lambda edit: edit.start)
        values.update(text[e.start : e.end] for e in group if e.replacement_kind in REDACTION_KINDS)
        text = apply_edits(
            text,
            [
                TextEdit(
                    e.start,
                    e.end,
                    e.replacement_kind,
                    replacement_for(e.replacement_kind, text[e.start : e.end], e.label),
                )
                for e in group
            ],
        )
    return sorted((value for value in values if value), key=lambda value: (-len(value), value))


def mask_values(text: str, values: Sequence[str]) -> str:
    """``text`` with every occurrence of each value replaced by as many mask characters, so
    offsets still hold and a slice never shows part of a value."""
    for value in values:
        text = text.replace(value, MASK * len(value))
    return text


@dataclass(frozen=True)
class Hunk:
    rule: str
    element_id: str
    start: int  # in the rule's input text
    end: int
    kind: str
    before: str  # the edited span with context, values masked
    after: str
    redaction: bool


def diff_hunks(
    document: ParsedDocument, edits: list[Edit], *, context_chars: int = 40, limit: int = 500
) -> tuple[list[Hunk], int]:
    """Each edit as a before/after hunk with context, for the diff viewer, and the total number
    of edits. Redacted values are masked in every string, the context included."""
    values = redacted_values(document, edits)
    hunks: list[Hunk] = []
    text = document.text
    total = 0
    for rule, run in groupby(edits, key=lambda edit: edit.rule):
        group = sorted(run, key=lambda edit: edit.start)
        replacements = [
            TextEdit(
                e.start,
                e.end,
                e.replacement_kind,
                replacement_for(e.replacement_kind, text[e.start : e.end], e.label),
                e.label,
            )
            for e in group
        ]
        masked = mask_values(text, values)
        for edit, replacement in zip(group, replacements, strict=True):
            total += 1
            if len(hunks) < limit:
                left = max(0, edit.start - context_chars)
                right = min(len(text), edit.end + context_chars)
                prefix, suffix = masked[left : edit.start], masked[edit.end : right]
                hunks.append(
                    Hunk(
                        rule=rule,
                        element_id=edit.element_id,
                        start=edit.start,
                        end=edit.end,
                        kind=edit.replacement_kind,
                        before=prefix + masked[edit.start : edit.end] + suffix,
                        after=prefix + mask_values(replacement.replacement, values) + suffix,
                        redaction=edit.replacement_kind in REDACTION_KINDS,
                    )
                )
        text = apply_edits(text, replacements)
    return hunks, total


COMPONENTS: list[type[Component]] = [
    Whitespace,
    UnicodeNormalize,
    Dehyphenate,
    Redact,
    RepeatedLines,
    Boilerplate,
    NearDuplicates,
    Language,
    MinContent,
]
