"""The chunkers beyond fixed windows.

- ``chunk/sections``, the default: heading-aware packing (``Sections`` says how).
- ``chunk/recursive``: splits at the first separator of a hierarchy (blank lines, lines,
  sentences, words, characters) that brings pieces under the size, then packs pieces into
  chunks with an overlap.
- ``chunk/sentence-window``: windows of ``window`` sentences, moving ``stride`` sentences at a
  time.
- ``chunk/semantic``: breakpoints where the similarity of adjacent sentences (TF-IDF over
  ``keywords.terms``, each sentence with its neighbours) drops below the
  ``breakpoint_percentile``-th largest drop, within a token range.
- ``chunk/table-aware``: text packed at element boundaries; a table stays whole up to
  ``table_max_tokens``, otherwise it becomes row groups that each repeat the header row.
- ``chunk/parent-child``: small child chunks, each carrying its parent section in
  ``metadata.parent``. Children are embedded and searched; retrieval returns their parents
  (``stratafi_components.retrieval``).

Every chunker records where each chunk came from (elements, offsets, pages, heading path) and
counts its tokens with its estimator (``stratafi_components.estimators``).
"""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stratafi_components.chunk import (
    ChunkComponent,
    _heading_path,
    body_elements,
    chunk_id,
    label_title,
    record_metrics,
    render_index_text,
)
from stratafi_components.estimators import (
    CHARS,
    EstimatorName,
    counter,
    estimator_env,
    sentence_spans,
)
from stratafi_components.keywords import terms
from stratafi_core.artifacts import ELEMENT_SEPARATOR, Chunk, Element, ParsedDocument
from stratafi_core.components import ComponentError, ComponentInfo, StageContext
from stratafi_core.specs import DEFAULT_INDEX_TEXT

Span = tuple[int, int]


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def trim_span(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


class Maker:
    """Turns spans of a document's text into chunks with their provenance."""

    def __init__(
        self,
        document: ParsedDocument,
        *,
        cache_key: str,
        label: str | None,
        index_text: str,
        estimator: str,
        count,
    ) -> None:
        self.document = document
        self.cache_key = cache_key
        self.index_text = index_text
        self.estimator = estimator
        self.count = count
        self.title = document.title or label_title(label)
        self.elements = body_elements(document)
        self.headings = [e for e in document.elements if e.type == "heading"]
        self.chunks: list[Chunk] = []

    def covered(self, start: int, end: int) -> list[Element]:
        return [e for e in self.elements if e.char_start < end and e.char_end > start]

    def add(
        self,
        start: int,
        end: int,
        *,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        path: list[str] | None = None,
    ) -> Chunk | None:
        """A chunk of the document's ``[start, end)`` span, or of ``text`` standing for it.
        Its heading path is the one in force at ``start`` unless ``path`` gives it."""
        body = self.document.text[start:end] if text is None else text
        if not body.strip():
            return None
        covered = self.covered(start, end)
        pages = [e.page for e in covered if e.page is not None]
        path = _heading_path(self.headings, start) if path is None else path
        ordinal = len(self.chunks)
        chunk = Chunk(
            chunk_id=chunk_id(self.document.document_id, self.cache_key, ordinal),
            document_id=self.document.document_id,
            ordinal=ordinal,
            text=body,
            index_text=render_index_text(
                self.index_text,
                {
                    "title": self.title,
                    "heading_path": " > ".join(path),
                    "section": path[-1] if path else None,
                    "text": body,
                },
            ),
            element_ids=[e.id for e in covered],
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            char_start=start,
            char_end=end,
            heading_path=path,
            tokens={self.estimator: self.count(body)},
            metadata=metadata or ({"section": path[-1]} if path else {}),
        )
        self.chunks.append(chunk)
        return chunk


def _start(document: ParsedDocument) -> int:
    elements = body_elements(document)
    if not elements:
        raise ComponentError("empty_document")
    return elements[0].char_start


def _finish(
    chunks: list[Chunk], context: StageContext | None, estimator: str, max_chunks: int = 100_000
) -> list[Chunk]:
    if not chunks:
        raise ComponentError("empty_document")
    if len(chunks) > max_chunks:
        raise ComponentError("too_many_chunks")
    record_metrics(context, chunks, estimator)
    return chunks


class _Chunker(ChunkComponent):
    """Shared plumbing: the estimator from the parameters, and its cache-key environment."""

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        return estimator_env(getattr(params, "token_estimator", CHARS))

    def maker(self, document, params, cache_key, label, index_text) -> Maker:
        estimator = getattr(params, "token_estimator", CHARS)
        return Maker(
            document,
            cache_key=cache_key,
            label=label,
            index_text=index_text,
            estimator=estimator,
            count=counter(estimator),
        )


# --- chunk/sections ----------------------------------------------------------------------------

_CONTENT = ("text", "table")
_SEARCH_CHARS_PER_TOKEN = 16  # how far a piece of max_tokens tokens can reach, in characters


class SectionsParams(_Params):
    max_tokens: int = Field(default=400, ge=16, le=32_000, description="Tokens per chunk at most")
    min_tokens: int = Field(
        default=200,
        ge=0,
        le=32_000,
        description="A chunk smaller than this takes in the next subsection, if it fits",
    )
    overlap: int = Field(
        default=0,
        ge=0,
        le=16_000,
        description="Tokens of whole sentences repeated when a section splits; 0 for none",
    )
    token_estimator: EstimatorName = Field(default=CHARS)
    max_chunks: int = Field(
        default=10_000, ge=1, le=100_000, description="More refuses the document"
    )

    @model_validator(mode="after")
    def _sizes(self) -> "SectionsParams":
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens can't be more than max_tokens")
        if self.overlap * 2 > self.max_tokens:
            raise ValueError("overlap can't be more than half of max_tokens")
        return self


@dataclass(frozen=True)
class _Part:
    """A span of the document in a chunk: a heading, text, a table (or rows of one) or a
    carried tail. ``text`` stands for the span when the chunk shows something else: a table's
    rows with its header repeated."""

    start: int
    end: int
    kind: Literal["heading", "text", "table", "carry"]
    text: str | None = None


class _SectionPacker:
    """Packs a document's elements into chunks (``Sections`` says how)."""

    def __init__(self, document: ParsedDocument, params: SectionsParams, count) -> None:
        self.text = document.text
        self.params = params
        self.count = count
        self.drafts: list[list[_Part]] = []
        self.parts: list[_Part] = []

    # --- the open chunk ----------------------------------------------------------------------

    def render(self, parts: Sequence[_Part]) -> str:
        """A chunk's text: the document's own text across adjacent spans, so a chunk of whole
        spans is a slice of it; a blank line on either side of a span shown otherwise."""
        out: list[str] = []
        for index, part in enumerate(parts):
            if index:
                before = parts[index - 1]
                gap = self.text[before.end : part.start] if before.end <= part.start else "x"
                verbatim = before.text is None and part.text is None and not gap.strip()
                out.append(gap if verbatim else ELEMENT_SEPARATOR)
            out.append(self.text[part.start : part.end] if part.text is None else part.text)
        return "".join(out)

    def fits(self, *extra: _Part) -> bool:
        return self.count(self.render([*self.parts, *extra])) <= self.params.max_tokens

    def has_content(self) -> bool:
        return any(part.kind in _CONTENT for part in self.parts)

    def flush(self, *, carry: bool) -> None:
        """Close the open chunk. Headings at its end move on to the next with the content they
        head; with ``carry``, the next starts with the end of the closed one's last text: whole
        sentences, up to ``overlap`` tokens."""
        trailing: list[_Part] = []
        while self.parts and self.parts[-1].kind == "heading":
            trailing.insert(0, self.parts.pop())
        closed = self.parts
        if any(part.kind in _CONTENT for part in closed):
            self.drafts.append(closed)
        self.parts = trailing
        last = closed[-1] if closed else None
        if carry and not trailing and last is not None and last.kind == "text":
            tail = self.tail(last)
            if tail is not None:
                self.parts.append(tail)

    def tail(self, part: _Part) -> _Part | None:
        if not self.params.overlap:
            return None
        start = None
        for sentence_start, _ in reversed(sentence_spans(self.text, part.start, part.end)):
            if self.count(self.text[sentence_start : part.end]) > self.params.overlap:
                break
            start = sentence_start
        return _Part(start, part.end, "carry") if start is not None else None

    # --- packing -----------------------------------------------------------------------------

    def close_section(self) -> None:
        """Close the open chunk at a top-level heading or the document's end. Headings after
        its content head nothing in their section, so they become a chunk of their own rather
        than join the next section's."""
        trailing: list[_Part] = []
        while self.parts and self.parts[-1].kind == "heading":
            trailing.insert(0, self.parts.pop())
        if self.has_content():
            self.drafts.append(self.parts)
        if trailing:
            self.drafts.append(trailing)
        self.parts = []

    def heading(self, element: Element, top_level: int) -> None:
        if (element.level or 1) <= top_level:
            self.close_section()
        elif self.has_content() and self.count(self.render(self.parts)) >= self.params.min_tokens:
            self.flush(carry=False)
        self.parts.append(_Part(element.char_start, element.char_end, "heading"))

    def text_element(self, element: Element) -> None:
        whole = _Part(element.char_start, element.char_end, "text")
        if self.fits(whole):
            self.parts.append(whole)
            return
        if self.has_content():
            self.flush(carry=True)
            if self.fits(whole):
                self.parts.append(whole)
                return
        start, end = whole.start, whole.end
        while start < end:
            cut = self.longest_piece(start, end)
            if cut is None and self.parts and self.parts[-1].kind == "carry":
                self.parts.pop()  # the carried tail makes room
                continue
            cut = cut or self.character_cut(start, end)
            self.parts.append(_Part(start, cut, "text"))
            start = cut
            while start < end and self.text[start].isspace():
                start += 1
            if start < end:
                self.flush(carry=True)

    def longest_piece(self, start: int, end: int) -> int | None:
        """The furthest sentence end in ``[start, end]`` up to which the text fits beside the
        open chunk's parts, else the furthest word end in the first sentence; None if no word
        fits."""
        reach = min(end, start + self.params.max_tokens * _SEARCH_CHARS_PER_TOKEN)
        spans = sentence_spans(self.text, start, reach)
        sentence_ends = [e for _, e in spans if reach == end or e < reach]
        found = self._furthest(start, sentence_ends)
        if found is not None or not spans:
            return found
        first_end = spans[0][1]
        word_ends = [start + m.end() for m in re.finditer(r"\S+", self.text[start:first_end])]
        return self._furthest(start, word_ends)

    def _furthest(self, start: int, ends: Sequence[int]) -> int | None:
        """The last of ``ends`` (ascending) whose piece fits, by binary search."""
        low, high, best = 0, len(ends) - 1, None
        while low <= high:
            middle = (low + high) // 2
            if self.fits(_Part(start, ends[middle], "text")):
                best, low = ends[middle], middle + 1
            else:
                high = middle - 1
        return best

    def character_cut(self, start: int, end: int) -> int:
        """A word longer than the room: as many characters as fit, one at least."""
        reach = min(end, start + self.params.max_tokens * _SEARCH_CHARS_PER_TOKEN)
        found = self._furthest(start, range(start + 1, reach + 1))
        return found if found is not None else start + 1

    def table(self, element: Element) -> None:
        whole = _Part(element.char_start, element.char_end, "table")
        if self.fits(whole):
            self.parts.append(whole)
            return
        if self.has_content():
            self.flush(carry=False)
            if self.fits(whole):
                self.parts.append(whole)
                return
        lines = _row_lines(element)
        header_rows = 1
        if element.table is not None and element.table.cells:
            header_rows = max(1, len({cell.r for cell in element.table.cells if cell.header}))
        if len(lines) <= header_rows:
            self.text_element(element)
            return
        header = self.text[lines[0][0] : lines[header_rows - 1][1]]
        rows = lines[header_rows:]

        def group(first: int, last: int) -> _Part:
            """Rows ``first`` to ``last``: from the table's start for the first group, else
            with the header in front."""
            if first == 0:
                return _Part(element.char_start, rows[last][1], "table")
            body = self.text[rows[first][0] : rows[last][1]]
            return _Part(rows[first][0], rows[last][1], "table", text=f"{header}\n{body}")

        first = 0
        while first < len(rows):
            last = first
            while last + 1 < len(rows) and self.fits(group(first, last + 1)):
                last += 1
            self.parts.append(group(first, last))  # a row at least, even one that's too long
            first = last + 1
            if first < len(rows):
                self.flush(carry=False)

    def pack(self, elements: Sequence[Element], checkpoint) -> list[list[_Part]]:
        levels = [element.level or 1 for element in elements if element.type == "heading"]
        top_level = min(levels, default=0)
        for element in elements:
            checkpoint()
            if element.type == "heading":
                self.heading(element, top_level)
            elif element.type == "table":
                self.table(element)
            else:
                self.text_element(element)
        self.close_section()
        return self.drafts


class Sections(_Chunker):
    """Heading-aware packing, the default chunker.

    Consecutive elements under the same heading path pack into a chunk of up to ``max_tokens``.
    A top-level heading (the shallowest level the document uses) always starts a new chunk, so
    no chunk crosses a top-level section; a deeper heading starts one only once the open chunk
    has ``min_tokens``, so short subsections share a chunk. A heading always stays with the
    content after it; headings with no content in their section are a chunk of their own.

    An element too big for the room left goes to the next chunk, and one too big for a chunk of
    its own splits at sentence ends, a sentence longer than a chunk at word ends, and a word
    longer than a chunk at characters. A table stays whole when it fits; otherwise its rows
    split into groups, each after the header row. With ``overlap``, a chunk that continues a
    section begins by repeating the previous chunk's last sentences, up to that many tokens.

    A chunk's text is the document's own text across its span, except a later row group, which
    shows the header row before its rows. Each chunk carries its elements, offsets and pages,
    and the heading path at its first content, whose last heading is its ``section``. Chunks
    stay within ``max_tokens`` except where a heading or a table row alone leaves no room.
    """

    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="chunk/sections",
        stage="chunk",
        version="1.0.0",
        title="Sections",
        description="Heading-aware chunks of up to 400 tokens: never across a top-level "
        "section, short subsections together, long elements split at sentences, tables whole "
        "or by rows under their header.",
        license="MIT",
        baseline=True,
    )
    Params: ClassVar[type[BaseModel]] = SectionsParams

    def chunk(
        self,
        document,
        params,
        *,
        cache_key,
        label=None,
        index_text=DEFAULT_INDEX_TEXT,
        context=None,
    ):
        params = cast(SectionsParams, params)
        maker = self.maker(document, params, cache_key, label, index_text)
        if not maker.elements:
            raise ComponentError("empty_document")
        packer = _SectionPacker(document, params, maker.count)
        drafts = packer.pack(maker.elements, context.checkpoint if context else lambda: None)
        if len(drafts) > params.max_chunks:
            raise ComponentError("too_many_chunks")
        for parts in drafts:
            first = next((part for part in parts if part.kind != "heading"), parts[0])
            path = _heading_path(maker.headings, first.start)
            metadata: dict[str, Any] = {"section": path[-1]} if path else {}
            if any(part.text is not None for part in parts):
                metadata["header_repeated"] = True
            maker.add(
                parts[0].start,
                parts[-1].end,
                text=packer.render(parts),
                metadata=metadata,
                path=path,
            )
        return _finish(maker.chunks, context, params.token_estimator, params.max_chunks)


# --- chunk/recursive ---------------------------------------------------------------------------


class RecursiveParams(_Params):
    size: int = Field(default=512, ge=8, le=32_000, description="Tokens per chunk at most")
    overlap: int = Field(default=64, ge=0, le=16_000, description="Tokens repeated")
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "],
        min_length=1,
        max_length=10,
        description="Tried in order; characters are the last resort",
    )
    token_estimator: EstimatorName = Field(default=CHARS)

    @model_validator(mode="after")
    def _overlap_fits(self) -> "RecursiveParams":
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        if any(not separator for separator in self.separators):
            raise ValueError("separators can't be empty")
        return self


def recursive_pieces(
    text: str, start: int, end: int, size: int, separators: Sequence[str], count
) -> list[Span]:
    """Spans of ``text[start:end]`` each within ``size`` tokens, cut at the first separator
    that's present, then the next, down to characters. Pieces keep their separators, so they
    tile the span."""
    if count(text[start:end]) <= size or end - start <= 1:
        return [(start, end)]
    for index, separator in enumerate(separators):
        cuts = [m.end() for m in re.finditer(re.escape(separator), text[start:end])]
        cuts = [start + cut for cut in cuts if start + cut < end]
        if not cuts:
            continue
        pieces: list[Span] = []
        position = start
        for cut in [*cuts, end]:
            if cut <= position:
                continue
            if count(text[position:cut]) <= size:
                pieces.append((position, cut))
            else:
                pieces += recursive_pieces(
                    text, position, cut, size, separators[index + 1 :], count
                )
            position = cut
        return pieces
    # No separator left: cut by characters, as many as fit.
    step = max(1, size * 4)
    return [(position, min(position + step, end)) for position in range(start, end, step)]


def pack(pieces: list[Span], text: str, size: int, overlap: int, count) -> list[Span]:
    """Consecutive pieces packed into spans within ``size`` tokens, each repeating pieces worth
    up to ``overlap`` tokens from the end of the one before."""
    spans: list[Span] = []
    current: list[Span] = []
    for piece in pieces:
        candidate = [*current, piece]
        if current and count(text[candidate[0][0] : candidate[-1][1]]) > size:
            spans.append((current[0][0], current[-1][1]))
            kept: list[Span] = []
            for previous in reversed(current):
                if count(text[previous[0] : (kept or [previous])[-1][1]]) > overlap:
                    break
                kept.insert(0, previous)
            current = kept
            while current and count(text[current[0][0] : piece[1]]) > size:
                current.pop(0)
            current.append(piece)
        else:
            current = candidate
    if current:
        spans.append((current[0][0], current[-1][1]))
    return spans


class Recursive(_Chunker):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="chunk/recursive",
        stage="chunk",
        version="1.0.0",
        title="Recursive separators",
        description="Splits at blank lines, then lines, sentences, words and characters until "
        "pieces fit, then packs them into chunks with an overlap.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = RecursiveParams

    def chunk(
        self,
        document,
        params,
        *,
        cache_key,
        label=None,
        index_text=DEFAULT_INDEX_TEXT,
        context=None,
    ):
        params = cast(RecursiveParams, params)
        maker = self.maker(document, params, cache_key, label, index_text)
        text = document.text
        pieces = recursive_pieces(
            text, _start(document), len(text), params.size, params.separators, maker.count
        )
        for start, end in pack(pieces, text, params.size, params.overlap, maker.count):
            if context is not None:
                context.checkpoint()
            maker.add(*trim_span(text, start, end))
        return _finish(maker.chunks, context, params.token_estimator)


# --- chunk/sentence-window ---------------------------------------------------------------------


class SentenceWindowParams(_Params):
    window: int = Field(default=3, ge=1, le=50, description="Sentences per chunk")
    stride: int = Field(default=2, ge=1, le=50, description="Sentences each window moves on")
    token_estimator: EstimatorName = Field(default=CHARS)

    @model_validator(mode="after")
    def _stride_fits(self) -> "SentenceWindowParams":
        if self.stride > self.window:
            raise ValueError("stride can't be more than window: sentences would be skipped")
        return self


class SentenceWindow(_Chunker):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="chunk/sentence-window",
        stage="chunk",
        version="1.0.0",
        title="Sentence windows",
        description="Windows of a number of sentences, moving a stride of sentences at a time; "
        "a stride below the window overlaps them.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = SentenceWindowParams

    def chunk(
        self,
        document,
        params,
        *,
        cache_key,
        label=None,
        index_text=DEFAULT_INDEX_TEXT,
        context=None,
    ):
        params = cast(SentenceWindowParams, params)
        maker = self.maker(document, params, cache_key, label, index_text)
        sentences = [
            span
            for element in maker.elements
            for span in sentence_spans(document.text, element.char_start, element.char_end)
        ]
        if not sentences:
            raise ComponentError("empty_document")
        first = 0
        while first < len(sentences):
            if context is not None:
                context.checkpoint()
            group = sentences[first : first + params.window]
            maker.add(
                group[0][0], group[-1][1], metadata={"sentences": [first, first + len(group)]}
            )
            if first + params.window >= len(sentences):
                break
            first += params.stride
        return _finish(maker.chunks, context, params.token_estimator)


# --- chunk/semantic ----------------------------------------------------------------------------


class SemanticParams(_Params):
    breakpoint_percentile: float = Field(
        default=90, ge=50, le=99, description="Drops above this percentile become breakpoints"
    )
    buffer: int = Field(default=1, ge=0, le=5, description="Neighbours each sentence is read with")
    min_tokens: int = Field(default=64, ge=0, le=16_000)
    max_tokens: int = Field(default=512, ge=16, le=32_000)
    token_estimator: EstimatorName = Field(default=CHARS)

    @model_validator(mode="after")
    def _ordered(self) -> "SemanticParams":
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens can't be more than max_tokens")
        return self


def _vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(terms(text))
    vector = {term: count * idf.get(term, 0.0) for term, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    return {term: value / norm for term, value in vector.items()}


def _cosine(first: dict[str, float], second: dict[str, float]) -> float:
    if len(first) > len(second):
        first, second = second, first
    return sum(value * second.get(term, 0.0) for term, value in first.items())


def similarity_drops(texts: list[str], buffer: int) -> list[float]:
    """1 - the TF-IDF cosine between each sentence (with ``buffer`` neighbours) and the next."""
    groups = [" ".join(texts[max(0, i - buffer) : i + buffer + 1]) for i in range(len(texts))]
    documents = [set(terms(group)) for group in groups]
    frequency = Counter(term for found in documents for term in found)
    total = len(groups)
    idf = {term: math.log((1 + total) / (1 + n)) + 1 for term, n in frequency.items()}
    vectors = [_vector(group, idf) for group in groups]
    return [1 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]


class Semantic(_Chunker):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="chunk/semantic",
        stage="chunk",
        version="1.1.0",
        title="Semantic breakpoints",
        description="Breaks where adjacent sentences stop sharing terms (TF-IDF cosine with "
        "neighbours, the largest drops), within a token range.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = SemanticParams

    def chunk(
        self,
        document,
        params,
        *,
        cache_key,
        label=None,
        index_text=DEFAULT_INDEX_TEXT,
        context=None,
    ):
        params = cast(SemanticParams, params)
        maker = self.maker(document, params, cache_key, label, index_text)
        text = document.text
        sentences = [
            piece
            for element in maker.elements
            for span in sentence_spans(text, element.char_start, element.char_end)
            # A "sentence" over the limit (a table, a list without full stops) is cut first.
            for piece in recursive_pieces(
                text, *span, params.max_tokens, ["\n", ". ", " "], maker.count
            )
        ]
        if not sentences:
            raise ComponentError("empty_document")
        drops = similarity_drops([text[s:e] for s, e in sentences], params.buffer)
        ranked = sorted(drops)
        cut = (
            ranked[min(len(ranked) - 1, int(len(ranked) * params.breakpoint_percentile / 100))]
            if ranked
            else 1.0
        )
        group: list[Span] = []
        groups: list[list[Span]] = []
        for index, sentence in enumerate(sentences):
            candidate = [*group, sentence]
            if group and maker.count(text[candidate[0][0] : candidate[-1][1]]) > params.max_tokens:
                groups.append(group)
                group = [sentence]
                continue
            group = candidate
            breaking = index < len(drops) and drops[index] >= cut and drops[index] > 0
            if breaking and maker.count(text[group[0][0] : group[-1][1]]) >= params.min_tokens:
                groups.append(group)
                group = []
        if group:
            groups.append(group)
        for spans in groups:
            if context is not None:
                context.checkpoint()
            maker.add(spans[0][0], spans[-1][1])
        return _finish(maker.chunks, context, params.token_estimator)


# --- chunk/table-aware -------------------------------------------------------------------------


class TableAwareParams(_Params):
    max_tokens: int = Field(default=512, ge=16, le=32_000, description="Text chunks at most")
    table_max_tokens: int = Field(
        default=1024, ge=16, le=64_000, description="A table this size or smaller stays whole"
    )
    token_estimator: EstimatorName = Field(default=CHARS)


def _row_lines(element: Element) -> list[Span]:
    """Each line of a table element's text as offsets in the document."""
    spans: list[Span] = []
    position = element.char_start
    for line in element.text.split("\n"):
        spans.append((position, position + len(line)))
        position += len(line) + 1
    return spans


class TableAware(_Chunker):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="chunk/table-aware",
        stage="chunk",
        version="1.0.0",
        title="Table-aware",
        description="Packs text at element boundaries; keeps a table whole up to a limit, "
        "otherwise splits it into row groups that each repeat the header row.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = TableAwareParams

    def chunk(
        self,
        document,
        params,
        *,
        cache_key,
        label=None,
        index_text=DEFAULT_INDEX_TEXT,
        context=None,
    ):
        params = cast(TableAwareParams, params)
        maker = self.maker(document, params, cache_key, label, index_text)
        text = document.text
        pending: list[Span] = []

        def flush() -> None:
            if not pending:
                return
            pieces: list[Span] = []
            for start, end in pending:
                pieces += recursive_pieces(
                    text, start, end, params.max_tokens, ["\n\n", "\n", ". ", " "], maker.count
                )
            for start, end in pack(pieces, text, params.max_tokens, 0, maker.count):
                maker.add(*trim_span(text, start, end))
            pending.clear()

        for element in maker.elements:
            if context is not None:
                context.checkpoint()
            if element.type != "table":
                pending.append((element.char_start, element.char_end))
                continue
            flush()
            if maker.count(element.text) <= params.table_max_tokens:
                maker.add(element.char_start, element.char_end, metadata={"table": element.id})
                continue
            lines = _row_lines(element)
            header_rows = 1
            if element.table is not None:
                header_rows = max(1, len({c.r for c in element.table.cells if c.header}))
            header = "\n".join(text[s:e] for s, e in lines[:header_rows])
            budget = params.table_max_tokens - maker.count(header)
            group: list[Span] = []
            first_group = True
            for line in lines[header_rows:]:
                candidate = [*group, line]
                if group and maker.count("\n".join(text[s:e] for s, e in candidate)) > budget:
                    self._rows(maker, text, header, group, element, first_group)
                    group, first_group = [line], False
                else:
                    group = candidate
            if group:
                self._rows(maker, text, header, group, element, first_group)
        flush()
        return _finish(maker.chunks, context, params.token_estimator)

    @staticmethod
    def _rows(
        maker: Maker, text: str, header: str, rows: list[Span], element: Element, first: bool
    ) -> None:
        """One row group with the header in front. The first group's span starts at the table,
        so the header's own text is covered once."""
        body = "\n".join(text[s:e] for s, e in rows)
        maker.add(
            element.char_start if first else rows[0][0],
            rows[-1][1],
            text=f"{header}\n{body}",
            metadata={"table": element.id, "header_repeated": True},
        )


# --- chunk/parent-child ------------------------------------------------------------------------


class ParentChildParams(_Params):
    child_tokens: int = Field(default=128, ge=8, le=8_000, description="Small chunks: searched")
    child_overlap: int = Field(default=16, ge=0, le=4_000)
    parent_max_tokens: int = Field(
        default=1024, ge=16, le=32_000, description="A section longer than this splits"
    )
    token_estimator: EstimatorName = Field(default=CHARS)

    @model_validator(mode="after")
    def _sizes(self) -> "ParentChildParams":
        if self.child_overlap >= self.child_tokens:
            raise ValueError("child_overlap must be smaller than child_tokens")
        if self.child_tokens > self.parent_max_tokens:
            raise ValueError("child_tokens can't be more than parent_max_tokens")
        return self


def sections(document: ParsedDocument, elements: list[Element]) -> list[Span]:
    """Spans from each heading to the next: a document's sections."""
    starts = [e.char_start for e in elements if e.type == "heading"]
    first = elements[0].char_start
    bounds = sorted({first, *(s for s in starts if s > first)})
    ends = [*bounds[1:], elements[-1].char_end]
    return [(start, end) for start, end in zip(bounds, ends, strict=True) if start < end]


def parent_id(document_id: str, cache_key: str, ordinal: int) -> str:
    return hashlib.sha256(f"{document_id}:{cache_key}:parent:{ordinal}".encode()).hexdigest()[:32]


class ParentChild(_Chunker):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="chunk/parent-child",
        stage="chunk",
        version="1.0.0",
        title="Parent and child",
        description="Small child chunks are embedded and searched; each carries its parent "
        "section, and retrieval returns the parents.",
        license="MIT",
    )
    Params: ClassVar[type[BaseModel]] = ParentChildParams

    def chunk(
        self,
        document,
        params,
        *,
        cache_key,
        label=None,
        index_text=DEFAULT_INDEX_TEXT,
        context=None,
    ):
        params = cast(ParentChildParams, params)
        maker = self.maker(document, params, cache_key, label, index_text)
        text = document.text
        if not maker.elements:
            raise ComponentError("empty_document")
        parents: list[Span] = []
        for start, end in sections(document, maker.elements):
            pieces = recursive_pieces(
                text, start, end, params.parent_max_tokens, ["\n\n", "\n", ". ", " "], maker.count
            )
            parents += pack(pieces, text, params.parent_max_tokens, 0, maker.count)
        for number, (start, end) in enumerate(parents):
            start, end = trim_span(text, start, end)
            if start >= end:
                continue
            covered = maker.covered(start, end)
            pages = [e.page for e in covered if e.page is not None]
            path = _heading_path(maker.headings, start)
            parent = {
                "id": parent_id(document.document_id, cache_key, number),
                "text": text[start:end],
                "char_start": start,
                "char_end": end,
                "heading_path": path,
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "element_ids": [e.id for e in covered],
            }
            pieces = recursive_pieces(
                text, start, end, params.child_tokens, ["\n\n", "\n", ". ", " "], maker.count
            )
            for child_start, child_end in pack(
                pieces, text, params.child_tokens, params.child_overlap, maker.count
            ):
                if context is not None:
                    context.checkpoint()
                span = trim_span(text, child_start, child_end)
                if span[0] < span[1]:
                    maker.add(
                        *span, metadata={"parent": parent, "section": path[-1] if path else None}
                    )
        return _finish(maker.chunks, context, params.token_estimator)


COMPONENTS = [Sections, Recursive, SentenceWindow, Semantic, TableAware, ParentChild]
