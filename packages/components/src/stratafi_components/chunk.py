"""The chunk and enrich stages.

A chunker turns a cleaned document into ``stratafi.chunk.v1`` chunks: the text, the index text
(exactly what is embedded and keyword-indexed), and where it came from: the elements it covers,
its offsets in the document's text, its pages and its heading path.

- ``chunk/fixed-tokens`` cuts fixed windows with an overlap, by the ``chars/4`` estimate (never
  inside a word) or by a tiktoken encoding (exact token windows).
- ``enrich/none`` passes chunks through.

The default chunker, ``chunk/sections``, and the others are in ``stratafi_components.chunkers``.
Offsets are a chunk's span of the document's text, from its first piece's start to its last
piece's end.
"""

import hashlib
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stratafi_components.estimators import CHARS as CHARS_ESTIMATOR
from stratafi_components.estimators import chars_estimate as estimate_tokens
from stratafi_core.artifacts import Chunk, Element, ParsedDocument
from stratafi_core.components import Component, ComponentError, ComponentInfo, StageContext
from stratafi_core.specs import DEFAULT_INDEX_TEXT

_WORD_RE = re.compile(r"\S+")
_FIELD_RE = re.compile(r"\{(title|heading_path|section|text|context)\}")
CHARS_PER_TOKEN = 4  # what chars/4 counts as a token


# --- shared --------------------------------------------------------------------------------------


def label_title(label: str | None) -> str | None:
    """A title for a document without one, from its key or file name: ``pto-leave`` reads
    ``Pto leave``."""
    return label.replace("-", " ").capitalize() if label else None


def render_index_text(template: str, fields: dict[str, str | None]) -> str:
    """Fill ``template`` one line at a time and drop lines that come out empty, so a document
    without a title or a chunk without a heading path leaves no blank line."""
    lines = []
    for line in template.split("\n"):
        rendered = _FIELD_RE.sub(lambda match: fields.get(match.group(1)) or "", line)
        if rendered:
            lines.append(rendered)
    return "\n".join(lines)


def chunk_id(document_id: str, cache_key: str, ordinal: int) -> str:
    """A hash of the document, the chunker's cache key and the ordinal (chunk.v1)."""
    return hashlib.sha256(f"{document_id}:{cache_key}:{ordinal}".encode()).hexdigest()[:32]


def body_elements(document: ParsedDocument) -> list[Element]:
    """The elements a chunker reads: everything but the title, and nothing empty."""
    return [e for e in document.elements if e.type != "title" and e.text.strip()]


class ChunkComponent(Component):
    """A chunker: cleaned document → chunks."""

    def chunk(
        self,
        document: ParsedDocument,
        params: BaseModel,
        *,
        cache_key: str,
        label: str | None = None,
        index_text: str = DEFAULT_INDEX_TEXT,
        context: StageContext | None = None,
    ) -> list[Chunk]:
        raise NotImplementedError


def record_metrics(context: StageContext | None, chunks: Sequence[Chunk], estimator: str) -> None:
    if context is None or not chunks:
        return
    counts = sorted(chunk.tokens.get(estimator, 0) for chunk in chunks)
    context.metric("chunks", len(chunks))
    context.metric("tokens_min", counts[0])
    context.metric("tokens_p50", counts[len(counts) // 2])
    context.metric("tokens_max", counts[-1])


# --- chunk/fixed-tokens --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def tiktoken_encoding(name: str) -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:  # an encoding that can't load (offline, unknown) is a component error
        raise ComponentError("tokenizer_unavailable") from None


class FixedTokensParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int = Field(default=512, ge=8, le=32_000, description="Tokens per chunk")
    overlap: int = Field(default=64, ge=0, le=16_000, description="Tokens repeated from the last")
    token_estimator: Literal["chars/4", "tiktoken:cl100k_base", "tiktoken:o200k_base"] = Field(
        default="chars/4", description="chars/4 never cuts a word; tiktoken counts exactly"
    )

    @model_validator(mode="after")
    def _overlap_fits(self) -> "FixedTokensParams":
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        return self


def _char_windows(text: str, start: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Windows of about ``size * 4`` characters that end and start at word boundaries."""
    words = [match.span() for match in _WORD_RE.finditer(text, start)]
    if not words:
        return []
    limit, back = size * CHARS_PER_TOKEN, overlap * CHARS_PER_TOKEN
    windows, first = [], 0
    while first < len(words):
        window_start = words[first][0]
        last = first
        while last + 1 < len(words) and words[last + 1][1] - window_start <= limit:
            last += 1
        windows.append((window_start, words[last][1]))
        if last + 1 >= len(words):
            break
        # The next window starts at the first word inside the overlap, but always moves on.
        next_first = last + 1
        while next_first - 1 > first and words[last][1] - words[next_first - 1][0] <= back:
            next_first -= 1
        first = next_first
    return windows


def _token_windows(text: str, start: int, encoding: Any, size: int, overlap: int):
    body = text[start:]
    tokens = encoding.encode(body, disallowed_special=())
    if not tokens:
        return []
    _, offsets = encoding.decode_with_offsets(tokens)
    windows, first = [], 0
    while first < len(tokens):
        last = min(first + size, len(tokens))
        begin = start + offsets[first]
        end = start + offsets[last] if last < len(tokens) else len(text)
        windows.append((begin, end, last - first))
        if last >= len(tokens):
            break
        first = last - overlap
    return windows


class FixedTokens(ChunkComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="chunk/fixed-tokens",
        stage="chunk",
        version="1.0.0",
        title="Fixed token windows",
        description="Windows of a fixed number of tokens with an overlap, counted as "
        "characters / 4 (never cutting a word) or with a tiktoken encoding.",
        license="MIT (tiktoken: MIT)",
    )
    Params: ClassVar[type[BaseModel]] = FixedTokensParams

    @classmethod
    def static_env(cls, params: BaseModel) -> dict[str, Any]:
        estimator = cast(FixedTokensParams, params).token_estimator
        if not estimator.startswith("tiktoken:"):
            return {}
        import tiktoken

        return {"tiktoken": tiktoken.__version__}

    def chunk(
        self,
        document: ParsedDocument,
        params: BaseModel,
        *,
        cache_key: str,
        label: str | None = None,
        index_text: str = DEFAULT_INDEX_TEXT,
        context: StageContext | None = None,
    ) -> list[Chunk]:
        params = cast(FixedTokensParams, params)
        elements = body_elements(document)
        if not elements:
            raise ComponentError("empty_document")
        text, start = document.text, elements[0].char_start
        estimator = params.token_estimator
        if estimator == CHARS_ESTIMATOR:
            windows = [
                (begin, end, estimate_tokens(text[begin:end]))
                for begin, end in _char_windows(text, start, params.size, params.overlap)
            ]
        else:
            encoding = tiktoken_encoding(estimator.removeprefix("tiktoken:"))
            windows = _token_windows(text, start, encoding, params.size, params.overlap)
        title = document.title or label_title(label)
        headings = [e for e in document.elements if e.type == "heading"]
        chunks = []
        for ordinal, (begin, end, count) in enumerate(windows):
            if context is not None:
                context.checkpoint()
            covered = [e for e in elements if e.char_start < end and e.char_end > begin]
            pages = [e.page for e in covered if e.page is not None]
            path = _heading_path(headings, begin)
            body = text[begin:end]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(document.document_id, cache_key, ordinal),
                    document_id=document.document_id,
                    ordinal=ordinal,
                    text=body,
                    index_text=render_index_text(
                        index_text,
                        {
                            "title": title,
                            "heading_path": " > ".join(path),
                            "section": path[-1] if path else None,
                            "text": body,
                        },
                    ),
                    element_ids=[e.id for e in covered],
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    char_start=begin,
                    char_end=end,
                    heading_path=path,
                    tokens={estimator: count},
                    metadata={},
                )
            )
        record_metrics(context, chunks, estimator)
        return chunks


def _heading_path(headings: list[Element], position: int) -> list[str]:
    """The headings in force at ``position``: each level's latest heading before it."""
    stack: list[tuple[int, str]] = []
    for heading in headings:
        if heading.char_start > position:
            break
        level = heading.level or 1
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading.text))
    return [text for _, text in stack]


# --- enrich/none ---------------------------------------------------------------------------------


class EnrichComponent(Component):
    def enrich(
        self, chunks: list[Chunk], params: BaseModel, context: StageContext | None = None
    ) -> list[Chunk]:
        raise NotImplementedError


class EnrichNone(EnrichComponent):
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="enrich/none",
        stage="enrich",
        version="1.0.0",
        title="No enrichment",
        description="Chunks pass through unchanged.",
        license="MIT",
        baseline=True,
    )

    def enrich(
        self, chunks: list[Chunk], params: BaseModel, context: StageContext | None = None
    ) -> list[Chunk]:
        return list(chunks)


COMPONENTS: list[type[Component]] = [FixedTokens, EnrichNone]
