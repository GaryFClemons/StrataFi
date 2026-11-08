"""Chunkers and enrichment: properties that hold for every chunker on any
document, parent-child retrieval returning parents, the estimators, and the context enricher
behind its switch."""

import json
import re
import shutil
from pathlib import Path

import httpx2
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stratafi_components import default_registry
from stratafi_components.chunkers import ParentChild
from stratafi_components.detect import detect
from stratafi_components.embed import MODELS_SERVICE
from stratafi_components.enrich import PROMPT_VERSION, LlmContext
from stratafi_components.estimators import chars_estimate, counter, sentence_spans
from stratafi_components.extract import raw_document
from stratafi_components.index import Bm25Index, IndexInput, KeywordIndex
from stratafi_components.models import GatewayProvider
from stratafi_components.retrieval import EntryMeta, RetrievalEngine, with_parents
from stratafi_core.artifacts import Chunk, ParsedDocument, build_text
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.hashing import content_hash
from stratafi_core.specs import RetrievalConfig

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = default_registry()
CHUNKERS = [
    "chunk/sections",
    "chunk/recursive",
    "chunk/sentence-window",
    "chunk/semantic",
    "chunk/table-aware",
    "chunk/parent-child",
]


def document_of(*elements: tuple[str, str]) -> ParsedDocument:
    texts = [text for _, text in elements]
    text, offsets = build_text(texts)
    return ParsedDocument(
        document_id="d-1",
        parser={"id": "parse/test", "version": "1.0.0", "params_hash": content_hash({})},
        title="Policy",
        media_type="text/plain",
        elements=[
            {"id": f"e{i}", "type": kind, "text": body, "page": 1, "char_start": s, "char_end": e}
            for i, ((kind, body), (s, e)) in enumerate(zip(elements, offsets, strict=True))
        ],
        text=text,
    )


def chunk(chunker: str, document: ParsedDocument, **params) -> list[Chunk]:
    component = REGISTRY.get(chunker)()
    return component.chunk(
        document, component.resolve_params(params), cache_key="k" * 64, label="p"
    )


WORDS = st.text(alphabet="abcdefghij", min_size=1, max_size=12)
SENTENCE = st.lists(WORDS, min_size=1, max_size=18).map(
    lambda words: " ".join(words).capitalize() + "."
)
PARAGRAPH = st.lists(SENTENCE, min_size=1, max_size=8).map(" ".join)
ELEMENT = st.one_of(
    PARAGRAPH.map(lambda text: ("paragraph", text)),
    st.lists(WORDS, min_size=1, max_size=4).map(lambda words: ("heading", " ".join(words).title())),
    st.lists(st.lists(WORDS, min_size=2, max_size=3).map(" | ".join), min_size=2, max_size=12).map(
        lambda rows: ("table", "\n".join(rows))
    ),
)
DOCUMENTS = st.lists(ELEMENT, min_size=1, max_size=12).filter(
    lambda elements: any(kind != "heading" for kind, _ in elements)
)
SMALL = {
    "chunk/recursive": {"size": 40, "overlap": 8},
    "chunk/sentence-window": {"window": 3, "stride": 2},
    "chunk/semantic": {"min_tokens": 8, "max_tokens": 48},
    "chunk/table-aware": {"max_tokens": 40, "table_max_tokens": 24},
    "chunk/parent-child": {"child_tokens": 24, "child_overlap": 4, "parent_max_tokens": 96},
    "chunk/sections": {"max_tokens": 48, "min_tokens": 16, "overlap": 8},
}


@pytest.mark.parametrize("chunker", CHUNKERS)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(elements=DOCUMENTS)
def test_every_chunker_keeps_its_properties(chunker: str, elements) -> None:
    """Property tests for every chunker: chunks are valid, in order, inside the text, with
    unique IDs; together they cover every word; the same input gives the same chunks."""
    document = document_of(*elements)
    chunks = chunk(chunker, document, **SMALL[chunker])
    assert chunks
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for item in chunks:
        assert item.text.strip()
        assert 0 <= item.char_start <= item.char_end <= len(document.text)
        assert item.document_id == "d-1"
        assert item.index_text.endswith(item.text) or "\n" in item.index_text
        assert all(count >= 0 for count in item.tokens.values())
    covered = [(c.char_start, c.char_end) for c in chunks]
    # Every word of every body element falls inside some chunk's span.
    for element in document.elements:
        for match in re.finditer(r"\S+", element.text):
            begin, finish = element.char_start + match.start(), element.char_start + match.end()
            assert any(s <= begin and finish <= e for s, e in covered), (chunker, match.group())
    assert chunk(chunker, document, **SMALL[chunker]) == chunks


@pytest.mark.parametrize(
    ("chunker", "limit_key"),
    [
        ("chunk/recursive", "size"),
        ("chunk/semantic", "max_tokens"),
        ("chunk/sections", "max_tokens"),
        ("chunk/parent-child", "child_tokens"),
    ],
)
@settings(max_examples=40, deadline=None)
@given(elements=DOCUMENTS)
def test_size_limits_hold(chunker: str, limit_key: str, elements) -> None:
    document = document_of(*elements)
    limit = SMALL[chunker][limit_key]
    for item in chunk(chunker, document, **SMALL[chunker]):
        # One word longer than the limit can't be split at a separator: characters cut it.
        assert (
            chars_estimate(item.text) <= limit
            or len(item.text.split()) == 1
            or max(chars_estimate(word) for word in item.text.split()) > limit // 2
        )


# --- tables and parents --------------------------------------------------------------------


def test_a_large_table_splits_into_row_groups_that_repeat_the_header() -> None:
    rows = ["Fee | Amount"] + [f"Item {i} | {i * 5} dollars" for i in range(40)]
    document = document_of(
        ("heading", "Fees"), ("table", "\n".join(rows)), ("paragraph", "After the table.")
    )
    chunks = chunk("chunk/table-aware", document, table_max_tokens=64)
    tables = [c for c in chunks if c.metadata.get("table") == "e1"]
    assert len(tables) > 1
    assert all(c.text.startswith("Fee | Amount\n") for c in tables)
    assert all(c.metadata["header_repeated"] for c in tables)
    small = chunk("chunk/table-aware", document, table_max_tokens=4000)
    assert [c.text for c in small if c.metadata.get("table")] == ["\n".join(rows)]


def test_parent_child_retrieval_returns_parents() -> None:
    """Small chunks are searched; the final passages are their parent sections, once
    each."""
    document = document_of(
        ("heading", "Wires"),
        ("paragraph", "Tellers confirm every wire request by calling the number on file. " * 6),
        ("heading", "Holds"),
        (
            "paragraph",
            "Deposits over five thousand dollars may be held for two business days. " * 6,
        ),
    )
    children = chunk(
        "chunk/parent-child", document, child_tokens=32, child_overlap=4, parent_max_tokens=400
    )
    parents = {c.metadata["parent"]["id"] for c in children}
    assert len(parents) == 2
    assert len(children) > len(parents)
    built = Bm25Index().build(
        [IndexInput("c" * 64, children)], Bm25Index.resolve_params({}), StageContext()
    )
    keyword = KeywordIndex.load(built.manifest, built.payload)
    config = RetrievalConfig.model_validate(
        {
            "schema": "stratafi.retrieval.v1",
            "name": "kw",
            "retrievers": {"dense": None, "keyword": {"top_n": 50}},
            "fusion": {"method": "none"},
            "candidates": 50,
            "top_k": 5,
        }
    )
    chunks = with_parents({c.chunk_id: c for c in children})
    engine = RetrievalEngine(
        config,
        entries=[EntryMeta(document_id="d-1", name="policy.md")],
        chunks=chunks,
        keyword=keyword,
    )
    result = engine.search("calling the number on file", asker=None, day=None)
    final = result.checkpoints["final"]
    assert final[0][0] in parents
    ids = [chunk_id for chunk_id, _ in final]
    assert len(ids) == len(set(ids))  # each parent once
    assert set(ids) <= parents
    assert chunks[final[0][0]].text.startswith("Wires")
    assert len(result.checkpoints["reranked"]) > len(final)


# --- estimators -------------------------------------------------------------------------------


def test_estimators() -> None:
    assert counter("chars/4")("abcdefgh") == 2
    with pytest.raises(ComponentError):
        counter("unknown")
    for name in ("tiktoken:cl100k_base", "hf:bert-base-uncased"):
        codes: list[str] = []
        try:
            count = counter(name)
        except ComponentError as error:
            codes.append(error.code)  # offline: the files can't download
        else:
            assert count("Tellers confirm wires.") > 0
        assert codes in ([], ["tokenizer_unavailable"])
    assert ParentChild.static_env(ParentChild.resolve_params({"token_estimator": "chars/4"})) == {}
    with pytest.raises(Exception, match="token_estimator"):
        ParentChild.resolve_params({"token_estimator": "hf:"})


def test_sentences_split_at_ends_and_blank_lines() -> None:
    text = "Call back first. Then send it! Is it 5 p.m.? Yes.\n\nNew block here"
    assert [text[s:e] for s, e in sentence_spans(text)] == [
        "Call back first.",
        "Then send it!",
        "Is it 5 p.m.?",
        "Yes.",
        "New block here",
    ]


# --- enrichment ---------------------------------------------------------------------------------


def chat_gateway(seen: list) -> GatewayProvider:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        last = body["messages"][-1]["content"].splitlines()[-1]
        return httpx2.Response(
            200,
            json={
                "choices": [{"message": {"content": f"About: {last[:20]}"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
            headers={"X-Model-Revision": "chat-rev-1"},
        )

    return GatewayProvider(
        "http://gw",
        client=httpx2.Client(base_url="http://gw", transport=httpx2.MockTransport(handler)),
    )


def test_context_enrichment_adds_a_line_and_records_its_cost() -> None:
    document = document_of(("heading", "Wires"), ("paragraph", "Call back first. " * 10))
    chunks = chunk("chunk/recursive", document, size=40, overlap=0)
    seen: list = []
    context = StageContext(
        services={MODELS_SERVICE: chat_gateway(seen), "enrichment_enabled": True}
    )
    enriched = LlmContext().enrich(chunks, LlmContext.resolve_params({}), context)
    assert len(enriched) == len(chunks) == len(seen)
    assert all(item.context and item.context.startswith("About:") for item in enriched)
    assert enriched[0].index_text.startswith(enriched[0].context)
    assert enriched[0].metadata["context_prompt"] == PROMPT_VERSION
    assert enriched[0].metadata["context_revision"] == "chat-rev-1"
    assert seen[0]["model"] == "chat-default"
    assert context.metrics["tokens"] == 55 * len(chunks)
    assert context.metrics["gateway.calls"] == len(chunks)
    assert LlmContext.static_env(LlmContext.resolve_params({})) == {"prompt": PROMPT_VERSION}
    assert LlmContext.info.deterministic is False


def test_context_enrichment_waits_for_its_switch() -> None:
    document = document_of(("paragraph", "Call back first."))
    chunks = chunk("chunk/recursive", document)
    with pytest.raises(ComponentError) as error:
        LlmContext().enrich(
            chunks,
            LlmContext.resolve_params({}),
            StageContext(services={MODELS_SERVICE: chat_gateway([])}),
        )
    assert error.value.code == "enrichment_not_enabled"


def test_chunkers_read_real_parsed_documents(tmp_path: Path) -> None:
    path = tmp_path / "food-safety.md"
    shutil.copy(ROOT / "fixtures" / "formats" / "food-safety.md", path)
    raw = raw_document("doc", "0" * 64, path.stat().st_size, detect(path, path.name))
    parser = REGISTRY.get("parse/markdown")()
    document = parser.parse(path, raw, parser.resolve_params({}), StageContext())
    for chunker in CHUNKERS:
        chunks = chunk(chunker, document)
        assert chunks, chunker
        assert any(c.heading_path for c in chunks), chunker
