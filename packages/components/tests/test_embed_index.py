"""Embedding and indexes: the format fixtures through the fake gateway, exact
search, BM25's analysis and scoring, and the revision guard."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stratafi_components import default_registry
from stratafi_components.chunkers import Sections
from stratafi_components.clean import Whitespace, run_clean
from stratafi_components.detect import detect
from stratafi_components.embed import (
    FakeHashEmbed,
    GatewayEmbed,
    Vectors,
    embed_query,
    hash_embedding,
    vectors_from_bytes,
    vectors_to_bytes,
)
from stratafi_components.extract import raw_document
from stratafi_components.index import (
    Bm25Index,
    DenseIndex,
    ExactIndex,
    IndexInput,
    KeywordIndex,
)
from stratafi_components.keywords import ANALYZER, terms
from stratafi_components.models import GatewayProvider
from stratafi_components.retrieval import reciprocal_rank_fusion
from stratafi_core.artifacts import Chunk, VectorsSidecar
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.hashing import content_hash
from stratafi_testing.fake_gateway import FakeGateway, create_app

ROOT = Path(__file__).resolve().parents[3]
FORMATS = ROOT / "fixtures" / "formats"
# One rendering of each format fixture, each through its default parser.
FIXTURES = {
    "tips.md": "parse/markdown",
    "wholesale.docx": "parse/markitdown",
    "food-safety.pdf": "parse/pypdf",
}


def chunks_of(name: str) -> list[Chunk]:
    path = FORMATS / name
    data = path.read_bytes()
    raw = raw_document(
        path.stem, hashlib.sha256(data).hexdigest(), len(data), detect(path, path.name)
    )
    parser = default_registry().get(FIXTURES[name])()
    parsed = parser.parse_local(path, raw, parser.resolve_params({}))
    cleaned = run_clean(parsed, [("clean/whitespace", Whitespace(), Whitespace.resolve_params({}))])
    chunker = Sections()
    return chunker.chunk(
        cleaned, chunker.resolve_params({}), cache_key=content_hash(name), label=path.stem
    )


CORPUS = {name: chunks_of(name) for name in FIXTURES}


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def context(gateway: FakeGateway) -> StageContext:
    client = TestClient(create_app(gateway), base_url="http://gateway")
    provider = GatewayProvider("http://gateway", client=client, sleep=lambda seconds: None)
    return StageContext(services={"models": provider})


def embed_corpus(context: StageContext, alias: str = "embed-default") -> list[IndexInput]:
    component = GatewayEmbed()
    params = component.resolve_params({"alias": alias, "batch_size": 4})
    revision = component.revision_for(params, context)
    return [
        IndexInput(
            content_hash(name),
            chunks,
            component.embed(chunks, params, expected_revision=revision, context=context),
        )
        for name, chunks in CORPUS.items()
    ]


# --- exact search -------------------------------------------------------------------------------


def test_the_fixture_corpus_embeds_and_exact_search_finds_the_expected_neighbours(
    context: StageContext, gateway: FakeGateway
) -> None:
    inputs = embed_corpus(context)
    assert all(item.vectors is not None and item.vectors.sidecar.dims == 384 for item in inputs)
    built = ExactIndex().build(inputs, ExactIndex.resolve_params({}), context)
    index = DenseIndex.load(built.manifest, built.payload)
    assert built.manifest.revision == gateway.aliases["embed-default"].revision
    component, params = GatewayEmbed(), GatewayEmbed.resolve_params({})
    # Every chunk is its own nearest neighbour.
    for item in inputs:
        for chunk in item.chunks:
            vector, revision = embed_query(component, params, chunk.index_text, context)
            assert index.search(vector, 1, revision=revision)[0][0] == chunk.chunk_id
    owner = {c.chunk_id: name for name, chunks in CORPUS.items() for c in chunks}
    expected = {
        "two people count the tip jar together at close": "tips.md",
        "tier prices are for coffee packed in 5 lb bags": "wholesale.docx",
        "keep allergen information and the allergy order steps": "food-safety.pdf",
        "the catering service charge is cafe revenue not a tip": "tips.md",
    }
    for query, document in expected.items():
        vector, revision = embed_query(component, params, query, context)
        assert owner[index.search(vector, 1, revision=revision)[0][0]] == document, query


def test_batches_go_through_the_rate_limit_and_record_usage(
    context: StageContext, gateway: FakeGateway
) -> None:
    waits: list[tuple[str, int]] = []

    class Slot:
        def __init__(self, name: str, tokens: int) -> None:
            waits.append((name, tokens))

        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    context.rate_limit = Slot
    chunks = CORPUS["tips.md"]
    component = GatewayEmbed()
    vectors = component.embed(
        chunks,
        component.resolve_params({"batch_size": 2}),
        expected_revision=None,
        context=context,
    )
    calls = [r for r in gateway.records if r.endpoint == "embeddings"]
    assert len(chunks) == 7
    assert [r.inputs for r in calls] == [2, 2, 2, 1]
    assert [name for name, _ in waits] == ["model_gateway"] * 4
    assert all(tokens > 0 for _, tokens in waits)
    assert vectors.sidecar.tokens > 0
    assert context.metrics["gateway.calls"] == 4


def test_the_index_refuses_mixed_revisions_and_dimensions(
    context: StageContext, gateway: FakeGateway
) -> None:
    default = embed_corpus(context, "embed-default")
    local = embed_corpus(context, "embed-local")
    with pytest.raises(ComponentError) as raised:
        ExactIndex().build([default[0], local[1]], ExactIndex.resolve_params({}), context)
    assert raised.value.code == "mixed_revisions"
    gateway.change_revision("embed-default", "2")
    moved = embed_corpus(context, "embed-default")
    with pytest.raises(ComponentError) as raised:
        ExactIndex().build([default[0], moved[1]], ExactIndex.resolve_params({}), context)
    assert raised.value.code == "mixed_revisions"
    built = ExactIndex().build(default, ExactIndex.resolve_params({}), context)
    index = DenseIndex.load(built.manifest, built.payload)
    with pytest.raises(ComponentError) as raised:
        index.search(np.ones(384, dtype=np.float32), 3, revision=moved[0].vectors.sidecar.revision)
    assert raised.value.code == "revision_mismatch"
    shape = moved[0].vectors.sidecar.model_copy(update={"revision": built.manifest.revision})
    odd = Vectors(
        np.ones((len(shape.chunk_ids), 8), dtype=np.float32), shape.model_copy(update={"dims": 8})
    )
    with pytest.raises(ComponentError) as raised:
        ExactIndex().build(
            [default[0], IndexInput(moved[0].chunks_key, moved[0].chunks, odd)],
            ExactIndex.resolve_params({}),
            context,
        )
    assert raised.value.code == "mixed_dimensions"


def test_a_revision_change_during_a_run_fails_the_task(
    context: StageContext, gateway: FakeGateway
) -> None:
    component = GatewayEmbed()
    params = component.resolve_params({})
    planned = component.revision_for(params, context)
    gateway.change_revision("embed-default", "2")
    with pytest.raises(ComponentError) as raised:
        component.embed(
            CORPUS["wholesale.docx"], params, expected_revision=planned, context=context
        )
    assert raised.value.code == "revision_changed"


# --- BM25 ---------------------------------------------------------------------------------


def keyword_index(chunks: list[Chunk], **params) -> KeywordIndex:
    built = Bm25Index().build(
        [IndexInput(content_hash("c"), chunks)], Bm25Index.resolve_params(params), StageContext()
    )
    return KeywordIndex.load(built.manifest, built.payload)


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id * 32,
        document_id="d",
        ordinal=0,
        text=text,
        index_text=text,
        char_start=0,
        char_end=len(text),
    )


def test_terms_are_unicode_words_without_stop_words_stemmed() -> None:
    assert terms("The baristas weren't counting the caf\u00e9\u2019s TIP jars") == [
        "barista",
        "count",
        "café",
        "tip",
        "jar",
    ]
    assert terms("Non-slip shoes: 18.5 g") == ["non", "slip", "shoe", "18", "5", "g"]
    assert terms("the and of if or") == []


def test_bm25_scores_by_the_okapi_formula() -> None:
    chunks = [
        chunk("a", "Steam milk for lattes. Milk foam is thin."),
        chunk("b", "Cold brew steeps for sixteen hours."),
        chunk("c", "Kids' drinks get milk steamed to 120 degrees."),
    ]
    index = keyword_index(chunks)
    lengths = [len(terms(item.text)) for item in chunks]
    average = sum(lengths) / len(lengths)

    def score(row: int, query: list[str]) -> float:
        counts = {term: terms(chunks[row].text).count(term) for term in query}
        total = 0.0
        for term in query:
            containing = sum(term in terms(item.text) for item in chunks)
            idf = math.log(1 + (len(chunks) - containing + 0.5) / (containing + 0.5))
            tf = counts[term]
            norm = 1.2 * (1 - 0.75 + 0.75 * lengths[row] / average)
            total += idf * tf * 2.2 / (tf + norm)
        return total

    hits = index.search("steamed milk", 10)
    assert [chunk_id for chunk_id, _ in hits] == ["a" * 32, "c" * 32]
    for chunk_id, value in hits:
        row = "abc".index(chunk_id[0])
        assert value == pytest.approx(score(row, ["steam", "milk"]))
    assert (index.k1, index.b) == (1.2, 0.75)
    assert index.manifest.version == "2.0.0"


def test_bm25_filters_inside_the_search_and_breaks_ties_by_id() -> None:
    chunks = [chunk(letter, "Batch brew is held for thirty minutes.") for letter in "dcba"]
    index = keyword_index(chunks)
    assert [chunk_id[0] for chunk_id, _ in index.search("batch brew", 10)] == list("abcd")
    allowed = np.array([True, False, True, False])
    assert [chunk_id[0] for chunk_id, _ in index.search("brew", 10, allowed=allowed)] == ["b", "d"]


def test_bm25_has_no_hits_without_matching_terms() -> None:
    index = keyword_index(CORPUS["wholesale.docx"])
    assert index.search("the and of", 5) == []
    assert index.search("zebra quartz", 5) == []
    assert index.search("Wholesale tiers", 5)


def test_the_bm25_payload_names_its_analyzer() -> None:
    built = Bm25Index().build(
        [IndexInput(content_hash("c"), CORPUS["tips.md"])],
        Bm25Index.resolve_params({}),
        StageContext(),
    )
    assert built.payload.startswith(b'{"analyzer": "' + ANALYZER.encode())


# --- the gateway client -------------------------------------------------------------------------


def test_transient_failures_retry_then_give_up_retryably(
    context: StageContext, gateway: FakeGateway
) -> None:
    provider: GatewayProvider = context.services["models"]
    gateway.inject_fault(503, 2)
    result = provider.embed(["one", "two"], alias="embed-default")
    assert result.vectors.shape == (2, 384)
    gateway.inject_fault(429, 4)  # every attempt
    with pytest.raises(ComponentError) as raised:
        provider.embed(["one"], alias="embed-default")
    assert (raised.value.code, raised.value.retryable) == ("gateway_rate_limited", True)
    gateway.inject_fault(400, 1)
    with pytest.raises(ComponentError) as raised:
        provider.embed(["one"], alias="embed-default")
    assert (raised.value.code, raised.value.retryable) == ("gateway_refused_400", False)


def test_unknown_aliases_and_missing_revisions_are_refused(context: StageContext) -> None:
    provider: GatewayProvider = context.services["models"]
    with pytest.raises(ComponentError) as raised:
        provider.revision("embed-nope")
    assert raised.value.code == "unknown_alias"
    with pytest.raises(ComponentError) as raised:
        provider.embed(["one"], alias="rerank-default")
    assert raised.value.code == "gateway_refused_404"

    def no_revision(request):
        import httpx2

        return httpx2.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    import httpx2

    bare = GatewayProvider(
        "http://x",
        client=httpx2.Client(transport=httpx2.MockTransport(no_revision), base_url="http://x"),
    )
    with pytest.raises(ComponentError) as raised:
        bare.embed(["one"], alias="any")
    assert raised.value.code == "missing_revision"


def test_a_service_identity_goes_with_every_call(gateway: FakeGateway) -> None:
    client = TestClient(create_app(gateway), base_url="http://gateway")
    provider = GatewayProvider(
        "http://gateway", client=client, headers=lambda: {"X-Stratafi-Identity": "svc-token"}
    )
    provider.revision("embed-default")
    provider.embed(["a"], alias="embed-default")
    assert {record.identity for record in gateway.records} == {"svc-token"}


# --- the fake embedder, vectors files and fusion -----------------------------------------------


def test_fake_hash_vectors_are_deterministic_and_round_trip() -> None:
    component = FakeHashEmbed()
    params = component.resolve_params({"dimensions": 64})
    chunks = CORPUS["food-safety.pdf"]
    first = component.embed(chunks, params, expected_revision=None, context=StageContext())
    again = component.embed(chunks, params, expected_revision=None, context=StageContext())
    assert np.array_equal(first.matrix, again.matrix)
    assert first.sidecar.revision == "fake-hash-v2-64"
    payload, sidecar = vectors_to_bytes(first)
    loaded = vectors_from_bytes(payload, sidecar)
    assert np.array_equal(loaded.matrix, first.matrix)
    assert loaded.sidecar == first.sidecar
    assert np.allclose(np.linalg.norm(loaded.matrix, axis=1), 1.0, atol=1e-5)
    assert hash_embedding("Wire Transfers", 16) == hash_embedding("wire transfers", 16)


def test_the_sidecar_refuses_repeated_chunks() -> None:
    with pytest.raises(ValueError, match="repeat"):
        VectorsSidecar(
            chunk_ids=["a", "a"],
            component="embed/x",
            model="m",
            revision="r",
            dims=2,
            normalized=True,
        )


def test_reciprocal_rank_fusion_sums_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    assert [item for item, _ in fused] == ["a", "c", "b"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_embed_and_index_components_are_registered() -> None:
    registry = default_registry()
    for component_id in ("embed/gateway", "embed/fake-hash", "index/exact", "index/bm25"):
        assert component_id in registry
    assert registry.get("embed/gateway").info.egress == "model_gateway"
    assert registry.get("embed/gateway").info.resource_class == "io"
