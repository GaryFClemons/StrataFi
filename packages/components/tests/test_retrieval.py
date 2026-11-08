"""The retrieval engine for evaluation.

The filter tests hold a knowledge base's promises: restricted chunks never come back and never
reach the reranker; superseded, future-dated and draft versions stay hidden for the as-of date,
on a miniature corpus and on the Halcyon demo corpus's own traps. RRF is checked against the
formula, worked by hand.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import BaseModel

from stratafi_components import default_registry
from stratafi_components.chunkers import Sections
from stratafi_components.clean import Whitespace, run_clean
from stratafi_components.detect import detect
from stratafi_components.embed import FakeHashEmbed
from stratafi_components.extract import raw_document
from stratafi_components.index import Bm25Index, DenseIndex, ExactIndex, IndexInput, KeywordIndex
from stratafi_components.models import GatewayProvider
from stratafi_components.parsers.markdown import MarkdownParser
from stratafi_components.rerank import (
    FakeOverlapRerank,
    GatewayRerank,
    RerankNone,
    overlap_score,
)
from stratafi_components.retrieval import (
    CHECKPOINTS,
    EntryMeta,
    QueryEmbedder,
    Reranker,
    RetrievalEngine,
    fuse,
    reciprocal_rank_fusion,
    visible_until,
    weighted_fusion,
)
from stratafi_core.artifacts import Chunk
from stratafi_core.components import ComponentError, StageContext
from stratafi_core.evalsets import Asker
from stratafi_core.hashing import content_hash
from stratafi_core.specs import Fusion, RetrievalConfig, resolve_retrieval
from stratafi_testing.fake_gateway import FakeGateway, create_app

ROOT = Path(__file__).resolve().parents[3]
DEMO_CORPUS = ROOT / "fixtures" / "demo-corpus"

# The demo set's askers (fixtures/demo-corpus/retrieval.yaml), with user IDs.
STAFF = Asker(name="staff", user="u-halcyon-staff", roles=["staff"], groups=["all-staff"])
MANAGER = Asker(
    name="manager", user="u-halcyon-manager", roles=["manager"], groups=["all-staff", "managers"]
)
EXEC = Asker(
    name="exec",
    user="u-halcyon-exec",
    roles=["exec"],
    groups=["all-staff", "managers", "executives"],
)
AS_OF = date(2026, 11, 1)


def config(**changes: Any) -> RetrievalConfig:
    document = {
        "schema": "stratafi.retrieval.v1",
        "name": "hybrid-test",
        "retrievers": {"dense": {"top_n": 50}, "keyword": {"top_n": 50}},
        "fusion": {"method": "rrf", "k": 60},
        "candidates": 50,
        "filters": {"current_only": True},
        "rerank": {"component": "rerank/fake-overlap"},
        "top_k": 8,
        **changes,
    }
    return resolve_retrieval(document, default_registry())


class RecordingRerank(FakeOverlapRerank):
    """The fake reranker, keeping every text it was sent: proof of what reached it."""

    def __init__(self) -> None:
        self.received: list[str] = []
        self.calls = 0

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        params: BaseModel,
        context: StageContext,
        *,
        expected_revision: str | None = None,
    ) -> list[float] | None:
        self.calls += 1
        self.received += list(texts)
        return super().rerank(query, texts, params, context)


@dataclass
class Corpus:
    entries: list[EntryMeta]
    chunks: dict[str, Chunk]
    dense: DenseIndex
    keyword: KeywordIndex
    texts: dict[str, list[str]] = field(default_factory=dict)  # document ID → chunk texts

    def engine(
        self,
        retrieval: RetrievalConfig | None = None,
        *,
        reranker: RecordingRerank | None = None,
        embedder: QueryEmbedder | None = None,
        entries: list[EntryMeta] | None = None,
    ) -> RetrievalEngine:
        context = StageContext()
        rerank = reranker or RecordingRerank()
        return RetrievalEngine(
            retrieval or config(),
            entries=entries if entries is not None else self.entries,
            chunks=self.chunks,
            dense=self.dense,
            keyword=self.keyword,
            embedder=embedder
            or QueryEmbedder(FakeHashEmbed(), FakeHashEmbed.resolve_params({}), context),
            reranker=Reranker.resolve(rerank, rerank.resolve_params({}), context),
        )


def build(documents: list[tuple[EntryMeta, list[tuple[str, str]]]]) -> Corpus:
    """Chunks from (entry, [(section, text)]), embedded with the fake and indexed."""
    embedder = FakeHashEmbed()
    params = embedder.resolve_params({})
    context = StageContext()
    inputs, chunks, texts = [], {}, {}
    for entry, sections in documents:
        document_chunks = []
        for ordinal, (section, text) in enumerate(sections):
            chunk_id = hashlib.sha256(f"{entry.name}#{ordinal}".encode()).hexdigest()[:32]
            title = entry.display_title or ""
            document_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=entry.document_id,
                    ordinal=ordinal,
                    text=text,
                    index_text=f"{title}\n{section}\n{text}",
                    char_start=0,
                    char_end=len(text),
                    heading_path=[section],
                    metadata={"section": section},
                )
            )
        vectors = embedder.embed(document_chunks, params, expected_revision=None, context=context)
        inputs.append(IndexInput(content_hash(entry.name), document_chunks, vectors))
        chunks.update({chunk.chunk_id: chunk for chunk in document_chunks})
        texts[entry.document_id] = [chunk.text for chunk in document_chunks]
    dense = ExactIndex().build(inputs, ExactIndex.resolve_params({}), context)
    keyword = Bm25Index().build(inputs, Bm25Index.resolve_params({}), context)
    return Corpus(
        [entry for entry, _ in documents],
        chunks,
        DenseIndex.load(dense.manifest, dense.payload),
        KeywordIndex.load(keyword.manifest, keyword.payload),
        texts,
    )


def entry(
    key: str,
    version: str,
    start: date,
    *,
    end: date | None = None,
    groups: tuple[str, ...] = ("all-staff",),
    collection: str = "policies",
    status: str = "published",
    title: str | None = None,
) -> EntryMeta:
    return EntryMeta.of(
        document_id=f"doc-{key}-{version}",
        name=f"{collection}/{key}-v{version}.md",
        doc_key=key,
        doc_version=version,
        collection=collection,
        acl={"groups": list(groups), "users": []},
        effective_from=start,
        effective_to=end,
        metadata={"status": status, **({"title": title} if title else {})},
        media_type="text/markdown",
    )


# A coffee company's knowledge base in miniature: each document says something checkable.
DEMO = [
    (
        entry("cafe-hours", "1", date(2026, 1, 1), title="Cafe Hours"),
        [("Friday hours", "Cafes close at 5:00 p.m. on Fridays.")],
    ),
    (
        entry("cafe-hours", "2", date(2026, 12, 1), title="Cafe Hours"),
        [("Friday hours", "Cafes close at 6:00 p.m. on Fridays.")],
    ),
    (
        entry("refund-limits", "1", date(2025, 1, 1)),
        [("Limits", "The refund limit for shift leads is $25.")],
    ),
    (
        entry("refund-limits", "2", date(2026, 3, 1)),
        [("Limits", "The refund limit for shift leads is $50.")],
    ),
    (
        entry("free-drinks", "1", date(2025, 6, 1)),
        [("Rewards", "Rewards members earn a free drink every 60 points.")],
    ),
    (
        entry("free-drinks", "2", date(2026, 6, 1), status="draft"),
        [("Rewards", "Rewards members earn a free drink every 50 points.")],
    ),
    (
        entry(
            "expansion-memo",
            "1",
            date(2025, 1, 1),
            groups=("executives",),
            collection="executive",
        ),
        [("Sites", "The expansion committee reviews new cafe sites every quarter.")],
    ),
    (
        entry("safe-procedure", "1", date(2025, 1, 1), groups=("managers",)),
        [("Approvals", "The cafe manager approves safe cash pickups.")],
    ),
    (
        entry("batch-brew", "1", date(2025, 1, 1)),
        [("Holding", "Batch brew is held in its thermal server for thirty minutes.")],
    ),
]


@pytest.fixture(scope="module")
def demo() -> Corpus:
    return build(DEMO)


def documents_in(corpus: Corpus, ranking: list[tuple[str, float]]) -> set[str]:
    return {corpus.chunks[chunk_id].document_id for chunk_id, _ in ranking}


def everywhere(corpus: Corpus, result: Any) -> set[str]:
    return set().union(*(documents_in(corpus, ranked) for ranked in result.checkpoints.values()))


# --- the filter ---------------------------------------------------------------------------


def test_a_future_dated_v2_leaves_v1_visible_until_its_date(demo: Corpus) -> None:
    engine = demo.engine()
    question = "What time do cafes close on Fridays?"
    for day in (AS_OF, date(2026, 11, 30)):
        result = engine.search(question, asker=STAFF, day=day)
        assert "doc-cafe-hours-1" in documents_in(demo, result.checkpoints["final"])
        assert "doc-cafe-hours-2" not in everywhere(demo, result)
        assert "5:00 p.m." in demo.chunks[result.checkpoints["final"][0][0]].text
    result = engine.search(question, asker=STAFF, day=date(2026, 12, 1))
    assert "doc-cafe-hours-2" in documents_in(demo, result.checkpoints["final"])
    assert "doc-cafe-hours-1" not in everywhere(demo, result)
    assert "6:00 p.m." in demo.chunks[result.checkpoints["final"][0][0]].text


def test_a_new_version_replaces_the_old_one(demo: Corpus) -> None:
    result = demo.engine().search(
        "What is the refund limit for shift leads?", asker=STAFF, day=AS_OF
    )
    assert "doc-refund-limits-1" not in everywhere(demo, result)
    first = demo.chunks[result.checkpoints["final"][0][0]]
    assert first.document_id == "doc-refund-limits-2"
    assert "$50" in first.text


def test_versions_listed_out_of_order_still_chain(demo: Corpus) -> None:
    ends = visible_until(reversed(demo.entries))
    assert ends == visible_until(demo.entries)
    assert ends["policies/refund-limits-v1.md"] == date(2026, 3, 1)
    assert ends["policies/refund-limits-v2.md"] is None
    result = demo.engine(entries=list(reversed(demo.entries))).search(
        "What is the refund limit for shift leads?", asker=STAFF, day=AS_OF
    )
    assert "doc-refund-limits-1" not in everywhere(demo, result)


def test_a_draft_never_surfaces_and_never_ends_the_published_version(demo: Corpus) -> None:
    result = demo.engine().search("How many points does a free drink take?", asker=STAFF, day=AS_OF)
    assert "doc-free-drinks-2" not in everywhere(demo, result)
    assert demo.chunks[result.checkpoints["final"][0][0]].document_id == "doc-free-drinks-1"
    assert visible_until(demo.entries)["policies/free-drinks-v1.md"] is None


def test_restricted_chunks_never_come_back_and_never_reach_the_reranker(demo: Corpus) -> None:
    restricted = {"doc-expansion-memo-1", "doc-safe-procedure-1"}
    restricted_text = [text for document in restricted for text in demo.texts[document]]
    for question in (
        "How often does the expansion committee review cafe sites?",
        "Who approves safe cash pickups?",
    ):
        reranker = RecordingRerank()
        result = demo.engine(config(filters={"current_only": True}), reranker=reranker).search(
            question, asker=STAFF, day=AS_OF
        )
        assert not everywhere(demo, result) & restricted
        sent = "\n".join(reranker.received)
        assert all(text not in sent for text in restricted_text)

    # The positive controls: an executive sees the executives' memo, a manager sees the
    # managers' procedure, and those chunks reach the reranker for them.
    reranker = RecordingRerank()
    memo = demo.engine(reranker=reranker).search(
        "How often does the expansion committee review cafe sites?", asker=EXEC, day=AS_OF
    )
    top = demo.chunks[memo.checkpoints["final"][0][0]]
    assert top.document_id == "doc-expansion-memo-1"
    assert any("expansion committee" in text for text in reranker.received)
    safe = demo.engine().search("Who approves safe cash pickups?", asker=MANAGER, day=AS_OF)
    top = demo.chunks[safe.checkpoints["final"][0][0]]
    assert top.document_id == "doc-safe-procedure-1"


def test_users_only_search_the_collections_the_config_names(demo: Corpus) -> None:
    executive_only = config(filters={"collections": ["executive"], "current_only": True})
    question = "How often does the expansion committee review cafe sites?"
    engine = demo.engine(executive_only)
    assert engine.search(question, asker=STAFF, day=AS_OF).checkpoints["final"] == []
    assert engine.search(question, asker=EXEC, day=AS_OF).checkpoints["final"]


def test_an_acl_can_name_users(demo: Corpus) -> None:
    named = EntryMeta.of(
        document_id="doc-private",
        name="private.md",
        doc_key="private",
        doc_version="1",
        acl={"groups": [], "users": ["u-halcyon-manager"]},
    )
    corpus = build([(named, [("Private", "The safe combination changes weekly.")])])
    question = "How often does the safe combination change?"
    engine = corpus.engine()
    assert engine.search(question, asker=MANAGER, day=AS_OF).checkpoints["final"]
    assert engine.search(question, asker=STAFF, day=AS_OF).checkpoints["final"] == []


def test_entries_without_metadata_are_visible_to_everyone_on_every_day() -> None:
    plain = EntryMeta.of(document_id="doc-plain", name="notes/plain.md")
    corpus = build([(plain, [("Notes", "Coffee is in the second-floor kitchen.")])])
    for asker in (Asker(), STAFF):
        for day in (date(1999, 1, 1), AS_OF):
            result = corpus.engine().search("Where is the coffee?", asker=asker, day=day)
            assert result.checkpoints["final"]
    scoped = config(filters={"collections": ["policies"], "current_only": True})
    assert corpus.engine(scoped).search("coffee", asker=STAFF, day=AS_OF).checkpoints["final"] == []


def test_without_current_only_every_published_version_is_searchable(demo: Corpus) -> None:
    every = config(filters={"current_only": False})
    result = demo.engine(every).search(
        "What time do cafes close on Fridays?", asker=STAFF, day=AS_OF
    )
    assert {"doc-cafe-hours-1", "doc-cafe-hours-2"} <= everywhere(demo, result)


# --- the demo corpus's traps ---------------------------------------------------------------


def demo_corpus() -> tuple[Corpus, dict[str, Asker]]:
    """The Halcyon demo corpus through parse/markdown and chunk/sections, and its eval set's
    askers by name."""
    manifest = yaml.safe_load((DEMO_CORPUS / "corpus" / "manifest.yaml").read_text("utf-8"))
    documents = []
    parser, chunker = MarkdownParser(), Sections()
    for item in manifest["documents"]:
        path = DEMO_CORPUS / "corpus" / item["path"]
        data = path.read_bytes()
        meta = EntryMeta.of(
            document_id=hashlib.sha256(data).hexdigest()[:16],
            name=item["path"],
            doc_key=item["doc_key"],
            doc_version=str(item["version"]),
            collection=item["collection"],
            acl=item["acl"],
            effective_from=item["effective_from"],
            effective_to=item.get("effective_to"),
            metadata={"status": item["status"], "title": item["title"]},
            media_type="text/markdown",
        )
        raw = raw_document(
            meta.document_id, hashlib.sha256(data).hexdigest(), len(data), detect(path, path.name)
        )
        parsed = parser.parse_local(path, raw, parser.resolve_params({}))
        cleaned = run_clean(
            parsed, [("clean/whitespace", Whitespace(), Whitespace.resolve_params({}))]
        )
        chunks = chunker.chunk(
            cleaned,
            chunker.resolve_params({}),
            cache_key=content_hash(item["path"]),
            label=meta.label,
        )
        documents.append((meta, [(section_or(chunk), chunk.text) for chunk in chunks]))
    eval_set = yaml.safe_load((DEMO_CORPUS / "retrieval.yaml").read_text("utf-8"))
    askers = {name: Asker(name=name, **fields) for name, fields in eval_set["askers"].items()}
    return build(documents), askers


def section_or(chunk: Chunk) -> str:
    return str(chunk.metadata.get("section") or "")


def test_superseded_and_future_dated_versions_stay_hidden_for_the_as_of_date() -> None:
    corpus, askers = demo_corpus()
    by_id = {meta.document_id: meta for meta in corpus.entries}
    eval_set = yaml.safe_load((DEMO_CORPUS / "retrieval.yaml").read_text(encoding="utf-8"))
    as_of = date.fromisoformat(eval_set["as_of"])
    engine = corpus.engine()
    hidden = {("pto-leave", "2"), ("loyalty", "2"), ("loyalty", "4")}
    unreadable = {"staff": {"management", "executive"}, "manager": {"executive"}, "exec": set()}
    for query in eval_set["queries"]:
        result = engine.search(query["query"], asker=askers[query["asker"]], day=as_of)
        seen = [by_id[document] for document in everywhere(corpus, result)]
        assert not {(meta.doc_key, meta.version) for meta in seen} & hidden, query["id"]
        assert not {meta.collection for meta in seen} & unreadable[query["asker"]], query["id"]
    # From January 1, 2027, version 4 of the Rewards program is current and version 3 is not.
    question = "How many points does it take to earn a free drink?"
    rewards = engine.search(question, asker=askers["staff"], day=date(2027, 1, 2))
    seen = {(by_id[d].doc_key, by_id[d].version) for d in everywhere(corpus, rewards)}
    assert ("loyalty", "4") in seen
    assert not seen & {("loyalty", "3"), ("loyalty", "2")}
    ends = visible_until(corpus.entries)
    assert ends["operations/loyalty-v3.md"] == date(2027, 1, 1)
    assert ends["handbook/pto-leave-v2.md"] == date(2026, 1, 1)


# --- fusion -------------------------------------------------------------------------------


def named(names: list[str]) -> list[tuple[str, float]]:
    return [(name, 0.0) for name in names]


def test_rrf_sums_one_over_k_plus_rank() -> None:
    """RRF scores each item by the sum of ``1 / (k + rank)`` over the lists that hold it
    (Cormack et al. 2009), worked by hand."""
    dense, keyword = named(["a", "b", "c"]), named(["c", "d"])
    for k in (1, 60):
        fused = fuse(Fusion(method="rrf", k=k), dense=dense, keyword=keyword)
        assert dict(fused) == pytest.approx(
            {"a": 1 / (k + 1), "b": 1 / (k + 2), "c": 1 / (k + 3) + 1 / (k + 1), "d": 1 / (k + 2)}
        )
        assert [name for name, _ in fused] == ["c", "a", "b", "d"]  # b and d tie: by ID
    # The lists' order doesn't matter, an item counts once per list, and one list is its order.
    swapped = fuse(Fusion(method="rrf", k=60), dense=keyword, keyword=dense)
    assert swapped == fuse(Fusion(method="rrf", k=60), dense=dense, keyword=keyword)
    assert reciprocal_rank_fusion([["a", "a", "b"]], k=60) == [("a", 1 / 61), ("b", 1 / 63)]
    assert [n for n, _ in fuse(Fusion(method="rrf"), dense=dense, keyword=None)] == ["a", "b", "c"]
    assert fuse(Fusion(method="rrf"), dense=[], keyword=[]) == []


def test_weighted_fusion_normalizes_each_list() -> None:
    dense = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    keyword = [("c", 12.0), ("d", 4.0)]
    fused = dict(weighted_fusion(dense, keyword, alpha=0.75))
    assert fused == pytest.approx({"a": 0.75, "b": 0.375, "c": 0.25, "d": 0.0})
    assert [name for name, _ in weighted_fusion(dense, keyword, alpha=0.75)] == ["a", "b", "c", "d"]
    assert weighted_fusion([("x", 3.0), ("y", 3.0)], None, alpha=1.0) == [("x", 1.0), ("y", 1.0)]


def test_one_retriever_needs_no_fusion(demo: Corpus) -> None:
    keyword_only = config(
        retrievers={"dense": None, "keyword": {"top_n": 5}}, fusion={"method": "none"}
    )
    engine = RetrievalEngine(
        keyword_only, entries=demo.entries, chunks=demo.chunks, keyword=demo.keyword
    )
    result = engine.search("batch brew holding", asker=STAFF, day=AS_OF)
    assert result.checkpoints["fused"] == result.checkpoints["keyword"]
    assert "dense" not in result.checkpoints
    assert result.embedding_revision is None


# --- checkpoints, caching and models ------------------------------------------------------


def test_every_checkpoint_and_latency_is_reported(demo: Corpus) -> None:
    small = config(candidates=4, top_k=2)
    result = demo.engine(small).search("batch brew holding", asker=STAFF, day=AS_OF)
    assert set(result.checkpoints) == set(CHECKPOINTS)
    assert len(result.checkpoints["candidates"]) <= 4
    assert result.checkpoints["final"] == result.checkpoints["reranked"][:2]
    assert {item for item, _ in result.checkpoints["reranked"]} == {
        item for item, _ in result.checkpoints["candidates"]
    }
    assert set(result.latency_ms) == {
        "keyword_ms", "embed_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms"
    }  # fmt: skip
    assert result.embedding_revision == demo.dense.manifest.revision
    assert result.rerank_revision == "fake-overlap-v2"
    # Out: cafe hours v2 (future), refund limits v1 (superseded), the draft, and the
    # executives' and managers' documents (restricted).
    assert result.visible_chunks == 4


def test_a_query_from_another_revision_is_refused(demo: Corpus) -> None:
    class OtherRevision(FakeHashEmbed):
        def revision_for(self, params: BaseModel, context: StageContext) -> str:
            return "another-model"

    embedder = QueryEmbedder(OtherRevision(), OtherRevision.resolve_params({}), StageContext())
    with pytest.raises(ComponentError) as refused:
        demo.engine(embedder=embedder).search("batch brew", asker=STAFF, day=AS_OF)
    assert refused.value.code == "revision_mismatch"


def test_query_vectors_and_rerank_scores_are_cached(demo: Corpus) -> None:
    calls: list[str] = []

    class Counting(FakeHashEmbed):
        def embed(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
            raise AssertionError

    class CountingEmbedder(QueryEmbedder):
        def __call__(self, query: str) -> tuple[np.ndarray, str]:
            calls.append(query)
            return super().__call__(query)

    context = StageContext()
    embedder = CountingEmbedder(Counting(), Counting.resolve_params({}), context)
    reranker = RecordingRerank()
    engine = demo.engine(embedder=embedder, reranker=reranker)
    first = engine.search("batch brew holding", asker=STAFF, day=AS_OF)
    second = engine.search("batch brew holding", asker=STAFF, day=AS_OF)
    assert first.checkpoints == second.checkpoints
    assert (len(calls), reranker.calls) == (1, 1)
    engine.search("Who approves safe cash pickups?", asker=MANAGER, day=AS_OF)
    assert (len(calls), reranker.calls) == (2, 2)
    assert context.metrics["query_embeddings"] == 2


def test_the_rerank_components(demo: Corpus) -> None:
    context = StageContext()
    assert RerankNone().rerank("q", ["a"], RerankNone.resolve_params({}), context) is None
    # "batch" (5 letters) is in the text, "brew" (4) isn't: 5 of 9.
    assert overlap_score("batch brew", "Batch brews only") == pytest.approx(5 / 9)
    keep = Reranker.resolve(RerankNone(), RerankNone.resolve_params({}), context)
    engine = RetrievalEngine(
        config(rerank="rerank/none"),
        entries=demo.entries,
        chunks=demo.chunks,
        dense=demo.dense,
        keyword=demo.keyword,
        embedder=QueryEmbedder(FakeHashEmbed(), FakeHashEmbed.resolve_params({}), context),
        reranker=keep,
    )
    result = engine.search("batch brew holding", asker=STAFF, day=AS_OF)
    assert result.checkpoints["reranked"] == result.checkpoints["candidates"]


def test_gateway_reranking_scores_every_candidate_once(demo: Corpus) -> None:
    gateway = FakeGateway()
    client = TestClient(create_app(gateway), base_url="http://gateway")
    provider = GatewayProvider("http://gateway", client=client, sleep=lambda seconds: None)
    context = StageContext(services={"models": provider})
    component = GatewayRerank()
    reranker = Reranker.resolve(component, component.resolve_params({}), context)
    assert reranker.revision == gateway.aliases["rerank-default"].revision
    engine = RetrievalEngine(
        config(rerank={"component": "rerank/gateway"}),
        entries=demo.entries,
        chunks=demo.chunks,
        dense=demo.dense,
        keyword=demo.keyword,
        embedder=QueryEmbedder(FakeHashEmbed(), FakeHashEmbed.resolve_params({}), context),
        reranker=reranker,
    )
    question = "Is batch brew held in its thermal server for thirty minutes?"
    result = engine.search(question, asker=STAFF, day=AS_OF)
    top = demo.chunks[result.checkpoints["final"][0][0]]
    assert top.document_id == "doc-batch-brew-1"
    reranks = [record for record in gateway.records if record.endpoint == "rerank"]
    assert len(reranks) == 1
    assert reranks[0].inputs == len(result.checkpoints["candidates"])
    engine.search(question, asker=STAFF, day=AS_OF)
    assert len([r for r in gateway.records if r.endpoint == "rerank"]) == 1
    assert context.metrics["gateway.calls"] == 1
    # A rerank model swapped at the gateway mid-run fails rather than mixing revisions.
    gateway.change_revision("rerank-default", "2")
    with pytest.raises(ComponentError) as changed:
        engine.search("How long is batch brew held?", asker=STAFF, day=AS_OF)
    assert changed.value.code == "revision_changed"


def test_the_default_registry_resolves_the_hybrid_retrieval_config() -> None:
    document = yaml.safe_load(
        (ROOT / "contracts" / "fixtures" / "retrieval.v1" / "hybrid-v1.yaml").read_text()
    )
    resolved = resolve_retrieval(document, default_registry())
    assert resolved.rerank.component == "rerank/gateway"
    assert resolved.rerank.params == {"alias": "rerank-default"}
    for component_id in ("rerank/none", "rerank/fake-overlap", "rerank/gateway"):
        assert default_registry().get(component_id).info.stage == "rerank"
