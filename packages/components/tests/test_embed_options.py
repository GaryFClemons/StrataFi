"""Embedders and embed options: quantization and truncation logic, the
in-process and OpenAI-compatible embedders behind their gates, and the model catalog."""

import json
import sys
import types
from typing import ClassVar

import httpx2
import numpy as np
import pytest

from stratafi_components.embed import (
    FakeHashEmbed,
    GatewayEmbed,
    embed_query,
    vectors_from_bytes,
    vectors_to_bytes,
)
from stratafi_components.embedders import (
    CATALOG,
    OpenAICompatibleEmbed,
    SentenceTransformersEmbed,
)
from stratafi_components.index import DenseIndex, ExactIndex, IndexInput
from stratafi_components.parsers.hosted import ProviderAccess
from stratafi_components.vectors import (
    INT8_SCALE,
    binarize,
    dequantize,
    fit_text,
    limit_inputs,
    quantize,
    shape,
)
from stratafi_core.artifacts import Chunk
from stratafi_core.components import ComponentError, StageContext

RNG = np.random.default_rng(0)


def chunk(i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{i:032x}",
        document_id="d-1",
        ordinal=i,
        text=text,
        index_text=text,
        char_start=0,
        char_end=len(text),
    )


TEXTS = [
    "Tellers confirm every wire request by calling the number on file.",
    "Deposits over five thousand dollars may be held for two business days.",
    "A youth debit card can withdraw one hundred dollars a day.",
    "Safe deposit boxes are drilled after twelve months of unpaid rent.",
]
CHUNKS = [chunk(i, text) for i, text in enumerate(TEXTS)]


# --- quantization and truncation -------------------------------------------------------------


def test_matryoshka_truncation_keeps_the_first_values_and_renormalizes() -> None:
    matrix = RNG.normal(size=(5, 64)).astype(np.float32)
    shaped = shape(matrix, dimensions=16, normalize=True)
    assert shaped.shape == (5, 16)
    np.testing.assert_allclose(np.linalg.norm(shaped, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(shaped[0] / shaped[0][0], matrix[0][:16] / matrix[0][0], rtol=1e-4)
    np.testing.assert_array_equal(shape(matrix, dimensions=None, normalize=False), matrix)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8", "binary"])
def test_quantized_vectors_round_trip(dtype: str) -> None:
    matrix = shape(RNG.normal(size=(20, 48)).astype(np.float32), dimensions=None, normalize=True)
    stored, meta = quantize(matrix, dtype)
    restored = dequantize(stored, dtype, 48)
    assert restored.shape == matrix.shape
    if dtype == "binary":
        assert stored.dtype == np.uint8
        assert stored.shape == (20, 6)
        assert meta == {"bits": 48}
        assert np.array_equal(np.sign(restored), np.where(matrix > 0, 1.0, -1.0))
        np.testing.assert_allclose(np.linalg.norm(restored, axis=1), 1.0, rtol=1e-5)
    elif dtype == "int8":
        assert stored.dtype == np.int8
        assert meta == {"scale": INT8_SCALE}
        assert np.abs(restored - matrix).max() <= 0.5 / INT8_SCALE + 1e-6
    else:
        np.testing.assert_allclose(restored, matrix, atol=1e-3)


def test_quantization_costs_little_ranking_on_unit_vectors() -> None:
    documents = shape(
        RNG.normal(size=(200, 64)).astype(np.float32), dimensions=None, normalize=True
    )
    query = documents[7] + RNG.normal(scale=0.3, size=64).astype(np.float32)
    exact = np.argsort(-(documents @ query))[:10]
    int8 = dequantize(quantize(documents, "int8")[0], "int8", 64)
    assert len(set(exact) & set(np.argsort(-(int8 @ query))[:10])) >= 9
    binary = dequantize(quantize(documents, "binary")[0], "binary", 64)
    rescored = np.argsort(-(binary @ query))[:10]
    signs_only = np.argsort(-(binary @ binarize(query)))[:10]
    assert 7 in rescored
    assert len(set(exact) & set(rescored)) >= len(set(exact) & set(signs_only)) - 2


def test_inputs_over_the_limit_are_reported_truncated_or_refused() -> None:
    count = lambda text: len(text.split())  # noqa: E731 - a word counter
    texts = ["one two three", "one two three four five six"]
    kept, over = limit_inputs(texts, max_tokens=4, policy="report", count=count)
    assert (kept, over) == (texts, 1)
    cut, over = limit_inputs(texts, max_tokens=4, policy="truncate", count=count)
    assert over == 1
    assert cut[1] == "one two three four"
    with pytest.raises(ComponentError) as error:
        limit_inputs(texts, max_tokens=4, policy="error", count=count)
    assert error.value.code == "input_too_long"
    assert limit_inputs(texts, max_tokens=None, policy="error", count=count) == (texts, 0)
    assert fit_text("a b c d e", 3, count) == "a b c"


def test_embed_options_reach_the_vectors_and_the_query() -> None:
    params = FakeHashEmbed.resolve_params(
        {
            "dimensions": 64,
            "truncate_to": 32,
            "dtype": "binary",
            "max_input_tokens": 12,
            "truncation": "truncate",
        }
    )
    context = StageContext()
    vectors = FakeHashEmbed().embed(CHUNKS, params, expected_revision=None, context=context)
    assert vectors.sidecar.dtype == "binary"
    assert (vectors.sidecar.dims, vectors.sidecar.full_dims) == (32, 64)
    assert vectors.matrix.dtype == np.uint8
    assert vectors.sidecar.truncated_inputs == 4
    assert context.metrics["truncated_inputs"] == 4
    payload, sidecar = vectors_to_bytes(vectors)
    loaded = vectors_from_bytes(payload, sidecar)
    assert loaded.matrix.dtype == np.float32
    assert loaded.matrix.shape == (4, 32)
    query, _ = embed_query(FakeHashEmbed(), params, "calling the number on file", context)
    assert query.shape == (1, 32)
    built = ExactIndex().build(
        [IndexInput("c" * 64, CHUNKS, loaded)], ExactIndex.resolve_params({}), StageContext()
    )
    assert built.manifest.dtype == "binary"
    index = DenseIndex.load(built.manifest, built.payload)
    hits = index.search(query, 2, revision=loaded.sidecar.revision)
    assert hits[0][0] == CHUNKS[0].chunk_id


def test_int8_needs_normalized_vectors() -> None:
    with pytest.raises(Exception, match="normalize"):
        GatewayEmbed.resolve_params({"dtype": "int8", "normalize": False})
    assert GatewayEmbed.resolve_params({"dtype": "int8", "normalize": True}).dtype == "int8"


# --- embed/sentence-transformers ---------------------------------------------------------------


class StandInModel:
    """sentence-transformers' surface, deterministically: vectors from the fake hash."""

    loaded: ClassVar[list[tuple]] = []

    def __init__(self, name: str, revision: str, device: str, trust_remote_code: bool) -> None:
        StandInModel.loaded.append((name, revision, trust_remote_code))
        self.tokenizer = lambda text, add_special_tokens=True: {"input_ids": text.split()}
        self.encoded: list[str] = []

    def encode(self, texts, batch_size=32, convert_to_numpy=True):
        from stratafi_components.embed import hash_embedding

        self.encoded += list(texts)
        return np.asarray([hash_embedding(text, 768) for text in texts], dtype=np.float32)


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StandInModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(SentenceTransformersEmbed, "unavailable", classmethod(lambda cls: None))
    monkeypatch.setattr(SentenceTransformersEmbed, "_models", {})
    StandInModel.loaded = []
    return StandInModel


def test_in_process_models_are_lab_only(stand_in) -> None:
    params = SentenceTransformersEmbed.resolve_params({"model": "nomic-ai/nomic-embed-text-v1.5"})
    with pytest.raises(ComponentError) as error:
        SentenceTransformersEmbed().embed(
            CHUNKS,
            params,
            expected_revision=None,
            context=StageContext(services={"in_process": False}),
        )
    assert error.value.code == "in_process_disabled"


def test_a_catalog_model_embeds_with_its_prefixes_and_its_tokenizer(stand_in) -> None:
    params = SentenceTransformersEmbed.resolve_params(
        {"model": "nomic-ai/nomic-embed-text-v1.5", "dimensions": 256, "max_input_tokens": 10}
    )
    context = StageContext(services={"in_process": True})
    vectors = SentenceTransformersEmbed().embed(
        CHUNKS, params, expected_revision=None, context=context
    )
    assert stand_in.loaded == [("nomic-ai/nomic-embed-text-v1.5", "main", True)]
    assert vectors.sidecar.prefix == "search_document: "
    assert (vectors.sidecar.dims, vectors.sidecar.full_dims) == (256, 768)
    assert vectors.sidecar.revision == "nomic-ai/nomic-embed-text-v1.5@main"
    assert vectors.sidecar.truncated_inputs == 4  # every text is over 10 words with its prefix
    query, revision = embed_query(SentenceTransformersEmbed(), params, "wire callbacks", context)
    assert query.shape == (1, 256)
    assert revision == vectors.sidecar.revision
    model = next(iter(SentenceTransformersEmbed._models.values()))
    assert model.encoded[-1] == "search_query: wire callbacks"


def test_uncataloged_models_are_refused(stand_in) -> None:
    params = SentenceTransformersEmbed.resolve_params({"model": "someone/unknown-model"})
    with pytest.raises(ComponentError) as error:
        SentenceTransformersEmbed().embed(
            CHUNKS,
            params,
            expected_revision=None,
            context=StageContext(services={"in_process": True}),
        )
    assert error.value.code == "model_not_cataloged"


def test_the_catalog_says_what_each_model_needs() -> None:
    assert len(CATALOG) >= 10
    for entry in CATALOG:
        assert entry.license in ("Apache-2.0", "MIT")
        assert entry.dims >= 384
        assert entry.max_tokens >= 256
        assert all(size < entry.dims for size in entry.matryoshka)
    e5 = next(entry for entry in CATALOG if entry.model == "intfloat/e5-base-v2")
    assert (e5.query_prefix, e5.document_prefix) == ("query: ", "passage: ")


# --- embed/openai-compatible -------------------------------------------------------------------


def endpoint(seen: list) -> ProviderAccess:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body, request.headers.get("authorization")))
        data = [{"index": i, "embedding": [float(i + 1)] * 8} for i in range(len(body["input"]))]
        return httpx2.Response(200, json={"data": data, "usage": {"total_tokens": 7}})

    return ProviderAccess(
        "http://tei.local", "secret", client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )


def test_an_openai_compatible_endpoint_embeds_only_when_allow_listed() -> None:
    seen: list = []
    params = OpenAICompatibleEmbed.resolve_params(
        {
            "provider": "tei",
            "model": "bge",
            "batch_size": 3,
            "request_dimensions": 8,
            "normalize": True,
        }
    )
    with pytest.raises(ComponentError) as error:
        OpenAICompatibleEmbed().embed(
            CHUNKS,
            params,
            expected_revision=None,
            context=StageContext(services={"provider:tei": endpoint(seen)}),
        )
    assert error.value.code == "provider_not_allowed"
    assert seen == []
    context = StageContext(
        services={"providers_allowed": frozenset({"tei"}), "provider:tei": endpoint(seen)}
    )
    vectors = OpenAICompatibleEmbed().embed(CHUNKS, params, expected_revision=None, context=context)
    assert [path for path, _, _ in seen] == ["/v1/embeddings", "/v1/embeddings"]
    assert seen[0][1]["dimensions"] == 8
    assert seen[0][2] == "Bearer secret"
    assert vectors.matrix.shape == (4, 8)
    assert vectors.sidecar.revision == "tei:bge:8"
    assert context.metrics["openai_compatible.calls"] == 2
    assert context.metrics["tokens"] == 14
