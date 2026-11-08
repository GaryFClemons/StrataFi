"""``embed/gateway`` and ``rerank/gateway`` through a real model gateway (the model tier).

Runs when ``STRATAFI_TEST_MODEL_URL`` names a model gateway (``services/gateway``, or another
OpenAI-compatible one that reports revisions); ``STRATAFI_TEST_MODEL_IDENTITY``, when set, is
the identity token sent with each call. ``STRATAFI_TEST_EMBED_ALIAS`` picks the embedding
alias, ``embed-default`` by default; ``STRATAFI_TEST_RERANK_ALIAS`` the rerank alias,
``rerank-default`` by default.
"""

import os

import pytest

from stratafi_components.embed import GatewayEmbed
from stratafi_components.models import GatewayProvider
from stratafi_components.rerank import GatewayRerank
from stratafi_core.artifacts import Chunk
from stratafi_core.components import StageContext

pytestmark = pytest.mark.model


def test_gateway_embedding_through_a_real_gateway_records_its_revision(
    model_endpoint: str,
) -> None:
    alias = os.environ.get("STRATAFI_TEST_EMBED_ALIAS", "embed-default")
    token = os.environ.get("STRATAFI_TEST_MODEL_IDENTITY")
    provider = GatewayProvider(
        model_endpoint, headers=lambda: {"X-Stratafi-Identity": token} if token else {}
    )
    context = StageContext(services={"models": provider})
    component = GatewayEmbed()
    params = component.resolve_params({"alias": alias, "batch_size": 2})
    revision = component.revision_for(params, context)
    chunks = [
        Chunk(
            chunk_id=f"{n:032x}",
            document_id="d-1",
            ordinal=n,
            text=text,
            index_text=text,
            char_start=0,
            char_end=len(text),
        )
        for n, text in enumerate(["Wire transfers need a callback.", "Holds on new accounts."])
    ]
    vectors = component.embed(chunks, params, expected_revision=revision, context=context)
    assert vectors.sidecar.revision == revision
    assert vectors.sidecar.model == alias
    assert vectors.matrix.shape[0] == 2


def test_gateway_reranking_through_a_real_gateway(model_endpoint: str) -> None:
    """``rerank/gateway`` once a real rerank model is configured. The alias is
    ``STRATAFI_TEST_RERANK_ALIAS``, ``rerank-default`` by default."""
    alias = os.environ.get("STRATAFI_TEST_RERANK_ALIAS", "rerank-default")
    token = os.environ.get("STRATAFI_TEST_MODEL_IDENTITY")
    provider = GatewayProvider(
        model_endpoint, headers=lambda: {"X-Stratafi-Identity": token} if token else {}
    )
    context = StageContext(services={"models": provider})
    component = GatewayRerank()
    params = component.resolve_params({"alias": alias})
    revision = component.revision_for(params, context)
    texts = [
        "Branch hours\nSaturdays\nLobbies open at 9 a.m. on Saturdays.",
        "Wire transfers\nCallbacks\nEmail requests are confirmed by a callback to the number on "
        "file before a wire is sent.",
    ]
    scores = component.rerank(
        "Are emailed wire requests confirmed by phone?", texts, params, context
    )
    assert revision
    assert scores is not None
    assert len(scores) == 2
    assert scores[1] > scores[0]
