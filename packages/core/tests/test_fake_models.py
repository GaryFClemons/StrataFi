"""The fake models' algorithms, shared by embed/fake-hash, rerank/fake-overlap and the fake
gateway: deterministic, and texts that share words score higher."""

import math

import pytest

from stratafi_core.fake_models import (
    EMBED_REVISION,
    RERANK_REVISION,
    hashed_vector,
    overlap_score,
    words,
)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_deterministic_unit_vectors() -> None:
    first = hashed_vector("Batch brew is held for 30 minutes.", 64)
    assert first == hashed_vector("Batch brew is held for 30 minutes.", 64)
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)
    # Case and punctuation don't matter; word order doesn't either.
    assert hashed_vector("WIRE transfers!", 32) == hashed_vector("transfers wire", 32)


def test_texts_that_share_words_score_higher() -> None:
    query = hashed_vector("how long is batch brew held", 384)
    related = hashed_vector("Batch brew is held for 30 minutes, then poured out.", 384)
    unrelated = hashed_vector("Cash drawers are counted by two people at close.", 384)
    assert cosine(query, related) > cosine(query, unrelated)
    assert cosine(query, related) > 0.3


def test_repeated_words_weigh_less_than_new_ones() -> None:
    once = hashed_vector("espresso tamping", 384)
    repeated = hashed_vector("espresso espresso espresso tamping", 384)
    # Repeating a word tilts the vector toward it, but only by its log count.
    assert 0.8 < cosine(once, repeated) < 1.0


def test_a_text_without_words_is_the_first_unit_vector() -> None:
    assert hashed_vector("  --  ", 8) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="at least one dimension"):
        hashed_vector("text", 0)


def test_overlap_weighs_the_query_words_a_text_covers() -> None:
    assert overlap_score("batch brew", "Batch brew only") == 1.0
    assert overlap_score("batch brew", "Batch brews only") == pytest.approx(5 / 9)
    assert overlap_score("batch brew", "nothing alike") == 0.0
    assert overlap_score("", "anything") == 0.0
    assert words("Café_hours: 7-5") == ["café", "hours", "7", "5"]


def test_the_revisions_name_this_version_of_the_algorithms() -> None:
    assert (EMBED_REVISION, RERANK_REVISION) == ("fake-hash-v2", "fake-overlap-v2")
