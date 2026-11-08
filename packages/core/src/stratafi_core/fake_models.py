"""Deterministic stand-ins for an embedding model and a reranker, for tests and offline work.

The in-process components (``embed/fake-hash``, ``rerank/fake-overlap``) and the fake model
gateway (``stratafi_testing.fake_gateway``) all score with these functions, so a pipeline gets
the same vectors and scores whichever of them it uses. Neither is a model: they only promise
that texts sharing words score higher than texts that don't.

- ``hashed_vector`` is signed feature hashing. Each distinct word of the text (lowercase
  letters and digits) is hashed with BLAKE2b, personalized for Stratafi, into ``PROBES``
  positions of the vector, each with its own sign. A word's weight grows with the log of how
  often it occurs, so repeating a word doesn't swamp the rest. The vector is then scaled to
  length 1; a text without words gets the first unit vector.
- ``overlap_score`` is how much of the query a text covers, from 0 to 1: the query's distinct
  words found in the text, each weighted by its length, a cheap stand-in for how rare it is.

Changing either changes every vector or score made with it: bump ``EMBED_REVISION`` or
``RERANK_REVISION`` with it, so nothing cached under the old one is reused.
"""

import hashlib
import math
import re
from collections import Counter

EMBED_REVISION = "fake-hash-v2"
RERANK_REVISION = "fake-overlap-v2"
PROBES = 2
_PERSON = b"stratafi-fake-v2"  # BLAKE2b's personalization: at most 16 bytes
_WORD_RE = re.compile(r"[^\W_]+")


def words(text: str) -> list[str]:
    """The text's words: runs of letters and digits, case-folded."""
    return _WORD_RE.findall(text.casefold())


def _positions(word: str, dimensions: int) -> list[tuple[int, float]]:
    """Where a word lands in the vector, and with which sign: one pair per probe."""
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8 * PROBES, person=_PERSON)
    raw = digest.digest()
    found = []
    for probe in range(PROBES):
        value = int.from_bytes(raw[8 * probe : 8 * (probe + 1)], "big")
        found.append(((value >> 1) % dimensions, 1.0 if value & 1 else -1.0))
    return found


def hashed_vector(text: str, dimensions: int) -> list[float]:
    """A unit vector of ``dimensions`` floats for ``text``; the module docstring says how."""
    if dimensions < 1:
        raise ValueError("a vector needs at least one dimension")
    vector = [0.0] * dimensions
    for word, count in Counter(words(text)).items():
        weight = 1.0 + math.log(count)
        for index, sign in _positions(word, dimensions):
            vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


def overlap_score(query: str, text: str) -> float:
    """The share of the query's distinct words, weighted by length, that the text holds."""
    wanted = set(words(query))
    if not wanted:
        return 0.0
    present = set(words(text))
    total = sum(len(word) for word in wanted)
    return sum(len(word) for word in wanted & present) / total
