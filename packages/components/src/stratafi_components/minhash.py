"""MinHash signatures and LSH buckets, for ``clean/near-duplicates``.

A document's text becomes a set of word shingles (``k`` consecutive words, lowercased); its
signature is the minimum of ``num_perm`` hash functions over that set. The share of equal
positions in two signatures estimates the Jaccard similarity of their shingle sets. LSH splits
a signature into ``bands`` of ``rows``; two documents that agree on every row of any band are
candidates, which finds pairs above about ``(1 / bands) ** (1 / rows)`` similarity without
comparing every pair. Candidates are then confirmed by their estimated similarity.

The hash functions are fixed (a seeded generator), so signatures are deterministic and the
same text always has the same one.
"""

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping

import numpy as np

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)
SEED = 20261030


def shingles(text: str, k: int) -> set[str]:
    tokens = [word.lower() for word in _WORD_RE.findall(text)]
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _permutations(num_perm: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.RandomState(SEED)
    a = generator.randint(1, (1 << 32) - 1, size=num_perm, dtype=np.uint64)
    b = generator.randint(0, (1 << 32) - 1, size=num_perm, dtype=np.uint64)
    return a, b


def signature(text: str, *, k: int = 5, num_perm: int = 128) -> np.ndarray:
    """The MinHash signature of ``text``'s ``k``-word shingles: ``num_perm`` uint64 values."""
    found = shingles(text, k)
    if not found:
        return np.full(num_perm, _MAX_HASH, dtype=np.uint64)
    values = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(item.encode(), digest_size=4).digest(), "little")
            for item in sorted(found)
        ),
        dtype=np.uint64,
        count=len(found),
    )
    a, b = _permutations(num_perm)
    with np.errstate(over="ignore"):
        hashed = (np.outer(values, a) + b) % _PRIME & _MAX_HASH
    return hashed.min(axis=0)


def similarity(first: np.ndarray, second: np.ndarray) -> float:
    """The estimated Jaccard similarity of two signatures."""
    return float(np.mean(first == second))


def bands_for(num_perm: int, threshold: float) -> tuple[int, int]:
    """The (bands, rows) split of ``num_perm`` whose LSH threshold is closest to
    ``threshold``."""
    best: tuple[float, int, int] | None = None
    for rows in range(1, num_perm + 1):
        if num_perm % rows:
            continue
        bands = num_perm // rows
        error = abs((1 / bands) ** (1 / rows) - threshold)
        if best is None or error < best[0]:
            best = (error, bands, rows)
    if best is None:  # pragma: no cover - rows = 1 always divides
        raise ValueError("num_perm must be positive")
    return best[1], best[2]


def candidates(signatures: Mapping[str, np.ndarray], *, threshold: float) -> dict[str, set[str]]:
    """For each key, the other keys sharing an LSH bucket with it."""
    if not signatures:
        return {}
    num_perm = len(next(iter(signatures.values())))
    bands, rows = bands_for(num_perm, threshold)
    buckets: dict[tuple[int, bytes], list[str]] = defaultdict(list)
    for key, values in signatures.items():
        for band in range(bands):
            buckets[(band, values[band * rows : (band + 1) * rows].tobytes())].append(key)
    found: dict[str, set[str]] = {key: set() for key in signatures}
    for members in buckets.values():
        if len(members) > 1:
            for key in members:
                found[key].update(member for member in members if member != key)
    return found


def near_duplicates(
    signatures: Mapping[str, np.ndarray], key: str, *, threshold: float
) -> list[tuple[str, float]]:
    """The keys whose estimated similarity to ``key`` is at least ``threshold``, best first."""
    pairs = candidates(signatures, threshold=threshold).get(key, set())
    scored = [(other, similarity(signatures[key], signatures[other])) for other in pairs]
    return sorted(
        ((other, score) for other, score in scored if score >= threshold),
        key=lambda pair: (-pair[1], pair[0]),
    )
