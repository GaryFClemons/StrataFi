"""Embed options: what happens to vectors after the model, and to texts before it.

- **Dimensions** (Matryoshka truncation): keep the first ``dimensions`` values of each vector.
  Models trained for it (nomic v1.5, mxbai, arctic) lose little; others lose more, which is what
  the lab measures.
- **Normalize**: scale each vector to length 1 (after truncation, so cosine still works).
- **dtype** (quantization): ``float32``; ``float16``; ``int8``, which needs normalized vectors
  and stores ``round(x * 127)`` so every document shares one scale; ``binary``, one sign bit
  per dimension, packed eight to a byte. Indexes read them back as float32
  (``dequantize``): int8 divided by 127, binary as ±1/√d. A binary index can rescore with the
  float query (the default) or compare signs only.
- **Input limits**: each text is counted with the model's tokenizer (or an estimator) against
  ``max_input_tokens``. ``truncation`` decides: ``report`` (count it, the model truncates
  silently), ``truncate`` (cut it to fit first, so what is embedded is known) or ``error``.
"""

import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from stratafi_core.components import ComponentError

Dtype = Literal["float32", "float16", "int8", "binary"]
INT8_SCALE = 127


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def shape(matrix: np.ndarray, *, dimensions: int | None, normalize: bool) -> np.ndarray:
    """Truncate to ``dimensions`` (when set and smaller) and normalize."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if dimensions is not None and dimensions < matrix.shape[1]:
        matrix = matrix[:, :dimensions]
    return normalize_rows(matrix) if normalize else matrix


def quantize(matrix: np.ndarray, dtype: Dtype) -> tuple[np.ndarray, dict[str, Any] | None]:
    """The stored array for ``dtype``, and what reading it back needs."""
    if dtype == "float32":
        return matrix.astype(np.float32), None
    if dtype == "float16":
        return matrix.astype(np.float16), None
    if dtype == "int8":
        clipped = np.clip(matrix, -1.0, 1.0)
        return np.round(clipped * INT8_SCALE).astype(np.int8), {"scale": INT8_SCALE}
    return np.packbits(matrix > 0, axis=1), {"bits": int(matrix.shape[1])}


def dequantize(stored: np.ndarray, dtype: str, dims: int) -> np.ndarray:
    """A stored array as float32 vectors of ``dims`` values."""
    if dtype in ("float32", "float16"):
        return stored.astype(np.float32)
    if dtype == "int8":
        return stored.astype(np.float32) / INT8_SCALE
    if dtype == "binary":
        bits = np.unpackbits(stored, axis=1)[:, :dims].astype(np.float32)
        return (bits * 2 - 1) / math.sqrt(dims)
    raise ComponentError("vectors_corrupt")


def binarize(vector: np.ndarray) -> np.ndarray:
    """A query as the binary index stores documents: signs, scaled to length 1."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return np.where(vector > 0, 1.0, -1.0).astype(np.float32) / math.sqrt(len(vector))


def fit_text(text: str, limit: int, count: Callable[[str], int]) -> str:
    """The longest prefix of ``text`` within ``limit`` tokens, cut at a word when it can be."""
    if count(text) <= limit:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    cut = text[:low]
    space = cut.rfind(" ")
    return cut[:space] if space > low // 2 else cut


def limit_inputs(
    texts: list[str],
    *,
    max_tokens: int | None,
    policy: Literal["report", "truncate", "error"],
    count: Callable[[str], int],
) -> tuple[list[str], int]:
    """Texts checked against the model's input limit: the texts to embed, and how many were
    over it."""
    if max_tokens is None:
        return texts, 0
    over = [count(text) > max_tokens for text in texts]
    if policy == "error" and any(over):
        raise ComponentError("input_too_long")
    if policy == "truncate":
        texts = [
            fit_text(t, max_tokens, count) if o else t for t, o in zip(texts, over, strict=True)
        ]
    return texts, sum(over)
