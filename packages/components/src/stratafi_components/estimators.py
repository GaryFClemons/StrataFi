"""Token estimators.

A chunker counts tokens with one of these, named in its parameters:

- ``chars/4``: characters / 4, rounded up: a quick estimate that needs no tokenizer.
- ``tiktoken:cl100k_base``, ``tiktoken:o200k_base``: OpenAI's encodings (MIT), exact for
  models that use them. The encoding files download once into tiktoken's cache.
- ``hf:<model>``: the Hugging Face tokenizer of a model (``tokenizers``, Apache-2.0, the ``hf``
  extra), loaded from the Hugging Face cache, or downloaded once. Counting with the embedding
  model's own tokenizer makes the truncation check exact.

An estimator that can't load (offline, unknown) fails with ``tokenizer_unavailable``. The
version of the library is part of a chunker's cache key (``estimator_env``).
"""

import math
import re
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated, Any

from pydantic import StringConstraints

from stratafi_core.components import ComponentError

CHARS = "chars/4"
TIKTOKEN_ENCODINGS = ("cl100k_base", "o200k_base")
ESTIMATOR_PATTERN = (
    r"^(chars/4|tiktoken:(cl100k_base|o200k_base)|hf:[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
    r"(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?)$"
)
EstimatorName = Annotated[str, StringConstraints(pattern=ESTIMATOR_PATTERN)]
Counter = Callable[[str], int]


def chars_estimate(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


@lru_cache(maxsize=8)
def _tiktoken(name: str) -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:
        raise ComponentError("tokenizer_unavailable") from None


@lru_cache(maxsize=8)
def hf_tokenizer(model: str) -> Any:
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_pretrained(model)
    except Exception:
        raise ComponentError("tokenizer_unavailable") from None


def counter(name: str) -> Counter:
    """The token counter an estimator name stands for."""
    if name == CHARS:
        return chars_estimate
    if name.startswith("tiktoken:"):
        encoding = _tiktoken(name.removeprefix("tiktoken:"))
        return lambda text: len(encoding.encode(text, disallowed_special=()))
    if name.startswith("hf:"):
        tokenizer = hf_tokenizer(name.removeprefix("hf:"))
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    raise ComponentError("unknown_estimator")


def estimator_env(name: str) -> dict[str, Any]:
    """What a chunker's cache key must hold for its estimator: the library's version."""
    if name.startswith("tiktoken:"):
        try:
            import tiktoken

            return {"tiktoken": tiktoken.__version__}
        except ImportError:
            return {}
    if name.startswith("hf:"):
        try:
            from importlib.metadata import version

            return {"tokenizers": version("tokenizers")}
        except Exception:
            return {}
    return {}


_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Sentences in ``text[start:end]`` as spans, without the space after each: a split at
    ``.``, ``!`` or ``?`` followed by space and a capital or digit, and at blank lines."""
    end = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    for block in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text[start:end]):
        block_start = start + block.start()
        position = block_start
        for match in _SENTENCE_END.finditer(block.group()):
            spans.append((position, block_start + match.start()))
            position = block_start + match.end()
        spans.append((position, block_start + block.end() - block.start()))
    return [(s, e) for s, e in spans if text[s:e].strip()]
