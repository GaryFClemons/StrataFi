"""Cross-encoder reranking for ``/v1/rerank`` (Ollama serves no rerankers).

Each alias names a Hugging Face cross-encoder, loaded with sentence-transformers the first time
it's used (and downloaded into the Hugging Face cache the first time ever). Without the
``rerank`` extra the pool is empty: ``/v1/models`` lists no rerankers and ``/v1/rerank`` answers
404, so a retrieval config that names one fails at once, not halfway through a run.

A reranker's revision is a hash of the model ID it names, so it's stable from the first call on,
before and after the download; name another model to change it.
"""

import hashlib
import importlib.util
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

Scorer = Callable[[str, Sequence[str]], list[float]]


def sentence_transformers_installed() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


class RerankerPool:
    def __init__(
        self,
        models: Mapping[str, str],
        *,
        device: str | None = None,
        loader: Callable[[str], Scorer] | None = None,
    ) -> None:
        # Tests pass a loader; otherwise the pool serves models only with the extra installed.
        self._models = dict(models) if (loader or sentence_transformers_installed()) else {}
        self._device = device
        self._loader = loader or self._load
        self._loaded: dict[str, Scorer] = {}
        self._lock = threading.Lock()

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._models)

    def has(self, alias: str) -> bool:
        return alias in self._models

    def revision(self, alias: str) -> str:
        return hashlib.sha256(f"cross-encoder:{self._models[alias]}".encode()).hexdigest()[:16]

    def score(self, alias: str, query: str, documents: Sequence[str]) -> list[float]:
        """Scores in the documents' order; blocking, so call it from a worker thread."""
        with self._lock:
            scorer = self._loaded.get(alias)
            if scorer is None:
                scorer = self._loaded[alias] = self._loader(self._models[alias])
        return [float(score) for score in scorer(query, documents)]

    def _load(self, model_id: str) -> Scorer:
        from sentence_transformers import CrossEncoder  # the rerank extra

        kwargs: dict[str, Any] = {"max_length": 512}
        if self._device:
            kwargs["device"] = self._device
        model = CrossEncoder(model_id, **kwargs)

        def scorer(query: str, documents: Sequence[str]) -> list[float]:
            scores = model.predict(
                [(query, text) for text in documents], batch_size=32, show_progress_bar=False
            )
            return [float(score) for score in scores]

        return scorer
