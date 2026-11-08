"""Indexes that live in Postgres.

- ``index/postgres-fts``: Postgres full-text search with the ``english`` configuration:
  ``to_tsvector('english', ...)`` over each chunk's index text (its title, heading path and
  text), queried with the query's words (``keywords.query_words``) OR-ed together in
  ``to_tsquery('english', ...)``, so Postgres stems them and drops its own stop words, and
  ranked by ``ts_rank_cd``, ties by chunk ID.
- ``index/pgvector-hnsw``: an HNSW index (``m``, ``ef_construction``) over the run's vectors,
  searched with ``ef_search`` and, where pgvector 0.8 has them, iterative scans, so a filter
  doesn't starve the scan. Building it measures **ANN recall@10 against exact search** on a
  seeded sample of its own vectors as queries, and records it with the index.

Each index is a table in the ``stratafi_index`` schema, named from a hash of what it was built
from, so the same inputs reuse it; the artifact holds its manifest (``backend: postgres``).
The worker and the API give these components the database through ``StageContext.services``
(``database``). Lab-only storage: a production deployment keeps its own engine.
"""

import hashlib
import random
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from stratafi_components.index import BuiltIndex, IndexComponent, IndexInput, _manifest
from stratafi_components.keywords import query_words
from stratafi_components.vectors import normalize_rows
from stratafi_core.artifacts import IndexManifest
from stratafi_core.components import ComponentError, ComponentInfo, StageContext

DATABASE_SERVICE = "database"
SCHEMA = "stratafi_index"
RECALL_SAMPLE = 100
RECALL_K = 10
TEXT_SEARCH_CONFIG = "english"


def table_name(prefix: str, inputs: list[IndexInput], params: dict[str, Any]) -> str:
    material = repr(([item.chunks_key for item in inputs], sorted(params.items())))
    return f"{prefix}_{hashlib.sha256(material.encode()).hexdigest()[:24]}"


def _engine(context: StageContext) -> Any:
    return context.service(DATABASE_SERVICE)


def _qualified(table: str) -> str:
    if not table.replace("_", "").isalnum() or len(table) > 63:
        raise ComponentError("index_corrupt")
    return f"{SCHEMA}.{table}"


def statement(template: str, **names: str) -> Any:
    """SQL with checked identifiers filled in: table names come from hashes and a schema
    constant, and the text search configuration from a fixed pair, never from a caller."""
    from sqlalchemy import text

    for value in names.values():
        if not all(part.replace("_", "").isalnum() for part in value.split(".")):
            raise ComponentError("index_corrupt")
    return text(template.format(**names))


def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in vector) + "]"


def pgvector_version(conn: Any) -> tuple[int, ...]:
    from sqlalchemy import text

    found = conn.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one_or_none()
    if found is None:
        raise ComponentError("pgvector_missing")
    return tuple(int(part) for part in str(found).split(".")[:3])


# --- index/postgres-fts ----------------------------------------------------------------------


class PostgresFts(IndexComponent):
    kind = "keyword"
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="index/postgres-fts",
        stage="index",
        version="2.0.0",
        title="Postgres full-text",
        description="Postgres full-text search over each chunk's index text with the english "
        "configuration: the query's words OR-ed, ranked by ts_rank_cd.",
        resource_class="io",
        license="PostgreSQL",
        services=("lab-index",),
    )

    def build(
        self, inputs: list[IndexInput], params: BaseModel, context: StageContext
    ) -> BuiltIndex:
        from sqlalchemy import text

        dumped = self.dump_params(params)
        table = table_name("fts", inputs, dumped)
        qualified = _qualified(table)
        rows = [
            {
                "row": row,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "body": chunk.index_text,
            }
            for row, chunk in enumerate(chunk for item in inputs for chunk in item.chunks)
        ]
        with _engine(context).begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {qualified}"))
            conn.execute(
                text(
                    f"CREATE TABLE {qualified} (row integer PRIMARY KEY, chunk_id text NOT NULL, "
                    "document_id text NOT NULL, tsv tsvector NOT NULL)"
                )
            )
            for start in range(0, len(rows), 500):
                context.checkpoint()
                conn.execute(
                    statement(
                        "INSERT INTO {table} (row, chunk_id, document_id, tsv) "
                        "VALUES (:row, :chunk_id, :document_id, to_tsvector('{config}', :body))",
                        table=qualified,
                        config=TEXT_SEARCH_CONFIG,
                    ),
                    rows[start : start + 500],
                )
            conn.execute(text(f"CREATE INDEX ON {qualified} USING gin (tsv)"))
        manifest = _manifest(self, params, inputs).model_copy(
            update={"backend": "postgres", "table": table}
        )
        context.metric("chunks", len(rows))
        payload = f'{{"table": "{table}", "config": "{TEXT_SEARCH_CONFIG}"}}'.encode()
        return BuiltIndex(manifest, payload, "application/json")


class PgKeywordIndex:
    """Searches a ``index/postgres-fts`` table."""

    def __init__(self, manifest: IndexManifest, engine: Any) -> None:
        self.manifest = manifest
        self.engine = engine
        self.qualified = _qualified(manifest.table or "")

    def search(
        self, query: str, top_n: int, *, allowed: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        words = " | ".join(query_words(query))
        if not words:
            return []
        rows_clause, bind = "", {"query": words, "limit": top_n}
        if allowed is not None and not bool(np.all(allowed)):
            rows_clause = "AND row = ANY(:rows)"
            bind["rows"] = [int(row) for row in np.flatnonzero(allowed)]
        # The filter is a fixed fragment and the configuration a constant; values are bound.
        query_sql = statement(
            "SELECT chunk_id, ts_rank_cd(tsv, q)::float8 AS score FROM {table}, "  # noqa: S608
            "to_tsquery('{config}', :query) AS q WHERE tsv @@ q "
            + rows_clause
            + " ORDER BY score DESC, chunk_id LIMIT :limit",
            table=self.qualified,
            config=TEXT_SEARCH_CONFIG,
        )
        with self.engine.connect() as conn:
            return [(row.chunk_id, float(row.score)) for row in conn.execute(query_sql, bind)]


# --- index/pgvector-hnsw ---------------------------------------------------------------------


class HnswParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    m: int = Field(default=16, ge=2, le=100, description="Links per node")
    ef_construction: int = Field(default=64, ge=4, le=1000, description="Build-time candidates")
    ef_search: int = Field(default=100, ge=1, le=1000, description="Search-time candidates")
    iterative_scan: Literal["relaxed_order", "strict_order", "off"] = Field(
        default="relaxed_order", description="pgvector 0.8's scans that keep filtered results full"
    )


class PgvectorHnsw(IndexComponent):
    kind = "dense"
    info: ClassVar[ComponentInfo] = ComponentInfo(
        id="index/pgvector-hnsw",
        stage="index",
        version="1.0.0",
        title="pgvector HNSW",
        description="An HNSW index in Postgres (m, ef_construction, ef_search, iterative scans) "
        "over cosine distance; building it reports ANN recall@10 against exact search.",
        resource_class="io",
        license="PostgreSQL (pgvector)",
        services=("lab-index",),
    )
    Params: ClassVar[type[BaseModel]] = HnswParams

    def build(
        self, inputs: list[IndexInput], params: BaseModel, context: StageContext
    ) -> BuiltIndex:
        from sqlalchemy import text

        params = cast(HnswParams, params)
        vectors = [item.vectors for item in inputs if item.vectors is not None]
        if len(vectors) != len(inputs) or not vectors:
            raise ComponentError("vectors_missing")
        if len({(v.sidecar.model, v.sidecar.revision) for v in vectors}) != 1:
            raise ComponentError("mixed_revisions")
        if len({v.sidecar.dims for v in vectors}) != 1:
            raise ComponentError("mixed_dimensions")
        matrix = normalize_rows(np.vstack([v.matrix for v in vectors]).astype(np.float32))
        first = vectors[0].sidecar
        dims = first.dims
        dumped = self.dump_params(params)
        table = table_name("hnsw", inputs, dumped)
        qualified = _qualified(table)
        chunks = [chunk for item in inputs for chunk in item.chunks]
        with _engine(context).begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {qualified}"))
            conn.execute(
                text(
                    f"CREATE TABLE {qualified} (row integer PRIMARY KEY, chunk_id text NOT NULL, "
                    f"document_id text NOT NULL, embedding vector({dims}) NOT NULL)"
                )
            )
            rows = [
                {
                    "row": row,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "embedding": vector_literal(matrix[row]),
                }
                for row, chunk in enumerate(chunks)
            ]
            for start in range(0, len(rows), 500):
                context.checkpoint()
                conn.execute(
                    statement(
                        "INSERT INTO {table} (row, chunk_id, document_id, embedding) "
                        "VALUES (:row, :chunk_id, :document_id, CAST(:embedding AS vector))",
                        table=qualified,
                    ),
                    rows[start : start + 500],
                )
            conn.execute(
                text(
                    f"CREATE INDEX ON {qualified} USING hnsw (embedding vector_cosine_ops) "
                    f"WITH (m = {params.m}, ef_construction = {params.ef_construction})"
                )
            )
        manifest = _manifest(
            self,
            params,
            inputs,
            model=first.model,
            revision=first.revision,
            dims=dims,
            metric="cosine",
            dtype=first.dtype,
        ).model_copy(update={"backend": "postgres", "table": table})
        index = PgDenseIndex(manifest, _engine(context))
        recall = index.recall_at_k(matrix, k=RECALL_K, sample=RECALL_SAMPLE)
        manifest = manifest.model_copy(update={"stats": {"ann_recall_at_10": recall}})
        context.metric("vectors", len(chunks))
        context.metric("ann_recall_at_10", recall)
        payload = f'{{"table": "{table}", "dims": {dims}}}'.encode()
        return BuiltIndex(manifest, payload, "application/json")


class PgDenseIndex:
    """Searches a ``index/pgvector-hnsw`` table."""

    def __init__(self, manifest: IndexManifest, engine: Any) -> None:
        self.manifest = manifest
        self.engine = engine
        self.qualified = _qualified(manifest.table or "")
        self.params = HnswParams.model_validate(manifest.params)

    def _settings(self, conn: Any) -> None:
        from sqlalchemy import text

        conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(self.params.ef_search)}"))
        if self.params.iterative_scan != "off" and pgvector_version(conn) >= (0, 8):
            conn.execute(text(f"SET LOCAL hnsw.iterative_scan = {self.params.iterative_scan}"))

    def search(
        self,
        query: np.ndarray,
        top_n: int,
        *,
        revision: str,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:

        if revision != self.manifest.revision:
            raise ComponentError("revision_mismatch")
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.manifest.dims:
            raise ComponentError("dimension_mismatch")
        vector = vector / (float(np.linalg.norm(vector)) or 1.0)
        rows_clause, bind = "", {"vector": vector_literal(vector), "limit": top_n}
        if allowed is not None and not bool(np.all(allowed)):
            rows_clause = "WHERE row = ANY(:rows)"
            bind["rows"] = [int(row) for row in np.flatnonzero(allowed)]
        nearest = statement(  # the filter is a fixed fragment; values are bound
            "WITH nearest AS MATERIALIZED (SELECT chunk_id, embedding <=> CAST(:vector AS vector) "  # noqa: S608
            "AS distance FROM {table} " + rows_clause + " ORDER BY distance LIMIT :limit) "
            "SELECT chunk_id, 1 - distance AS score FROM nearest ORDER BY distance, chunk_id",
            table=self.qualified,
        )
        with self.engine.begin() as conn:
            self._settings(conn)
            return [(row.chunk_id, float(row.score)) for row in conn.execute(nearest, bind)]

    def recall_at_k(self, matrix: np.ndarray, *, k: int, sample: int) -> float:
        """The share of exact search's top ``k`` that the HNSW search finds, over a seeded
        sample of the index's own vectors as queries."""

        rows = len(matrix)
        if rows == 0:
            return 1.0
        chosen = sorted(random.Random(0).sample(range(rows), min(sample, rows)))  # noqa: S311
        found = 0
        total = 0
        with self.engine.begin() as conn:
            self._settings(conn)
            for row in chosen:
                exact = np.argsort(-(matrix @ matrix[row]), kind="stable")[: min(k, rows)]
                approximate = conn.execute(
                    statement(
                        "SELECT row FROM {table} "
                        "ORDER BY embedding <=> CAST(:vector AS vector) LIMIT :k",
                        table=self.qualified,
                    ),
                    {"vector": vector_literal(matrix[row]), "k": min(k, rows)},
                ).scalars()
                found += len(set(int(r) for r in exact) & set(int(r) for r in approximate))
                total += min(k, rows)
        return round(found / total, 6)


COMPONENTS = [PostgresFts, PgvectorHnsw]
