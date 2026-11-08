"""The baseline: StrataFi's whole schema in one migration.

Everything lives in the ``stratafi`` schema, and every table that holds a workspace's data has a
``workspace_id``. The tables, in the order they're created:

- workspaces, their members and the audit trail
- artifacts and their lineage, and the job queue
- the component registry, pipeline specs, retrieval configs, corpora and their frozen versions
- sources, documents and where each one came from, and the entries of corpora and their versions
- pipeline runs, their entries and their stage tasks
- eval sets with their versions and items, and eval runs with their shards and query results
- experiments and their trials, test-split confirmations, and knowledge profiles with reviews
- answer runs, calibration labels, and drafted questions with their review queue
- test suites with their runs, schedules and check results, and webhooks with their deliveries
- API tokens and workspace features
- provider limits that every worker process shares
- training sets, training runs and the trained model registry
- annotations of runs, eval runs and experiments, and registry aliases

The ``stratafi_index`` schema holds the indexes that live in Postgres: one table per index, made
by the index components.

Triggers keep frozen versions frozen. A sealed corpus version and its entries can't change, and
a frozen eval set version can't change except for its test-split lock and read counter. Neither
can be deleted directly, only with the corpus, eval set or workspace it belongs to: the trigger
lets a cascade through, by its depth.

Row-level security covers every table with a ``workspace_id``. It is forced, so the tables'
owner, the role the services connect as, is bound by it too. One policy, ``workspace_isolation``,
makes a row visible and writable when the transaction's scope is ``system`` or names the row's
workspace (``stratafi_store.scope``). ``workspaces`` is matched by its ``id``, and
``answer_results``, which has no ``workspace_id``, through its answer run. A transaction that
sets no scope sees nothing. The two provider limit tables hold configuration names only, so
they have no policy.

A later change is a new migration after this one.

Revision ID: 0001
Revises:
Create Date: 2026-11-06
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

SCHEMA = "stratafi"

NAME = r"^[a-z0-9][a-z0-9._-]{0,63}$"
HASH = "^[0-9a-f]{64}$"
RUN_STATUSES = "('queued', 'running', 'succeeded', 'failed', 'cancelling', 'cancelled')"
JOB_STATUSES = "('queued', 'running', 'succeeded', 'failed', 'cancelled')"
EXPERIMENT_STATUSES = (
    "('planned', 'running', 'stopping', 'succeeded', 'stopped', 'failed', 'cancelling', "
    "'cancelled')"
)
TRIAL_STATUSES = "('pending', 'running', 'succeeded', 'failed', 'skipped', 'cancelled', 'pruned')"
TEST_RUN_STATUSES = (
    "('queued', 'running', 'evaluating', 'passed', 'failed', 'error', 'cancelling', 'cancelled')"
)
TEST_TRIGGERS = "('api', 'schedule', 'corpus_version_frozen', 'target_changed', 'ci')"
MODEL_KINDS = "('query-adapter', 'embedder', 'reranker')"
ALIAS_KINDS = "('trained_model', 'knowledge_profile')"

# Every table with a workspace_id: each gets row-level security by workspace.
WORKSPACE_TABLES = (
    "memberships",
    "audit_events",
    "artifacts",
    "artifact_inputs",
    "jobs",
    "pipeline_specs",
    "retrieval_configs",
    "corpora",
    "corpus_versions",
    "sources",
    "documents",
    "document_origins",
    "corpus_documents",
    "corpus_version_documents",
    "runs",
    "run_entries",
    "stage_tasks",
    "eval_sets",
    "eval_set_versions",
    "eval_items",
    "eval_runs",
    "eval_shards",
    "eval_query_results",
    "experiments",
    "trials",
    "confirmations",
    "knowledge_profiles",
    "profile_approvals",
    "answer_runs",
    "calibration_labels",
    "question_batches",
    "question_candidates",
    "test_suites",
    "test_runs",
    "test_check_results",
    "test_schedules",
    "webhooks",
    "webhook_deliveries",
    "api_tokens",
    "workspace_features",
    "training_sets",
    "training_runs",
    "trained_models",
    "annotations",
    "registry_aliases",
)
POLICY = "workspace_isolation"
SYSTEM = "current_setting('stratafi.scope', true) = 'system'"
WORKSPACE = "NULLIF(current_setting('stratafi.workspace_id', true), '')::uuid"

CORPUS_VERSION_TRIGGERS = """
CREATE FUNCTION stratafi.corpus_version_frozen() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.sealed THEN
        RAISE EXCEPTION 'a frozen corpus version cannot change' USING ERRCODE = '23001';
    END IF;
    -- A direct delete runs this at depth 1; a cascade from its corpus runs it deeper.
    IF TG_OP = 'DELETE' AND OLD.sealed AND pg_trigger_depth() = 1 THEN
        RAISE EXCEPTION 'a frozen corpus version cannot be deleted' USING ERRCODE = '23001';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END $$;

CREATE FUNCTION stratafi.corpus_version_entry_frozen() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    version_id uuid;
BEGIN
    version_id := CASE WHEN TG_OP = 'INSERT' THEN NEW.corpus_version_id
                       ELSE OLD.corpus_version_id END;
    -- In a cascade the version row is already gone, so its entries may go too.
    IF EXISTS (SELECT 1 FROM stratafi.corpus_versions WHERE id = version_id AND sealed) THEN
        RAISE EXCEPTION 'a frozen corpus version cannot change' USING ERRCODE = '23001';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END $$;

CREATE TRIGGER corpus_versions_frozen BEFORE UPDATE OR DELETE ON stratafi.corpus_versions
    FOR EACH ROW EXECUTE FUNCTION stratafi.corpus_version_frozen();

CREATE TRIGGER corpus_version_documents_frozen
    BEFORE INSERT OR UPDATE OR DELETE ON stratafi.corpus_version_documents
    FOR EACH ROW EXECUTE FUNCTION stratafi.corpus_version_entry_frozen();
"""

EVAL_SET_VERSION_TRIGGERS = """
CREATE FUNCTION stratafi.eval_set_version_frozen() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    -- Only the test-split lock and its read counter may change once a version is frozen.
    IF TG_OP = 'UPDATE' AND OLD.frozen AND
        (to_jsonb(NEW) - ARRAY['test_locked', 'test_reads', 'last_test_read_at'])
        IS DISTINCT FROM
        (to_jsonb(OLD) - ARRAY['test_locked', 'test_reads', 'last_test_read_at'])
    THEN
        RAISE EXCEPTION 'a frozen eval set version cannot change' USING ERRCODE = '23001';
    END IF;
    -- A direct delete runs this at depth 1; a cascade from its eval set runs it deeper.
    IF TG_OP = 'DELETE' AND OLD.frozen AND pg_trigger_depth() = 1 THEN
        RAISE EXCEPTION 'a frozen eval set version cannot be deleted' USING ERRCODE = '23001';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END $$;

CREATE FUNCTION stratafi.eval_item_frozen() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    version_id uuid;
BEGIN
    version_id := CASE WHEN TG_OP = 'INSERT' THEN NEW.eval_set_version_id
                       ELSE OLD.eval_set_version_id END;
    -- In a cascade the version row is already gone, so its items may go too.
    IF EXISTS (SELECT 1 FROM stratafi.eval_set_versions WHERE id = version_id AND frozen) THEN
        RAISE EXCEPTION 'a frozen eval set version cannot change' USING ERRCODE = '23001';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END $$;

CREATE TRIGGER eval_set_versions_frozen BEFORE UPDATE OR DELETE ON stratafi.eval_set_versions
    FOR EACH ROW EXECUTE FUNCTION stratafi.eval_set_version_frozen();

CREATE TRIGGER eval_items_frozen BEFORE INSERT OR UPDATE OR DELETE ON stratafi.eval_items
    FOR EACH ROW EXECUTE FUNCTION stratafi.eval_item_frozen();
"""


# --- helpers: constraint names follow tables.NAMING_CONVENTION ----------------------------------


def _now() -> sa.Function:
    return sa.func.now()


def _at(name: str, *, nullable: bool = False, now: bool = True) -> sa.Column:
    """A timestamp. One that can't be null defaults to now(), unless ``now`` is False."""
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        nullable=nullable,
        server_default=_now() if now and not nullable else None,
    )


def _timestamps() -> list[sa.Column]:
    return [_at("created_at"), _at("updated_at")]


def _jsonb(name: str, default: str | None = "'{}'::jsonb") -> sa.Column:
    """A jsonb column: NOT NULL with ``default``, or nullable with no default (None)."""
    if default is None:
        return sa.Column(name, postgresql.JSONB(), nullable=True)
    return sa.Column(name, postgresql.JSONB(), nullable=False, server_default=sa.text(default))


def _text(name: str, *, nullable: bool = False, default: str | None = None) -> sa.Column:
    return sa.Column(name, sa.Text(), nullable=nullable, server_default=default)


def _uuid(name: str, *, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.Uuid(), nullable=nullable)


def _int(name: str, *, nullable: bool = False, default: str | None = None) -> sa.Column:
    return sa.Column(name, sa.Integer(), nullable=nullable, server_default=default)


def _bool(name: str, default: sa.ColumnElement[bool] | None = None) -> sa.Column:
    return sa.Column(name, sa.Boolean(), nullable=False, server_default=default)


def _float(name: str, *, nullable: bool = True) -> sa.Column:
    return sa.Column(name, sa.Float(), nullable=nullable)


def _pk(table: str, *columns: str) -> sa.PrimaryKeyConstraint:
    return sa.PrimaryKeyConstraint(*columns, name=op.f(f"pk_{table}"))


def _uq(table: str, *columns: str) -> sa.UniqueConstraint:
    return sa.UniqueConstraint(*columns, name=op.f(f"uq_{table}_{'_'.join(columns)}"))


def _ck(table: str, name: str, condition: str) -> sa.CheckConstraint:
    return sa.CheckConstraint(condition, name=op.f(f"ck_{table}_{name}"))


def _fk(
    table: str,
    columns: list[str],
    referred: str,
    referred_columns: list[str],
    **kwargs: object,
) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        columns,
        [f"{SCHEMA}.{referred}.{column}" for column in referred_columns],
        name=op.f(f"fk_{table}_{'_'.join(columns)}_{referred}"),
        **kwargs,
    )


def _in_workspace(table: str) -> sa.ForeignKeyConstraint:
    """``workspace_id`` to ``workspaces``: deleting a workspace deletes the row."""
    return _fk(table, ["workspace_id"], "workspaces", ["id"], ondelete="CASCADE")


def _of(table: str, column: str, referred: str, **kwargs: object) -> sa.ForeignKeyConstraint:
    """``(workspace_id, column)`` to ``referred(workspace_id, id)``, so a row can't point into
    another workspace."""
    return _fk(table, ["workspace_id", column], referred, ["workspace_id", "id"], **kwargs)


def _deferred(**kwargs: object) -> dict[str, object]:
    """Checked at commit, so one transaction can delete a workspace's rows in any order."""
    return {"deferrable": True, "initially": "DEFERRED", **kwargs}


def _index(name: str, table: str, *columns: str | sa.TextClause, where: str | None = None) -> None:
    op.create_index(
        name,
        table,
        list(columns),
        schema=SCHEMA,
        postgresql_where=sa.text(where) if where is not None else None,
    )


def _unique_index(name: str, table: str, *columns: str, where: str) -> None:
    op.create_index(
        name, table, list(columns), unique=True, schema=SCHEMA, postgresql_where=sa.text(where)
    )


def _table(name: str, *items: sa.Column | sa.Constraint) -> None:
    op.create_table(name, *items, schema=SCHEMA)


def _protect(table: str, condition: str) -> None:
    op.execute(f"ALTER TABLE {SCHEMA}.{table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {SCHEMA}.{table} FORCE ROW LEVEL SECURITY")
    # Fixed names and conditions from this file, never input.
    op.execute(
        f"CREATE POLICY {POLICY} ON {SCHEMA}.{table} USING ({condition}) WITH CHECK ({condition})"
    )


# --- the schema ---------------------------------------------------------------------------------


def _tenancy() -> None:
    _table(
        "workspaces",
        _uuid("id"),
        _text("slug"),
        _text("name"),
        _text("external_tenant_id", nullable=True),
        *_timestamps(),
        _pk("workspaces", "id"),
        _uq("workspaces", "slug"),
        _uq("workspaces", "external_tenant_id"),
        _ck("workspaces", "slug_shape", r"slug ~ '^[a-z][a-z0-9_-]{0,62}$'"),
    )
    _table(
        "memberships",
        _uuid("workspace_id"),
        _text("subject"),
        _text("role"),
        *_timestamps(),
        _pk("memberships", "workspace_id", "subject"),
        _in_workspace("memberships"),
        _ck("memberships", "role", "role IN ('owner', 'editor', 'viewer')"),
    )
    # No foreign key: audit events outlive their workspace.
    _table(
        "audit_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        _uuid("workspace_id"),
        _at("occurred_at"),
        _text("actor_kind"),
        _text("actor_subject"),
        _text("action"),
        _text("resource_type", nullable=True),
        _text("resource_id", nullable=True),
        _text("outcome"),
        _text("request_id", nullable=True),
        _jsonb("details"),
        _pk("audit_events", "id"),
        _ck("audit_events", "actor_kind", "actor_kind IN ('user', 'service', 'token', 'system')"),
        _ck("audit_events", "outcome", "outcome IN ('success', 'denied', 'failed')"),
    )
    _index("ix_audit_events_workspace_occurred", "audit_events", "workspace_id", "occurred_at")


def _artifacts_and_jobs() -> None:
    _table(
        "artifacts",
        _uuid("workspace_id"),
        _text("cache_key"),
        _text("kind"),
        _text("storage_uri"),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        _text("content_sha256"),
        _text("media_type"),
        _jsonb("summary"),
        _at("created_at"),
        _pk("artifacts", "workspace_id", "cache_key"),
        _in_workspace("artifacts"),
        _ck("artifacts", "cache_key_shape", f"cache_key ~ '{HASH}'"),
        _ck("artifacts", "content_sha256_shape", f"content_sha256 ~ '{HASH}'"),
        _ck("artifacts", "size_bytes", "size_bytes >= 0"),
        _ck("artifacts", "kind_shape", r"kind ~ '^[a-z][a-z0-9_.]{0,62}$'"),
    )
    _table(
        "artifact_inputs",
        _uuid("workspace_id"),
        _text("artifact_key"),
        _text("input_key"),
        _text("role", default="input"),
        _int("ordinal", default="0"),
        _pk("artifact_inputs", "workspace_id", "artifact_key", "input_key"),
        _fk(
            "artifact_inputs",
            ["workspace_id", "artifact_key"],
            "artifacts",
            ["workspace_id", "cache_key"],
            ondelete="CASCADE",
        ),
        # An input with dependents can't be deleted before them; checked at commit, so one
        # transaction can delete a whole graph in any order.
        _fk(
            "artifact_inputs",
            ["workspace_id", "input_key"],
            "artifacts",
            ["workspace_id", "cache_key"],
            **_deferred(),
        ),
        _ck("artifact_inputs", "not_self", "artifact_key <> input_key"),
    )
    _index(
        "ix_artifact_inputs_workspace_id_input_key", "artifact_inputs", "workspace_id", "input_key"
    )

    _table(
        "jobs",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("kind"),
        _jsonb("payload"),
        _int("priority", default="0"),
        _text("resource_class"),
        _text("status", default="queued"),
        _text("dedupe_key", nullable=True),
        _int("attempts", default="0"),
        _int("max_attempts", default="3"),
        _at("run_after"),
        _text("lease_owner", nullable=True),
        _at("lease_expires_at", nullable=True),
        _bool("cancel_requested", sa.false()),
        _jsonb("progress", None),
        _jsonb("result", None),
        _text("error_code", nullable=True),
        *_timestamps(),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _pk("jobs", "id"),
        _in_workspace("jobs"),
        _ck("jobs", "resource_class", "resource_class IN ('cpu', 'gpu', 'io')"),
        _ck("jobs", "status", f"status IN {JOB_STATUSES}"),
        _ck("jobs", "max_attempts", "max_attempts >= 1"),
        _ck("jobs", "attempts", "attempts >= 0"),
        _ck("jobs", "kind_shape", r"kind ~ '^[a-z][a-z0-9_.]{0,62}$'"),
    )
    # A dedupe key is unique among active jobs only.
    _unique_index(
        "uq_jobs_active_dedupe",
        "jobs",
        "workspace_id",
        "dedupe_key",
        where="dedupe_key IS NOT NULL AND status IN ('queued', 'running')",
    )
    _index(
        "ix_jobs_claim",
        "jobs",
        "resource_class",
        sa.text("priority DESC"),
        "run_after",
        "created_at",
        where="status = 'queued'",
    )
    _index("ix_jobs_lease", "jobs", "lease_expires_at", where="status = 'running'")
    _index("ix_jobs_workspace_created", "jobs", "workspace_id", "created_at", "id")


def _versioned_documents(name: str) -> None:
    """Pipeline specs, retrieval configs and test suites: named chains of immutable versions."""
    _table(
        name,
        _uuid("id"),
        _uuid("workspace_id"),
        _text("name"),
        _int("version"),
        sa.Column("document", postgresql.JSONB(), nullable=False),
        _text("spec_hash"),
        _text("document_hash"),
        _text("created_by"),
        _at("created_at"),
        _pk(name, "id"),
        _in_workspace(name),
        _uq(name, "workspace_id", "name", "version"),
        _ck(name, "version", "version >= 1"),
        _ck(name, "name_shape", f"name ~ '{NAME}'"),
        _ck(name, "spec_hash_shape", f"spec_hash ~ '{HASH}'"),
        _ck(name, "document_hash_shape", f"document_hash ~ '{HASH}'"),
    )
    _index(f"ix_{name}_workspace_id_spec_hash", name, "workspace_id", "spec_hash")


def _configuration() -> None:
    # The component registry, the same for every workspace.
    _table(
        "components",
        _text("component_id"),
        _text("version"),
        _text("stage"),
        sa.Column("info", postgresql.JSONB(), nullable=False),
        sa.Column("params_schema", postgresql.JSONB(), nullable=False),
        _bool("available", sa.true()),
        _at("first_seen_at"),
        _at("last_seen_at"),
        _pk("components", "component_id", "version"),
        _ck(
            "components",
            "id_shape",
            r"component_id ~ '^[a-z][a-z0-9-]{0,31}/[a-z0-9][a-z0-9.-]{0,63}$'",
        ),
    )
    _versioned_documents("pipeline_specs")
    _versioned_documents("retrieval_configs")

    _table(
        "corpora",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("name"),
        _text("description", default=""),
        _text("created_by"),
        *_timestamps(),
        _pk("corpora", "id"),
        _in_workspace("corpora"),
        _uq("corpora", "workspace_id", "name"),
        _uq("corpora", "workspace_id", "id"),
        _ck("corpora", "name_shape", f"name ~ '{NAME}'"),
    )
    _table(
        "corpus_versions",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("corpus_id"),
        _int("version"),
        _text("manifest_hash"),
        _int("document_count"),
        sa.Column("total_bytes", sa.BigInteger(), nullable=False),
        _bool("sealed", sa.false()),
        _text("created_by"),
        _at("frozen_at"),
        _pk("corpus_versions", "id"),
        _of("corpus_versions", "corpus_id", "corpora", ondelete="CASCADE"),
        _uq("corpus_versions", "corpus_id", "version"),
        _uq("corpus_versions", "workspace_id", "id"),
        _ck("corpus_versions", "version", "version >= 1"),
        _ck("corpus_versions", "document_count", "document_count >= 0"),
        _ck("corpus_versions", "total_bytes", "total_bytes >= 0"),
        _ck("corpus_versions", "manifest_hash_shape", f"manifest_hash ~ '{HASH}'"),
    )


def _entry_columns() -> list[sa.Column]:
    """A corpus entry's metadata, as a manifest import gives it."""
    return [
        _text("doc_key", nullable=True),
        _text("doc_version", nullable=True),
        _text("collection", nullable=True),
        _jsonb("acl", None),
        sa.Column("effective_from", sa.Date(), nullable=True),
        sa.Column("effective_to", sa.Date(), nullable=True),
        _jsonb("metadata"),
    ]


def _documents() -> None:
    _table(
        "sources",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("kind"),
        _jsonb("details"),
        _text("created_by"),
        _at("created_at"),
        _pk("sources", "id"),
        _in_workspace("sources"),
        _uq("sources", "workspace_id", "id"),
        _ck("sources", "kind", "kind IN ('upload', 'folder', 'archive', 'manifest', 's3', 'url')"),
    )
    # A withdrawn document was purged: its row stays as a tombstone.
    _table(
        "documents",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("sha256"),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        _text("name"),
        _text("extension"),
        _text("media_type"),
        _text("kind"),
        _bool("mismatch"),
        _bool("known"),
        _jsonb("traits"),
        _text("status"),
        _text("reason_code", nullable=True),
        _text("storage_key"),
        _text("created_by"),
        _at("created_at"),
        _pk("documents", "id"),
        _in_workspace("documents"),
        _uq("documents", "workspace_id", "sha256"),
        _uq("documents", "workspace_id", "id"),
        _ck("documents", "sha256_shape", f"sha256 ~ '{HASH}'"),
        _ck("documents", "size_bytes", "size_bytes >= 0"),
        _ck("documents", "status", "status IN ('stored', 'refused', 'withdrawn')"),
    )
    _index("ix_documents_workspace_created", "documents", "workspace_id", "created_at", "id")
    _table(
        "document_origins",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        _uuid("workspace_id"),
        _uuid("document_id"),
        _uuid("source_id"),
        _text("name"),
        _uuid("parent_document_id", nullable=True),
        _text("member_path", nullable=True),
        _at("created_at"),
        _pk("document_origins", "id"),
        _of("document_origins", "document_id", "documents", ondelete="CASCADE"),
        _of("document_origins", "source_id", "sources", ondelete="CASCADE"),
        _of("document_origins", "parent_document_id", "documents", ondelete="CASCADE"),
    )
    _index(
        "ix_document_origins_workspace_id_document_id",
        "document_origins",
        "workspace_id",
        "document_id",
    )
    _table(
        "corpus_documents",
        _uuid("workspace_id"),
        _uuid("corpus_id"),
        _text("name"),
        _uuid("document_id"),
        *_entry_columns(),
        _text("added_by"),
        _at("added_at"),
        _pk("corpus_documents", "corpus_id", "name"),
        _of("corpus_documents", "corpus_id", "corpora", ondelete="CASCADE"),
        _of("corpus_documents", "document_id", "documents", ondelete="CASCADE"),
    )
    _table(
        "corpus_version_documents",
        _uuid("workspace_id"),
        _uuid("corpus_version_id"),
        _text("name"),
        _uuid("document_id"),
        _text("sha256"),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        _text("media_type"),
        _jsonb("traits"),
        *_entry_columns(),
        _pk("corpus_version_documents", "corpus_version_id", "name"),
        _of("corpus_version_documents", "corpus_version_id", "corpus_versions", ondelete="CASCADE"),
        _of("corpus_version_documents", "document_id", "documents", **_deferred()),
    )
    op.execute(CORPUS_VERSION_TRIGGERS)


def _runs() -> None:
    _table(
        "runs",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("kind", default="pipeline"),
        _text("mode", default="batch"),
        _text("status", default="queued"),
        _uuid("pipeline_spec_id"),
        _uuid("corpus_version_id"),
        _text("until_stage", nullable=True),
        _jsonb("sample", None),
        _uuid("rerun_of", nullable=True),
        _text("rerun_from", nullable=True),
        _text("spec_hash"),
        _jsonb("model_env"),
        _jsonb("manifest", None),
        _text("manifest_hash", nullable=True),
        _jsonb("counts"),
        _text("error_code", nullable=True),
        _text("created_by"),
        *_timestamps(),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _pk("runs", "id"),
        _in_workspace("runs"),
        _fk("runs", ["pipeline_spec_id"], "pipeline_specs", ["id"], **_deferred()),
        _of("runs", "corpus_version_id", "corpus_versions", **_deferred()),
        _uq("runs", "workspace_id", "id"),
        _ck("runs", "status", f"status IN {RUN_STATUSES}"),
        _ck("runs", "mode", "mode IN ('interactive', 'batch')"),
        _ck("runs", "kind", "kind = 'pipeline'"),
    )
    _index("ix_runs_workspace_created", "runs", "workspace_id", "created_at", "id")
    _table(
        "run_entries",
        _uuid("workspace_id"),
        _uuid("run_id"),
        _int("ordinal"),
        _text("name"),
        _uuid("document_id"),
        _text("sha256"),
        _bool("duplicate"),
        _text("label", nullable=True),
        _text("state", default="active"),
        _text("stop_code", nullable=True),
        _pk("run_entries", "run_id", "ordinal"),
        _of("run_entries", "run_id", "runs", ondelete="CASCADE"),
        _ck(
            "run_entries",
            "state",
            "state IN ('active', 'done', 'excluded', 'unsupported_type', 'failed', 'cancelled')",
        ),
    )
    # A run's active entries and tasks: each completion asks whether any are left.
    _index("ix_run_entries_run_active", "run_entries", "run_id", where="state = 'active'")
    _table(
        "stage_tasks",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("run_id"),
        _int("ordinal"),
        _text("stage"),
        _text("component_id"),
        _text("component_version"),
        _text("cache_key"),
        _jsonb("inputs", "'[]'::jsonb"),
        _jsonb("params"),
        _jsonb("env"),
        _text("route", nullable=True),
        _text("resource_class"),
        _text("status"),
        _bool("cache_hit", sa.false()),
        _uuid("job_id", nullable=True),
        _int("attempts", default="0"),
        _text("error_code", nullable=True),
        _jsonb("metrics"),
        _at("created_at"),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _float("duration_ms"),
        _pk("stage_tasks", "id"),
        _of("stage_tasks", "run_id", "runs", ondelete="CASCADE"),
        _uq("stage_tasks", "run_id", "stage", "component_id", "ordinal"),
        _ck(
            "stage_tasks",
            "status",
            "status IN ('pending', 'queued', 'running', 'succeeded', 'cached', 'skipped', "
            "'failed', 'cancelled')",
        ),
        _ck("stage_tasks", "cache_key_shape", f"cache_key ~ '{HASH}'"),
    )
    _index("ix_stage_tasks_workspace_id_cache_key", "stage_tasks", "workspace_id", "cache_key")
    _index("ix_stage_tasks_run_id_ordinal", "stage_tasks", "run_id", "ordinal")
    _index(
        "ix_stage_tasks_active_job_id",
        "stage_tasks",
        "job_id",
        where="status IN ('queued', 'running')",
    )
    _index(
        "ix_stage_tasks_run_active",
        "stage_tasks",
        "run_id",
        where="status IN ('pending', 'queued', 'running')",
    )


def _eval_sets() -> None:
    _table(
        "eval_sets",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("name"),
        _text("description", default=""),
        _text("created_by"),
        *_timestamps(),
        _pk("eval_sets", "id"),
        _in_workspace("eval_sets"),
        _uq("eval_sets", "workspace_id", "name"),
        _uq("eval_sets", "workspace_id", "id"),
        _ck("eval_sets", "name_shape", f"name ~ '{NAME}'"),
    )
    _table(
        "eval_set_versions",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("eval_set_id"),
        _int("version"),
        sa.Column("document", postgresql.JSONB(), nullable=False),
        _text("document_hash"),
        _int("item_count"),
        _jsonb("split_counts"),
        _jsonb("split_params", None),
        _jsonb("source"),
        _bool("frozen", sa.false()),
        _at("frozen_at", nullable=True),
        _bool("test_locked", sa.true()),
        _int("test_reads", default="0"),
        _at("last_test_read_at", nullable=True),
        _text("created_by"),
        _at("created_at"),
        _pk("eval_set_versions", "id"),
        _of("eval_set_versions", "eval_set_id", "eval_sets", ondelete="CASCADE"),
        _uq("eval_set_versions", "eval_set_id", "version"),
        _uq("eval_set_versions", "workspace_id", "id"),
        _ck("eval_set_versions", "version", "version >= 1"),
        _ck("eval_set_versions", "item_count", "item_count >= 0"),
        _ck("eval_set_versions", "test_reads", "test_reads >= 0"),
        _ck("eval_set_versions", "document_hash_shape", f"document_hash ~ '{HASH}'"),
    )
    _table(
        "eval_items",
        _uuid("workspace_id"),
        _uuid("eval_set_version_id"),
        _text("item_id"),
        _int("ordinal"),
        _text("query"),
        sa.Column("asker", postgresql.JSONB(), nullable=False),
        _jsonb("expected", "'[]'::jsonb"),
        _jsonb("avoid", "'[]'::jsonb"),
        _jsonb("must_not_return", "'[]'::jsonb"),
        _jsonb("near_misses", "'[]'::jsonb"),
        _text("reference_answer", nullable=True),
        _jsonb("tags", "'[]'::jsonb"),
        _text("split"),
        _pk("eval_items", "eval_set_version_id", "item_id"),
        _of("eval_items", "eval_set_version_id", "eval_set_versions", ondelete="CASCADE"),
        _uq("eval_items", "eval_set_version_id", "ordinal"),
        _ck("eval_items", "split", "split IN ('dev', 'test')"),
    )
    op.execute(EVAL_SET_VERSION_TRIGGERS)


def _eval_runs() -> None:
    _table(
        "eval_runs",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("status", default="queued"),
        _uuid("pipeline_run_id"),
        _uuid("retrieval_config_id"),
        _uuid("eval_set_version_id"),
        _text("split"),
        _jsonb("options"),
        _text("retrieval_hash"),
        _text("results_key", nullable=True),
        _bool("results_cached", sa.false()),
        _bool("test_read", sa.false()),
        _jsonb("context"),
        _jsonb("manifest", None),
        _text("manifest_hash", nullable=True),
        _jsonb("metrics", None),
        _jsonb("counts"),
        _text("error_code", nullable=True),
        _text("created_by"),
        *_timestamps(),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _pk("eval_runs", "id"),
        _in_workspace("eval_runs"),
        _of("eval_runs", "pipeline_run_id", "runs", **_deferred()),
        _fk("eval_runs", ["retrieval_config_id"], "retrieval_configs", ["id"], **_deferred()),
        _of("eval_runs", "eval_set_version_id", "eval_set_versions", **_deferred()),
        _uq("eval_runs", "workspace_id", "id"),
        _ck("eval_runs", "status", f"status IN {RUN_STATUSES}"),
        _ck("eval_runs", "split", "split IN ('dev', 'test', 'all')"),
    )
    _index("ix_eval_runs_workspace_created", "eval_runs", "workspace_id", "created_at", "id")
    _table(
        "eval_shards",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("eval_run_id"),
        _int("ordinal"),
        _text("cache_key"),
        _jsonb("item_ids", "'[]'::jsonb"),
        _text("status"),
        _bool("cache_hit", sa.false()),
        _uuid("job_id", nullable=True),
        _int("attempts", default="0"),
        _text("error_code", nullable=True),
        _jsonb("metrics"),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _float("duration_ms"),
        _pk("eval_shards", "id"),
        _of("eval_shards", "eval_run_id", "eval_runs", ondelete="CASCADE"),
        _uq("eval_shards", "eval_run_id", "ordinal"),
        _ck(
            "eval_shards",
            "status",
            "status IN ('queued', 'running', 'succeeded', 'cached', 'failed', 'cancelled')",
        ),
        _ck("eval_shards", "cache_key_shape", f"cache_key ~ '{HASH}'"),
    )
    _table(
        "eval_query_results",
        _uuid("workspace_id"),
        _uuid("eval_run_id"),
        _text("item_id"),
        _int("ordinal"),
        _int("shard"),
        _text("split"),
        _jsonb("tags", "'[]'::jsonb"),
        _jsonb("file_types", "'[]'::jsonb"),
        _bool("has_evidence"),
        sa.Column("hit", sa.Boolean(), nullable=True),
        _int("first_hit_rank", nullable=True),
        _text("lost_at", nullable=True),
        sa.Column("avoid_violation", sa.Boolean(), nullable=True),
        _bool("must_not_return_violation"),
        sa.Column("score", postgresql.JSONB(), nullable=False),
        _pk("eval_query_results", "eval_run_id", "item_id"),
        _of("eval_query_results", "eval_run_id", "eval_runs", ondelete="CASCADE"),
        _uq("eval_query_results", "eval_run_id", "ordinal"),
        _ck("eval_query_results", "lost_at", "lost_at IN ('parse', 'retrieval', 'rerank')"),
    )
    # Indexes kept in Postgres: index/postgres-fts and index/pgvector-hnsw make one table per
    # index here, named from a hash of what it was built from.
    op.execute("CREATE SCHEMA IF NOT EXISTS stratafi_index")


def _experiments() -> None:
    _table(
        "experiments",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("name"),
        _text("status", default="planned"),
        # json keeps the factors' order, which is the grid's; jsonb would sort the keys.
        sa.Column("document", postgresql.JSON(), nullable=False),
        _text("document_hash"),
        _uuid("corpus_version_id"),
        _uuid("eval_set_version_id"),
        _uuid("pipeline_spec_id"),
        _uuid("retrieval_config_id"),
        sa.Column("plan", postgresql.JSONB(), nullable=False),
        _jsonb("spent"),
        _jsonb("counts"),
        _text("stop_reason", nullable=True),
        _text("error_code", nullable=True),
        _text("created_by"),
        _text("started_by", nullable=True),
        *_timestamps(),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _pk("experiments", "id"),
        _in_workspace("experiments"),
        _of("experiments", "corpus_version_id", "corpus_versions", **_deferred()),
        _of("experiments", "eval_set_version_id", "eval_set_versions", **_deferred()),
        _fk("experiments", ["pipeline_spec_id"], "pipeline_specs", ["id"], **_deferred()),
        _fk("experiments", ["retrieval_config_id"], "retrieval_configs", ["id"], **_deferred()),
        _uq("experiments", "workspace_id", "id"),
        _ck("experiments", "status", f"status IN {EXPERIMENT_STATUSES}"),
        _ck("experiments", "name_shape", f"name ~ '{NAME}'"),
        _ck("experiments", "document_hash_shape", f"document_hash ~ '{HASH}'"),
    )
    _index("ix_experiments_workspace_created", "experiments", "workspace_id", "created_at", "id")
    _table(
        "trials",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("experiment_id"),
        _int("ordinal"),
        _int("grid_index"),
        sa.Column("levels", postgresql.JSONB(), nullable=False),
        sa.Column("factor_values", postgresql.JSONB(), nullable=False),
        _text("pipeline_hash"),
        _text("retrieval_hash"),
        _uuid("pipeline_spec_id", nullable=True),
        _uuid("retrieval_config_id", nullable=True),
        _uuid("pipeline_run_id", nullable=True),
        _uuid("eval_run_id", nullable=True),
        _text("status", default="pending"),
        _jsonb("objective", None),
        _jsonb("measures"),
        _jsonb("constraints", "'[]'::jsonb"),
        sa.Column("feasible", sa.Boolean(), nullable=True),
        _text("error_code", nullable=True),
        _at("created_at"),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        # Successive halving: the rung a trial is at, and each finished rung's result.
        _int("rung", default="0"),
        _jsonb("rungs", "'[]'::jsonb"),
        _pk("trials", "id"),
        _of("trials", "experiment_id", "experiments", ondelete="CASCADE"),
        _fk("trials", ["pipeline_spec_id"], "pipeline_specs", ["id"], **_deferred()),
        _fk("trials", ["retrieval_config_id"], "retrieval_configs", ["id"], **_deferred()),
        _of("trials", "pipeline_run_id", "runs", **_deferred()),
        _of("trials", "eval_run_id", "eval_runs", **_deferred()),
        _uq("trials", "experiment_id", "ordinal"),
        _ck("trials", "status", f"status IN {TRIAL_STATUSES}"),
        _ck("trials", "rung", "rung >= 0"),
    )
    _index("ix_trials_pipeline_run_id", "trials", "pipeline_run_id")
    _index("ix_trials_eval_run_id", "trials", "eval_run_id")

    _table(
        "confirmations",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("experiment_id"),
        _uuid("trial_id"),
        _uuid("eval_run_id"),
        _uuid("eval_set_version_id"),
        _int("ordinal"),
        _text("reason", nullable=True),
        _text("note", nullable=True),
        _text("created_by"),
        _at("created_at"),
        _pk("confirmations", "id"),
        _of("confirmations", "experiment_id", "experiments", ondelete="CASCADE"),
        _fk("confirmations", ["trial_id"], "trials", ["id"], ondelete="CASCADE"),
        _of("confirmations", "eval_run_id", "eval_runs", **_deferred()),
        _uq("confirmations", "experiment_id", "ordinal"),
        _ck("confirmations", "ordinal", "ordinal >= 1"),
        _ck(
            "confirmations",
            "reason",
            "(ordinal = 1 AND reason IS NULL) OR "
            "reason IN ('run_failed', 'candidate_changed', 'other')",
        ),
        _ck("confirmations", "note_length", "note IS NULL OR length(note) <= 500"),
    )

    _table(
        "knowledge_profiles",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("name"),
        _int("version"),
        sa.Column("document", postgresql.JSONB(), nullable=False),
        _text("document_hash"),
        _text("target"),
        _text("level"),
        _text("status", default="draft"),
        _text("source"),
        _uuid("experiment_id", nullable=True),
        _uuid("trial_id", nullable=True),
        _uuid("confirmation_id", nullable=True),
        _text("created_by"),
        *_timestamps(),
        _pk("knowledge_profiles", "id"),
        _in_workspace("knowledge_profiles"),
        _of(
            "knowledge_profiles",
            "experiment_id",
            "experiments",
            **_deferred(ondelete="SET NULL"),
        ),
        _uq("knowledge_profiles", "workspace_id", "name", "version"),
        _uq("knowledge_profiles", "workspace_id", "id"),
        _ck("knowledge_profiles", "status", "status IN ('draft', 'approved', 'rejected')"),
        _ck("knowledge_profiles", "level", "level IN ('A', 'B', 'C')"),
        _ck("knowledge_profiles", "source", "source IN ('trial', 'import')"),
        _ck("knowledge_profiles", "version", "version >= 1"),
        _ck("knowledge_profiles", "name_shape", f"name ~ '{NAME}'"),
        _ck("knowledge_profiles", "document_hash_shape", f"document_hash ~ '{HASH}'"),
    )
    _table(
        "profile_approvals",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("profile_id"),
        _text("decision"),
        _text("decided_by"),
        _at("decided_at"),
        # The version's own author decided it.
        _bool("self_review", sa.false()),
        _pk("profile_approvals", "id"),
        _of("profile_approvals", "profile_id", "knowledge_profiles", ondelete="CASCADE"),
        _ck("profile_approvals", "decision", "decision IN ('approved', 'rejected')"),
    )


def _answers_and_questions() -> None:
    _table(
        "answer_runs",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("eval_run_id"),
        _text("status", default="queued"),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        _text("results_key", nullable=True),
        _bool("results_cached", sa.false()),
        _jsonb("metrics", None),
        _jsonb("usage"),
        _text("error_code", nullable=True),
        _text("created_by"),
        *_timestamps(),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _pk("answer_runs", "id"),
        _of("answer_runs", "eval_run_id", "eval_runs", ondelete="CASCADE"),
        _uq("answer_runs", "workspace_id", "id"),
        _ck("answer_runs", "status", f"status IN {JOB_STATUSES}"),
    )
    _index("ix_answer_runs_eval_run_id", "answer_runs", "eval_run_id")
    _table(
        "answer_results",
        _uuid("answer_run_id"),
        _text("item_id"),
        _int("ordinal"),
        _text("answer_key"),
        sa.Column("score", postgresql.JSONB(), nullable=False),
        _pk("answer_results", "answer_run_id", "item_id"),
        _fk("answer_results", ["answer_run_id"], "answer_runs", ["id"], ondelete="CASCADE"),
        _ck("answer_results", "answer_key_shape", f"answer_key ~ '{HASH}'"),
    )
    _index("ix_answer_results_answer_key", "answer_results", "answer_key")
    _table(
        "calibration_labels",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("answer_key"),
        _text("item_id"),
        _text("dimension"),
        _bool("label"),
        _text("labeled_by"),
        _at("labeled_at"),
        _pk("calibration_labels", "id"),
        _in_workspace("calibration_labels"),
        _uq("calibration_labels", "workspace_id", "answer_key", "dimension", "labeled_by"),
        _ck(
            "calibration_labels",
            "dimension",
            "dimension IN ('faithful', 'correct', 'abstained')",
        ),
        _ck("calibration_labels", "answer_key_shape", f"answer_key ~ '{HASH}'"),
    )
    _table(
        "question_batches",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("eval_set_id"),
        _uuid("pipeline_run_id"),
        _text("status", default="queued"),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        _jsonb("counts"),
        _text("error_code", nullable=True),
        _text("created_by"),
        *_timestamps(),
        _at("finished_at", nullable=True),
        _pk("question_batches", "id"),
        _of("question_batches", "eval_set_id", "eval_sets", ondelete="CASCADE"),
        _of("question_batches", "pipeline_run_id", "runs", ondelete="CASCADE"),
        _uq("question_batches", "workspace_id", "id"),
        _ck("question_batches", "status", f"status IN {JOB_STATUSES}"),
    )
    _table(
        "question_candidates",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("eval_set_id"),
        _uuid("batch_id"),
        _int("ordinal"),
        _uuid("document_id", nullable=True),
        _text("chunk_id", nullable=True),
        _text("doc_key"),
        _text("version", nullable=True),
        _text("section", nullable=True),
        _text("query"),
        _text("evidence"),
        _text("drafted_query"),
        _text("split", default="dev"),
        _jsonb("tags", "'[]'::jsonb"),
        _text("status", default="proposed"),
        _text("decided_by", nullable=True),
        _at("decided_at", nullable=True),
        _text("item_id", nullable=True),
        _uuid("added_version_id", nullable=True),
        *_timestamps(),
        _pk("question_candidates", "id"),
        _of("question_candidates", "batch_id", "question_batches", ondelete="CASCADE"),
        _uq("question_candidates", "batch_id", "ordinal"),
        _ck(
            "question_candidates",
            "status",
            "status IN ('proposed', 'accepted', 'rejected', 'added')",
        ),
        _ck("question_candidates", "split", "split IN ('dev', 'test')"),
    )
    _index(
        "ix_question_candidates_eval_set_id_status",
        "question_candidates",
        "eval_set_id",
        "status",
    )


def _tests_and_automation() -> None:
    _versioned_documents("test_suites")
    _table(
        "test_runs",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("suite_id"),
        _text("suite_name"),
        _int("suite_version"),
        _text("status", default="queued"),
        _text("trigger"),
        _jsonb("triggers", "'[]'::jsonb"),
        _jsonb("resolved"),
        _uuid("pipeline_run_id", nullable=True),
        _uuid("eval_run_id", nullable=True),
        _uuid("baseline_pipeline_run_id", nullable=True),
        _uuid("baseline_eval_run_id", nullable=True),
        _uuid("baseline_test_run_id", nullable=True),
        _jsonb("summary"),
        _jsonb("record", None),
        _text("record_hash", nullable=True),
        _text("error_code", nullable=True),
        _text("created_by"),
        *_timestamps(),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _pk("test_runs", "id"),
        _in_workspace("test_runs"),
        _fk("test_runs", ["suite_id"], "test_suites", ["id"], **_deferred(ondelete="CASCADE")),
        _uq("test_runs", "workspace_id", "id"),
        _ck("test_runs", "status", f"status IN {TEST_RUN_STATUSES}"),
        _ck("test_runs", "trigger", f"trigger IN {TEST_TRIGGERS}"),
        _ck("test_runs", "suite_name_shape", f"suite_name ~ '{NAME}'"),
    )
    _index("ix_test_runs_workspace_created", "test_runs", "workspace_id", "created_at", "id")
    _index("ix_test_runs_suite", "test_runs", "workspace_id", "suite_name", "created_at")
    # One run per suite is active at a time, and at most one waits behind it.
    _unique_index(
        "uq_test_runs_active_suite",
        "test_runs",
        "workspace_id",
        "suite_name",
        where="status IN ('running', 'evaluating', 'cancelling')",
    )
    _unique_index(
        "uq_test_runs_queued_suite",
        "test_runs",
        "workspace_id",
        "suite_name",
        where="status = 'queued'",
    )
    for column in (
        "pipeline_run_id",
        "eval_run_id",
        "baseline_pipeline_run_id",
        "baseline_eval_run_id",
    ):
        _index(f"ix_test_runs_{column}", "test_runs", column)
    _table(
        "test_check_results",
        _uuid("test_run_id"),
        _text("check_id"),
        _uuid("workspace_id"),
        _int("ordinal"),
        _text("type"),
        _text("result"),
        _text("code", nullable=True),
        _float("value"),
        _float("baseline_value"),
        _jsonb("threshold"),
        _jsonb("detail"),
        _jsonb("evidence", "'[]'::jsonb"),
        _float("duration_ms"),
        _pk("test_check_results", "test_run_id", "check_id"),
        _of("test_check_results", "test_run_id", "test_runs", ondelete="CASCADE"),
        _uq("test_check_results", "test_run_id", "ordinal"),
        _ck("test_check_results", "result", "result IN ('pass', 'fail', 'error', 'skipped')"),
    )
    _index(
        "ix_test_check_results_workspace_id_check_id",
        "test_check_results",
        "workspace_id",
        "check_id",
    )

    _table(
        "test_schedules",
        _uuid("workspace_id"),
        _text("suite_name"),
        _text("cron"),
        _text("time_zone"),
        _at("next_fire_at", now=False),
        _at("last_fired_at", nullable=True),
        _uuid("last_test_run_id", nullable=True),
        _at("created_at"),
        _pk("test_schedules", "workspace_id", "suite_name", "cron"),
        _in_workspace("test_schedules"),
    )
    _index("ix_test_schedules_next_fire_at", "test_schedules", "next_fire_at")
    # A webhook's secret is null when it's derived from the deployment's key for each delivery.
    _table(
        "webhooks",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("url"),
        _text("secret", nullable=True),
        _text("suite_name", nullable=True),
        sa.Column("events", postgresql.JSONB(), nullable=False),
        _bool("enabled", sa.true()),
        _text("created_by"),
        *_timestamps(),
        _pk("webhooks", "id"),
        _in_workspace("webhooks"),
        _uq("webhooks", "workspace_id", "id"),
        _ck("webhooks", "url_length", "length(url) <= 2048"),
    )
    _table(
        "webhook_deliveries",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("webhook_id"),
        _text("event"),
        _uuid("test_run_id", nullable=True),
        _text("body"),
        _text("status", default="pending"),
        _int("attempts", default="0"),
        _int("response_status", nullable=True),
        _text("error_code", nullable=True),
        _at("created_at"),
        _at("delivered_at", nullable=True),
        _pk("webhook_deliveries", "id"),
        _of("webhook_deliveries", "webhook_id", "webhooks", ondelete="CASCADE"),
        _ck("webhook_deliveries", "status", "status IN ('pending', 'delivered', 'failed')"),
    )
    _index(
        "ix_webhook_deliveries_webhook_id_created_at",
        "webhook_deliveries",
        "webhook_id",
        "created_at",
    )

    _table(
        "api_tokens",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("name"),
        _text("prefix"),
        _text("token_hash"),
        sa.Column("scopes", postgresql.JSONB(), nullable=False),
        _text("created_by"),
        _at("created_at"),
        _at("expires_at", now=False),
        _at("revoked_at", nullable=True),
        _text("revoked_by", nullable=True),
        _at("last_used_at", nullable=True),
        _pk("api_tokens", "id"),
        _in_workspace("api_tokens"),
        _uq("api_tokens", "token_hash"),
        _uq("api_tokens", "workspace_id", "id"),
        _ck("api_tokens", "name_shape", f"name ~ '{NAME}'"),
        _ck("api_tokens", "token_hash_shape", f"token_hash ~ '{HASH}'"),
        _ck("api_tokens", "expiry", "expires_at > created_at"),
    )
    _index("ix_api_tokens_workspace_id_created_at", "api_tokens", "workspace_id", "created_at")
    _table(
        "workspace_features",
        _uuid("workspace_id"),
        _text("feature"),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        _text("review", nullable=True),
        _text("changed_by"),
        _at("changed_at"),
        _pk("workspace_features", "workspace_id", "feature"),
        _in_workspace("workspace_features"),
        _ck("workspace_features", "feature", "feature IN ('custom_checks')"),
    )


def _provider_limits() -> None:
    # A provider's request and token buckets, and the calls in flight, each a lease that lapses
    # if its worker dies. Provider names come from configuration: no workspace data.
    _table(
        "provider_quota",
        _text("provider"),
        _float("requests"),
        _float("tokens"),
        _at("refilled_at"),
        _pk("provider_quota", "provider"),
    )
    _table(
        "provider_leases",
        _uuid("id"),
        _text("provider"),
        _text("holder"),
        _at("expires_at", now=False),
        _pk("provider_leases", "id"),
        _fk("provider_leases", ["provider"], "provider_quota", ["provider"], ondelete="CASCADE"),
    )
    _index("ix_provider_leases_provider_expires_at", "provider_leases", "provider", "expires_at")


def _training() -> None:
    _table(
        "training_sets",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("name"),
        _uuid("eval_run_id"),
        _text("status", default="queued"),
        sa.Column("options", postgresql.JSONB(), nullable=False),
        _text("cache_key", nullable=True),
        _bool("cached", sa.false()),
        _jsonb("context"),
        _jsonb("stats", None),
        _text("error_code", nullable=True),
        _text("created_by"),
        *_timestamps(),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _pk("training_sets", "id"),
        _of("training_sets", "eval_run_id", "eval_runs", ondelete="CASCADE"),
        _uq("training_sets", "workspace_id", "id"),
        _ck("training_sets", "status", f"status IN {JOB_STATUSES}"),
        _ck("training_sets", "cache_key_shape", f"cache_key IS NULL OR cache_key ~ '{HASH}'"),
    )
    _index(
        "ix_training_sets_workspace_created", "training_sets", "workspace_id", "created_at", "id"
    )
    _index("ix_training_sets_eval_run_id", "training_sets", "eval_run_id")
    _table(
        "training_runs",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("training_set_id"),
        _text("name"),
        _text("kind"),
        _text("trainer"),
        _text("trainer_version"),
        sa.Column("params", postgresql.JSONB(), nullable=False),
        _jsonb("base"),
        _text("status", default="queued"),
        _jsonb("progress"),
        _jsonb("metrics", None),
        _jsonb("manifest", None),
        _text("manifest_hash", nullable=True),
        _text("model_key", nullable=True),
        _bool("model_cached", sa.false()),
        _uuid("model_id", nullable=True),
        _uuid("job_id", nullable=True),
        _text("error_code", nullable=True),
        _text("created_by"),
        *_timestamps(),
        _at("started_at", nullable=True),
        _at("finished_at", nullable=True),
        _pk("training_runs", "id"),
        _of("training_runs", "training_set_id", "training_sets", ondelete="CASCADE"),
        _uq("training_runs", "workspace_id", "id"),
        _ck("training_runs", "status", f"status IN {RUN_STATUSES}"),
        _ck("training_runs", "kind", f"kind IN {MODEL_KINDS}"),
        _ck("training_runs", "model_key_shape", f"model_key IS NULL OR model_key ~ '{HASH}'"),
    )
    _index(
        "ix_training_runs_workspace_created", "training_runs", "workspace_id", "created_at", "id"
    )
    _index("ix_training_runs_training_set_id", "training_runs", "training_set_id")
    _table(
        "trained_models",
        _uuid("id"),
        _uuid("workspace_id"),
        _text("name"),
        _int("version"),
        _text("kind"),
        _uuid("training_run_id"),
        _text("model_key"),
        sa.Column("base", postgresql.JSONB(), nullable=False),
        sa.Column("card", postgresql.JSONB(), nullable=False),
        _text("status", default="ready"),
        _text("withdrawn_by", nullable=True),
        _at("withdrawn_at", nullable=True),
        _text("created_by"),
        *_timestamps(),
        _pk("trained_models", "id"),
        _of("trained_models", "training_run_id", "training_runs", ondelete="CASCADE"),
        _uq("trained_models", "workspace_id", "id"),
        _uq("trained_models", "workspace_id", "name", "version"),
        _ck("trained_models", "kind", f"kind IN {MODEL_KINDS}"),
        _ck("trained_models", "status", "status IN ('ready', 'withdrawn')"),
        _ck("trained_models", "model_key_shape", f"model_key ~ '{HASH}'"),
        _ck("trained_models", "name_shape", f"name ~ '{NAME}'"),
        _ck("trained_models", "version", "version >= 1"),
    )


def _tracking() -> None:
    # People's titles, tags, notes and archive marks, in a table of their own so the rows that
    # workers rewrite never carry them.
    _table(
        "annotations",
        _uuid("id"),
        _uuid("workspace_id"),
        _uuid("run_id", nullable=True),
        _uuid("eval_run_id", nullable=True),
        _uuid("experiment_id", nullable=True),
        _text("title", nullable=True),
        _jsonb("tags"),
        _text("note", nullable=True),
        _at("archived_at", nullable=True),
        _text("updated_by"),
        *_timestamps(),
        _pk("annotations", "id"),
        _of("annotations", "run_id", "runs", ondelete="CASCADE"),
        _of("annotations", "eval_run_id", "eval_runs", ondelete="CASCADE"),
        _of("annotations", "experiment_id", "experiments", ondelete="CASCADE"),
        _uq("annotations", "run_id"),
        _uq("annotations", "eval_run_id"),
        _uq("annotations", "experiment_id"),
        _ck("annotations", "one_target", "num_nonnulls(run_id, eval_run_id, experiment_id) = 1"),
        _ck(
            "annotations",
            "title_length",
            "title IS NULL OR char_length(title) BETWEEN 1 AND 120",
        ),
        _ck("annotations", "note_length", "note IS NULL OR char_length(note) <= 20000"),
        _ck("annotations", "tags_object", "jsonb_typeof(tags) = 'object'"),
    )
    _table(
        "registry_aliases",
        _uuid("workspace_id"),
        _text("kind"),
        _text("name"),
        _text("alias"),
        _uuid("target_id"),
        _text("set_by"),
        *_timestamps(),
        _pk("registry_aliases", "workspace_id", "kind", "name", "alias"),
        _in_workspace("registry_aliases"),
        _ck("registry_aliases", "kind", f"kind IN {ALIAS_KINDS}"),
        _ck("registry_aliases", "alias_shape", r"alias ~ '^[a-z][a-z0-9_-]{0,31}$'"),
        _ck("registry_aliases", "name_shape", f"name ~ '{NAME}'"),
    )
    _index("ix_registry_aliases_target_id", "registry_aliases", "target_id")


def _row_level_security() -> None:
    for table in WORKSPACE_TABLES:
        _protect(table, f"{SYSTEM} OR workspace_id = {WORKSPACE}")
    _protect("workspaces", f"{SYSTEM} OR id = {WORKSPACE}")
    through_run = (
        "EXISTS (SELECT 1 FROM stratafi.answer_runs AS run "
        "WHERE run.id = answer_results.answer_run_id)"
    )
    _protect("answer_results", f"{SYSTEM} OR {through_run}")


def upgrade() -> None:
    _tenancy()
    _artifacts_and_jobs()
    _configuration()
    _documents()
    _runs()
    _eval_sets()
    _eval_runs()
    _experiments()
    _answers_and_questions()
    _tests_and_automation()
    _provider_limits()
    _training()
    _tracking()
    _row_level_security()


def downgrade() -> None:
    raise NotImplementedError("StrataFi's migrations are forward-only")
