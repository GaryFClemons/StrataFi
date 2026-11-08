"""StrataFi's tables, as SQLAlchemy Core metadata.

Everything lives in the Postgres schema ``stratafi``, and every workspace's row carries
``workspace_id``. Migrations are written by hand in ``stratafi_store/migrations``; a test checks
that they build exactly this metadata.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

SCHEMA = "stratafi"

# Constraint names that are stable across migrations and readable in errors.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = sa.MetaData(schema=SCHEMA, naming_convention=NAMING_CONVENTION)

WORKSPACE_ROLES = ("owner", "editor", "viewer")
ACTOR_KINDS = ("user", "service", "token", "system")
AUDIT_OUTCOMES = ("success", "denied", "failed")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(value) for value in values)})"


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    ]


workspaces = sa.Table(
    "workspaces",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("slug", sa.Text, nullable=False, unique=True),
    sa.Column("name", sa.Text, nullable=False),
    # The tenant an identity token names (its ``tid``), which signs in to this workspace.
    sa.Column("external_tenant_id", sa.Text, nullable=True, unique=True),
    *_timestamps(),
    sa.CheckConstraint(r"slug ~ '^[a-z][a-z0-9_-]{0,62}$'", name="slug_shape"),
)

memberships = sa.Table(
    "memberships",
    metadata,
    sa.Column(
        "workspace_id",
        sa.Uuid,
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    # The identity's subject (its ``sub`` claim). No user table is the source of truth.
    sa.Column("subject", sa.Text, primary_key=True),
    sa.Column("role", sa.Text, nullable=False),
    *_timestamps(),
    sa.CheckConstraint(_in("role", WORKSPACE_ROLES), name="role"),
)

# Who did what, with IDs and counts only: no document, chunk or query text. Rows outlive their
# workspace, so there is no foreign key.
audit_events = sa.Table(
    "audit_events",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column(
        "occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Column("actor_kind", sa.Text, nullable=False),
    sa.Column("actor_subject", sa.Text, nullable=False),
    sa.Column("action", sa.Text, nullable=False),
    sa.Column("resource_type", sa.Text, nullable=True),
    sa.Column("resource_id", sa.Text, nullable=True),
    sa.Column("outcome", sa.Text, nullable=False),
    sa.Column("request_id", sa.Text, nullable=True),
    sa.Column(
        "details",
        postgresql.JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    ),
    sa.CheckConstraint(_in("actor_kind", ACTOR_KINDS), name="actor_kind"),
    sa.CheckConstraint(_in("outcome", AUDIT_OUTCOMES), name="outcome"),
    # B-tree indexes scan backwards, so this serves "newest first" too.
    sa.Index("ix_audit_events_workspace_occurred", "workspace_id", "occurred_at"),
)

# Every stage output, addressed by its cache key; raw documents are artifacts too, under
# their content hash. Payloads live in the artifact store at ``storage_uri``; ``summary`` holds
# small counts for the UI, never text.
artifacts = sa.Table(
    "artifacts",
    metadata,
    sa.Column(
        "workspace_id",
        sa.Uuid,
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("cache_key", sa.Text, primary_key=True),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("storage_uri", sa.Text, nullable=False),
    sa.Column("size_bytes", sa.BigInteger, nullable=False),
    sa.Column("content_sha256", sa.Text, nullable=False),
    sa.Column("media_type", sa.Text, nullable=False),
    sa.Column(
        "summary",
        postgresql.JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    ),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.CheckConstraint("cache_key ~ '^[0-9a-f]{64}$'", name="cache_key_shape"),
    sa.CheckConstraint("content_sha256 ~ '^[0-9a-f]{64}$'", name="content_sha256_shape"),
    sa.CheckConstraint("size_bytes >= 0", name="size_bytes"),
    sa.CheckConstraint(r"kind ~ '^[a-z][a-z0-9_.]{0,62}$'", name="kind_shape"),
)

# Lineage: which input artifacts each artifact was made from. Deleting an artifact deletes its
# own edges. An input that still has dependents can't be deleted until they are (purging
# follows lineage). That check waits for the commit, so one transaction can delete a
# whole graph, or a whole workspace, in any order.
artifact_inputs = sa.Table(
    "artifact_inputs",
    metadata,
    sa.Column("workspace_id", sa.Uuid, primary_key=True),
    sa.Column("artifact_key", sa.Text, primary_key=True),
    sa.Column("input_key", sa.Text, primary_key=True),
    sa.Column("role", sa.Text, nullable=False, server_default="input"),
    sa.Column("ordinal", sa.Integer, nullable=False, server_default="0"),
    sa.ForeignKeyConstraint(
        ["workspace_id", "artifact_key"],
        ["artifacts.workspace_id", "artifacts.cache_key"],
        ondelete="CASCADE",
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "input_key"],
        ["artifacts.workspace_id", "artifacts.cache_key"],
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.CheckConstraint("artifact_key <> input_key", name="not_self"),
    sa.Index("ix_artifact_inputs_workspace_id_input_key", "workspace_id", "input_key"),
)

# Provider limits every worker process shares: the buckets' levels, and the calls in
# flight, each a lease that lapses if its worker dies. Configuration's names only, no tenant data.
provider_quota = sa.Table(
    "provider_quota",
    metadata,
    sa.Column("provider", sa.Text, primary_key=True),
    sa.Column("requests", sa.Float, nullable=True),
    sa.Column("tokens", sa.Float, nullable=True),
    sa.Column(
        "refilled_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
)

provider_leases = sa.Table(
    "provider_leases",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "provider",
        sa.Text,
        sa.ForeignKey("provider_quota.provider", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("holder", sa.Text, nullable=False),
    sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    sa.Index("ix_provider_leases_provider_expires_at", "provider", "expires_at"),
)

RESOURCE_CLASSES = ("cpu", "gpu", "io")
JOB_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
ACTIVE_JOB_STATUSES = ("queued", "running")

# The job queue: claimed with SKIP LOCKED, held by a lease that a heartbeat renews,
# retried with backoff until max_attempts. A dedupe key is unique among active jobs only, so the
# same work can be queued again after it finishes. Times come from the database's clock.
jobs = sa.Table(
    "jobs",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("payload", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
    sa.Column("resource_class", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="queued"),
    sa.Column("dedupe_key", sa.Text, nullable=True),
    sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
    sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
    sa.Column(
        "run_after", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Column("lease_owner", sa.Text, nullable=True),
    sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("cancel_requested", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("progress", postgresql.JSONB, nullable=True),
    sa.Column("result", postgresql.JSONB, nullable=True),
    sa.Column("error_code", sa.Text, nullable=True),
    *_timestamps(),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint(_in("resource_class", RESOURCE_CLASSES), name="resource_class"),
    sa.CheckConstraint(_in("status", JOB_STATUSES), name="status"),
    sa.CheckConstraint("max_attempts >= 1", name="max_attempts"),
    sa.CheckConstraint("attempts >= 0", name="attempts"),
    sa.CheckConstraint(r"kind ~ '^[a-z][a-z0-9_.]{0,62}$'", name="kind_shape"),
    sa.Index(
        "uq_jobs_active_dedupe",
        "workspace_id",
        "dedupe_key",
        unique=True,
        postgresql_where=sa.text("dedupe_key IS NOT NULL AND status IN ('queued', 'running')"),
    ),
    sa.Index(
        "ix_jobs_claim",
        "resource_class",
        sa.text("priority DESC"),
        "run_after",
        "created_at",
        postgresql_where=sa.text("status = 'queued'"),
    ),
    sa.Index(
        "ix_jobs_lease",
        "lease_expires_at",
        postgresql_where=sa.text("status = 'running'"),
    ),
    sa.Index("ix_jobs_workspace_created", "workspace_id", "created_at", "id"),
)

# --- configuration -----------------------------------------------------------------------

# A snapshot of the component registry, synced when the API starts. Components are code, the same
# for every workspace, so this table is global. Rows are kept after a version leaves the
# code, so old runs can still name it; ``available`` says whether this process's code has it.
components = sa.Table(
    "components",
    metadata,
    sa.Column("component_id", sa.Text, primary_key=True),
    sa.Column("version", sa.Text, primary_key=True),
    sa.Column("stage", sa.Text, nullable=False),
    sa.Column("info", postgresql.JSONB, nullable=False),
    sa.Column("params_schema", postgresql.JSONB, nullable=False),
    sa.Column("available", sa.Boolean, nullable=False, server_default=sa.true()),
    sa.Column(
        "first_seen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Column(
        "last_seen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.CheckConstraint(
        r"component_id ~ '^[a-z][a-z0-9-]{0,31}/[a-z0-9][a-z0-9.-]{0,63}$'", name="id_shape"
    ),
)


def _versioned_documents(name: str) -> sa.Table:
    """Pipeline specs and retrieval configs: named chains of immutable versions. A version holds
    the resolved document (every default filled in), its configuration hash and its document
    hash."""
    return sa.Table(
        name,
        metadata,
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "workspace_id",
            sa.Uuid,
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("document", postgresql.JSONB, nullable=False),
        sa.Column("spec_hash", sa.Text, nullable=False),
        sa.Column("document_hash", sa.Text, nullable=False),
        sa.Column("created_by", sa.Text, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("workspace_id", "name", "version"),
        sa.CheckConstraint("version >= 1", name="version"),
        sa.CheckConstraint(r"name ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name="name_shape"),
        sa.CheckConstraint("spec_hash ~ '^[0-9a-f]{64}$'", name="spec_hash_shape"),
        sa.CheckConstraint("document_hash ~ '^[0-9a-f]{64}$'", name="document_hash_shape"),
        sa.Index(f"ix_{name}_workspace_id_spec_hash", "workspace_id", "spec_hash"),
    )


pipeline_specs = _versioned_documents("pipeline_specs")
retrieval_configs = _versioned_documents("retrieval_configs")

# --- documents ---------------------------------------------------------------------

# A corpus is a named, editable set of documents; a corpus version is a frozen snapshot of it,
# with a manifest hash.
corpora = sa.Table(
    "corpora",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("description", sa.Text, nullable=False, server_default=""),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.UniqueConstraint("workspace_id", "name"),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(r"name ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name="name_shape"),
)

corpus_versions = sa.Table(
    "corpus_versions",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("corpus_id", sa.Uuid, nullable=False),
    sa.Column("version", sa.Integer, nullable=False),
    sa.Column("manifest_hash", sa.Text, nullable=False),
    sa.Column("document_count", sa.Integer, nullable=False),
    sa.Column("total_bytes", sa.BigInteger, nullable=False),
    sa.Column("sealed", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("created_by", sa.Text, nullable=False),
    sa.Column(
        "frozen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "corpus_id"], ["corpora.workspace_id", "corpora.id"], ondelete="CASCADE"
    ),
    sa.UniqueConstraint("corpus_id", "version"),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint("version >= 1", name="version"),
    sa.CheckConstraint("document_count >= 0", name="document_count"),
    sa.CheckConstraint("total_bytes >= 0", name="total_bytes"),
    sa.CheckConstraint("manifest_hash ~ '^[0-9a-f]{64}$'", name="manifest_hash_shape"),
)

# A withdrawn document was purged: its row stays as a tombstone.
DOCUMENT_STATUSES = ("stored", "refused", "withdrawn")
SOURCE_KINDS = ("upload", "folder", "archive", "manifest", "s3", "url")

# Where documents came from: one row per upload request, folder import or unpacked archive.
sources = sa.Table(
    "sources",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("details", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("created_by", sa.Text, nullable=False),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("kind", SOURCE_KINDS), name="kind"),
)

# A document is a file's bytes, unique per workspace by SHA-256: arriving twice is one document.
# The bytes sit in the quarantine prefix at ``storage_key``. Its type and traits are as detected
# when it first arrived. A refused archive is recorded with its reason, so the user can see why.
documents = sa.Table(
    "documents",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("sha256", sa.Text, nullable=False),
    sa.Column("size_bytes", sa.BigInteger, nullable=False),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("extension", sa.Text, nullable=False),
    sa.Column("media_type", sa.Text, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("mismatch", sa.Boolean, nullable=False),
    sa.Column("known", sa.Boolean, nullable=False),
    sa.Column("traits", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("reason_code", sa.Text, nullable=True),
    sa.Column("storage_key", sa.Text, nullable=False),
    sa.Column("created_by", sa.Text, nullable=False),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.UniqueConstraint("workspace_id", "sha256"),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_shape"),
    sa.CheckConstraint("size_bytes >= 0", name="size_bytes"),
    sa.CheckConstraint(_in("status", DOCUMENT_STATUSES), name="status"),
    sa.Index("ix_documents_workspace_created", "workspace_id", "created_at", "id"),
)

# Provenance: every arrival of a document, with its name, its source and, for an archive member,
# the archive it came from and its path inside it.
document_origins = sa.Table(
    "document_origins",
    metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("document_id", sa.Uuid, nullable=False),
    sa.Column("source_id", sa.Uuid, nullable=False),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("parent_document_id", sa.Uuid, nullable=True),
    sa.Column("member_path", sa.Text, nullable=True),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "document_id"],
        ["documents.workspace_id", "documents.id"],
        ondelete="CASCADE",
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "source_id"], ["sources.workspace_id", "sources.id"], ondelete="CASCADE"
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "parent_document_id"],
        ["documents.workspace_id", "documents.id"],
        ondelete="CASCADE",
    ),
    sa.Index("ix_document_origins_workspace_id_document_id", "workspace_id", "document_id"),
)


def _entry_columns() -> list[sa.Column]:
    """A corpus entry's metadata, as a manifest import gives it."""
    return [
        sa.Column("doc_key", sa.Text, nullable=True),
        sa.Column("doc_version", sa.Text, nullable=True),
        sa.Column("collection", sa.Text, nullable=True),
        sa.Column("acl", postgresql.JSONB, nullable=True),
        sa.Column("effective_from", sa.Date, nullable=True),
        sa.Column("effective_to", sa.Date, nullable=True),
        sa.Column(
            "metadata", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
    ]


# A corpus's working set: entry names for documents. The same document may appear under two
# names; scan/dedupe excludes the later ones in a run.
corpus_documents = sa.Table(
    "corpus_documents",
    metadata,
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("corpus_id", sa.Uuid, primary_key=True),
    sa.Column("name", sa.Text, primary_key=True),
    sa.Column("document_id", sa.Uuid, nullable=False),
    *_entry_columns(),
    sa.Column("added_by", sa.Text, nullable=False),
    sa.Column("added_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    sa.ForeignKeyConstraint(
        ["workspace_id", "corpus_id"], ["corpora.workspace_id", "corpora.id"], ondelete="CASCADE"
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "document_id"],
        ["documents.workspace_id", "documents.id"],
        ondelete="CASCADE",
    ),
)

# A frozen version's entries, with what each document was when it was frozen. A trigger refuses
# any change once the version is sealed (the baseline migration).
corpus_version_documents = sa.Table(
    "corpus_version_documents",
    metadata,
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("corpus_version_id", sa.Uuid, primary_key=True),
    sa.Column("name", sa.Text, primary_key=True),
    sa.Column("document_id", sa.Uuid, nullable=False),
    sa.Column("sha256", sa.Text, nullable=False),
    sa.Column("size_bytes", sa.BigInteger, nullable=False),
    sa.Column("media_type", sa.Text, nullable=False),
    sa.Column("traits", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    *_entry_columns(),
    sa.ForeignKeyConstraint(
        ["workspace_id", "corpus_version_id"],
        ["corpus_versions.workspace_id", "corpus_versions.id"],
        ondelete="CASCADE",
    ),
    # Deferred, so one transaction can delete a workspace's documents and versions in any order.
    sa.ForeignKeyConstraint(
        ["workspace_id", "document_id"],
        ["documents.workspace_id", "documents.id"],
        deferrable=True,
        initially="DEFERRED",
    ),
)

# --- runs --------------------------------------------------------------------------------

RUN_STATUSES = ("queued", "running", "succeeded", "failed", "cancelling", "cancelled")
ACTIVE_RUN_STATUSES = ("queued", "running", "cancelling")
RUN_MODES = ("interactive", "batch")
TASK_STATUSES = (
    "pending",
    "queued",
    "running",
    "succeeded",
    "cached",
    "skipped",
    "failed",
    "cancelled",
)
ENTRY_STATES = ("active", "done", "excluded", "unsupported_type", "failed", "cancelled")

# A run: one pipeline spec version over one frozen corpus version, to a stage, on a sample. Its
# manifest is written when it starts and finalized when it ends; manifest_hash is the final
# one's hash, the run's fingerprint.
runs = sa.Table(
    "runs",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("kind", sa.Text, nullable=False, server_default="pipeline"),
    sa.Column("mode", sa.Text, nullable=False, server_default="batch"),
    sa.Column("status", sa.Text, nullable=False, server_default="queued"),
    sa.Column("pipeline_spec_id", sa.Uuid, nullable=False),
    sa.Column("corpus_version_id", sa.Uuid, nullable=False),
    sa.Column("until_stage", sa.Text, nullable=True),
    sa.Column("sample", postgresql.JSONB, nullable=True),
    sa.Column("rerun_of", sa.Uuid, nullable=True),
    sa.Column("rerun_from", sa.Text, nullable=True),
    sa.Column("spec_hash", sa.Text, nullable=False),
    sa.Column("model_env", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("manifest", postgresql.JSONB, nullable=True),
    sa.Column("manifest_hash", sa.Text, nullable=True),
    sa.Column("counts", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    # Deferred, so a workspace's runs, specs and corpus versions can go in one transaction.
    sa.ForeignKeyConstraint(
        ["pipeline_spec_id"], ["pipeline_specs.id"], deferrable=True, initially="DEFERRED"
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "corpus_version_id"],
        ["corpus_versions.workspace_id", "corpus_versions.id"],
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("status", RUN_STATUSES), name="status"),
    sa.CheckConstraint(_in("mode", RUN_MODES), name="mode"),
    sa.CheckConstraint("kind = 'pipeline'", name="kind"),
    sa.Index("ix_runs_workspace_created", "workspace_id", "created_at", "id"),
)

# Each corpus entry a run works on, with where it ended up.
run_entries = sa.Table(
    "run_entries",
    metadata,
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("run_id", sa.Uuid, primary_key=True),
    sa.Column("ordinal", sa.Integer, primary_key=True),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("document_id", sa.Uuid, nullable=False),
    sa.Column("sha256", sa.Text, nullable=False),
    sa.Column("duplicate", sa.Boolean, nullable=False),
    sa.Column("label", sa.Text, nullable=True),
    sa.Column("state", sa.Text, nullable=False, server_default="active"),
    sa.Column("stop_code", sa.Text, nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "run_id"], ["runs.workspace_id", "runs.id"], ondelete="CASCADE"
    ),
    sa.CheckConstraint(_in("state", ENTRY_STATES), name="state"),
    # A run's active entries, for the checks each completion makes.
    sa.Index("ix_run_entries_run_active", "run_id", postgresql_where=sa.text("state = 'active'")),
)

# One task per run, stage and entry (ordinal -1 for corpus-level tasks), addressed by the cache
# key of the artifact it makes. Its parameters and environment are what its key hashed.
stage_tasks = sa.Table(
    "stage_tasks",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("run_id", sa.Uuid, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("stage", sa.Text, nullable=False),
    sa.Column("component_id", sa.Text, nullable=False),
    sa.Column("component_version", sa.Text, nullable=False),
    sa.Column("cache_key", sa.Text, nullable=False),
    sa.Column("inputs", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.Column("params", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("env", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("route", sa.Text, nullable=True),
    sa.Column("resource_class", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("cache_hit", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("job_id", sa.Uuid, nullable=True),
    sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("metrics", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("duration_ms", sa.Float, nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "run_id"], ["runs.workspace_id", "runs.id"], ondelete="CASCADE"
    ),
    sa.UniqueConstraint("run_id", "stage", "component_id", "ordinal"),
    sa.CheckConstraint(_in("status", TASK_STATUSES), name="status"),
    sa.CheckConstraint("cache_key ~ '^[0-9a-f]{64}$'", name="cache_key_shape"),
    sa.Index("ix_stage_tasks_workspace_id_cache_key", "workspace_id", "cache_key"),
    sa.Index("ix_stage_tasks_run_id_ordinal", "run_id", "ordinal"),
    # A run's active tasks, for the checks each completion makes.
    sa.Index(
        "ix_stage_tasks_run_active",
        "run_id",
        postgresql_where=sa.text("status IN ('pending', 'queued', 'running')"),
    ),
    # The reconcile sweep looks up active tasks' jobs.
    sa.Index(
        "ix_stage_tasks_active_job_id",
        "job_id",
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    ),
)

# --- eval sets ---------------------------------------------------------------------------

EVAL_SPLITS = ("dev", "test")

# A named set of questions with known evidence; its versions hold the questions.
eval_sets = sa.Table(
    "eval_sets",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("description", sa.Text, nullable=False, server_default=""),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.UniqueConstraint("workspace_id", "name"),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(r"name ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name="name_shape"),
)

# One version of an eval set: a draft can be split again, a frozen one never changes (a trigger
# refuses it, in the baseline migration), except for its test-split lock and read counter.
# ``document`` holds the set's own fields (as_of, defaults, askers); ``document_hash`` covers the
# whole resolved set, items included.
eval_set_versions = sa.Table(
    "eval_set_versions",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("eval_set_id", sa.Uuid, nullable=False),
    sa.Column("version", sa.Integer, nullable=False),
    sa.Column("document", postgresql.JSONB, nullable=False),
    sa.Column("document_hash", sa.Text, nullable=False),
    sa.Column("item_count", sa.Integer, nullable=False),
    sa.Column(
        "split_counts", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
    ),
    sa.Column("split_params", postgresql.JSONB, nullable=True),
    sa.Column("source", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("frozen", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("test_locked", sa.Boolean, nullable=False, server_default=sa.true()),
    sa.Column("test_reads", sa.Integer, nullable=False, server_default="0"),
    sa.Column("last_test_read_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_set_id"],
        ["eval_sets.workspace_id", "eval_sets.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("eval_set_id", "version"),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint("version >= 1", name="version"),
    sa.CheckConstraint("item_count >= 0", name="item_count"),
    sa.CheckConstraint("test_reads >= 0", name="test_reads"),
    sa.CheckConstraint("document_hash ~ '^[0-9a-f]{64}$'", name="document_hash_shape"),
)

# A version's items: the query, who asks it, its gold evidence and what it must not return.
# Text lives here and in API responses only, never in logs or audit events.
eval_items = sa.Table(
    "eval_items",
    metadata,
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("eval_set_version_id", sa.Uuid, primary_key=True),
    sa.Column("item_id", sa.Text, primary_key=True),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("query", sa.Text, nullable=False),
    sa.Column("asker", postgresql.JSONB, nullable=False),
    sa.Column("expected", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.Column("avoid", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.Column(
        "must_not_return", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")
    ),
    sa.Column(
        "near_misses", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")
    ),
    sa.Column("reference_answer", sa.Text, nullable=True),
    sa.Column("tags", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.Column("split", sa.Text, nullable=False),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_set_version_id"],
        ["eval_set_versions.workspace_id", "eval_set_versions.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("eval_set_version_id", "ordinal"),
    sa.CheckConstraint(_in("split", EVAL_SPLITS), name="split"),
)

# --- eval runs ---------------------------------------------------------------------------

EVAL_RUN_SPLITS = ("dev", "test", "all")
EVAL_SHARD_STATUSES = ("queued", "running", "succeeded", "cached", "failed", "cancelled")

# An eval run: a retrieval config version over a finished pipeline run's indexes, scored on a
# frozen eval set version's split. Its record (stratafi.eval-run.v1) is written when it ends;
# ``metrics`` holds the aggregates with their intervals, slices and latencies (numbers only).
eval_runs = sa.Table(
    "eval_runs",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("status", sa.Text, nullable=False, server_default="queued"),
    sa.Column("pipeline_run_id", sa.Uuid, nullable=False),
    sa.Column("retrieval_config_id", sa.Uuid, nullable=False),
    sa.Column("eval_set_version_id", sa.Uuid, nullable=False),
    sa.Column("split", sa.Text, nullable=False),
    sa.Column("options", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("retrieval_hash", sa.Text, nullable=False),
    sa.Column("results_key", sa.Text, nullable=True),
    sa.Column("results_cached", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("test_read", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("context", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("manifest", postgresql.JSONB, nullable=True),
    sa.Column("manifest_hash", sa.Text, nullable=True),
    sa.Column("metrics", postgresql.JSONB, nullable=True),
    sa.Column("counts", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    # Deferred, so a workspace's runs, configs and eval sets can go in one transaction.
    sa.ForeignKeyConstraint(
        ["workspace_id", "pipeline_run_id"],
        ["runs.workspace_id", "runs.id"],
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.ForeignKeyConstraint(
        ["retrieval_config_id"], ["retrieval_configs.id"], deferrable=True, initially="DEFERRED"
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_set_version_id"],
        ["eval_set_versions.workspace_id", "eval_set_versions.id"],
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("status", RUN_STATUSES), name="status"),
    sa.CheckConstraint(_in("split", EVAL_RUN_SPLITS), name="split"),
    sa.Index("ix_eval_runs_workspace_created", "workspace_id", "created_at", "id"),
)

# A shard of an eval run: a batch of queries whose ranked lists are one artifact.
eval_shards = sa.Table(
    "eval_shards",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("eval_run_id", sa.Uuid, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("cache_key", sa.Text, nullable=False),
    sa.Column("item_ids", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("cache_hit", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("job_id", sa.Uuid, nullable=True),
    sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("metrics", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("duration_ms", sa.Float, nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_run_id"],
        ["eval_runs.workspace_id", "eval_runs.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("eval_run_id", "ordinal"),
    sa.CheckConstraint(_in("status", EVAL_SHARD_STATUSES), name="status"),
    sa.CheckConstraint("cache_key ~ '^[0-9a-f]{64}$'", name="cache_key_shape"),
)

# Each query's outcome in an eval run: its metrics at every checkpoint (per-query, so
# comparisons can pair them), where a miss was lost, and its violations. Ranked lists live in
# the shards' artifacts.
eval_query_results = sa.Table(
    "eval_query_results",
    metadata,
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("eval_run_id", sa.Uuid, primary_key=True),
    sa.Column("item_id", sa.Text, primary_key=True),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("shard", sa.Integer, nullable=False),
    sa.Column("split", sa.Text, nullable=False),
    sa.Column("tags", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.Column(
        "file_types", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")
    ),
    sa.Column("has_evidence", sa.Boolean, nullable=False),
    sa.Column("hit", sa.Boolean, nullable=True),
    sa.Column("first_hit_rank", sa.Integer, nullable=True),
    sa.Column("lost_at", sa.Text, nullable=True),
    sa.Column("avoid_violation", sa.Boolean, nullable=True),
    sa.Column("must_not_return_violation", sa.Boolean, nullable=False),
    sa.Column("score", postgresql.JSONB, nullable=False),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_run_id"],
        ["eval_runs.workspace_id", "eval_runs.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("eval_run_id", "ordinal"),
    sa.CheckConstraint("lost_at IN ('parse', 'retrieval', 'rerank')", name="lost_at"),
)

# --- experiments -------------------------------------------------------------------------

EXPERIMENT_STATUSES = (
    "planned",
    "running",
    "stopping",
    "succeeded",
    "stopped",
    "failed",
    "cancelling",
    "cancelled",
)
ACTIVE_EXPERIMENT_STATUSES = ("running", "stopping", "cancelling")
TRIAL_STATUSES = ("pending", "running", "succeeded", "failed", "skipped", "cancelled", "pruned")

# An experiment: a space of configurations against one eval set's dev split. ``document`` is
# its stratafi.experiment.v1, as json rather than jsonb: jsonb reorders keys, and the order of
# the factors is the order of the grid. The base documents, the corpus version and the eval set
# version are pinned when it's planned. ``plan`` holds the factors, the unique work and the
# estimate; ``spent`` what its runs cost so far.
experiments = sa.Table(
    "experiments",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="planned"),
    sa.Column("document", postgresql.JSON, nullable=False),
    sa.Column("document_hash", sa.Text, nullable=False),
    sa.Column("corpus_version_id", sa.Uuid, nullable=False),
    sa.Column("eval_set_version_id", sa.Uuid, nullable=False),
    sa.Column("pipeline_spec_id", sa.Uuid, nullable=False),
    sa.Column("retrieval_config_id", sa.Uuid, nullable=False),
    sa.Column("plan", postgresql.JSONB, nullable=False),
    sa.Column("spent", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("counts", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("stop_reason", sa.Text, nullable=True),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    sa.Column("started_by", sa.Text, nullable=True),
    *_timestamps(),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "corpus_version_id"],
        ["corpus_versions.workspace_id", "corpus_versions.id"],
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_set_version_id"],
        ["eval_set_versions.workspace_id", "eval_set_versions.id"],
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.ForeignKeyConstraint(
        ["pipeline_spec_id"], ["pipeline_specs.id"], deferrable=True, initially="DEFERRED"
    ),
    sa.ForeignKeyConstraint(
        ["retrieval_config_id"], ["retrieval_configs.id"], deferrable=True, initially="DEFERRED"
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("status", EXPERIMENT_STATUSES), name="status"),
    sa.CheckConstraint(r"name ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name="name_shape"),
    sa.CheckConstraint("document_hash ~ '^[0-9a-f]{64}$'", name="document_hash_shape"),
    sa.Index("ix_experiments_workspace_created", "workspace_id", "created_at", "id"),
)

# One configuration of an experiment: its level per factor, the saved variants it runs, the
# pipeline run it shares with the trials of the same pipeline, its eval run, and its results.
trials = sa.Table(
    "trials",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("experiment_id", sa.Uuid, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("grid_index", sa.Integer, nullable=False),
    sa.Column("levels", postgresql.JSONB, nullable=False),
    sa.Column("factor_values", postgresql.JSONB, nullable=False),
    sa.Column("pipeline_hash", sa.Text, nullable=False),
    sa.Column("retrieval_hash", sa.Text, nullable=False),
    sa.Column("pipeline_spec_id", sa.Uuid, nullable=True),
    sa.Column("retrieval_config_id", sa.Uuid, nullable=True),
    sa.Column("pipeline_run_id", sa.Uuid, nullable=True),
    sa.Column("eval_run_id", sa.Uuid, nullable=True),
    sa.Column("status", sa.Text, nullable=False, server_default="pending"),
    sa.Column("objective", postgresql.JSONB, nullable=True),
    sa.Column("measures", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column(
        "constraints", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")
    ),
    sa.Column("feasible", sa.Boolean, nullable=True),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    # Successive halving: the rung a trial is at, and each finished rung's result.
    sa.Column("rung", sa.Integer, nullable=False, server_default="0"),
    sa.Column("rungs", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.ForeignKeyConstraint(
        ["workspace_id", "experiment_id"],
        ["experiments.workspace_id", "experiments.id"],
        ondelete="CASCADE",
    ),
    sa.ForeignKeyConstraint(
        ["pipeline_spec_id"], ["pipeline_specs.id"], deferrable=True, initially="DEFERRED"
    ),
    sa.ForeignKeyConstraint(
        ["retrieval_config_id"], ["retrieval_configs.id"], deferrable=True, initially="DEFERRED"
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "pipeline_run_id"],
        ["runs.workspace_id", "runs.id"],
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_run_id"],
        ["eval_runs.workspace_id", "eval_runs.id"],
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.UniqueConstraint("experiment_id", "ordinal"),
    sa.CheckConstraint(_in("status", TRIAL_STATUSES), name="status"),
    sa.CheckConstraint("rung >= 0", name="rung"),
    # A finished run tells the experiments waiting on it (stratafi_store.experiments).
    sa.Index("ix_trials_pipeline_run_id", "pipeline_run_id"),
    sa.Index("ix_trials_eval_run_id", "eval_run_id"),
)

# --- test-split confirmations ------------------------------------------------------------

REREAD_REASONS = ("run_failed", "candidate_changed", "other")

# A read of the locked test split to confirm a trial chosen on dev: the eval run that
# read it, and its place among the experiment's reads. The first needs no reason; every later one
# gives a reason code (and may add a note, kept here, never logged), so no second read goes
# unlogged.
confirmations = sa.Table(
    "confirmations",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("experiment_id", sa.Uuid, nullable=False),
    sa.Column("trial_id", sa.Uuid, nullable=False),
    sa.Column("eval_run_id", sa.Uuid, nullable=False),
    sa.Column("eval_set_version_id", sa.Uuid, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("reason", sa.Text, nullable=True),
    sa.Column("note", sa.Text, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "experiment_id"],
        ["experiments.workspace_id", "experiments.id"],
        ondelete="CASCADE",
    ),
    sa.ForeignKeyConstraint(["trial_id"], ["trials.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_run_id"],
        ["eval_runs.workspace_id", "eval_runs.id"],
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.UniqueConstraint("experiment_id", "ordinal"),
    sa.CheckConstraint("ordinal >= 1", name="ordinal"),
    sa.CheckConstraint(
        f"(ordinal = 1 AND reason IS NULL) OR {_in('reason', REREAD_REASONS)}",
        name="reason",
    ),
    sa.CheckConstraint("note IS NULL OR length(note) <= 500", name="note_length"),
)

# --- knowledge profiles ------------------------------------------------------------------

PROFILE_STATUSES = ("draft", "approved", "rejected")
PROFILE_LEVELS = ("A", "B", "C")

# A knowledge profile's versions: each holds its stratafi.knowledge-profile.v1, as exported from
# a confirmed trial or imported, and its review's status. ``document_hash`` is the profile's
# hash, which its review doesn't change.
knowledge_profiles = sa.Table(
    "knowledge_profiles",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("version", sa.Integer, nullable=False),
    sa.Column("document", postgresql.JSONB, nullable=False),
    sa.Column("document_hash", sa.Text, nullable=False),
    sa.Column("target", sa.Text, nullable=False),
    sa.Column("level", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="draft"),
    sa.Column("source", sa.Text, nullable=False),
    sa.Column("experiment_id", sa.Uuid, nullable=True),
    sa.Column("trial_id", sa.Uuid, nullable=True),
    sa.Column("confirmation_id", sa.Uuid, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.ForeignKeyConstraint(
        ["workspace_id", "experiment_id"],
        ["experiments.workspace_id", "experiments.id"],
        ondelete="SET NULL",
        deferrable=True,
        initially="DEFERRED",
    ),
    sa.UniqueConstraint("workspace_id", "name", "version"),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("status", PROFILE_STATUSES), name="status"),
    sa.CheckConstraint(_in("level", PROFILE_LEVELS), name="level"),
    sa.CheckConstraint("source IN ('trial', 'import')", name="source"),
    sa.CheckConstraint("version >= 1", name="version"),
    sa.CheckConstraint(r"name ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name="name_shape"),
    sa.CheckConstraint("document_hash ~ '^[0-9a-f]{64}$'", name="document_hash_shape"),
)

# Each review decision on a profile version: who, when, which way. An owner decides, and not the
# person who exported it.
profile_approvals = sa.Table(
    "profile_approvals",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("profile_id", sa.Uuid, nullable=False),
    sa.Column("decision", sa.Text, nullable=False),
    sa.Column("decided_by", sa.Text, nullable=False),
    sa.Column(
        "decided_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    # The version's author decided it themselves.
    sa.Column("self_review", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.ForeignKeyConstraint(
        ["workspace_id", "profile_id"],
        ["knowledge_profiles.workspace_id", "knowledge_profiles.id"],
        ondelete="CASCADE",
    ),
    sa.CheckConstraint("decision IN ('approved', 'rejected')", name="decision"),
)

# --- answer-level evaluation and question generation ------------------------------------

ANSWER_RUN_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
LABEL_DIMENSIONS = ("faithful", "correct", "abstained")
CANDIDATE_STATUSES = ("proposed", "accepted", "rejected", "added")

# An answer run: a generator answers an eval run's queries from their final passages, and a
# judge grades the answers. The answers live in an artifact; the database holds numbers.
answer_runs = sa.Table(
    "answer_runs",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("eval_run_id", sa.Uuid, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="queued"),
    sa.Column("config", postgresql.JSONB, nullable=False),
    sa.Column("results_key", sa.Text, nullable=True),
    sa.Column("results_cached", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("metrics", postgresql.JSONB, nullable=True),
    sa.Column("usage", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_run_id"],
        ["eval_runs.workspace_id", "eval_runs.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("status", ANSWER_RUN_STATUSES), name="status"),
    sa.Index("ix_answer_runs_eval_run_id", "eval_run_id"),
)

# One answer's scores: numbers and flags, keyed by the hash of the answer, which labels attach to.
answer_results = sa.Table(
    "answer_results",
    metadata,
    sa.Column("answer_run_id", sa.Uuid, primary_key=True),
    sa.Column("item_id", sa.Text, primary_key=True),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("answer_key", sa.Text, nullable=False),
    sa.Column("score", postgresql.JSONB, nullable=False),
    sa.ForeignKeyConstraint(["answer_run_id"], ["answer_runs.id"], ondelete="CASCADE"),
    sa.CheckConstraint("answer_key ~ '^[0-9a-f]{64}$'", name="answer_key_shape"),
    sa.Index("ix_answer_results_answer_key", "answer_key"),
)

# A person's label for one answer on one dimension: the calibration set judges are measured
# against. One label per person, answer and dimension; the latest replaces the earlier one.
calibration_labels = sa.Table(
    "calibration_labels",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("answer_key", sa.Text, nullable=False),
    sa.Column("item_id", sa.Text, nullable=False),
    sa.Column("dimension", sa.Text, nullable=False),
    sa.Column("label", sa.Boolean, nullable=False),
    sa.Column("labeled_by", sa.Text, nullable=False),
    sa.Column(
        "labeled_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.UniqueConstraint("workspace_id", "answer_key", "dimension", "labeled_by"),
    sa.CheckConstraint(_in("dimension", LABEL_DIMENSIONS), name="dimension"),
    sa.CheckConstraint("answer_key ~ '^[0-9a-f]{64}$'", name="answer_key_shape"),
)

# A batch of drafted questions from one pipeline run's passages, for one eval set.
question_batches = sa.Table(
    "question_batches",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("eval_set_id", sa.Uuid, nullable=False),
    sa.Column("pipeline_run_id", sa.Uuid, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="queued"),
    sa.Column("config", postgresql.JSONB, nullable=False),
    sa.Column("counts", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_set_id"],
        ["eval_sets.workspace_id", "eval_sets.id"],
        ondelete="CASCADE",
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "pipeline_run_id"], ["runs.workspace_id", "runs.id"], ondelete="CASCADE"
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("status", ANSWER_RUN_STATUSES), name="status"),
)

# A drafted question waiting for a person: accepted (maybe edited) or rejected, then added to
# a new version of its eval set.
question_candidates = sa.Table(
    "question_candidates",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("eval_set_id", sa.Uuid, nullable=False),
    sa.Column("batch_id", sa.Uuid, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("document_id", sa.Uuid, nullable=True),
    sa.Column("chunk_id", sa.Text, nullable=True),
    sa.Column("doc_key", sa.Text, nullable=False),
    sa.Column("version", sa.Text, nullable=True),
    sa.Column("section", sa.Text, nullable=True),
    sa.Column("query", sa.Text, nullable=False),
    sa.Column("evidence", sa.Text, nullable=False),
    sa.Column("drafted_query", sa.Text, nullable=False),
    sa.Column("split", sa.Text, nullable=False, server_default="dev"),
    sa.Column("tags", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.Column("status", sa.Text, nullable=False, server_default="proposed"),
    sa.Column("decided_by", sa.Text, nullable=True),
    sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("item_id", sa.Text, nullable=True),
    sa.Column("added_version_id", sa.Uuid, nullable=True),
    *_timestamps(),
    sa.ForeignKeyConstraint(
        ["workspace_id", "batch_id"],
        ["question_batches.workspace_id", "question_batches.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("batch_id", "ordinal"),
    sa.CheckConstraint(_in("status", CANDIDATE_STATUSES), name="status"),
    sa.CheckConstraint("split IN ('dev', 'test')", name="split"),
    sa.Index("ix_question_candidates_eval_set_id_status", "eval_set_id", "status"),
)

# --- test suites ------------------------------------------------------------------------

# A test suite's versions: each holds its stratafi.test-suite.v1, resolved, with its
# configuration hash and its document hash, like pipeline specs.
test_suites = _versioned_documents("test_suites")

TEST_RUN_STATUSES = (
    "queued",
    "running",
    "evaluating",
    "passed",
    "failed",
    "error",
    "cancelling",
    "cancelled",
)
ACTIVE_TEST_RUN_STATUSES = ("running", "evaluating", "cancelling")
TEST_TRIGGERS = ("api", "schedule", "corpus_version_frozen", "target_changed", "ci")
CHECK_RESULTS = ("pass", "fail", "error", "skipped")

# A test run: one suite version, resolved to fixed inputs when it starts, the runs it waits for,
# and its record (stratafi.test-run.v1) once its checks are evaluated. One run per suite is
# active at a time, and at most one waits behind it: triggers that arrive meanwhile join the
# waiting one (``triggers``).
test_runs = sa.Table(
    "test_runs",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("suite_id", sa.Uuid, nullable=False),
    sa.Column("suite_name", sa.Text, nullable=False),
    sa.Column("suite_version", sa.Integer, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="queued"),
    sa.Column("trigger", sa.Text, nullable=False),
    sa.Column("triggers", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.Column("resolved", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("pipeline_run_id", sa.Uuid, nullable=True),
    sa.Column("eval_run_id", sa.Uuid, nullable=True),
    sa.Column("baseline_pipeline_run_id", sa.Uuid, nullable=True),
    sa.Column("baseline_eval_run_id", sa.Uuid, nullable=True),
    sa.Column("baseline_test_run_id", sa.Uuid, nullable=True),
    sa.Column("summary", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("record", postgresql.JSONB, nullable=True),
    sa.Column("record_hash", sa.Text, nullable=True),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(
        ["suite_id"], ["test_suites.id"], ondelete="CASCADE", deferrable=True, initially="DEFERRED"
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("status", TEST_RUN_STATUSES), name="status"),
    sa.CheckConstraint(_in("trigger", TEST_TRIGGERS), name="trigger"),
    sa.CheckConstraint(r"suite_name ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name="suite_name_shape"),
    sa.Index("ix_test_runs_workspace_created", "workspace_id", "created_at", "id"),
    sa.Index("ix_test_runs_suite", "workspace_id", "suite_name", "created_at"),
    sa.Index(
        "uq_test_runs_active_suite",
        "workspace_id",
        "suite_name",
        unique=True,
        postgresql_where=sa.text("status IN ('running', 'evaluating', 'cancelling')"),
    ),
    sa.Index(
        "uq_test_runs_queued_suite",
        "workspace_id",
        "suite_name",
        unique=True,
        postgresql_where=sa.text("status = 'queued'"),
    ),
    # A finished run tells the test runs waiting on it (stratafi_store.testruns).
    sa.Index("ix_test_runs_pipeline_run_id", "pipeline_run_id"),
    sa.Index("ix_test_runs_eval_run_id", "eval_run_id"),
    sa.Index("ix_test_runs_baseline_pipeline_run_id", "baseline_pipeline_run_id"),
    sa.Index("ix_test_runs_baseline_eval_run_id", "baseline_eval_run_id"),
)

# One check's result in a test run: what was measured, what it was held to, and where the
# evidence is. Numbers, codes and IDs only, so a suite's history and trends read from here.
test_check_results = sa.Table(
    "test_check_results",
    metadata,
    sa.Column("test_run_id", sa.Uuid, primary_key=True),
    sa.Column("check_id", sa.Text, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("type", sa.Text, nullable=False),
    sa.Column("result", sa.Text, nullable=False),
    sa.Column("code", sa.Text, nullable=True),
    sa.Column("value", sa.Float, nullable=True),
    sa.Column("baseline_value", sa.Float, nullable=True),
    sa.Column("threshold", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("detail", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("evidence", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    sa.Column("duration_ms", sa.Float, nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "test_run_id"],
        ["test_runs.workspace_id", "test_runs.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("test_run_id", "ordinal"),
    sa.CheckConstraint(_in("result", CHECK_RESULTS), name="result"),
    sa.Index("ix_test_check_results_workspace_id_check_id", "workspace_id", "check_id"),
)

# --- triggers, schedules and webhooks ---------------------------------------------------

# A suite's cron schedules, kept in step with its latest version: each schedule's next time, in
# UTC, and what it last started. The worker that holds the scheduler's advisory lock fires them.
test_schedules = sa.Table(
    "test_schedules",
    metadata,
    sa.Column(
        "workspace_id",
        sa.Uuid,
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("suite_name", sa.Text, primary_key=True),
    sa.Column("cron", sa.Text, primary_key=True),
    sa.Column("time_zone", sa.Text, nullable=False),
    sa.Column("next_fire_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("last_test_run_id", sa.Uuid, nullable=True),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Index("ix_test_schedules_next_fire_at", "next_fire_at"),
)

WEBHOOK_DELIVERY_STATUSES = ("pending", "delivered", "failed")

# Where a workspace's finished test runs are announced: a URL, the events it wants, optionally
# one suite's runs only, and the secret that signs each delivery (stratafi_core.webhooks). The
# secret is shown once, when the webhook is made, and never logged or returned again.
webhooks = sa.Table(
    "webhooks",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("url", sa.Text, nullable=False),
    # None when the secret is derived from the deployment's key.
    sa.Column("secret", sa.Text, nullable=True),
    sa.Column("suite_name", sa.Text, nullable=True),
    sa.Column("events", postgresql.JSONB, nullable=False),
    sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint("length(url) <= 2048", name="url_length"),
)

# One announcement to one webhook, with its signed body kept as sent, so every retry carries the
# same body and delivery ID. Numbers and IDs only.
webhook_deliveries = sa.Table(
    "webhook_deliveries",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("webhook_id", sa.Uuid, nullable=False),
    sa.Column("event", sa.Text, nullable=False),
    sa.Column("test_run_id", sa.Uuid, nullable=True),
    sa.Column("body", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="pending"),
    sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
    sa.Column("response_status", sa.Integer, nullable=True),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "webhook_id"], ["webhooks.workspace_id", "webhooks.id"], ondelete="CASCADE"
    ),
    sa.CheckConstraint(_in("status", WEBHOOK_DELIVERY_STATUSES), name="status"),
    sa.Index("ix_webhook_deliveries_webhook_id_created_at", "webhook_id", "created_at"),
)

# --- API tokens ------------------------------------------------------------------------

TOKEN_SCOPES = ("read", "runs:write", "tests:run")

# A workspace's API tokens, for scripts and CI: stored as the SHA-256 of the token, which is
# shown once when it's made. Each has scopes, an expiry, and maybe a revocation.
api_tokens = sa.Table(
    "api_tokens",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column(
        "workspace_id", sa.Uuid, sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    ),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("prefix", sa.Text, nullable=False),
    sa.Column("token_hash", sa.Text, nullable=False, unique=True),
    sa.Column("scopes", postgresql.JSONB, nullable=False),
    sa.Column("created_by", sa.Text, nullable=False),
    sa.Column(
        "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("revoked_by", sa.Text, nullable=True),
    sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(r"name ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name="name_shape"),
    sa.CheckConstraint("token_hash ~ '^[0-9a-f]{64}$'", name="token_hash_shape"),
    sa.CheckConstraint("expires_at > created_at", name="expiry"),
    sa.Index("ix_api_tokens_workspace_id_created_at", "workspace_id", "created_at"),
)

# --- workspace features ----------------------------------------------------------------

WORKSPACE_FEATURES = ("custom_checks",)

# Features a workspace's owner turns on or off; a missing row is off. Turning ``custom_checks``
# on records the sandbox's security review it relied on.
workspace_features = sa.Table(
    "workspace_features",
    metadata,
    sa.Column(
        "workspace_id",
        sa.Uuid,
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("feature", sa.Text, primary_key=True),
    sa.Column("enabled", sa.Boolean, nullable=False),
    sa.Column("review", sa.Text, nullable=True),
    sa.Column("changed_by", sa.Text, nullable=False),
    sa.Column(
        "changed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
    ),
    sa.CheckConstraint(_in("feature", WORKSPACE_FEATURES), name="feature"),
)


# --- post-training ---------------------------------------------------------------------------

TRAINING_SET_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
TRAINING_RUN_STATUSES = ("queued", "running", "succeeded", "failed", "cancelling", "cancelled")
ACTIVE_TRAINING_RUN_STATUSES = ("queued", "running", "cancelling")
MODEL_KINDS = ("query-adapter", "embedder", "reranker")
MODEL_STATUSES = ("ready", "withdrawn")

# A training set mined from a dev eval run: its options, the key of its records
# (``stratafi.training-set.v1``, an artifact whose text never reaches this row), what it was
# mined from (``context``) and its counts, including the leak guard's.
training_sets = sa.Table(
    "training_sets",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("eval_run_id", sa.Uuid, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="queued"),
    sa.Column("options", postgresql.JSONB, nullable=False),
    sa.Column("cache_key", sa.Text, nullable=True),
    sa.Column("cached", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("context", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("stats", postgresql.JSONB, nullable=True),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_run_id"],
        ["eval_runs.workspace_id", "eval_runs.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("status", TRAINING_SET_STATUSES), name="status"),
    sa.CheckConstraint("cache_key IS NULL OR cache_key ~ '^[0-9a-f]{64}$'", name="cache_key_shape"),
    sa.Index("ix_training_sets_workspace_created", "workspace_id", "created_at", "id"),
    sa.Index("ix_training_sets_eval_run_id", "eval_run_id"),
)

# One training run: a trainer and its parameters over a training set, from a
# base model; its progress, metrics and record (``stratafi.training-run.v1``), and the model it
# made (an artifact keyed by what made it, so an identical run reuses it).
training_runs = sa.Table(
    "training_runs",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("training_set_id", sa.Uuid, nullable=False),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("trainer", sa.Text, nullable=False),
    sa.Column("trainer_version", sa.Text, nullable=False),
    sa.Column("params", postgresql.JSONB, nullable=False),
    sa.Column("base", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("status", sa.Text, nullable=False, server_default="queued"),
    sa.Column("progress", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("metrics", postgresql.JSONB, nullable=True),
    sa.Column("manifest", postgresql.JSONB, nullable=True),
    sa.Column("manifest_hash", sa.Text, nullable=True),
    sa.Column("model_key", sa.Text, nullable=True),
    sa.Column("model_cached", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("model_id", sa.Uuid, nullable=True),
    sa.Column("job_id", sa.Uuid, nullable=True),
    sa.Column("error_code", sa.Text, nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(
        ["workspace_id", "training_set_id"],
        ["training_sets.workspace_id", "training_sets.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.CheckConstraint(_in("status", TRAINING_RUN_STATUSES), name="status"),
    sa.CheckConstraint(_in("kind", MODEL_KINDS), name="kind"),
    sa.CheckConstraint("model_key IS NULL OR model_key ~ '^[0-9a-f]{64}$'", name="model_key_shape"),
    sa.Index("ix_training_runs_workspace_created", "workspace_id", "created_at", "id"),
    sa.Index("ix_training_runs_training_set_id", "training_set_id"),
)

# The model registry: a trained model by name and version, its kind, base, run,
# artifact and card (``stratafi.model-card.v1``). Withdrawing one stops new runs from using it.
trained_models = sa.Table(
    "trained_models",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("version", sa.Integer, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("training_run_id", sa.Uuid, nullable=False),
    sa.Column("model_key", sa.Text, nullable=False),
    sa.Column("base", postgresql.JSONB, nullable=False),
    sa.Column("card", postgresql.JSONB, nullable=False),
    sa.Column("status", sa.Text, nullable=False, server_default="ready"),
    sa.Column("withdrawn_by", sa.Text, nullable=True),
    sa.Column("withdrawn_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("created_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.ForeignKeyConstraint(
        ["workspace_id", "training_run_id"],
        ["training_runs.workspace_id", "training_runs.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("workspace_id", "id"),
    sa.UniqueConstraint("workspace_id", "name", "version"),
    sa.CheckConstraint(_in("kind", MODEL_KINDS), name="kind"),
    sa.CheckConstraint(_in("status", MODEL_STATUSES), name="status"),
    sa.CheckConstraint("model_key ~ '^[0-9a-f]{64}$'", name="model_key_shape"),
    sa.CheckConstraint(r"name ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name="name_shape"),
    sa.CheckConstraint("version >= 1", name="version"),
)

# --- tracking and comparison -----------------------------------------------------------------

# A pipeline run's, eval run's or experiment's title, tags, Markdown note and archive:
# people's text, never logged or audited. One row per annotated resource, in a table of its own,
# so the rows workers rewrite never carry it and the run tables keep their columns; deleting the
# resource deletes its annotation.
annotations = sa.Table(
    "annotations",
    metadata,
    sa.Column("id", sa.Uuid, primary_key=True),
    sa.Column("workspace_id", sa.Uuid, nullable=False),
    sa.Column("run_id", sa.Uuid, nullable=True),
    sa.Column("eval_run_id", sa.Uuid, nullable=True),
    sa.Column("experiment_id", sa.Uuid, nullable=True),
    sa.Column("title", sa.Text, nullable=True),
    sa.Column("tags", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("note", sa.Text, nullable=True),
    sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("updated_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.ForeignKeyConstraint(
        ["workspace_id", "run_id"], ["runs.workspace_id", "runs.id"], ondelete="CASCADE"
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "eval_run_id"],
        ["eval_runs.workspace_id", "eval_runs.id"],
        ondelete="CASCADE",
    ),
    sa.ForeignKeyConstraint(
        ["workspace_id", "experiment_id"],
        ["experiments.workspace_id", "experiments.id"],
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("run_id"),
    sa.UniqueConstraint("eval_run_id"),
    sa.UniqueConstraint("experiment_id"),
    sa.CheckConstraint("num_nonnulls(run_id, eval_run_id, experiment_id) = 1", name="one_target"),
    sa.CheckConstraint(
        "title IS NULL OR char_length(title) BETWEEN 1 AND 120", name="title_length"
    ),
    sa.CheckConstraint("note IS NULL OR char_length(note) <= 20000", name="note_length"),
    sa.CheckConstraint("jsonb_typeof(tags) = 'object'", name="tags_object"),
)

ALIAS_KINDS = ("trained_model", "knowledge_profile")

# An alias names one version of a trained model or a knowledge profile: one per name and
# alias, so setting it again moves it. The target is a plain ID, as models and profiles are
# withdrawn, never deleted, until their workspace's purge removes the aliases with them.
registry_aliases = sa.Table(
    "registry_aliases",
    metadata,
    sa.Column(
        "workspace_id",
        sa.Uuid,
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("kind", sa.Text, primary_key=True),
    sa.Column("name", sa.Text, primary_key=True),
    sa.Column("alias", sa.Text, primary_key=True),
    sa.Column("target_id", sa.Uuid, nullable=False),
    sa.Column("set_by", sa.Text, nullable=False),
    *_timestamps(),
    sa.CheckConstraint(_in("kind", ALIAS_KINDS), name="kind"),
    sa.CheckConstraint(r"alias ~ '^[a-z][a-z0-9_-]{0,31}$'", name="alias_shape"),
    sa.CheckConstraint(r"name ~ '^[a-z0-9][a-z0-9._-]{0,63}$'", name="name_shape"),
    sa.Index("ix_registry_aliases_target_id", "target_id"),
)
