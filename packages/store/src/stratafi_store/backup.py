"""Backup and restore: Postgres and the object store together.

    stratafi-store backup --out /backups/2026-11-04
    stratafi-store restore --from /backups/2026-11-04     # into an empty database and store
    stratafi-store verify --from /backups/2026-11-04      # checksums only, touches nothing

A backup is a directory:

- ``database.dump``: ``pg_dump --format=custom`` of the ``stratafi`` and ``stratafi_index``
  schemas, one consistent snapshot. It runs as the services' role with row-level security on
  and the ``system`` scope, so every workspace's rows are in it
- ``objects/<key>``: every object under ``ws/``, copied after the dump, each with its SHA-256 in
  ``objects.jsonl``
- ``manifest.json``: ``stratafi.backup.v1``, the migration revision, the dump's SHA-256, and
  the counts

**Consistency.** Objects are content-addressed and written before the rows that name them, so
a row in the dump always has its object in the store; copying objects after the dump only adds
some no row names (``purge.orphans`` finds them). What could break it is a purge deleting an
object between the dump and its copy, so a backup holds an exclusive advisory lock for its
whole run: ``purge.remove`` takes it shared before it deletes objects, and the retention sweep
skips while a backup runs.

**Restore** checks every checksum first, refuses a database that already has Stratafi's schema,
restores the dump with ``pg_restore``, migrates it to this release's head (the upgrade from
the backup's release), and writes the objects back. ``pg_dump`` and ``pg_restore`` must be on
the path, at the server's major version or later (16): the host's, or the Postgres container's.
Passwords reach them through the environment, never the command line. Output is counts, never
names or text.
"""

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from sqlalchemy.engine import Engine

from stratafi_store.artifacts import ArtifactStore, check_key, workspace_prefix
from stratafi_store.db import libpq_url
from stratafi_store.migrations import current_revision, head_revision, upgrade
from stratafi_store.purge import BACKUP_LOCK
from stratafi_store.retention import SWEEP_LOCK
from stratafi_store.scope import system_scope
from stratafi_store.tables import SCHEMA, workspaces

FORMAT = "stratafi.backup.v1"
INDEX_SCHEMA = "stratafi_index"
CHUNK = 1024 * 1024


class BackupError(Exception):
    """A backup or restore failed. ``code`` is short and holds no secret."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class Manifest:
    format: str
    created_at: str
    revision: str | None
    database: dict[str, Any]
    objects: dict[str, int]
    workspaces: int
    rows: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RestoreReport:
    revision_from: str | None
    revision_to: str
    objects: int
    rows: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _client(engine: Engine) -> tuple[str, dict[str, str]]:
    """A libpq connection string without its password, and the environment that carries it."""
    params = conninfo_to_dict(libpq_url(engine))
    password = params.pop("password", None)
    environment = {**os.environ}
    if password:
        environment["PGPASSWORD"] = str(password)
    return make_conninfo(**{k: str(v) for k, v in params.items()}), environment


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise BackupError(f"{name.replace('_', '')}_missing")
    return found


def row_counts(engine: Engine) -> dict[str, int]:
    """Rows in each of Stratafi's tables, every workspace's."""
    counts: dict[str, int] = {}
    with system_scope(), engine.connect() as conn:
        tables = conn.execute(
            sa.text(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_schema IN (:a, :b) AND table_type = 'BASE TABLE' "
                "ORDER BY table_schema, table_name"
            ),
            {"a": SCHEMA, "b": INDEX_SCHEMA},
        ).all()
        for schema, table in tables:
            counts[f"{schema}.{table}"] = conn.execute(
                sa.text(f'SELECT count(*) FROM "{schema}"."{table}"')  # noqa: S608 - catalog names
            ).scalar_one()
    return counts


@contextmanager
def _exclusive(engine: Engine) -> Iterator[None]:
    """Hold the retention sweep's lock, then the backup lock, for the block. In that order: a
    sweep holds its own lock when it takes the backup lock shared, so the reverse would wait in
    a circle."""
    with system_scope(), engine.connect() as conn:
        conn.execute(sa.text("SELECT pg_advisory_lock(:key)"), {"key": SWEEP_LOCK})
        conn.execute(sa.text("SELECT pg_advisory_lock(:key)"), {"key": BACKUP_LOCK})
        conn.commit()
        try:
            yield
        finally:
            conn.execute(sa.text("SELECT pg_advisory_unlock(:key)"), {"key": BACKUP_LOCK})
            conn.execute(sa.text("SELECT pg_advisory_unlock(:key)"), {"key": SWEEP_LOCK})
            conn.commit()


def _workspace_ids(engine: Engine) -> list[uuid.UUID]:
    with system_scope(), engine.connect() as conn:
        return list(conn.execute(sa.select(workspaces.c.id).order_by(workspaces.c.id)).scalars())


def backup(engine: Engine, store: ArtifactStore, out: Path) -> Manifest:
    """Write a consistent backup of the database and the object store to ``out``, which must
    not exist or be empty."""
    if out.exists() and any(out.iterdir()):
        raise BackupError("backup_directory_not_empty")
    out.mkdir(parents=True, exist_ok=True)
    pg_dump = _tool("pg_dump")
    connection, environment = _client(engine)
    # Every workspace's rows: row-level security stays on, with the system scope.
    environment["PGOPTIONS"] = "-c stratafi.scope=system"
    dump = out / "database.dump"
    with _exclusive(engine):
        revision = current_revision(engine)
        rows = row_counts(engine)
        completed = subprocess.run(  # noqa: S603 - fixed arguments; the password is in the env
            [
                pg_dump,
                "--format=custom",
                "--no-owner",
                "--no-privileges",
                "--enable-row-security",
                f"--schema={SCHEMA}",
                f"--schema={INDEX_SCHEMA}",
                f"--file={dump}",
                f"--dbname={connection}",
            ],
            env=environment,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise BackupError("pg_dump_failed")
        objects = out / "objects"
        count = total = 0
        with (out / "objects.jsonl").open("w", encoding="utf-8") as listing:
            for workspace_id in _workspace_ids(engine):
                for key in store.keys(workspace_prefix(workspace_id)):
                    target = objects / check_key(key)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    size = 0
                    with store.open(key) as source, target.open("wb") as sink:
                        while block := source.read(CHUNK):
                            digest.update(block)
                            sink.write(block)
                            size += len(block)
                    listing.write(
                        json.dumps({"key": key, "sha256": digest.hexdigest(), "bytes": size}) + "\n"
                    )
                    count += 1
                    total += size
    manifest = Manifest(
        format=FORMAT,
        created_at=datetime.now(UTC).isoformat(),
        revision=revision,
        database={"file": dump.name, "sha256": _sha256(dump), "bytes": dump.stat().st_size},
        objects={"count": count, "bytes": total},
        workspaces=len(_workspace_ids(engine)),
        rows=rows,
    )
    (out / "manifest.json").write_text(json.dumps(manifest.as_dict(), indent=2), encoding="utf-8")
    return manifest


def read_manifest(source: Path) -> Manifest:
    try:
        data = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise BackupError("manifest_unreadable") from None
    if data.get("format") != FORMAT:
        raise BackupError("unknown_backup_format")
    return Manifest(**data)


def _listing(source: Path) -> Iterator[dict[str, Any]]:
    with (source / "objects.jsonl").open(encoding="utf-8") as listing:
        for line in listing:
            if line.strip():
                yield json.loads(line)


def verify(source: Path) -> list[str]:
    """What's wrong with the backup at ``source``: a missing file or a checksum that differs.
    Codes and keys only."""
    manifest = read_manifest(source)
    problems = []
    dump = source / str(manifest.database["file"])
    if not dump.is_file() or _sha256(dump) != manifest.database["sha256"]:
        problems.append("database_dump_damaged")
    count = 0
    for entry in _listing(source):
        count += 1
        path = source / "objects" / check_key(entry["key"])
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            problems.append(f"object_damaged:{entry['key']}")
    if count != manifest.objects["count"]:
        problems.append("object_count_differs")
    return problems


def restore(engine: Engine, store: ArtifactStore, source: Path) -> RestoreReport:
    """Restore the backup at ``source`` into ``engine``'s empty database and ``store``, then
    migrate the database to this release's head."""
    manifest = read_manifest(source)
    if problems := verify(source):
        raise BackupError(problems[0].split(":", 1)[0])
    with engine.connect() as conn:
        present = conn.execute(
            sa.text("SELECT count(*) FROM information_schema.schemata WHERE schema_name = :s"),
            {"s": SCHEMA},
        ).scalar_one()
    if present:
        raise BackupError("database_not_empty")
    pg_restore = _tool("pg_restore")
    connection, environment = _client(engine)
    environment["PGOPTIONS"] = "-c stratafi.scope=system"
    completed = subprocess.run(  # noqa: S603 - fixed arguments; the password is in the env
        [
            pg_restore,
            "--no-owner",
            "--no-privileges",
            "--exit-on-error",
            "--single-transaction",
            f"--dbname={connection}",
            str(source / str(manifest.database["file"])),
        ],
        env=environment,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BackupError("pg_restore_failed")
    # The backup's release may be older: its schema moves forward like any installation's.
    upgrade(engine)
    count = 0
    for entry in _listing(source):
        path = source / "objects" / check_key(entry["key"])
        with path.open("rb") as handle:
            store.put(entry["key"], handle)
        count += 1
    return RestoreReport(manifest.revision, head_revision(), count, row_counts(engine))
