"""``stratafi-store``: bootstrap the database, run migrations, purge, and back up.

    uv run --env-file .env stratafi-store bootstrap   # role, database and pgvector (admin DSN)
    uv run --env-file .env stratafi-store migrate     # apply every migration
    uv run --env-file .env stratafi-store current     # the applied and the head revision
    uv run stratafi-store revision -m "add runs"      # start a new migration (development)

Deletion, with the settings' storage:

    stratafi-store purge-document --workspace <id> --document <id>
    stratafi-store purge-workspace --workspace <id> --confirm <slug>
    stratafi-store retention                          # one retention sweep now

Backup and restore, with the settings' storage; ``pg_dump`` and ``pg_restore`` 16 on the
path:

    stratafi-store backup --out <directory>
    stratafi-store verify --from <directory>
    stratafi-store restore --from <directory>         # an empty database and store

Commands read ``STRATAFI_DATABASE_URL``, and ``bootstrap`` also ``STRATAFI_PG_ADMIN_DSN``. Output
names databases, revisions, IDs and counts, never URLs, passwords or text.
"""

import argparse
import os
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

from stratafi_store.bootstrap import BootstrapError, bootstrap, target_from_url
from stratafi_store.db import create_engine
from stratafi_store.migrations import current_revision, head_revision, new_revision, upgrade

DATABASE_URL_ENV = "STRATAFI_DATABASE_URL"
ADMIN_DSN_ENV = "STRATAFI_PG_ADMIN_DSN"


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"stratafi-store: {name} is not set (run with `uv run --env-file .env`)")
    return value


def _store():
    from stratafi_core.settings import Settings
    from stratafi_store.artifacts import artifact_store_from_settings

    return artifact_store_from_settings(Settings().storage)


def _purge_document(engine, args: argparse.Namespace) -> int:
    from stratafi_store import purge
    from stratafi_store.audit import record_audit_event
    from stratafi_store.scope import workspace_scope

    store = _store()
    with workspace_scope(args.workspace), engine.begin() as conn:
        plan = purge.plan_purge(conn, store, args.workspace, args.document, reason="deleted")
        if plan is None:
            print("stratafi-store: no such document in that workspace", file=sys.stderr)
            return 1
        record_audit_event(
            conn,
            workspace_id=args.workspace,
            actor_kind="system",
            actor_subject="system:stratafi-store",
            action="document.purge",
            resource_type="document",
            resource_id=str(args.document),
            details={"reason": "deleted", **plan.counts()},
        )
    objects = purge.remove(engine, store, plan)
    counts = ", ".join(f"{name} {count}" for name, count in plan.counts().items())
    print(f"document {args.document}: purged ({counts}, objects {objects})")
    return 0


def _purge_workspace(engine, args: argparse.Namespace) -> int:
    import sqlalchemy as sa

    from stratafi_store import purge
    from stratafi_store.scope import workspace_scope
    from stratafi_store.tables import workspaces

    store = _store()
    with workspace_scope(args.workspace), engine.begin() as conn:
        slug = conn.execute(
            sa.select(workspaces.c.slug).where(workspaces.c.id == args.workspace)
        ).scalar_one_or_none()
        if slug is None:
            print("stratafi-store: no such workspace", file=sys.stderr)
            return 1
        if slug != args.confirm:
            print("stratafi-store: --confirm must be the workspace's slug", file=sys.stderr)
            return 2
        plan = purge.plan_workspace_purge(conn, store, args.workspace)
    assert plan is not None  # noqa: S101 - it exists: the slug was read in this transaction
    objects = purge.remove(engine, store, plan)
    print(f"workspace {args.workspace}: purged (rows {plan.rows}, objects {objects})")
    return 0


def _retention(engine, args: argparse.Namespace) -> int:
    from stratafi_core.settings import Settings
    from stratafi_store.retention import sweep

    report = sweep(engine, _store(), Settings().retention)
    print(", ".join(f"{name} {count}" for name, count in report.as_dict().items()))
    return 0


def _backup(engine, args: argparse.Namespace) -> int:
    from stratafi_store.backup import BackupError, backup

    try:
        manifest = backup(engine, _store(), args.out)
    except BackupError as error:
        print(f"stratafi-store: backup failed: {error.code}", file=sys.stderr)
        return 1
    print(
        f"backed up revision {manifest.revision}: {sum(manifest.rows.values())} rows, "
        f"{manifest.objects['count']} objects ({manifest.objects['bytes']} bytes)"
    )
    return 0


def _verify(engine, args: argparse.Namespace) -> int:
    from stratafi_store.backup import BackupError, verify

    try:
        problems = verify(args.source)
    except BackupError as error:
        print(f"stratafi-store: {error.code}", file=sys.stderr)
        return 1
    for problem in problems:
        print(problem)
    print("backup intact" if not problems else f"{len(problems)} problems")
    return 0 if not problems else 1


def _restore(engine, args: argparse.Namespace) -> int:
    from stratafi_store.backup import BackupError, restore

    try:
        report = restore(engine, _store(), args.source)
    except BackupError as error:
        print(f"stratafi-store: restore failed: {error.code}", file=sys.stderr)
        return 1
    print(
        f"restored revision {report.revision_from}, migrated to {report.revision_to}: "
        f"{sum(report.rows.values())} rows, {report.objects} objects"
    )
    return 0


COMMANDS: dict[str, Callable[..., int]] = {
    "purge-document": _purge_document,
    "purge-workspace": _purge_workspace,
    "retention": _retention,
    "backup": _backup,
    "verify": _verify,
    "restore": _restore,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stratafi-store", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap", help="create the role, the database and pgvector")
    commands.add_parser("migrate", help="apply every migration")
    commands.add_parser("current", help="show the applied and the head revision")
    revision = commands.add_parser("revision", help="start a new migration")
    revision.add_argument("-m", "--message", required=True)
    document = commands.add_parser("purge-document", help="purge a document by lineage")
    document.add_argument("--workspace", type=uuid.UUID, required=True)
    document.add_argument("--document", type=uuid.UUID, required=True)
    workspace = commands.add_parser("purge-workspace", help="delete a workspace and its data")
    workspace.add_argument("--workspace", type=uuid.UUID, required=True)
    workspace.add_argument("--confirm", required=True, help="the workspace's slug")
    commands.add_parser("retention", help="run one retention sweep")
    made = commands.add_parser("backup", help="back up the database and the object store")
    made.add_argument("--out", type=Path, required=True)
    for name, text in (("verify", "check a backup's checksums"), ("restore", "restore a backup")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--from", dest="source", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "revision":
        next_id = f"{int(head_revision()) + 1:04d}"
        print(f"wrote {new_revision(args.message, next_id).name}")
        return 0
    url = _required(DATABASE_URL_ENV)
    if args.command == "bootstrap":
        try:
            database, role, password = target_from_url(url)
            result = bootstrap(_required(ADMIN_DSN_ENV), database, role, password=password)
        except BootstrapError as exc:
            print(f"stratafi-store: {exc}", file=sys.stderr)
            return 2
        print(
            f"database {result.database}: {'created' if result.created_database else 'kept'}; "
            f"role {result.role}: {'created' if result.created_role else 'kept'}"
        )
        return 0
    engine = create_engine(url, pool_size=1, max_overflow=1)
    try:
        if args.command in COMMANDS:
            return COMMANDS[args.command](engine, args)
        if args.command == "migrate":
            upgrade(engine)
        print(f"applied {current_revision(engine) or 'none'}; head {head_revision()}")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
