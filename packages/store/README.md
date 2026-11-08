# stratafi-store

Persistence for StrataFi: SQLAlchemy tables, repositories, the baseline Alembic migration, the
artifact store (files on disk or any S3 API), the Postgres job queue, row-level security's
scope, purge by lineage, retention and backups.

The `stratafi-store` command bootstraps the database and its role, runs migrations, purges
documents and workspaces, and backs up and restores
([local development](../../docs/guides/local-development.md#the-database)).
[The architecture](../../docs/architecture.md#the-job-queue) describes the queue and
row-level security.
