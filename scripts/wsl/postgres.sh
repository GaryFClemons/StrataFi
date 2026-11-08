#!/usr/bin/env bash
# PostgreSQL 16 with pgvector inside WSL (Ubuntu 22.04 or 24.04): StrataFi's database on Windows
# without Docker. From the repository, in a Windows terminal:
#
#   wsl -d Ubuntu-24.04 -- bash scripts/wsl/postgres.sh
#
# It asks for your Linux password once (sudo). If you don't know it, run it as the distribution's
# root user instead, which Windows lets you do without one:
#
#   wsl -d Ubuntu-24.04 -u root -- bash scripts/wsl/postgres.sh
#
# Then it:
# - installs PostgreSQL 16 and pgvector from the PostgreSQL project's apt repository
#   (apt.postgresql.org), which carries the current pgvector;
# - sets the cluster's port to STRATAFI_PG_PORT from .env (5442) and keeps statement text out of
#   its logs, as compose.yaml's Postgres does;
# - creates the administrator .env names (STRATAFI_PG_ADMIN_USER and _PASSWORD), which
#   `stratafi-store bootstrap` uses to create StrataFi's own role and database.
#
# Windows reaches the database on 127.0.0.1 through WSL's localhost forwarding. With systemd
# (Ubuntu 24.04's default in WSL) it starts whenever the distribution does. Safe to run again.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

cd "$(dirname "$0")/../.."
if [ ! -f .env ]; then
  echo "No .env here: run 'uv run python scripts/dev_setup.py' first." >&2
  exit 2
fi

env_value() { grep -E "^$1=" .env | tail -n 1 | cut -d= -f2- | tr -d '\r'; }
PORT=$(env_value STRATAFI_PG_PORT)
PORT=${PORT:-5442}
ADMIN=$(env_value STRATAFI_PG_ADMIN_USER)
ADMIN=${ADMIN:-stratafi_admin}
PASSWORD=$(env_value STRATAFI_PG_ADMIN_PASSWORD)
if [ -z "$PASSWORD" ]; then
  echo "STRATAFI_PG_ADMIN_PASSWORD is missing from .env: run scripts/dev_setup.py again." >&2
  exit 2
fi

echo "==> Installing PostgreSQL 16 and pgvector (needs sudo)"
sudo apt-get update -qq
sudo apt-get install -y -qq postgresql-common ca-certificates curl >/dev/null
if ! ls /etc/apt/sources.list.d/pgdg* >/dev/null 2>&1; then
  sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y >/dev/null
fi
sudo apt-get install -y -qq postgresql-16 postgresql-16-pgvector >/dev/null

echo "==> Configuring the cluster on port $PORT"
sudo pg_conftool 16 main set port "$PORT"
sudo pg_conftool 16 main set listen_addresses localhost
sudo pg_conftool 16 main set log_min_error_statement panic
sudo pg_conftool 16 main set log_error_verbosity terse
sudo systemctl enable postgresql@16-main >/dev/null 2>&1 || true
sudo systemctl restart postgresql@16-main

echo "==> Creating the administrator role $ADMIN"
cd /tmp # the postgres user can't read the Windows folder
sudo -u postgres psql -p "$PORT" -v ON_ERROR_STOP=1 -q \
  -v admin="$ADMIN" -v password="$PASSWORD" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN SUPERUSER', :'admin')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'admin') \gexec
SELECT format('ALTER ROLE %I LOGIN SUPERUSER PASSWORD %L', :'admin', :'password') \gexec
SQL

VECTOR=$(sudo -u postgres psql -p "$PORT" -tAc \
  "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'")
SERVER=$(sudo -u postgres psql -p "$PORT" -tAc "SHOW server_version")
echo "PostgreSQL $SERVER with pgvector $VECTOR is ready on 127.0.0.1:$PORT."
echo "Next, on Windows: uv run --env-file .env stratafi-store bootstrap"
