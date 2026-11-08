"""Throwaway Postgres databases for integration tests.

Each test gets its own database, named ``t_<random>``, with the ``vector`` extension, created
through the admin DSN and dropped (``WITH (FORCE)``) when the test ends. As in
``stratafi-store bootstrap``, the database belongs to its own role of the same name, which isn't
a superuser, and the URL ``create`` returns connects as it: superusers skip row-level security,
so the tests run under it as the services do.
"""

import secrets
from urllib.parse import quote, urlsplit, urlunsplit

import psycopg
from psycopg import sql


def with_database(url: str, database: str) -> str:
    """``url`` pointing at another database."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{database}"))


def as_role(url: str, role: str, password: str) -> str:
    """``url`` signing in as another role."""
    parts = urlsplit(url)
    host = parts.hostname or "localhost"
    if ":" in host:
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit(parts._replace(netloc=f"{quote(role)}:{quote(password)}@{host}{port}"))


class DatabaseFactory:
    def __init__(self, admin_url: str) -> None:
        self.admin_url = admin_url
        self.created: list[str] = []

    def create(self, *, extensions: tuple[str, ...] = ("vector",)) -> str:
        """A new, empty database owned by a new role of its name; returns its URL as that role."""
        name = f"t_{secrets.token_hex(6)}"
        password = secrets.token_urlsafe(18)
        with psycopg.connect(self.admin_url, autocommit=True) as conn:
            conn.execute(
                sql.SQL("CREATE ROLE {} WITH LOGIN NOSUPERUSER PASSWORD {}").format(
                    sql.Identifier(name), sql.Literal(password)
                )
            )
            self.created.append(name)
            conn.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(name), sql.Identifier(name)
                )
            )
        admin = with_database(self.admin_url, name)
        with psycopg.connect(admin, autocommit=True) as conn:
            for extension in extensions:
                conn.execute(
                    sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(sql.Identifier(extension))
                )
        return as_role(admin, name, password)

    def admin_url_of(self, url: str) -> str:
        """The admin's URL for a database ``create`` made."""
        return with_database(self.admin_url, urlsplit(url).path.lstrip("/"))

    def drop_all(self) -> None:
        with psycopg.connect(self.admin_url, autocommit=True) as conn:
            for name in self.created:
                conn.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
                )
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(name)))
        self.created.clear()
