"""
Alembic environment for Hindsight.

Supports two dialects:

* PostgreSQL (sync psycopg2 driver) — default; uses ``search_path`` for
  multi-tenant schema isolation and forces read-write transactions to work
  around Supabase's read-only-by-default sessions.
* Oracle 23ai (``oracledb`` driver) — uses ``CURRENT_SCHEMA`` for tenant
  isolation; no equivalent of ``search_path`` or read-only session quirks.

Each migration file dispatches its DDL through ``alembic._dialect.run_for_dialect``
so a single revision tree serves both backends.
"""

import logging
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import Connection, engine_from_config, pool
from sqlalchemy.engine import Engine

from hindsight_api.db_url import is_oracle_url, to_libpq_url
from hindsight_api.models import Base


def load_env() -> None:
    """Load environment variables from .env (skipped if already configured)."""
    if os.getenv("HINDSIGHT_API_DATABASE_URL"):
        return

    root_dir = Path(__file__).parent.parent.parent
    env_file = root_dir / ".env"

    if env_file.exists():
        load_dotenv(env_file)


load_env()

config = context.config
target_metadata = Base.metadata


def _oracle_connect_params(dsn: str) -> dict[str, any]:
    """Turn the configured database URL into oracledb connect kwargs.

    Accepts ``oracle://user:pass@host:port/service`` and, for Autonomous
    Database / TCPS setups that need a full connect descriptor or TNS alias,
    ``oracle://user:pass@/?dsn=<descriptor-or-alias>``. Credentials are
    URL-decoded, so passwords containing ``#``, ``@`` or ``%`` work when
    percent-encoded in the URL. Anything that is not an ``oracle://`` URL is
    passed through as the dsn.
    """
    from urllib.parse import parse_qsl, unquote, urlparse

    parsed = urlparse(dsn)
    if parsed.scheme not in ("oracle", "oracle+oracledb"):
        return {"dsn": dsn}
    params: dict[str, any] = {
        "user": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password) if parsed.password else None,
    }
    descriptor = parse_qsl(parsed.query).get("dsn")
    if descriptor and descriptor[0]:
        params["dsn"] = descriptor[0]
    else:
        host = parsed.hostname or "localhost"
        port = parsed.port or 1521
        service = parsed.path.lstrip("/") if parsed.path else "FREEPDB1"
        params["dsn"] = f"{host}:{port}/{service}"
    return params


def _normalize_oracle_url(url: str) -> str:
    """Coerce an Oracle URL into the SQLAlchemy form the oracledb dialect expects.

    Three issues to handle:

    1. Force the ``oracle+oracledb`` driver — bare ``oracle://`` defaults to
       cx_Oracle.
    2. Map a path-style service to ``?service_name=...``. SQLAlchemy's oracledb
       dialect treats the URL path as a *SID* (legacy), but Oracle Free /
       Autonomous DB only register a service name. Without this rewrite we get
       ``DPY-6003: SID "FREEPDB1" is not registered`` even though the listener
       is happy to accept the same name as a service.
    3. Handle ``dsn`` query parameter with full Oracle descriptor — preserve it for oracledb thin mode.
    """
    parts = urlsplit(url)
    if not parts.scheme.startswith("oracle"):
        return url

    new_scheme = "oracle+oracledb" if parts.scheme == "oracle" else parts.scheme

    # Check if URL has a dsn query parameter with full descriptor
    query_params = parse_qsl(parts.query, keep_blank_values=True)
    dsn_param = next((v for k, v in query_params if k.lower() == "dsn"), None)

    if dsn_param:
        # URL has a full descriptor in dsn parameter - preserve it for oracledb thin mode
        # The descriptor already contains all connection details
        # Keep the dsn parameter but remove path
        new_query = parts.query  # Keep the dsn parameter intact
        new_path = ""
    else:
        # Standard case: promote /SERVICE to ?service_name=SERVICE
        service = parts.path.lstrip("/")
        new_query = parts.query
        new_path = parts.path
        if service and "service_name=" not in new_query and "sid=" not in new_query:
            params = [(k, v) for k, v in parse_qsl(new_query, keep_blank_values=True)]
            params.append(("service_name", service))
            new_query = urlencode(params)
            new_path = ""

    return urlunsplit((new_scheme, parts.netloc, new_path, new_query, parts.fragment))


def get_database_url() -> str:
    """Resolve the migration URL from Alembic config or env, normalizing per-dialect."""
    database_url = config.get_main_option("sqlalchemy.url")
    if not database_url:
        database_url = os.getenv("HINDSIGHT_API_DATABASE_URL")
        if not database_url:
            raise ValueError(
                "Database URL not found. "
                "Set HINDSIGHT_API_DATABASE_URL environment variable or pass database_url to run_migrations()."
            )

    if is_oracle_url(database_url):
        database_url = _normalize_oracle_url(database_url)
    else:
        # PG: convert SQLAlchemy-style asyncpg URLs and ?ssl= params to libpq form
        # for the sync engine used during migrations.
        database_url = to_libpq_url(database_url)

    # Alembic stores options through ConfigParser, where '%' is interpolation.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return database_url


def run_migrations_offline() -> None:
    logging.info("running offline")
    database_url = get_database_url()

    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def _configure_pg_session(engine: Engine, connection: Connection, target_schema: str | None) -> None:
    """PG-only: ensure the session is RW (Supabase) and bind ``search_path``."""
    from sqlalchemy import event, text

    @event.listens_for(engine, "connect")
    def set_read_write_mode(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE")
        if target_schema:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{target_schema}"')
            cursor.execute(f'SET search_path TO "{target_schema}", public')
        cursor.close()

    connection.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE"))
    if target_schema:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{target_schema}"'))
        connection.execute(text(f'SET search_path TO "{target_schema}", public'))
    connection.commit()


def _configure_oracle_session(connection: Connection, target_schema: str | None) -> None:
    """Oracle: switch the session's default schema; tolerate DDL contention."""
    from sqlalchemy import text

    # Wait up to 30s for DDL locks instead of failing immediately (ORA-00054).
    connection.execute(text("ALTER SESSION SET DDL_LOCK_TIMEOUT = 30"))
    if target_schema:
        connection.execute(text(f'ALTER SESSION SET CURRENT_SCHEMA = "{target_schema}"'))


def run_migrations_online() -> None:
    database_url = get_database_url()
    target_schema = config.get_main_option("target_schema")
    is_oracle = is_oracle_url(database_url)

    # For Oracle, use custom creator to properly handle dsn parameter
    if is_oracle:
        from hindsight_api.engine.db.oracle import _oracle_connect_params

        connect_params = _oracle_connect_params(database_url)
        oracledb = __import__("oracledb")
        oracledb.defaults.fetch_lobs = False

        def creator():
            return oracledb.connect(**connect_params)

        from sqlalchemy import create_engine
        connectable = create_engine(
            "oracle+oracledb://",
            creator=creator,
            poolclass=pool.NullPool,
        )
    else:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        if is_oracle:
            _configure_oracle_session(connection, target_schema)
        else:
            _configure_pg_session(connectable, connection, target_schema)

        context_opts = {
            "connection": connection,
            "target_metadata": target_metadata,
        }
        if target_schema and not is_oracle:
            # Oracle has no equivalent of PG's per-schema version table; the
            # ``alembic_version`` table lives in CURRENT_SCHEMA implicitly.
            context_opts["version_table_schema"] = target_schema

        context.configure(**context_opts)

        with context.begin_transaction():
            context.run_migrations()

        # Always commit. PG needs it for the explicit RW-mode SET to persist;
        # Oracle needs it because each DDL auto-commits but the trailing
        # ``UPDATE alembic_version`` is plain DML that would otherwise stay in
        # an open transaction and roll back when the connection closes —
        # producing the "schema is created but the version row is one revision
        # behind" failure mode.
        connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()