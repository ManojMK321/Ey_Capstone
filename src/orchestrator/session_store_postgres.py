"""
session_store_postgres.py
-------------------------
PostgreSQL-backed session management.

"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Generator, Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

from src.observability import metrics

load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

logger = logging.getLogger(__name__)

MAX_HISTORY_ITEMS = 20


def _build_dsn() -> str:
    url = os.environ.get("DATABASE_URL")
    if url and url.startswith(("postgresql://", "postgres://")):
        return url.replace("postgres://", "postgresql://", 1)
    dsn = (
        "host={host} port={port} dbname={db} user={user} password={password}"
    ).format(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ.get("POSTGRES_PORT", "5432"),
        db=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    sslmode = os.environ.get("POSTGRES_SSLMODE")
    if sslmode:
        dsn += f" sslmode={sslmode}"
    return dsn


_pool: Optional[ThreadedConnectionPool] = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        try:
            dsn = _build_dsn()
        except KeyError as exc:
            raise RuntimeError(
                f"PostgreSQL credentials missing from .env — add DATABASE_URL or "
                f"POSTGRES_HOST / POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD. "
                f"Missing key: {exc}"
            ) from exc
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=dsn)
        logger.info("Session store: PostgreSQL connected.")
    return _pool


@contextmanager
def _get_conn() -> Generator:
    pool = _get_pool()
    conn = pool.getconn()

    # A pooled connection can sit idle for hours between requests. Azure (or
    # an intermediate proxy) closes it server-side after its own idle
    # timeout, but the pool has no way to know that until something tries to
    # use it — ping first so a stale connection is discarded and replaced
    # here instead of failing deep inside a real query.
    try:
        with conn.cursor() as ping:
            ping.execute("SELECT 1")
        conn.rollback()  # clear the ping's implicit transaction before real use
    except psycopg2.Error:
        pool.putconn(conn, close=True)
        conn = pool.getconn()

    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        # Connection died mid-use (e.g. closed between the ping and the real
        # query) — discard it so the pool doesn't hand the same dead
        # connection to the next caller.
        pool.putconn(conn, close=True)
        raise
    else:
        pool.putconn(conn)


def _init_db() -> None:
    with _get_conn() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_active  TIMESTAMPTZ NOT NULL DEFAULT now(),
                turn_count   INTEGER     NOT NULL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id          BIGSERIAL   PRIMARY KEY,
                session_id  TEXT        NOT NULL
                                REFERENCES sessions(session_id) ON DELETE CASCADE,
                role        TEXT        NOT NULL,
                text        TEXT        NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_history_session ON history(session_id, id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                file_id       TEXT        PRIMARY KEY,
                session_id    TEXT        NOT NULL
                                  REFERENCES sessions(session_id) ON DELETE CASCADE,
                original_name TEXT        NOT NULL,
                size_bytes    BIGINT      NOT NULL,
                status        TEXT        NOT NULL DEFAULT 'indexed',
                uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id)")


_db_initialised = False


def _ensure_db() -> None:
    global _db_initialised
    if not _db_initialised:
        _init_db()
        _db_initialised = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(result) -> list[dict]:
    if result is None:
        return []
    return [dict(r) for r in result]


class SessionStore:

    def make_session(self, session_id: Optional[str] = None, reset: bool = False) -> str:
        _ensure_db()
        if not reset and session_id and self.session_exists(session_id):
            return session_id
        new_id = str(uuid.uuid4())
        with metrics.timed_db_query("postgres", "make_session"):
            with _get_conn() as cur:
                cur.execute(
                    "INSERT INTO sessions (session_id, created_at, last_active, turn_count) VALUES (%s, now(), now(), 0)",
                    (new_id,),
                )
        metrics.ACTIVE_SESSIONS.inc()
        return new_id

    def session_exists(self, session_id: str) -> bool:
        with metrics.timed_db_query("postgres", "session_exists"):
            with _get_conn() as cur:
                cur.execute("SELECT 1 FROM sessions WHERE session_id = %s", (session_id,))
                return cur.fetchone() is not None

    def append_history(self, session_id: str, role: str, text: str) -> None:
        with metrics.timed_db_query("postgres", "append_history"):
            with _get_conn() as cur:
                cur.execute(
                    "INSERT INTO history (session_id, role, text, created_at) VALUES (%s, %s, %s, now())",
                    (session_id, role, text),
                )
                cur.execute(
                    "DELETE FROM history WHERE session_id = %s AND id NOT IN "
                    "(SELECT id FROM history WHERE session_id = %s ORDER BY id DESC LIMIT %s)",
                    (session_id, session_id, MAX_HISTORY_ITEMS),
                )
                cur.execute(
                    "UPDATE sessions SET last_active = now(), "
                    "turn_count = (SELECT COUNT(*)/2 FROM history WHERE session_id = %s) "
                    "WHERE session_id = %s",
                    (session_id, session_id),
                )

    def get_history(self, session_id: str) -> list[dict]:
        with metrics.timed_db_query("postgres", "get_history"):
            with _get_conn() as cur:
                cur.execute(
                    "SELECT role, text, created_at AS timestamp FROM history "
                    "WHERE session_id = %s ORDER BY id ASC",
                    (session_id,),
                )
                return _rows(cur.fetchall())

    def history_context(self, session_id: str) -> Optional[str]:
        history = self.get_history(session_id)
        if not history:
            return None
        return "\n".join(f"{h['role'].capitalize()}: {h['text']}" for h in history)

    def add_document(self, session_id: str, file_id: str, original_name: str, size_bytes: int) -> None:
        with metrics.timed_db_query("postgres", "add_document"):
            with _get_conn() as cur:
                cur.execute(
                    "INSERT INTO documents (file_id, session_id, original_name, size_bytes, status, uploaded_at) "
                    "VALUES (%s, %s, %s, %s, 'indexed', now()) ON CONFLICT (file_id) DO NOTHING",
                    (file_id, session_id, original_name, size_bytes),
                )

    def list_documents(self, session_id: str) -> list[dict]:
        with metrics.timed_db_query("postgres", "list_documents"):
            with _get_conn() as cur:
                cur.execute(
                    "SELECT file_id, original_name, size_bytes, status, uploaded_at "
                    "FROM documents WHERE session_id = %s ORDER BY uploaded_at DESC",
                    (session_id,),
                )
                return _rows(cur.fetchall())

    def has_documents(self, session_id: str) -> bool:
        with metrics.timed_db_query("postgres", "has_documents"):
            with _get_conn() as cur:
                cur.execute("SELECT 1 FROM documents WHERE session_id = %s LIMIT 1", (session_id,))
                return cur.fetchone() is not None

    def delete_document(self, session_id: str, file_id: str) -> bool:
        with metrics.timed_db_query("postgres", "delete_document"):
            with _get_conn() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE session_id = %s AND file_id = %s",
                    (session_id, file_id),
                )
                return cur.rowcount > 0

    def get_session_info(self, session_id: str) -> Optional[dict]:
        with metrics.timed_db_query("postgres", "get_session_info"):
            with _get_conn() as cur:
                cur.execute(
                    "SELECT session_id, created_at, last_active, turn_count "
                    "FROM sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def list_sessions(self) -> list[dict]:
        with metrics.timed_db_query("postgres", "list_sessions"):
            with _get_conn() as cur:
                cur.execute(
                    "SELECT session_id, created_at, last_active, turn_count "
                    "FROM sessions ORDER BY last_active DESC"
                )
                return _rows(cur.fetchall())

    def delete_session(self, session_id: str) -> None:
        with metrics.timed_db_query("postgres", "delete_session"):
            with _get_conn() as cur:
                cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        metrics.ACTIVE_SESSIONS.dec()

    def cleanup_inactive(self, max_age_minutes: int = 60) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
        with metrics.timed_db_query("postgres", "cleanup_inactive"):
            with _get_conn() as cur:
                cur.execute("DELETE FROM sessions WHERE last_active < %s", (cutoff,))
                removed = cur.rowcount
        if removed:
            metrics.ACTIVE_SESSIONS.dec(removed)
        return removed


session_store = SessionStore()
