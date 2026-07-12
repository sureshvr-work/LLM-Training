"""
db.py — one connection pool to Postgres.

Each new connection:
  1. makes sure the `vector` extension exists (CREATE EXTENSION IF NOT EXISTS vector)
  2. registers the pgvector adapter so we can pass/read Python vectors directly
"""
import os
import time
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@db:5432/postgres",
)


def _configure(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)


# Wait for the database to accept connections (compose healthcheck usually handles
# this, but we retry to be safe on a cold start).
_pool = None


def pool():
    global _pool
    if _pool is None:
        last = None
        for _ in range(30):
            try:
                _pool = ConnectionPool(DSN, min_size=1, max_size=5,
                                       configure=_configure, open=True)
                # smoke test
                with _pool.connection() as c, c.cursor() as cur:
                    cur.execute("SELECT 1")
                break
            except Exception as e:  # noqa
                last = e
                _pool = None
                time.sleep(2)
        if _pool is None:
            raise RuntimeError(f"could not connect to Postgres: {last}")
    return _pool
