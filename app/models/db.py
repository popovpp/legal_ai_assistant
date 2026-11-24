# app/models/db.py
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    mime TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    num_pages INTEGER NOT NULL DEFAULT 0,
    num_chunks INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    page INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    query TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_analyses_document ON analyses(document_id);
"""


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    settings.ensure_dirs()
    with get_conn() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(documents)")]
        if "language" not in cols:
            conn.execute("ALTER TABLE documents ADD COLUMN language TEXT NOT NULL DEFAULT 'en'")
