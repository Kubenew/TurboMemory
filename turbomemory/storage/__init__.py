"""Storage module with retry logic and connection pooling."""

import os
import sqlite3
import threading
import time
import logging
from contextlib import contextmanager
from typing import Iterator, List, Optional, Any, Dict
from pathlib import Path
from dataclasses import dataclass
import base64
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 0.1
    max_delay: float = 2.0
    backoff_factor: float = 2.0


def with_retry(func):
    """Decorator to add retry logic to database operations."""
    def wrapper(*args, **kwargs):
        config = RetryConfig()
        delay = config.initial_delay
        last_exception = None
        
        for attempt in range(config.max_retries):
            try:
                return func(*args, **kwargs)
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                last_exception = e
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    logger.warning(f"Database locked, retry {attempt + 1}/{config.max_retries}")
                    time.sleep(delay)
                    delay = min(delay * config.backoff_factor, config.max_delay)
                else:
                    raise
        raise last_exception
    return wrapper


class SQLitePool:
    """Thread-safe SQLite connection pool with WAL mode."""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: List[sqlite3.Connection] = []

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            with self._lock:
                if len(self._connections) < self.pool_size:
                    conn = self._create_connection()
                    self._connections.append(conn)
                    self._local.conn = conn
                else:
                    conn = self._connections.pop(0)
                    self._connections.append(conn)
                    self._local.conn = conn
        return self._local.conn

    def close_all(self) -> None:
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
        self._local.conn = None


class MigrationManager:
    """Database migration system for schema versioning."""

    MIGRATIONS_TABLE = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        )
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def ensure_migrations_table(self) -> None:
        self.conn.execute(self.MIGRATIONS_TABLE)

    def get_current_version(self) -> int:
        try:
            result = self.conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()[0]
            return result or 0
        except sqlite3.OperationalError:
            return 0

    def apply_migration(self, version: int, description: str, sql: str) -> None:
        current = self.get_current_version()
        if version <= current:
            logger.info(f"Migration {version} already applied")
            return
        
        logger.info(f"Applying migration {version}: {description}")
        self.conn.executescript(sql)
        self.conn.execute(
            "INSERT INTO schema_migrations(version, applied_at, description) VALUES (?, ?, ?)",
            (version, self._get_timestamp(), description)
        )

    def _get_timestamp(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


CURRENT_SCHEMA_VERSION = 2

SCHEMA_V1 = """
-- Initial schema
CREATE TABLE IF NOT EXISTS topics (
    topic TEXT PRIMARY KEY,
    file_ref TEXT NOT NULL,
    updated TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    centroid_bits INTEGER NOT NULL,
    centroid_scale REAL NOT NULL,
    centroid_qmax INTEGER NOT NULL,
    centroid_blob BLOB NOT NULL,
    centroid_shape TEXT NOT NULL DEFAULT '[384]'
);

CREATE TABLE IF NOT EXISTS chunks (
    topic TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    staleness REAL NOT NULL,
    entropy REAL NOT NULL,
    ts TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    text_preview TEXT NOT NULL,
    emb_bits INTEGER NOT NULL,
    emb_scale REAL NOT NULL,
    emb_qmax INTEGER NOT NULL,
    emb_blob BLOB NOT NULL,
    emb_shape TEXT NOT NULL DEFAULT '[384]',
    ttl_ts TEXT,
    quality_score REAL DEFAULT 0.5,
    verified INTEGER DEFAULT 0,
    verification_score REAL DEFAULT 0.0,
    cross_refs INTEGER DEFAULT 0,
    PRIMARY KEY(topic, chunk_id)
);

CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    topic TEXT,
    action TEXT NOT NULL,
    details TEXT,
    chunks_affected INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS exclusion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    topic TEXT,
    reason TEXT NOT NULL,
    text_preview TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic);
CREATE INDEX IF NOT EXISTS idx_chunks_confidence ON chunks(confidence);
CREATE INDEX IF NOT EXISTS idx_chunks_ttl ON chunks(ttl_ts);
CREATE INDEX IF NOT EXISTS idx_chunks_quality ON chunks(quality_score);
CREATE INDEX IF NOT EXISTS idx_consolidation_ts ON consolidation_log(ts);
"""

SCHEMA_V2 = """
-- Add audit log table for sensitive operations
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    operation TEXT NOT NULL,
    user_id TEXT,
    topic TEXT,
    details TEXT,
    success INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_log(operation);

-- Add source tracking for chunks
ALTER TABLE chunks ADD COLUMN source_ref TEXT;
"""

MIGRATIONS = [
    (1, "Initial schema", SCHEMA_V1),
    (2, "Add audit log and source tracking", SCHEMA_V2),
]


def get_current_schema() -> int:
    return CURRENT_SCHEMA_VERSION


class StorageManager:
    """High-level storage manager with migrations and retry logic."""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self._pool = SQLitePool(db_path, pool_size)
        self._init_db()

    @contextmanager
    def get_conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._pool.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise

    def _init_db(self) -> None:
        with self.get_conn() as conn:
            mm = MigrationManager(conn)
            mm.ensure_migrations_table()
            
            current = mm.get_current_version()
            for version, description, sql in MIGRATIONS:
                if version > current:
                    mm.apply_migration(version, description, sql)

    def execute_with_retry(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL with retry logic."""
        config = RetryConfig()
        delay = config.initial_delay
        last_exception = None
        
        for attempt in range(config.max_retries):
            try:
                with self.get_conn() as conn:
                    return conn.execute(sql, params)
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                last_exception = e
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    logger.warning(f"Database locked, retry {attempt + 1}/{config.max_retries}")
                    time.sleep(delay)
                    delay = min(delay * config.backoff_factor, config.max_delay)
                else:
                    raise
        raise last_exception

    def close(self) -> None:
        self._pool.close_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False