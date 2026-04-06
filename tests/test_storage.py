"""Tests for the storage module."""

import pytest
import os
import shutil
import tempfile
import sqlite3

from turbomemory.storage import (
    SQLitePool,
    StorageManager,
    RetryConfig,
    MigrationManager,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def db_path(temp_dir):
    return os.path.join(temp_dir, "test.db")


class TestSQLitePool:
    def test_pool_creation(self, db_path):
        pool = SQLitePool(db_path, pool_size=3)
        conn = pool.get_connection()
        assert conn is not None
        pool.close_all()

    def test_wal_mode(self, db_path):
        pool = SQLitePool(db_path)
        conn = pool.get_connection()
        result = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert result == "wal"
        pool.close_all()

    def test_thread_local(self, db_path):
        pool = SQLitePool(db_path, pool_size=2)
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        assert conn1 is conn2
        pool.close_all()


class TestStorageManager:
    def test_creation(self, db_path):
        sm = StorageManager(db_path)
        assert os.path.exists(db_path)
        sm.close()

    def test_migrations_applied(self, db_path):
        sm = StorageManager(db_path)
        with sm.get_conn() as conn:
            version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            assert version >= 1
        sm.close()

    def test_execute_with_retry(self, db_path):
        sm = StorageManager(db_path)
        result = sm.execute_with_retry("SELECT 1")
        assert result.fetchone()[0] == 1
        sm.close()


class TestRetryConfig:
    def test_default_values(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 0.1
        assert config.max_delay == 2.0
        assert config.backoff_factor == 2.0

    def test_custom_values(self):
        config = RetryConfig(max_retries=5, initial_delay=0.5)
        assert config.max_retries == 5
        assert config.initial_delay == 0.5