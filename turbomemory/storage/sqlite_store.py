"""TurboMemory v3 SQLite storage with tiered schema."""

import os
import sqlite3
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-64000;
PRAGMA mmap_size=268435456;

CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid            TEXT UNIQUE NOT NULL,
    text            TEXT NOT NULL,
    topic_id         INTEGER,
    source           TEXT,
    agent_id         TEXT,
    created_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL,
    confidence       REAL DEFAULT 0.5,
    importance       REAL DEFAULT 0.5,
    ttl_seconds      INTEGER,
    access_count     INTEGER DEFAULT 0,
    last_accessed    INTEGER,
    hash             TEXT,
    extra_json       TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic_id);
CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(hash);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id       INTEGER PRIMARY KEY,
    dim             INTEGER NOT NULL,
    dtype           TEXT NOT NULL,
    blob            BLOB NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tags (
    memory_id       INTEGER NOT NULL,
    tag             TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag),
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_tags_mem ON tags(memory_id);

CREATE TABLE IF NOT EXISTS topics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT UNIQUE NOT NULL,
    centroid_dim    INTEGER,
    centroid_dtype TEXT,
    centroid_blob   BLOB,
    health_score    REAL DEFAULT 1.0,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    id_a            INTEGER NOT NULL,
    id_b            INTEGER NOT NULL,
    relation_type   TEXT NOT NULL,
    score           REAL NOT NULL,
    created_at      INTEGER NOT NULL,
    FOREIGN KEY(id_a) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY(id_b) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_a ON edges(id_a);
CREATE INDEX IF NOT EXISTS idx_edges_b ON edges(id_b);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relation_type);
"""


class SQLiteStoreV3:
    """TurboMemory v3 SQLite storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_dir = os.path.dirname(db_path) or "."
        os.makedirs(self.db_dir, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
    
    def _init_schema(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        conn.close()
    
    @contextmanager
    def _conn(self):
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def add_memory(self, data: Dict[str, Any]) -> int:
        """Add a memory and return its ID."""
        now = int(datetime.now(timezone.utc).timestamp())
        
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO memories (
                    uuid, text, topic_id, source, agent_id, created_at, updated_at,
                    confidence, importance, ttl_seconds, hash, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("uuid"),
                data["text"],
                data.get("topic_id"),
                data.get("source"),
                data.get("agent_id"),
                now, now,
                data.get("confidence", 0.5),
                data.get("importance", 0.5),
                data.get("ttl_seconds"),
                data.get("hash"),
                data.get("extra_json"),
            ))
            mem_id = cur.lastrowid
            
            for tag in data.get("tags", []):
                conn.execute("INSERT INTO tags (memory_id, tag) VALUES (?, ?)", (mem_id, tag))
            
            return mem_id
    
    def update_memory(self, mem_id: int, fields: Dict[str, Any]) -> None:
        """Update memory fields."""
        now = int(datetime.now(timezone.utc).timestamp())
        sets = ["updated_at = ?"]
        params = [now]
        
        for k in ["text", "confidence", "importance", "ttl_seconds", "source", "agent_id"]:
            if k in fields:
                sets.append(f"{k} = ?")
                params.append(fields[k])
        
        params.append(mem_id)
        
        with self._conn() as conn:
            conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params)
    
    def delete_memory(self, mem_id: int) -> None:
        """Hard delete a memory."""
        with self._conn() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
    
    def get_memory(self, mem_id: int) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,))
            row = cur.fetchone()
            if not row:
                return None
            
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    
    def get_memories(
        self,
        topic_id: Optional[int] = None,
        agent_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get memories with filters."""
        where = []
        params = []
        
        if topic_id:
            where.append("topic_id = ?")
            params.append(topic_id)
        if agent_id:
            where.append("agent_id = ?")
            params.append(agent_id)
        if min_confidence is not None:
            where.append("confidence >= ?")
            params.append(min_confidence)
        
        sql = "SELECT * FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    
    def search_by_text(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Simple text search."""
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT * FROM memories 
                WHERE text LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (f"%{query}%", limit))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    
    def get_topics(self) -> List[Dict[str, Any]]:
        """Get all topics."""
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM topics")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    
    def get_or_create_topic(self, name: str) -> int:
        """Get topic ID by name, create if not exists."""
        now = int(datetime.now(timezone.utc).timestamp())
        
        with self._conn() as conn:
            cur = conn.execute("SELECT id FROM topics WHERE name = ?", (name,))
            row = cur.fetchone()
            if row:
                return row[0]
            
            cur = conn.execute("""
                INSERT INTO topics (name, created_at, updated_at)
                VALUES (?, ?, ?)
            """, (name, now, now))
            return cur.lastrowid
    
    def update_topic_centroid(
        self, 
        topic_id: int, 
        dim: int, 
        dtype: str, 
        blob: bytes
    ) -> None:
        """Update topic centroid."""
        now = int(datetime.now(timezone.utc).timestamp())
        
        with self._conn() as conn:
            conn.execute("""
                UPDATE topics 
                SET centroid_dim = ?, centroid_dtype = ?, centroid_blob = ?, updated_at = ?
                WHERE id = ?
            """, (dim, dtype, blob, now, topic_id))
    
    def get_topic_centroid(self, topic_id: int) -> Optional[bytes]:
        """Get topic centroid blob."""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT centroid_blob FROM topics WHERE id = ?", 
                (topic_id,)
            )
            row = cur.fetchone()
            return row[0] if row and row[0] else None
    
    def add_edge(
        self, 
        id_a: int, 
        id_b: int, 
        relation_type: str, 
        score: float
    ) -> None:
        """Add edge (contradiction/support)."""
        now = int(datetime.now(timezone.utc).timestamp())
        
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO edges (id_a, id_b, relation_type, score, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (id_a, id_b, relation_type, score, now))
    
    def get_edges(
        self, 
        memory_id: int, 
        relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get edges for a memory."""
        with self._conn() as conn:
            if relation_type:
                cur = conn.execute("""
                    SELECT * FROM edges 
                    WHERE (id_a = ? OR id_b = ?) AND relation_type = ?
                """, (memory_id, memory_id, relation_type))
            else:
                cur = conn.execute("""
                    SELECT * FROM edges 
                    WHERE id_a = ? OR id_b = ?
                """, (memory_id, memory_id))
            
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    
    def get_embedding(self, memory_id: int) -> Optional[Tuple[int, str, bytes]]:
        """Get embedding blob."""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT dim, dtype, blob FROM embeddings WHERE memory_id = ?",
                (memory_id,)
            )
            row = cur.fetchone()
            return row if row else None
    
    def set_embedding(
        self, 
        memory_id: int, 
        dim: int, 
        dtype: str, 
        blob: bytes
    ) -> None:
        """Set embedding blob."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embeddings (memory_id, dim, dtype, blob)
                VALUES (?, ?, ?, ?)
            """, (memory_id, dim, dtype, blob))
    
    def increment_access(self, memory_id: int) -> None:
        """Increment access count."""
        now = int(datetime.now(timezone.utc).timestamp())
        
        with self._conn() as conn:
            conn.execute("""
                UPDATE memories 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, (now, memory_id))
    
    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        with self._conn() as conn:
            cur = conn.execute("SELECT COUNT(*), AVG(confidence), AVG(importance) FROM memories")
            total, avg_conf, avg_imp = cur.fetchone()
            
            cur = conn.execute("SELECT COUNT(*) FROM topics")
            topics = cur.fetchone()[0]
            
            cur = conn.execute("SELECT COUNT(*) FROM edges WHERE relation_type = 'contradict'")
            contradictions = cur.fetchone()[0]
            
            return {
                "total_memories": total or 0,
                "total_topics": topics or 0,
                "total_edges": contradictions or 0,
                "avg_confidence": avg_conf or 0.0,
                "avg_importance": avg_imp or 0.0,
            }
    
    def close(self) -> None:
        """Close connection."""
        conn = getattr(self._local, 'conn', None)
        if conn:
            conn.close()
            self._local.conn = None