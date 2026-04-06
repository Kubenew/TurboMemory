"""TMF index operations using SQLite."""

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Tuple
from contextlib import contextmanager


INDEX_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL UNIQUE,
    file_ref TEXT,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    centroid_bits INTEGER,
    centroid_scale REAL,
    centroid_qmax INTEGER,
    centroid_blob BLOB,
    centroid_shape TEXT DEFAULT '[384]'
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    chunk_key TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    staleness REAL DEFAULT 0.0,
    entropy REAL DEFAULT 0.0,
    ts TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    text_preview TEXT,
    emb_offset INTEGER NOT NULL,
    emb_length INTEGER NOT NULL,
    emb_bits INTEGER,
    emb_scale REAL,
    emb_qmax INTEGER,
    emb_shape TEXT DEFAULT '[384]',
    ttl_ts TEXT,
    quality_score REAL DEFAULT 0.5,
    verified INTEGER DEFAULT 0,
    verification_score REAL DEFAULT 0.0,
    source_refs TEXT,
    PRIMARY KEY(topic, chunk_key),
    FOREIGN KEY(topic) REFERENCES topics(topic)
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    topic TEXT,
    chunk_key TEXT,
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic);
CREATE INDEX IF NOT EXISTS idx_chunks_confidence ON chunks(confidence);
CREATE INDEX IF NOT EXISTS idx_chunks_ttl ON chunks(ttl_ts);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(text_hash);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


class TMFIndex:
    """SQLite index for TMF storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(INDEX_SCHEMA_V1)
    
    # Topic operations
    def upsert_topic(self, topic: str, **kwargs) -> None:
        """Insert or update a topic."""
        defaults = {
            "created": kwargs.get("created", ""),
            "updated": kwargs.get("updated", ""),
            "chunk_count": kwargs.get("chunk_count", 0),
            "centroid_bits": kwargs.get("centroid_bits"),
            "centroid_scale": kwargs.get("centroid_scale"),
            "centroid_qmax": kwargs.get("centroid_qmax"),
            "centroid_blob": kwargs.get("centroid_blob"),
            "centroid_shape": kwargs.get("centroid_shape", "[384]"),
            "file_ref": kwargs.get("file_ref"),
        }
        
        cols = ["topic"] + [k for k in defaults if defaults[k] is not None]
        vals = [topic] + [v for v in defaults.values() if v is not None]
        placeholders = ",".join(["?"] * len(cols))
        
        sql = f"""
            INSERT INTO topics({','.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(topic) DO UPDATE SET
                {','.join(f'{k}=excluded.{k}' for k in defaults if defaults[k] is not None)}
        """
        
        with self._conn() as conn:
            conn.execute(sql, vals)
    
    def get_topic(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get topic metadata."""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM topics WHERE topic = ?",
                (topic,)
            )
            row = cur.fetchone()
            if not row:
                return None
            
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))
    
    def list_topics(self) -> List[Dict[str, Any]]:
        """List all topics."""
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM topics ORDER BY topic")
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    
    def delete_topic(self, topic: str) -> int:
        """Delete a topic and its chunks."""
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE topic = ?", (topic,))
            conn.execute("DELETE FROM topics WHERE topic = ?", (topic,))
            return conn.total_changes
    
    # Chunk operations
    def upsert_chunk(self, topic: str, chunk_key: str, **kwargs) -> None:
        """Insert or update a chunk."""
        defaults = {
            "confidence": kwargs.get("confidence", 0.5),
            "staleness": kwargs.get("staleness", 0.0),
            "entropy": kwargs.get("entropy", 0.0),
            "ts": kwargs.get("ts", ""),
            "text_hash": kwargs.get("text_hash", ""),
            "text_preview": kwargs.get("text_preview", "")[:200],
            "emb_offset": kwargs.get("emb_offset", 0),
            "emb_length": kwargs.get("emb_length", 0),
            "emb_bits": kwargs.get("emb_bits"),
            "emb_scale": kwargs.get("emb_scale"),
            "emb_qmax": kwargs.get("emb_qmax"),
            "emb_shape": kwargs.get("emb_shape", "[384]"),
            "ttl_ts": kwargs.get("ttl_ts"),
            "quality_score": kwargs.get("quality_score", 0.5),
            "verified": kwargs.get("verified", 0),
            "verification_score": kwargs.get("verification_score", 0.0),
            "source_refs": kwargs.get("source_refs"),
        }
        
        cols = ["topic", "chunk_key"] + [k for k in defaults if defaults[k] is not None]
        vals = [topic, chunk_key] + [v for v in defaults.values() if v is not None]
        placeholders = ",".join(["?"] * len(cols))
        
        sql = f"""
            INSERT INTO chunks({','.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(topic, chunk_key) DO UPDATE SET
                {','.join(f'excluded.{k}={k}' for k in defaults if defaults[k] is not None)}
        """
        
        with self._conn() as conn:
            conn.execute(sql, vals)
            
            # Update topic chunk count
            conn.execute("""
                UPDATE topics 
                SET chunk_count = (SELECT COUNT(*) FROM chunks WHERE topic = ?),
                    updated = datetime('now')
                WHERE topic = ?
            """, (topic, topic))
    
    def get_chunk(self, topic: str, chunk_key: str) -> Optional[Dict[str, Any]]:
        """Get chunk metadata."""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM chunks WHERE topic = ? AND chunk_key = ?",
                (topic, chunk_key)
            )
            row = cur.fetchone()
            if not row:
                return None
            
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))
    
    def get_chunks_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all chunks for a topic."""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM chunks WHERE topic = ? ORDER BY chunk_key",
                (topic,)
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    
    def search_chunks(
        self,
        topics: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search chunks with filters."""
        sql = "SELECT * FROM chunks WHERE confidence >= ?"
        params = [min_confidence]
        
        if topics:
            placeholders = ",".join(["?"] * len(topics))
            sql += f" AND topic IN ({placeholders})"
            params.extend(topics)
        
        sql += " ORDER BY confidence DESC LIMIT ?"
        params.append(limit)
        
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    
    def delete_chunk(self, topic: str, chunk_key: str) -> int:
        """Delete a chunk."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE topic = ? AND chunk_key = ?",
                (topic, chunk_key)
            )
            
            # Update topic chunk count
            conn.execute("""
                UPDATE topics 
                SET chunk_count = (SELECT COUNT(*) FROM chunks WHERE topic = ?)
                WHERE topic = ?
            """, (topic, topic))
            
            return conn.total_changes
    
    # Event logging
    def log_event(self, event_type: str, topic: Optional[str] = None, 
                  chunk_key: Optional[str] = None, details: Optional[str] = None) -> None:
        """Log an event."""
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events(ts, event_type, topic, chunk_key, details) VALUES (?, ?, ?, ?, ?)",
                (ts, event_type, topic, chunk_key, details)
            )
    
    def get_events(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get events."""
        sql = "SELECT * FROM events"
        params = []
        
        if event_type:
            sql += " WHERE event_type = ?"
            params.append(event_type)
        
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    
    # Stats
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._conn() as conn:
            topic_count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            avg_confidence = conn.execute("SELECT AVG(confidence) FROM chunks").fetchone()[0] or 0
            event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            
            return {
                "topic_count": topic_count,
                "chunk_count": chunk_count,
                "avg_confidence": round(avg_confidence, 3),
                "event_count": event_count,
            }
    
    def close(self):
        """Close the index."""
        pass  # Connections are context-managed
