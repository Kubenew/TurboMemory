"""TMF v1 (TurboMemory Format) specification and implementation.

This module implements the TMF v1 portable storage format:
- .tmmeta.json - Global metadata
- .tmindex.sqlite - SQLite metadata index  
- .tmlog - Append-only event log
- .tmvec - Packed vector files

Design principles:
- Portability: Copy directory → load instantly
- Durability: Append-only log is source of truth
- Efficiency: 4/6/8-bit TurboQuant vectors
- Self-describing: Versioning, checksums, schema
"""

import os
import json
import sqlite3
import hashlib
import struct
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone


TMF_VERSION = "1.0"
INDEX_FILENAME = "tm.index.sqlite"
LOG_FILENAME = "tm.log"
VECTORS_DIR = "vectors"
CENTROIDS_DIR = "centroids"
META_FILENAME = ".tmmeta.json"


@dataclass
class TMFMetadata:
    """Global metadata for TMF store."""
    tmf_version: str = TMF_VERSION
    created_at: str = ""
    last_modified: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    default_bit_width: int = 6
    topics: List[str] = field(default_factory=list)
    checksum_algorithm: str = "sha256"
    global_checksum: str = ""
    migration_history: List[Dict[str, str]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tmf_version": self.tmf_version,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "default_bit_width": self.default_bit_width,
            "topics": self.topics,
            "checksum_algorithm": self.checksum_algorithm,
            "global_checksum": self.global_checksum,
            "migration_history": self.migration_history,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TMFMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TMFVectorHeader:
    """Header for .tmvec packed vector file (128 bytes)."""
    magic: bytes = b"TMFV1"
    version: int = 1
    bit_width: int = 6
    dim: int = 384
    num_vectors: int = 0
    checksum_header: str = ""
    reserved: bytes = b"\x00" * 32

    def to_bytes(self) -> bytes:
        return struct.pack(
            "<5sIBHH32s32x",
            self.magic,
            self.version,
            self.bit_width,
            self.dim,
            self.num_vectors,
            self.reserved[:32]
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TMFVectorHeader":
        values = struct.unpack("<5sIBHH32s32x", data[:128])
        return cls(
            magic=values[0],
            version=values[1],
            bit_width=values[2],
            dim=values[3],
            num_vectors=values[4],
            reserved=values[5],
        )


class TMFIndex:
    """SQLite metadata index for TMF."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        topic TEXT NOT NULL,
        text TEXT,
        timestamp INTEGER,
        ttl INTEGER,
        confidence REAL,
        verification INTEGER DEFAULT 0,
        bit_width INTEGER DEFAULT 6,
        vector_offset INTEGER,
        vector_length INTEGER,
        tags TEXT,
        checksum TEXT
    );

    CREATE TABLE IF NOT EXISTS topics (
        name TEXT PRIMARY KEY,
        centroid_id TEXT,
        vector_count INTEGER DEFAULT 0,
        last_consolidated INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_topic ON chunks(topic);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON chunks(timestamp);
    CREATE INDEX IF NOT EXISTS idx_confidence ON chunks(confidence);
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(self.SCHEMA)
        conn.commit()
        conn.close()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_chunk(self, chunk_data: Dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO chunks(id, topic, text, timestamp, ttl, confidence, 
                                   verification, bit_width, vector_offset, vector_length, tags, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text, timestamp=excluded.timestamp, ttl=excluded.ttl,
                    confidence=excluded.confidence, verification=excluded.verification,
                    bit_width=excluded.bit_width, vector_offset=excluded.vector_offset,
                    vector_length=excluded.vector_length, tags=excluded.tags, checksum=excluded.checksum
            """, (
                chunk_data.get("id"),
                chunk_data.get("topic"),
                chunk_data.get("text"),
                chunk_data.get("timestamp"),
                chunk_data.get("ttl"),
                chunk_data.get("confidence"),
                chunk_data.get("verification", 0),
                chunk_data.get("bit_width", 6),
                chunk_data.get("vector_offset"),
                chunk_data.get("vector_length"),
                json.dumps(chunk_data.get("tags", [])),
                chunk_data.get("checksum"),
            ))

    def get_chunks_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM chunks WHERE topic = ? ORDER BY timestamp DESC",
                (topic,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_topics(self) -> List[str]:
        with self._conn() as conn:
            cur = conn.execute("SELECT name FROM topics")
            return [r[0] for r in cur.fetchall()]

    def update_topic_count(self, topic: str, count: int) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO topics(name, vector_count, last_consolidated)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET vector_count=excluded.vector_count
            """, (topic, count, int(time.time())))


class TMFLog:
    """Append-only event log for TMF."""

    LOG_MAGIC = b"TMLOG"

    def __init__(self, log_path: str):
        self.log_path = log_path

    def append(self, event_type: str, data: Dict[str, Any]) -> int:
        """Append an event to the log."""
        import uuid
        
        entry_id = str(uuid.uuid4())[:16]
        timestamp = int(time.time() * 1000)
        
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        
        with open(self.log_path, "ab") as f:
            f.write(self.LOG_MAGIC)
            f.write(struct.pack("<I", len(payload)))
            f.write(struct.pack("<Q", timestamp))
            f.write(entry_id.encode("utf-8"))
            f.write(payload)
        
        return timestamp

    def get_events(self, since: int = 0) -> List[Dict[str, Any]]:
        """Get events since timestamp (ms)."""
        if not os.path.exists(self.log_path):
            return []
        
        events = []
        with open(self.log_path, "rb") as f:
            while True:
                magic = f.read(5)
                if not magic or magic != self.LOG_MAGIC:
                    break
                
                size = struct.unpack("<I", f.read(4))[0]
                ts = struct.unpack("<Q", f.read(8))[0]
                entry_id = f.read(16).decode("utf-8")
                payload = f.read(size)
                
                if ts > since:
                    events.append({
                        "entry_id": entry_id,
                        "timestamp": ts,
                        "data": json.loads(payload.decode("utf-8")),
                    })
        
        return events


class TMFVectors:
    """Packed vector file reader/writer (.tmvec)."""

    @staticmethod
    def write_topic_vectors(
        path: str,
        vectors: List[Dict[str, Any]],
        bit_width: int = 6,
        dim: int = 384,
    ) -> None:
        """Write vectors to .tmvec file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        header = TMFVectorHeader(
            bit_width=bit_width,
            dim=dim,
            num_vectors=len(vectors),
        )
        
        with open(path, "wb") as f:
            f.write(header.to_bytes())
            
            for v in vectors:
                import base64
                q = v.get("embedding_q", {})
                if isinstance(q, dict) and "data" in q:
                    packed = base64.b64decode(q["data"])
                    f.write(packed)
                elif isinstance(q, bytes):
                    f.write(q)

    @staticmethod
    def read_topic_vectors(path: str) -> Tuple[TMFVectorHeader, List[bytes]]:
        """Read vectors from .tmvec file."""
        if not os.path.exists(path):
            return TMFVectorHeader(), []
        
        with open(path, "rb") as f:
            header_data = f.read(128)
            header = TMFVectorHeader.from_bytes(header_data)
            
            vectors = []
            packed_size = header.dim * header.bit_width // 8
            
            while True:
                data = f.read(packed_size)
                if not data:
                    break
                vectors.append(data)
        
        return header, vectors


class TMFStore:
    """Complete TMF v1 store implementation."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.meta_path = self.root / META_FILENAME
        self.index_path = self.root / INDEX_FILENAME
        self.log_path = self.root / LOG_FILENAME
        self.vectors_dir = self.root / VECTORS_DIR
        self.centroids_dir = self.root / CENTROIDS_DIR

    def create(self, config: Optional[Dict[str, Any]] = None) -> TMFMetadata:
        """Create a new TMF store."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(exist_ok=True)
        self.centroids_dir.mkdir(exist_ok=True)
        
        meta = TMFMetadata(
            created_at=datetime.now(timezone.utc).isoformat(),
            last_modified=datetime.now(timezone.utc).isoformat(),
            config=config or {},
        )
        
        self._save_metadata(meta)
        self._init_index()
        
        return meta

    def _init_index(self) -> None:
        TMFIndex(str(self.index_path))

    def _save_metadata(self, meta: TMFMetadata) -> None:
        with open(self.meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

    def load_metadata(self) -> TMFMetadata:
        """Load TMF metadata."""
        if not self.meta_path.exists():
            raise FileNotFoundError(f"No TMF store at {self.root}")
        
        with open(self.meta_path, "r") as f:
            data = json.load(f)
        return TMFMetadata.from_dict(data)

    def validate(self) -> Dict[str, Any]:
        """Validate TMF store integrity."""
        result = {"valid": True, "errors": []}
        
        if not self.meta_path.exists():
            result["valid"] = False
            result["errors"].append("Missing .tmmeta.json")
        
        if not self.index_path.exists():
            result["valid"] = False
            result["errors"].append("Missing .tmindex.sqlite")
        
        if result["valid"]:
            meta = self.load_metadata()
            result["metadata"] = meta.to_dict()
        
        return result

    def compute_checksum(self) -> str:
        """Compute global checksum of all TMF files."""
        hasher = hashlib.sha256()
        
        for fpath in sorted(self.root.rglob("*")):
            if fpath.is_file() and not fpath.name.startswith("."):
                with open(fpath, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
        
        return hasher.hexdigest()

    def export_bundle(self, output_path: str) -> str:
        """Export TMF store to a single .tm bundle."""
        import tarfile
        
        checksum = self.compute_checksum()
        
        meta = self.load_metadata()
        meta.global_checksum = checksum
        self._save_metadata(meta)
        
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(self.root, arcname="tm_data")
        
        return output_path

    def import_bundle(self, bundle_path: str) -> None:
        """Import TMF store from a .tm bundle."""
        import tarfile
        
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(self.root.parent)


def create_tmf_store(root: str, **config) -> TMFStore:
    """Create a new TMF store."""
    store = TMFStore(root)
    store.create(config)
    return store


def open_tmf_store(root: str) -> TMFStore:
    """Open an existing TMF store."""
    store = TMFStore(root)
    store.load_metadata()
    return store


def verify_tmf_store(root: str) -> Dict[str, Any]:
    """Verify TMF store integrity."""
    store = TMFStore(root)
    return store.validate()


from contextlib import contextmanager