"""TurboMemory storage interface for plugins like AutoStructurer.

Provides:
- add() with multiple vector types per chunk (text, clip)
- SQLite metadata index with filtering (source, schema, topic)
- Unified search across multiple vector types with merge ranking
"""

import os
import json
import sqlite3
import base64
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..quantization import quantize_packed, dequantize_packed

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a chunk."""
    chunk_id: str
    doc_id: Optional[str] = None
    source: str = "unknown"
    schema: str = "generic_text"
    topic: str = "default"
    t_start: float = 0.0
    t_end: float = 0.0
    ref_path: Optional[str] = None
    entities_json: str = "{}"
    contradiction: float = 0.0
    confidence: float = 0.8
    created_at: float = field(default_factory=lambda: __import__("time").time())


@dataclass
class VectorRecord:
    """Vector storage record."""
    vector_id: int
    chunk_id: str
    kind: str
    dim: int
    packed: bytes
    scale: float
    zero: float


class TurboMemoryWriter:
    """Write interface for TurboMemory - used by AutoStructurer ETL pipeline.
    
    Usage:
        from turbomemory.plugins import TurboMemoryWriter
        
        writer = TurboMemoryWriter("data/")
        writer.add(
            text="OCR text from image",
            vector=text_embedding,
            metadata=ChunkMetadata(source="ocr", schema="invoice")
        )
        writer.add(
            vector=clip_embedding,
            vector_type="clip",
            chunk_id=same_chunk_id
        )
    """

    def __init__(self, root: str = "turbomemory_data"):
        self.root = root
        self.db_dir = os.path.join(root, "db")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "plugin_index.sqlite")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks(
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                source TEXT,
                schema TEXT,
                topic TEXT,
                t_start REAL,
                t_end REAL,
                text TEXT,
                entities_json TEXT,
                contradiction REAL,
                confidence REAL,
                ref_path TEXT,
                created_at REAL
            )
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_schema ON chunks(schema)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic)
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vectors(
                vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT,
                kind TEXT,
                dim INTEGER,
                packed BLOB,
                scale REAL,
                zero REAL,
                FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
            )
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_kind ON vectors(kind)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_chunk ON vectors(chunk_id)
        """)
        
        conn.commit()
        conn.close()

    def add(
        self,
        text: Optional[str],
        vector: np.ndarray,
        metadata: ChunkMetadata,
        bits: int = 4,
    ) -> str:
        """Add a chunk with text and vector.
        
        Args:
            text: Text content
            vector: Embedding vector (float32)
            metadata: Chunk metadata
            bits: Quantization bits (4, 6, or 8)
            
        Returns:
            chunk_id
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT OR REPLACE INTO chunks(
                chunk_id, doc_id, source, schema, topic, t_start, t_end,
                text, entities_json, contradiction, confidence, ref_path, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            metadata.chunk_id,
            metadata.doc_id,
            metadata.source,
            metadata.schema,
            metadata.topic,
            metadata.t_start,
            metadata.t_end,
            text,
            metadata.entities_json,
            metadata.contradiction,
            metadata.confidence,
            metadata.ref_path,
            metadata.created_at,
        ))
        
        packed, scale, zero = self._pack_vector(vector, bits)
        cur.execute("""
            INSERT INTO vectors(chunk_id, kind, dim, packed, scale, zero)
            VALUES (?,?,?,?,?,?)
        """, (metadata.chunk_id, "text", vector.shape[0], packed, scale, zero))
        
        conn.commit()
        conn.close()
        return metadata.chunk_id

    def add_vector(
        self,
        chunk_id: str,
        vector: np.ndarray,
        vector_type: str = "text",
        bits: int = 4,
    ) -> int:
        """Add an additional vector type to an existing chunk.
        
        Args:
            chunk_id: Existing chunk ID
            vector: Embedding vector
            vector_type: Type of vector (text, clip, etc.)
            bits: Quantization bits
            
        Returns:
            vector_id
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        packed, scale, zero = self._pack_vector(vector, bits)
        cur.execute("""
            INSERT INTO vectors(chunk_id, kind, dim, packed, scale, zero)
            VALUES (?,?,?,?,?,?)
        """, (chunk_id, vector_type, vector.shape[0], packed, scale, zero))
        
        vector_id = cur.lastrowid
        conn.commit()
        conn.close()
        return vector_id

    def _pack_vector(self, vec: np.ndarray, bits: int) -> Tuple[bytes, float, float]:
        """Pack vector using TurboQuant."""
        q = quantize_packed(vec.astype(np.float32), bits=bits)
        packed = base64.b64decode(q["data"])
        return packed, q["scale"], q["qmax"]

    def _unpack_vector(self, packed: bytes, dim: int, scale: float, zero: float) -> np.ndarray:
        """Unpack vector."""
        q = {
            "bits": 4,
            "scale": scale,
            "qmax": int(zero),
            "shape": [dim],
            "data": base64.b64encode(packed).decode("utf-8")
        }
        return dequantize_packed(q)

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "chunk_id": row[0],
            "doc_id": row[1],
            "source": row[2],
            "schema": row[3],
            "topic": row[4],
            "t_start": row[5],
            "t_end": row[6],
            "text": row[7],
            "entities_json": row[8],
            "contradiction": row[9],
            "confidence": row[10],
            "ref_path": row[11],
            "created_at": row[12],
        }

    def get_vectors(self, chunk_id: str) -> List[VectorRecord]:
        """Get all vectors for a chunk."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT vector_id, chunk_id, kind, dim, packed, scale, zero
            FROM vectors WHERE chunk_id = ?
        """, (chunk_id,))
        rows = cur.fetchall()
        conn.close()
        
        return [VectorRecord(*row) for row in rows]

    def count(self) -> int:
        """Total chunk count."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
        conn.close()
        return count


class TurboMemorySearch:
    """Search interface for TurboMemory with multi-vector support.
    
    Supports:
    - Filter by source, schema, topic
    - Search across multiple vector types
    - Merge ranking (max score across types)
    """

    def __init__(self, root: str = "turbomemory_data", use_faiss: bool = False):
        self.root = root
        self.db_dir = os.path.join(root, "db")
        self.db_path = os.path.join(self.db_dir, "plugin_index.sqlite")
        self.use_faiss = use_faiss
        
        self._text_index = None
        self._clip_index = None
        self._init_indexes()

    def _init_indexes(self):
        if not self.use_faiss:
            return
        try:
            import faiss
            text_path = os.path.join(self.db_dir, "text.index")
            clip_path = os.path.join(self.db_dir, "clip.index")
            
            if os.path.exists(text_path):
                self._text_index = faiss.read_index(text_path)
            else:
                self._text_index = faiss.IndexFlatIP(384)
                
            if os.path.exists(clip_path):
                self._clip_index = faiss.read_index(clip_path)
            else:
                self._clip_index = faiss.IndexFlatIP(512)
        except ImportError:
            logger.warning("faiss not available, using SQLite fallback")

    def search(
        self,
        query_vector: np.ndarray,
        vector_type: str = "text",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Search with optional filters.
        
        Args:
            query_vector: Query embedding
            vector_type: Which vector type to search (text, clip)
            top_k: Number of results
            filters: Optional filters (source, schema, topic)
            
        Returns:
            List of (score, chunk_dict) tuples
        """
        conn = sqlite3.connect(self.db_path)
        
        if filters:
            where_clauses = []
            params = []
            if "source" in filters:
                where_clauses.append("source = ?")
                params.append(filters["source"])
            if "schema" in filters:
                where_clauses.append("schema = ?")
                params.append(filters["schema"])
            if "topic" in filters:
                where_clauses.append("topic = ?")
                params.append(filters["topic"])
            
            if where_clauses:
                where = "WHERE " + " AND ".join(where_clauses)
                cur = conn.execute(f"""
                    SELECT chunk_id FROM chunks {where}
                """, params)
                allowed_chunks = {row[0] for row in cur.fetchall()}
            else:
                allowed_chunks = None
        else:
            allowed_chunks = None
        
        cur = conn.execute("""
            SELECT v.vector_id, v.chunk_id, v.dim, v.packed, v.scale, v.zero
            FROM vectors v
            WHERE v.kind = ?
        """, (vector_type,))
        
        from ..quantization import dequantize_packed
        
        results = []
        for row in cur.fetchall():
            vector_id, chunk_id, dim, packed, scale, zero = row
            
            if allowed_chunks and chunk_id not in allowed_chunks:
                continue
            
            q = {
                "bits": 4,
                "scale": scale,
                "qmax": int(zero),
                "shape": [dim],
                "data": base64.b64encode(packed).decode("utf-8")
            }
            vec = dequantize_packed(q)
            
            score = float(np.dot(query_vector, vec))
            results.append((score, chunk_id))
        
        conn.close()
        
        results.sort(key=lambda x: x[0], reverse=True)
        
        final_results = []
        for score, chunk_id in results[:top_k]:
            chunk = self._get_chunk(chunk_id)
            if chunk:
                final_results.append((score, chunk))
        
        return final_results

    def search_hybrid(
        self,
        text_query: np.ndarray,
        clip_query: Optional[np.ndarray] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        text_weight: float = 0.5,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Hybrid search across text and clip vectors with merge ranking.
        
        Args:
            text_query: Text embedding
            clip_query: Optional CLIP embedding
            top_k: Number of results
            filters: Optional filters
            text_weight: Weight for text scores (clip_weight = 1 - text_weight)
            
        Returns:
            List of (score, chunk_dict) tuples with merged scores
        """
        text_results = self.search(text_query, "text", top_k * 2, filters)
        
        if clip_query is not None:
            clip_results = self.search(clip_query, "clip", top_k * 2, filters)
        else:
            clip_results = []
        
        merged = {}
        for score, chunk in text_results:
            cid = chunk["chunk_id"]
            if cid not in merged or score > merged[cid][0]:
                merged[cid] = (score * text_weight, chunk)
        
        for score, chunk in clip_results:
            cid = chunk["chunk_id"]
            clip_score = score * (1 - text_weight)
            if cid not in merged or clip_score > merged[cid][0]:
                merged[cid] = (clip_score, chunk)
        
        sorted_results = sorted(merged.values(), key=lambda x: x[0], reverse=True)
        return sorted_results[:top_k]

    def _get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "chunk_id": row[0],
            "doc_id": row[1],
            "source": row[2],
            "schema": row[3],
            "topic": row[4],
            "t_start": row[5],
            "t_end": row[6],
            "text": row[7],
            "entities_json": row[8],
            "contradiction": row[9],
            "confidence": row[10],
            "ref_path": row[11],
            "created_at": row[12],
        }
    
    def filter_values(self, field: str) -> List[str]:
        """Get unique values for a filter field (source, schema, topic)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT {field} FROM chunks")
        values = [row[0] for row in cur.fetchall()]
        conn.close()
        return values