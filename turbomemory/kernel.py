"""TurboMemory v3 Kernel - Core engine."""

import os
import uuid
import time
import logging
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from .config import TurboMemoryConfig
from .storage.sqlite_store import SQLiteStore
from .storage.wal import WAL
from .embeddings.st_backend import SentenceTransformersBackend
from .quant.qpack import pack_q4, pack_q6, pack_q8
from .quant.dotprod import BITS_MAP
from .retrieval.pipeline import RetrievalPipeline

# Quantization function map
QUANT_FUNCS = {
    "q4": pack_q4,
    "q6": pack_q6,
    "q8": pack_q8,
}

class TurboMemoryKernel:
    """TurboMemory v3 kernel."""
    
    def __init__(self, root: str, config: TurboMemoryConfig):
        self.root = root
        self.config = config
        
        os.makedirs(root, exist_ok=True)
        
        # Load schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_v3.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        
        db_path = os.path.join(root, config.sqlite_file)
        self.store = SQLiteStore(db_path, schema_sql)
        
        # WAL
        self.wal = WAL(os.path.join(root, config.wal_dir)) if config.enable_wal else None
        
        # Embedding backend
        self.embedder = SentenceTransformersBackend(
            config.embedding_model,
            use_gpu=config.use_gpu
        )
        
        # Retrieval pipeline
        self.pipeline = RetrievalPipeline(self.store)
    
    def add(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        topic: Optional[str] = None,
        confidence: float = 0.5,
        ttl_seconds: Optional[int] = None,
        source: str = "manual",
        agent_id: str = "default",
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a memory."""
        mem_uuid = str(uuid.uuid4())
        
        # Embed and quantize
        vec = self.embedder.encode(text)
        
        dtype = self.config.quantization
        if dtype == "q8":
            blob = pack_q8(vec)
        else:
            # Fallback to q8
            blob = pack_q8(vec)
            dtype = "q8"
        
        mem_id = self.store.add_memory(
            uuid=mem_uuid,
            text=text,
            topic=topic,
            source=source,
            agent_id=agent_id,
            confidence=float(confidence),
            importance=0.5,
            ttl_seconds=ttl_seconds,
            tags=tags or [],
            extra=extra or {},
            embedding_blob=blob,
            embedding_dim=int(vec.shape[0]),
            embedding_dtype=dtype,
        )
        
        # Write to WAL
        if self.wal:
            self.wal.append({
                "op": "add",
                "id": mem_id,
                "uuid": mem_uuid,
                "text": text,
                "topic": topic,
                "tags": tags or [],
                "source": source,
                "agent_id": agent_id,
                "confidence": confidence,
                "ttl_seconds": ttl_seconds,
            })
        
        logger.info(f"Added memory {mem_id}")
        return mem_id
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        tags_any: Optional[List[str]] = None,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None,
        topic: Optional[str] = None,
        verify: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for memories."""
        qvec = self.embedder.encode(query)
        
        # Get candidates via FTS
        candidates = self.store.search_fts(query, limit=200)
        if not candidates:
            return []
        
        # Apply metadata filters
        if topic or source or min_confidence is not None:
            candidates = self.store.filter_memories(
                candidates,
                topic=topic,
                source=source,
                min_confidence=min_confidence,
            )
        
        # Vector rerank
        results = self.pipeline.search(qvec, candidates, top_k=top_k)
        
        out = []
        for score, mem_id in results:
            item = self.store.get_memory(mem_id)
            if not item:
                continue
            out.append({
                "id": mem_id,
                "score": float(score),
                "text": item["text"],
                "topic": item.get("topic"),
                "confidence": item["confidence"],
                "source": item.get("source"),
            })
        return out
    
    def reinforce(self, mem_id: int, delta: float = 0.1):
        """Increase confidence."""
        m = self.store.get_memory(mem_id)
        if not m:
            return
        new_conf = min(1.0, float(m["confidence"]) + delta)
        self.store.update_fields(mem_id, {"confidence": new_conf})
        
        if self.wal:
            self.wal.append({"op": "reinforce", "memory_id": mem_id, "delta": delta})
    
    def penalize(self, mem_id: int, delta: float = 0.2):
        """Decrease confidence."""
        m = self.store.get_memory(mem_id)
        if not m:
            return
        new_conf = max(0.0, float(m["confidence"]) - delta)
        self.store.update_fields(mem_id, {"confidence": new_conf})
        
        if self.wal:
            self.wal.append({"op": "penalize", "memory_id": mem_id, "delta": delta})
    
    def forget(self, mem_id: int):
        """Soft forget - reduce confidence near zero."""
        self.store.update_fields(mem_id, {"confidence": 0.01, "importance": 0.01})
        
        if self.wal:
            self.wal.append({"op": "forget", "memory_id": mem_id})
    
    def delete(self, mem_id: int):
        """Hard delete."""
        self.store.delete_memory(mem_id)
        
        if self.wal:
            self.wal.append({"op": "delete", "memory_id": mem_id})
    
    def stats(self) -> Dict[str, int]:
        """Get statistics."""
        return self.store.get_stats()
    
    def get_memory(self, mem_id: int) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        return self.store.get_memory(mem_id)
    
    def close(self):
        """Close stores."""
        if self.wal:
            self.wal.close()
        self.store.close()