"""TurboMemory v3 Multi-stage retrieval pipeline."""

import os
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result container."""
    memory_id: int
    text: str
    score: float
    topic: str
    confidence: float
    importance: float
    source: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]


class RetrievalPipeline:
    """Multi-stage retrieval pipeline for TurboMemory v3."""
    
    def __init__(
        self,
        store,
        embedding_service,
        faiss_index=None,
        use_gpu: bool = False,
    ):
        self.store = store
        self.embed_service = embedding_service
        self.faiss_index = faiss_index
        self.use_gpu = use_gpu
        
        self._centroid_cache = {}
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        topic: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        agent_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        stages: List[str] = None,
    ) -> List[SearchResult]:
        """Multi-stage search pipeline."""
        if stages is None:
            stages = ["centroid", "faiss", "exact", "verify"]
        
        results = []
        candidates = set()
        
        if "centroid" in stages:
            candidates = self._stage_centroid_filter(query, candidates, limit=100)
        
        candidates = candidates or None
        
        if "faiss" in stages and self.faiss_index:
            candidates = self._stage_faiss_search(
                query, candidates, limit=top_k * 4
            )
        
        if "exact" in stages:
            candidates = self._stage_exact_rerank(
                query, candidates, limit=top_k * 2
            )
        
        for mem in candidates:
            result = self._to_search_result(mem)
            if result:
                results.append(result)
        
        if "verify" in stages:
            results = self._stage_verify(query, results)
        
        results = self._apply_filters(
            results, 
            tags=tags, 
            source=source, 
            agent_id=agent_id,
            min_confidence=min_confidence,
        )
        
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
    
    def _stage_centroid_filter(
        self, 
        query: str, 
        candidates: set,
        limit: int = 100,
    ) -> set:
        """Stage A: Topic centroid prefilter."""
        q_emb = self.embed_service.encode([query])[0]
        
        topics = self.store.get_topics()
        
        scored_topics = []
        for t in topics:
            centroid = self.store.get_topic_centroid(t["id"])
            if not centroid:
                continue
            
            c_emb = np.frombuffer(centroid, dtype=np.float32)
            score = float(np.dot(q_emb, c_emb))
            scored_topics.append((score, t["id"]))
        
        scored_topics.sort(reverse=True)
        
        for _, topic_id in scored_topics[:20]:
            mems = self.store.get_memories(topic_id=topic_id, limit=limit)
            for m in mems:
                candidates.add(m["id"])
        
        return candidates
    
    def _stage_faiss_search(
        self,
        query: str,
        candidate_ids: Optional[set],
        limit: int = 40,
    ) -> set:
        """Stage B: FAISS approximate search."""
        if not self.faiss_index:
            return candidate_ids or set()
        
        q_emb = self.embed_service.encode([query])[0]
        
        faiss_results = self.faiss_index.search(q_emb, k=limit)
        
        candidates = candidate_ids or set()
        for mem_id, score in faiss_results:
            candidates.add(mem_id)
        
        return candidates
    
    def _stage_exact_rerank(
        self,
        query: str,
        candidate_ids: Optional[set],
        limit: int = 20,
    ) -> set:
        """Stage C: Exact cosine rerank."""
        if not candidate_ids:
            return set()
        
        q_emb = self.embed_service.encode([query])[0]
        
        scored = []
        for mem_id in candidate_ids:
            emb_info = self.store.get_embedding(mem_id)
            if not emb_info:
                continue
            
            dim, dtype, blob = emb_info
            
            if dtype.startswith("q"):
                from ..quantization import dequantize_packed
                import base64
                emb = dequantize_packed({
                    "bits": int(dtype[1:]),
                    "scale": 1.0,
                    "zero": 0,
                    "shape": [dim],
                    "data": base64.b64encode(blob).decode()
                })
            else:
                emb = np.frombuffer(blob, dtype=np.float32)
            
            score = float(np.dot(q_emb, emb))
            scored.append((score, mem_id))
        
        scored.sort(reverse=True)
        
        return set([m[1] for m in scored[:limit]])
    
    def _stage_verify(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Stage D: Semantic verification (optional)."""
        # TODO: Integrate cross-encoder or LLM verification
        return results
    
    def _apply_filters(
        self,
        results: List[SearchResult],
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        agent_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[SearchResult]:
        """Apply metadata filters."""
        filtered = []
        
        for r in results:
            if tags:
                if not any(t in r.tags for t in tags):
                    continue
            
            if source and r.source != source:
                continue
            
            if agent_id and r.metadata.get("agent_id") != agent_id:
                continue
            
            if min_confidence and r.confidence < min_confidence:
                continue
            
            filtered.append(r)
        
        return filtered
    
    def _to_search_result(self, mem_id: int) -> Optional[SearchResult]:
        """Convert memory to SearchResult."""
        mem = self.store.get_memory(mem_id)
        if not mem:
            return None
        
        return SearchResult(
            memory_id=mem["id"],
            text=mem["text"],
            score=1.0,
            topic=mem.get("topic_id", ""),
            confidence=mem.get("confidence", 0.5),
            importance=mem.get("importance", 0.5),
            source=mem.get("source"),
            tags=[],  # TODO: fetch tags
            metadata=mem,
        )
    
    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        topic_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Direct vector search."""
        if not self.faiss_index:
            return []
        
        return self.faiss_index.search(
            query_embedding, 
            k=top_k,
            topic_ids=topic_ids,
        )
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """BM25-style keyword search."""
        memories = self.store.search_by_text(query, limit=top_k)
        
        return [
            SearchResult(
                memory_id=m["id"],
                text=m["text"],
                score=0.5,
                topic=m.get("topic_id", ""),
                confidence=m.get("confidence", 0.5),
                importance=m.get("importance", 0.5),
                source=m.get("source"),
                tags=[],
                metadata=m,
            )
            for m in memories
        ]
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
    ) -> List[SearchResult]:
        """Hybrid search combining vector + keyword."""
        vec_results = self.search(query, top_k=top_k * 2)
        
        keyword_results = self.keyword_search(query, top_k=top_k * 2)
        
        merged = {}
        
        for r in vec_results:
            if r.memory_id not in merged:
                merged[r.memory_id] = r
            else:
                merged[r.memory_id].score = (
                    merged[r.memory_id].score * vector_weight + 
                    r.score * (1 - vector_weight)
                )
        
        for r in keyword_results:
            if r.memory_id not in merged:
                merged[r.memory_id] = r
            else:
                merged[r.memory_id].score = (
                    merged[r.memory_id].score * (1 - vector_weight) + 
                    r.score * vector_weight
                )
        
        results = list(merged.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]


class CentroidFilter:
    """Topic centroid prefilter for fast search space reduction."""
    
    def __init__(self, store):
        self.store = store
        self._cache = {}
    
    def get_top_topics(
        self, 
        query_embedding: np.ndarray, 
        top_n: int = 5,
    ) -> List[Tuple[int, float]]:
        """Get top N topics by centroid similarity."""
        topics = self.store.get_topics()
        
        scored = []
        for t in topics:
            centroid = self.store.get_topic_centroid(t["id"])
            if not centroid:
                continue
            
            c_emb = np.frombuffer(centroid, dtype=np.float32)
            score = float(np.dot(query_embedding, c_emb))
            scored.append((score, t["id"]))
        
        scored.sort(reverse=True)
        return scored[:top_n]
    
    def filter_candidates(
        self,
        query_embedding: np.ndarray,
        candidate_ids: set,
        prefilter_topics: int = 5,
    ) -> set:
        """Filter candidates by topic centroids."""
        top_topics = self.get_top_topics(query_embedding, top_n=prefilter_topics)
        
        _, topic_ids = zip(*top_topics)
        
        return {
            c for c in candidate_ids 
            if self._belongs_to_topics(c, topic_ids)
        }
    
    def _belongs_to_topics(self, mem_id: int, topic_ids: Tuple[int, ...]) -> bool:
        """Check if memory belongs to any of the topics."""
        mem = self.store.get_memory(mem_id)
        if not mem:
            return False
        return mem.get("topic_id") in topic_ids