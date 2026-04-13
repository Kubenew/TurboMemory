"""TurboMemory v3 FAISS IVF-PQ index for GPU-accelerated retrieval."""

import os
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available, falling back to numpy")


class FAISSIndex:
    """FAISS IVF-PQ index for vector search."""
    
    def __init__(
        self,
        dim: int = 384,
        nlist: int = 4096,
        m: int = 32,
        nbits: int = 8,
        use_gpu: bool = False,
    ):
        """Initialize FAISS index.
        
        Args:
            dim: embedding dimension
            nlist: number of clusters (4096-16384 for millions)
            m: number of subquantizers (32-64)
            nbits: bits per subquantizer (6-8)
            use_gpu: use GPU index
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu or faiss-gpu required")
        
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.use_gpu = use_gpu
        
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFPQ(
            quantizer, 
            dim, 
            nlist, 
            m, 
            nbits
        )
        self.index.nprobe = 16
        
        self._gpu_resources = None
        if use_gpu:
            self._to_gpu()
        
        self.id_to_idx = {}
        self.idx_to_id = {}
    
    def _to_gpu(self):
        """Move index to GPU."""
        if not FAISS_AVAILABLE:
            return
        try:
            self._gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, self.index)
        except Exception as e:
            logger.warning(f"Failed to move to GPU: {e}")
            self.use_gpu = False
    
    def _to_cpu(self):
        """Move index back to CPU."""
        if self._gpu_resources is not None:
            self.index = faiss.index_gpu_to_cpu(self.index)
            self._gpu_resources = None
    
    def add_with_ids(
        self, 
        embeddings: np.ndarray, 
        ids: List[int]
    ) -> None:
        """Add embeddings with IDs."""
        if not self.index.is_trained:
            logger.info(f"Training index with {len(embeddings)} vectors")
            self.index.train(embeddings.astype(np.float32))
        
        start = len(self.id_to_idx)
        idxs = np.arange(start, start + len(embeddings))
        
        self.index.add_with_ids(embeddings.astype(np.float32), idxs)
        
        for idx, id_ in zip(idxs, ids):
            self.id_to_idx[id_] = idx
            self.idx_to_id[idx] = id_
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 10,
        topic_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Search for similar vectors.
        
        Returns:
            List of (id, distance) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        q = query.reshape(1, -1).astype(np.float32)
        distances, idxs = self.index.search(q, min(k * 4, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], idxs[0]):
            if idx < 0:
                continue
            id_ = self.idx_to_id.get(int(idx))
            if id_ is not None:
                results.append((id_, float(dist)))
                if len(results) >= k:
                    break
        
        return results
    
    def search_batch(
        self, 
        queries: np.ndarray, 
        k: int = 10,
    ) -> List[List[Tuple[int, float]]]:
        """Batch search."""
        if self.index.ntotal == 0:
            return [[] for _ in queries]
        
        q = queries.astype(np.float32)
        distances, idxs = self.index.search(q, min(k * 4, self.index.ntotal))
        
        batch_results = []
        for dist_row, idx_row in zip(distances, idxs):
            results = []
            for dist, idx in zip(dist_row, idx_row):
                if idx < 0:
                    continue
                id_ = self.idx_to_id.get(int(idx))
                if id_ is not None:
                    results.append((id_, float(dist)))
                    if len(results) >= k:
                        break
            batch_results.append(results)
        
        return batch_results
    
    def remove_ids(self, ids: List[int]) -> None:
        """Remove vectors by ID."""
        idxs_to_remove = [self.id_to_idx[id_] for id_ in ids if id_ in self.id_to_idx]
        
        if idxs_to_remove:
            self.index.remove_ids(np.array(idxs_to_remove, dtype=np.int64))
            
            for id_ in ids:
                idx = self.id_to_idx.pop(id_, None)
                if idx is not None:
                    self.idx_to_id.pop(idx, None)
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        if self.use_gpu:
            self._to_cpu()
        
        faiss.write_index(self.index, path)
        
        with open(path + ".ids", "w") as f:
            for id_, idx in sorted(self.id_to_idx.items(), key=lambda x: x[1]):
                f.write(f"{id_},{idx}\n")
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        self.index = faiss.read_index(path)
        
        if self.use_gpu:
            self._to_gpu()
        
        ids_path = path + ".ids"
        if os.path.exists(ids_path):
            self.id_to_idx = {}
            self.idx_to_id = {}
            with open(ids_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        id_, idx = int(parts[0]), int(parts[1])
                        self.id_to_idx[id_] = idx
                        self.idx_to_id[idx] = id_
    
    def reset(self) -> None:
        """Reset index."""
        if self.use_gpu:
            self._to_cpu()
        
        self.index.reset()
        self.id_to_idx = {}
        self.idx_to_id = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "ntotal": self.index.ntotal,
            "dim": self.dim,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "is_trained": self.index.is_trained,
            "use_gpu": self.use_gpu,
        }


class TopicFAISSIndex:
    """Topic-specific FAISS indexes for fast filtering."""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.topic_indexes = {}
    
    def get_or_create_topic_index(
        self, 
        topic_id: int, 
        dim: int = 384,
        use_gpu: bool = False,
    ) -> FAISSIndex:
        """Get or create topic index."""
        if topic_id not in self.topic_indexes:
            path = os.path.join(self.index_dir, f"topic_{topic_id}.index")
            
            idx = FAISSIndex(dim=dim, use_gpu=use_gpu)
            
            if os.path.exists(path):
                idx.load(path)
            
            self.topic_indexes[topic_id] = idx
        
        return self.topic_indexes[topic_id]
    
    def search_topic(
        self,
        topic_id: int,
        query: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Search within a specific topic."""
        if topic_id not in self.topic_indexes:
            return []
        
        return self.topic_indexes[topic_id].search(query, k=k)
    
    def save_all(self) -> None:
        """Save all topic indexes."""
        for topic_id, idx in self.topic_indexes.items():
            path = os.path.join(self.index_dir, f"topic_{topic_id}.index")
            idx.save(path)
    
    def close(self) -> None:
        """Close all indexes."""
        for idx in self.topic_indexes.values():
            if hasattr(idx, 'close'):
                idx.close()
        self.topic_indexes = {}


def create_faiss_index(
    dim: int = 384,
    vector_count: int = 1000000,
    use_gpu: bool = False,
) -> FAISSIndex:
    """Factory function to create optimally configured index."""
    if vector_count < 10000:
        nlist = 256
    elif vector_count < 100000:
        nlist = 1024
    elif vector_count < 1000000:
        nlist = 4096
    else:
        nlist = 8192
    
    m = min(64, max(16, dim // 16))
    nbits = 8
    
    return FAISSIndex(
        dim=dim,
        nlist=nlist,
        m=m,
        nbits=nbits,
        use_gpu=use_gpu,
    )