"""IVF (Inverted File) index for accelerated vector search."""

import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class IVFConfig:
    """Configuration for IVF index."""
    dim: int = 384
    n_lists: int = 100
    n_probes: int = 10
    metric: str = "cosine"
    storage_path: Optional[str] = None


class IVFIndex:
    """Inverted File index with quantization for fast vector search.
    
    IVF partitions vectors into clusters using k-means and stores
    inverted lists for each cluster. Query searches only relevant clusters.
    """
    
    def __init__(self, config: Optional[IVFConfig] = None):
        self.config = config or IVFConfig()
        self._centroids: Optional[np.ndarray] = None
        self._lists: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._is_fitted = False
    
    def fit(self, vectors: np.ndarray, ids: Optional[List[int]] = None):
        """Fit the IVF index on training vectors."""
        vectors = np.array(vectors, dtype=np.float32)
        n_vectors = len(vectors)
        
        if n_vectors < self.config.n_lists:
            self.config.n_lists = max(1, n_vectors)
        
        ids = ids if ids is not None else list(range(n_vectors))
        
        logger.info(f"Fitting IVF index with {n_vectors} vectors into {self.config.n_lists} clusters")
        
        self._centroids, labels = self._kmeans(
            vectors,
            self.config.n_lists,
            max_iter=100,
        )
        
        self._lists = {i: [] for i in range(self.config.n_lists)}
        
        for vector, label, vec_id in zip(vectors, labels, ids):
            self._lists[int(label)].append((int(vec_id), vector.astype(np.float32)))
            self._metadata[int(vec_id)] = {}
        
        self._is_fitted = True
        logger.info(f"IVF index fitted with {self.config.n_lists} centroids")
    
    def _kmeans(
        self,
        vectors: np.ndarray,
        k: int,
        max_iter: int = 100,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering."""
        np.random.seed(seed)
        
        centroids = vectors[np.random.choice(len(vectors), k, replace=False)]
        
        for _ in range(max_iter):
            distances = self._compute_distances(vectors, centroids)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = vectors[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return centroids, labels
    
    def _compute_distances(self, vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute distances between vectors and centroids."""
        if self.config.metric == "cosine":
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
            return 1 - np.dot(vectors_norm, centroids_norm.T)
        else:
            return np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
    
    def add_vector(
        self,
        vector: np.ndarray,
        id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a vector to the index."""
        if not self._is_fitted:
            raise RuntimeError("IVF index must be fitted before adding vectors")
        
        vector = np.array(vector, dtype=np.float32).flatten()
        
        if id is None:
            id = max(self._metadata.keys(), default=-1) + 1
        
        distances = self._compute_distances(vector[np.newaxis], self._centroids)
        cluster = int(np.argmin(distances[0]))
        
        self._lists[cluster].append((id, vector))
        self._metadata[id] = metadata or {}
        
        return id
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probes: Optional[int] = None,
    ) -> List[Tuple[int, float, Dict]]:
        """Search for k nearest neighbors."""
        if not self._is_fitted:
            return []
        
        query = np.array(query, dtype=np.float32).flatten()
        
        n_probes = n_probes or self.config.n_probes
        
        distances = self._compute_distances(query[np.newaxis], self._centroids)[0]
        
        cluster_order = np.argsort(distances)[:n_probes]
        
        all_candidates = []
        for cluster_id in cluster_order:
            for vec_id, vector in self._lists[cluster_id]:
                if self.config.metric == "cosine":
                    dist = 1 - np.dot(query, vector) / (
                        np.linalg.norm(query) * np.linalg.norm(vector) + 1e-8
                    )
                else:
                    dist = np.linalg.norm(query - vector)
                all_candidates.append((vec_id, dist, self._metadata.get(vec_id, {})))
        
        all_candidates.sort(key=lambda x: x[1])
        return all_candidates[:k]
    
    def assign_clusters(self, vectors: np.ndarray) -> np.ndarray:
        """Assign vectors to clusters without searching."""
        if not self._is_fitted:
            raise RuntimeError("IVF index must be fitted first")
        
        vectors = np.array(vectors, dtype=np.float32)
        distances = self._compute_distances(vectors, self._centroids)
        return np.argmin(distances, axis=1)
    
    def get_cluster_centroids(self) -> Dict[int, np.ndarray]:
        """Get cluster centroids."""
        if self._centroids is None:
            return {}
        return {i: self._centroids[i] for i in range(len(self._centroids))}
    
    def get_cluster_size(self, cluster_id: int) -> int:
        """Get the number of vectors in a cluster."""
        return len(self._lists.get(cluster_id, []))
    
    def delete_vector(self, id: int) -> bool:
        """Delete a vector from the index."""
        if id not in self._metadata:
            return False
        
        for cluster_id, vectors in self._lists.items():
            self._lists[cluster_id] = [(vid, vec) for vid, vec in vectors if vid != id]
        
        del self._metadata[id]
        return True
    
    def get_vector(self, id: int) -> Optional[np.ndarray]:
        """Get vector by ID."""
        for vectors in self._lists.values():
            for vid, vector in vectors:
                if vid == id:
                    return vector
        return None
    
    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        lists_data = {}
        for cluster_id, vectors in self._lists.items():
            lists_data[str(cluster_id)] = [
                (int(vid), vec.tolist(), self._metadata.get(vid, {}))
                for vid, vec in vectors
            ]
        
        data = {
            "config": {
                "dim": self.config.dim,
                "n_lists": self.config.n_lists,
                "n_probes": self.config.n_probes,
                "metric": self.config.metric,
            },
            "centroids": self._centroids.tolist() if self._centroids is not None else [],
            "lists": lists_data,
            "metadata": {str(k): v for k, v in self._metadata.items()},
            "is_fitted": self._is_fitted,
        }
        
        with open(path / "ivf.json", "w") as f:
            json.dump(data, f)
        
        logger.info(f"Saved IVF index to {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        path = Path(path) / "ivf.json"
        
        if not path.exists():
            raise FileNotFoundError(f"IVF index not found at {path}")
        
        with open(path) as f:
            data = json.load(f)
        
        self.config = IVFConfig(**data["config"])
        
        if data["centroids"]:
            self._centroids = np.array(data["centroids"], dtype=np.float32)
        
        self._lists = {}
        for cluster_id, vectors in data["lists"].items():
            self._lists[int(cluster_id)] = [
                (vid, np.array(vec, dtype=np.float32))
                for vid, vec, _ in vectors
            ]
            for vid, _, meta in vectors:
                self._metadata[vid] = meta
        
        self._is_fitted = data["is_fitted"]
        
        logger.info(f"Loaded IVF index from {path.parent}")
    
    @property
    def size(self) -> int:
        """Return total number of vectors in index."""
        return len(self._metadata)
    
    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self.config.n_lists
    
    def clear(self):
        """Clear all entries."""
        self._lists.clear()
        self._metadata.clear()
        self._centroids = None
        self._is_fitted = False