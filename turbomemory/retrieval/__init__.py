"""Retrieval module with optional FAISS support for fast vector search."""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

# v3 components
from .pipeline import RetrievalPipeline, CentroidFilter, SearchResult
from .faiss_index import FAISSIndex, TopicFAISSIndex, create_faiss_index


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 0.0
    a = a / (a_norm + 1e-9)
    b = b / (b_norm + 1e-9)
    return float(np.dot(a, b))


class VectorIndex(ABC):
    """Abstract base class for vector indices."""

    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the index."""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the index."""
        pass


class NumpyIndex(VectorIndex):
    """Simple numpy-based vector index (fallback)."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._vectors: List[np.ndarray] = []
        self._ids: List[str] = []

    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        for vec, id_ in zip(vectors, ids):
            self._vectors.append(vec)
            self._ids.append(id_)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self._vectors:
            return []
        
        similarities = [
            (id_, cosine_similarity(query, vec)) 
            for vec, id_ in zip(self._vectors, self._ids)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def save(self, path: str) -> None:
        import pickle
        data = {"vectors": self._vectors, "ids": self._ids, "dimension": self.dimension}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._vectors = data["vectors"]
        self._ids = data["ids"]
        self.dimension = data["dimension"]

    def clear(self) -> None:
        self._vectors = []
        self._ids = []


class FAISSIndex(VectorIndex):
    """FAISS-based vector index for fast similarity search."""

    def __init__(self, dimension: int, index_type: str = "IVF"):
        self.dimension = dimension
        self.index_type = index_type
        self._index = None
        self._ids: List[str] = []
        self._init_index()

    def _init_index(self) -> None:
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, falling back to numpy")
            self._fallback = NumpyIndex(self.dimension)
            return

        if self.index_type == "Flat":
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "PQ":
            self._index = faiss.IndexIVFPQ(quantizer, self.dimension, 100, 8, 8)
        else:
            self._index = faiss.IndexFlatIP(self.dimension)

    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        if hasattr(self, "_fallback"):
            self._fallback.add_vectors(vectors, ids)
            return

        vectors = vectors.astype("float32")
        if self.index_type != "Flat" and not self._index.is_trained:
            logger.info("Training index...")
            self._index.train(vectors)
        
        self._index.add(vectors)
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if hasattr(self, "_fallback"):
            return self._fallback.search(query, k)

        query = query.reshape(1, -1).astype("float32")
        distances, indices = self._index.search(query, min(k, self._index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self._ids):
                results.append((self._ids[idx], float(dist)))
        return results

    def save(self, path: str) -> None:
        if hasattr(self, "_fallback"):
            self._fallback.save(path)
            return
        
        import faiss
        faiss.write_index(self._index, f"{path}.index")
        with open(f"{path}.ids", "w") as f:
            f.write("\n".join(self._ids))

    def load(self, path: str) -> None:
        if hasattr(self, "_fallback"):
            self._fallback.load(path)
            return
        
        import faiss
        self._index = faiss.read_index(f"{path}.index")
        with open(f"{path}.ids", "r") as f:
            self._ids = f.read().split("\n")

    def clear(self) -> None:
        self._init_index()
        self._ids = []


class RetrievalEngine:
    """Retrieval engine with pluggable index backends."""

    def __init__(
        self, 
        dimension: int = 384,
        use_faiss: bool = True,
        index_type: str = "Flat"
    ):
        self.dimension = dimension
        self.use_faiss = use_faiss
        
        if use_faiss:
            try:
                import faiss
                self._index = FAISSIndex(dimension, index_type)
                logger.info(f"Using FAISS index ({index_type})")
            except ImportError:
                logger.warning("FAISS not available, using numpy fallback")
                self._index = NumpyIndex(dimension)
        else:
            self._index = NumpyIndex(dimension)

    def add_vectors(
        self, 
        vectors: np.ndarray, 
        ids: List[str],
        topic: Optional[str] = None
    ) -> None:
        """Add vectors to the index."""
        self._index.add_vectors(vectors, ids)

    def search(
        self, 
        query: np.ndarray, 
        k: int = 5,
        filter_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        results = self._index.search(query, k * 2)
        
        if filter_ids is not None:
            filter_set = set(filter_ids)
            results = [r for r in results if r[0] in filter_set]
        
        return results[:k]

    def save_index(self, path: str) -> None:
        """Save index to disk."""
        self._index.save(path)

    def load_index(self, path: str) -> None:
        """Load index from disk."""
        self._index.load(path)

    def clear(self) -> None:
        """Clear the index."""
        self._index.clear()


class TopicPrefilter:
    """Fast topic selection using centroid similarity."""

    def __init__(self, storage_manager):
        self.storage = storage_manager

    def select_topics(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[str]:
        """Select top topics by centroid similarity."""
        try:
            import faiss
            return self._faiss_prefilter(query_vector, top_k)
        except ImportError:
            return self._numpy_prefilter(query_vector, top_k)

    def _faiss_prefilter(self, query: np.ndarray, top_k: int) -> List[str]:
        import faiss
        centroids = self._get_all_centroids()
        if not centroids:
            return []
        
        topics, vectors = zip(*centroids)
        vectors = np.array(vectors).astype("float32")
        
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        
        query = query.reshape(1, -1).astype("float32")
        _, indices = index.search(query, min(top_k, len(topics)))
        
        return [topics[i] for i in indices[0] if i < len(topics)]

    def _numpy_prefilter(self, query: np.ndarray, top_k: int) -> List[str]:
        centroids = self._get_all_centroids()
        if not centroids:
            return []
        
        scored = [
            (topic, cosine_similarity(query, vec))
            for topic, vec in centroids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:top_k]]

    def _get_all_centroids(self) -> List[Tuple[str, np.ndarray]]:
        from .quantization import dequantize_packed
        import base64
        import json
        
        centroids = []
        try:
            with self.storage.get_conn() as conn:
                cur = conn.execute(
                    "SELECT topic, centroid_bits, centroid_scale, centroid_qmax, centroid_blob, centroid_shape FROM topics"
                )
                for topic, bits, scale, qmax, blob, shape_str in cur.fetchall():
                    try:
                        shape = json.loads(shape_str)
                    except (json.JSONDecodeError, TypeError):
                        shape = [384]
                    qobj = {
                        "bits": int(bits),
                        "scale": float(scale),
                        "qmax": int(qmax),
                        "shape": shape,
                        "data": base64.b64encode(blob).decode("utf-8")
                    }
                    centroids.append((topic, dequantize_packed(qobj)))
        except Exception as e:
            logger.warning(f"Failed to get centroids: {e}")
        
        return centroids