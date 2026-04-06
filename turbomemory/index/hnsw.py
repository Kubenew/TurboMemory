"""HNSW index for accelerated vector search."""

import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HNSWConfig:
    """Configuration for HNSW index."""
    dim: int = 384
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    max_elements: int = 100000
    storage_path: Optional[str] = None


@dataclass
class HNSWEntry:
    """Represents an entry in HNSW graph."""
    id: int
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class HNSWIndex:
    """Hierarchical Navigable Small World index for fast vector search.
    
    This is a simplified implementation that provides HNSW-like functionality
    using numpy for distance calculations. For production use, consider using
    hnswlib or faiss with HNSW support.
    """
    
    def __init__(self, config: Optional[HNSWConfig] = None):
        self.config = config or HNSWConfig()
        self._entries: Dict[int, HNSWEntry] = {}
        self._level_mult = 1 / np.log(self.config.m)
        self._graph: Dict[int, Dict[int, List[int]]] = {}
        self._entry_point: Optional[int] = None
        self._max_level = 0
    
    def _get_random_level(self) -> float:
        """Get random level for new element."""
        return -np.log(np.random.random()) * self._level_mult
    
    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine distance."""
        return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    
    def _search_layer(
        self,
        query: np.ndarray,
        ep: int,
        ef: int,
        level: int,
    ) -> List[Tuple[int, float]]:
        """Search in a single HNSW layer."""
        visited = {ep}
        candidates = [(ep, self._distance(query, self._entries[ep].vector))]
        results = []
        
        while candidates:
            _, current = min(candidates, key=lambda x: x[1])
            candidates.remove((current, self._distance(query, self._entries[current].vector)))
            
            results.append((current, self._distance(query, self._entries[current].vector)))
            
            if len(results) > ef:
                break
            
            neighbors = self._graph.get(current, {}).get(level, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query, self._entries[neighbor].vector)
                    
                    if results and dist > results[-1][1] and len(results) >= ef:
                        continue
                    
                    candidates.append((neighbor, dist))
        
        return results[:ef]
    
    def add_vector(
        self,
        vector: np.ndarray,
        id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a vector to the index."""
        if id is None:
            id = len(self._entries)
        
        entry = HNSWEntry(
            id=id,
            vector=vector.astype(np.float32),
            metadata=metadata or {},
        )
        
        self._entries[id] = entry
        
        if self._entry_point is None:
            self._entry_point = id
            self._graph[id] = {l: [] for l in range(1)}
            return id
        
        level = int(self._get_random_level())
        self._max_level = max(self._max_level, level)
        
        for l in range(level + 1):
            if l not in self._graph.get(self._entry_point, {}):
                if self._entry_point not in self._graph:
                    self._graph[self._entry_point] = {}
                self._graph[self._entry_point][l] = []
        
        ep = self._entry_point
        for l in range(self._max_level, level, -1):
            ep_candidates = self._search_layer(vector, ep, 1, l)
            ep = ep_candidates[0][0] if ep_candidates else ep
        
        for l in range(level, -1, -1):
            candidates = self._search_layer(vector, ep, self.config.ef_construction, l)
            neighbors = [c[0] for c in candidates[:self.config.m]]
            
            if id not in self._graph:
                self._graph[id] = {}
            if l not in self._graph[id]:
                self._graph[id][l] = []
            
            self._graph[id][l].extend(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in self._graph:
                    self._graph[neighbor] = {}
                if l not in self._graph[neighbor]:
                    self._graph[neighbor][l] = []
                
                if id not in self._graph[neighbor][l]:
                    self._graph[neighbor][l].append(id)
            
            ep = candidates[0][0] if candidates else ep
        
        if level > self._max_level:
            self._entry_point = id
        
        return id
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
    ) -> List[Tuple[int, float, Dict]]:
        """Search for k nearest neighbors."""
        if not self._entries:
            return []
        
        ef = ef or self.config.ef_search
        
        ep = self._entry_point or list(self._entries.keys())[0]
        
        for l in range(self._max_level, 0, -1):
            candidates = self._search_layer(query, ep, 1, l)
            ep = candidates[0][0] if candidates else ep
        
        results = self._search_layer(query, ep, ef, 0)
        
        results = sorted(results, key=lambda x: x[1])[:k]
        
        return [
            (id, dist, self._entries[id].metadata)
            for id, dist in results
            if id in self._entries
        ]
    
    def delete_vector(self, id: int) -> bool:
        """Delete a vector from the index."""
        if id not in self._entries:
            return False
        
        del self._entries[id]
        
        if id in self._graph:
            for level, neighbors in self._graph[id].items():
                for neighbor in neighbors:
                    if neighbor in self._graph and level in self._graph[neighbor]:
                        if id in self._graph[neighbor][level]:
                            self._graph[neighbor][level].remove(id)
            del self._graph[id]
        
        if self._entry_point == id:
            self._entry_point = list(self._entries.keys())[0] if self._entries else None
        
        return True
    
    def get_vector(self, id: int) -> Optional[np.ndarray]:
        """Get vector by ID."""
        if id in self._entries:
            return self._entries[id].vector
        return None
    
    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": {
                "dim": self.config.dim,
                "m": self.config.m,
                "ef_construction": self.config.ef_construction,
                "ef_search": self.config.ef_search,
                "max_elements": self.config.max_elements,
            },
            "entries": {
                str(k): {
                    "vector": v.vector.tolist(),
                    "metadata": v.metadata,
                }
                for k, v in self._entries.items()
            },
            "graph": {
                str(k): {str(l): neighbors for l, neighbors in v.items()}
                for k, v in self._graph.items()
            },
            "entry_point": self._entry_point,
            "max_level": self._max_level,
        }
        
        with open(path / "hnsw.json", "w") as f:
            json.dump(data, f)
        
        logger.info(f"Saved HNSW index with {len(self._entries)} entries to {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        path = Path(path) / "hnsw.json"
        
        if not path.exists():
            raise FileNotFoundError(f"HNSW index not found at {path}")
        
        with open(path) as f:
            data = json.load(f)
        
        self.config = HNSWConfig(**data["config"])
        self._entries = {
            int(k): HNSWEntry(
                id=int(k),
                vector=np.array(v["vector"], dtype=np.float32),
                metadata=v["metadata"],
            )
            for k, v in data["entries"].items()
        }
        self._graph = {
            int(k): {int(l): neighbors for l, neighbors in v.items()}
            for k, v in data["graph"].items()
        }
        self._entry_point = data["entry_point"]
        self._max_level = data["max_level"]
        
        logger.info(f"Loaded HNSW index with {len(self._entries)} entries from {path.parent}")
    
    @property
    def size(self) -> int:
        """Return number of entries in index."""
        return len(self._entries)
    
    def clear(self):
        """Clear all entries."""
        self._entries.clear()
        self._graph.clear()
        self._entry_point = None
        self._max_level = 0