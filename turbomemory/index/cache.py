"""Index cache layer for hot topic acceleration."""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policy."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"


@dataclass
class CacheEntry:
    """Represents a cached index entry."""
    data: Any
    access_count: int = 1
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0


class IndexCache:
    """LRU/LFU/TTL cache for accelerated index access.
    
    Caches frequently accessed topic indexes and search results
    to reduce disk I/O and computation overhead.
    """
    
    def __init__(
        self,
        max_size: int = 100,
        max_memory_mb: int = 512,
        policy: CachePolicy = CachePolicy.LRU,
        ttl_seconds: Optional[float] = None,
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            if self.ttl_seconds and (time.time() - entry.created_at) > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None
            
            entry.access_count += 1
            entry.last_access = time.time()
            
            if self.policy == CachePolicy.LRU:
                self._cache.move_to_end(key)
            elif self.policy == CachePolicy.LFU:
                pass
            
            self._hits += 1
            return entry.data
    
    def set(self, key: str, value: Any, size_bytes: Optional[int] = None):
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                self._cache[key].data = value
                self._cache[key].access_count += 1
                self._cache[key].last_access = time.time()
                if size_bytes:
                    self._cache[key].size_bytes = size_bytes
                return
            
            entry = CacheEntry(
                data=value,
                size_bytes=size_bytes or 0,
            )
            
            self._evict_if_needed(entry.size_bytes)
            
            self._cache[key] = entry
            
            if self.policy == CachePolicy.LRU:
                self._cache.move_to_end(key)
    
    def _evict_if_needed(self, needed_bytes: int = 0):
        """Evict entries if cache is full."""
        current_memory = sum(e.size_bytes for e in self._cache.values())
        
        while (
            len(self._cache) >= self.max_size or
            current_memory + needed_bytes > self.max_memory_bytes
        ) and self._cache:
            if self.policy == CachePolicy.LRU:
                self._cache.popitem(last=False)
            elif self.policy == CachePolicy.LFU:
                min_entry = min(self._cache.items(), key=lambda x: x[1].access_count)
                del self._cache[min_entry[0]]
            elif self.policy == CachePolicy.TTL:
                oldest = min(self._cache.items(), key=lambda x: x[1].created_at)
                del self._cache[oldest[0]]
            
            current_memory = sum(e.size_bytes for e in self._cache.values())
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            current_memory = sum(e.size_bytes for e in self._cache.values())
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "memory_bytes": current_memory,
                "memory_mb": current_memory / (1024 * 1024),
                "policy": self.policy.value,
            }
    
    def warm_topic(self, topic: str, index_data: Any, size_bytes: int = 0):
        """Pre-warm cache with topic index data."""
        self.set(f"topic:{topic}", index_data, size_bytes)
    
    def get_topic(self, topic: str) -> Optional[Any]:
        """Get cached topic index."""
        return self.get(f"topic:{topic}")
    
    def warm_search_result(
        self,
        query_hash: str,
        result: Any,
        size_bytes: int = 0,
    ):
        """Cache search result."""
        self.set(f"search:{query_hash}", result, size_bytes)
    
    def get_search_result(self, query_hash: str) -> Optional[Any]:
        """Get cached search result."""
        return self.get(f"search:{query_hash}")


class HotTopicTracker:
    """Tracks hot topics for intelligent caching.
    
    Monitors query patterns to identify frequently accessed
    topics and prioritize their caching.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._topic_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def record_query(self, topics: List[str]):
        """Record topics accessed in a query."""
        with self._lock:
            for topic in topics:
                self._topic_counts[topic] = self._topic_counts.get(topic, 0) + 1
            
            if len(self._topic_counts) > self.window_size:
                sorted_topics = sorted(
                    self._topic_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                self._topic_counts = dict(sorted_topics[:self.window_size])
    
    def get_hot_topics(self, n: int = 10) -> List[str]:
        """Get top N hot topics."""
        with self._lock:
            sorted_topics = sorted(
                self._topic_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            return [topic for topic, _ in sorted_topics[:n]]
    
    def get_topics_by_access(self, threshold: int = 10) -> List[str]:
        """Get topics with access count above threshold."""
        with self._lock:
            return [
                topic for topic, count in self._topic_counts.items()
                if count >= threshold
            ]
    
    def reset(self):
        """Reset tracking data."""
        with self._lock:
            self._topic_counts.clear()


class IndexCacheManager:
    """Manages multiple index caches for different data types."""
    
    def __init__(self):
        self._caches: Dict[str, IndexCache] = {}
        self._lock = threading.Lock()
        
        self.create_cache("topic", max_size=50, max_memory_mb=256)
        self.create_cache("search", max_size=100, max_memory_mb=128)
        self.create_cache("vectors", max_size=20, max_memory_mb=512)
        
        self._hot_tracker = HotTopicTracker()
    
    def create_cache(
        self,
        name: str,
        max_size: int = 100,
        max_memory_mb: int = 512,
        policy: CachePolicy = CachePolicy.LRU,
        ttl_seconds: Optional[float] = None,
    ):
        """Create a named cache."""
        with self._lock:
            self._caches[name] = IndexCache(
                max_size=max_size,
                max_memory_mb=max_memory_mb,
                policy=policy,
                ttl_seconds=ttl_seconds,
            )
    
    def get_cache(self, name: str) -> Optional[IndexCache]:
        """Get cache by name."""
        return self._caches.get(name)
    
    def cache_topic(self, topic: str, data: Any, size_bytes: int = 0):
        """Cache topic data."""
        cache = self.get_cache("topic")
        if cache:
            cache.warm_topic(topic, data, size_bytes)
            self._hot_tracker.record_query([topic])
    
    def get_cached_topic(self, topic: str) -> Optional[Any]:
        """Get cached topic data."""
        cache = self.get_cache("topic")
        return cache.get_topic(topic) if cache else None
    
    def cache_search(
        self,
        query: str,
        result: Any,
        size_bytes: int = 0,
    ):
        """Cache search result."""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache = self.get_cache("search")
        if cache:
            cache.warm_search_result(query_hash, result, size_bytes)
    
    def get_cached_search(self, query: str) -> Optional[Any]:
        """Get cached search result."""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache = self.get_cache("search")
        return cache.get_search_result(query_hash) if cache else None
    
    def cache_vectors(self, key: str, vectors: Any, size_bytes: int = 0):
        """Cache vector data."""
        cache = self.get_cache("vectors")
        if cache:
            cache.set(key, vectors, size_bytes)
    
    def get_cached_vectors(self, key: str) -> Optional[Any]:
        """Get cached vector data."""
        cache = self.get_cache("vectors")
        return cache.get(key) if cache else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all caches."""
        stats = {"caches": {}, "hot_topics": self._hot_tracker.get_hot_topics()}
        
        for name, cache in self._caches.items():
            stats["caches"][name] = cache.get_stats()
        
        return stats
    
    def clear_all(self):
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()
        self._hot_tracker.reset()
    
    def prefetch_hot_topics(self, fetch_callback: Callable[[str], Any]):
        """Pre-fetch hot topics into cache."""
        hot_topics = self._hot_tracker.get_topics_by_access(threshold=5)
        cache = self.get_cache("topic")
        
        if cache:
            for topic in hot_topics:
                if cache.get_topic(topic) is None:
                    data = fetch_callback(topic)
                    if data:
                        cache.warm_topic(topic, data)