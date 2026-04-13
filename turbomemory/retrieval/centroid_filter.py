"""TurboMemory v3.1 Topic Centroid Filter."""

import numpy as np
from .dotprod import dot_packed


BITS_MAP = {"q4": 4, "q6": 6, "q8": 8}


class TopicCentroidFilter:
    """Fast topic selection using centroid similarity."""
    
    def __init__(self, store):
        self.store = store
    
    def get_topic_centroids(self):
        """Get all topic centroids from database."""
        cur = self.store.conn.cursor()
        cur.execute("""
            SELECT topic, centroid_blob, centroid_dim, centroid_dtype
            FROM topic_centroids
        """)
        rows = cur.fetchall()
        return [(topic, blob, dim, dtype) for topic, blob, dim, dtype in rows]
    
    def top_topics(self, query_vec: np.ndarray, top_n: int = 10):
        """Get top N topics by centroid similarity."""
        scored = []
        for topic, blob, dim, dtype in self.get_topic_centroids():
            try:
                score = dot_packed(query_vec, blob, dim, dtype)
                scored.append((score, topic))
            except Exception:
                continue
        
        scored.sort(reverse=True)
        return [t for _, t in scored[:top_n]]
    
    def update_centroid(self, topic: str, embed_vec: np.ndarray, quantization: str = "q6"):
        """Update topic centroid with new embedding."""
        bits = BITS_MAP.get(quantization, 6)
        
        from .qpack import pack_q
        blob = pack_q(embed_vec, bits)
        
        cur = self.store.conn.cursor()
        import time
        now = int(time.time())
        
        cur.execute("""
            INSERT OR REPLACE INTO topic_centroids 
            (topic, centroid_dim, centroid_dtype, centroid_blob, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (topic, embed_vec.shape[0], quantization, blob, now))
        
        self.store.conn.commit()