from dataclasses import dataclass

@dataclass
class TurboMemoryConfig:
    """TurboMemory v3.1 configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    quantization: str = "q6"   # q4|q6|q8|fp16|fp32
    use_gpu: bool = False

    # FAISS settings
    enable_faiss: bool = False
    faiss_gpu: bool = False
    faiss_nlist: int = 4096
    faiss_m: int = 32
    faiss_nbits: int = 8

    # Storage settings
    enable_bm25: bool = True
    enable_segments: bool = True
    enable_wal: bool = True

    # Topic centroids
    enable_topic_centroids: bool = True
    centroid_top_topics: int = 12

    # Graph & policy
    enable_contradiction_graph: bool = True
    decay_half_life_days: float = 90.0

    # Paths
    sqlite_file: str = "index.sqlite"
    wal_dir: str = "tmlog"
    segments_dir: str = "segments"
    faiss_index_file: str = "faiss.index"