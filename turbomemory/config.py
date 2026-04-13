from dataclasses import dataclass

@dataclass
class TurboMemoryConfig:
    """TurboMemory v3 configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    quantization: str = "q6"   # q4|q6|q8|fp16|fp32
    use_gpu: bool = False

    enable_faiss: bool = False
    faiss_gpu: bool = False

    enable_bm25: bool = True
    enable_segments: bool = True
    enable_wal: bool = True

    enable_contradiction_graph: bool = True
    decay_half_life_days: float = 90.0

    sqlite_file: str = "index.sqlite"
    wal_dir: str = "tmlog"
    segments_dir: str = "segments"