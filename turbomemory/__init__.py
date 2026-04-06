"""TurboMemory - Lightweight semantic storage engine with compressed embeddings.

TurboMemory is a lightweight semantic storage engine that combines:
- SQLite metadata indexing
- Packed embedding storage (4/6/8-bit TurboQuant compression)
- Append-only transaction logs for replication
- Hybrid search (BM25 + vector fusion)
- Portable TMF format for easy data portability

Usage:
    from turbomemory import TurboMemory
    
    tm = TurboMemory(root="./data")
    tm.add_memory("topic", "Your text here")
    results = tm.query("search query")
"""

from .core import TurboMemory, TurboMemoryConfig, ExclusionRules, QualityScore, VerificationResult, MemoryMetrics
from .quantization import quantize_packed, dequantize_packed, Quantizer
from .storage import StorageManager, SQLitePool, RetryConfig, MigrationManager
from .retrieval import RetrievalEngine, cosine_similarity
from .formats import TMFFormat, TMFIndex, TMFVectorStore, TMFEventLog, validate_format
from .replication import TurboSync, create_sync
from .hybrid_search import HybridSearch, BM25, HybridSearchEngine

__version__ = "0.5.0"
__author__ = "Kubenew"
__description__ = "Lightweight semantic storage with TurboQuant compression"

__all__ = [
    # Core
    "TurboMemory",
    "TurboMemoryConfig",
    "ExclusionRules",
    "QualityScore",
    "VerificationResult",
    "MemoryMetrics",
    
    # Quantization
    "quantize_packed",
    "dequantize_packed",
    "Quantizer",
    
    # Storage
    "StorageManager",
    "SQLitePool",
    "RetryConfig",
    "MigrationManager",
    
    # Retrieval
    "RetrievalEngine",
    "cosine_similarity",
    
    # TMF Format
    "TMFFormat",
    "TMFIndex",
    "TMFVectorStore",
    "TMFEventLog",
    "validate_format",
    
    # Replication
    "TurboSync",
    "create_sync",
    
    # Hybrid Search
    "HybridSearch",
    "BM25",
    "HybridSearchEngine",
]