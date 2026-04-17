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

# Try to use turboquant if available, fallback to internal
try:
    from turboquant import quantize as _quantize, dequantize as _dequantize, Quantizer as _Quantizer
    quantize_packed = _quantize
    dequantize_packed = _dequantize
    Quantizer = _Quantizer
except ImportError:
    from .quantization import quantize_packed, dequantize_packed, Quantizer

from .core import TurboMemory, TurboMemoryConfig, ExclusionRules, QualityScore, VerificationResult, MemoryMetrics
from .storage import StorageManager, SQLitePool, RetryConfig, MigrationManager
from .retrieval import RetrievalEngine, cosine_similarity
from .formats import TMFFormat, TMFIndex, TMFVectorStore, TMFEventLog, validate_format
from .replication import TurboSync, create_sync
from .hybrid_search import HybridSearch, BM25, HybridSearchEngine

__version__ = "0.5.1"
__author__ = "Kubenew"
__description__ = "Lightweight semantic storage with TurboQuant compression"

# Alias for compatibility
cosine_sim = cosine_similarity

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
    "cosine_sim",
    
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