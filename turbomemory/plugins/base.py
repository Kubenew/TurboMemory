"""Base plugin interfaces for TurboMemory extensibility."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Plugin(ABC):
    """Base class for all TurboMemory plugins."""

    name: str = "base_plugin"
    version: str = "0.1.0"
    description: str = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def initialize(self) -> None:
        """Called when plugin is loaded. Override for setup."""
        pass

    def cleanup(self) -> None:
        """Called when plugin is unloaded. Override for cleanup."""
        pass


class QualityScorer(Plugin):
    """Custom quality scoring plugin.

    Override this to implement your own memory quality assessment logic.
    The default scorer uses confidence + freshness + specificity + verification.

    Example:
        class MyScorer(QualityScorer):
            name = "my_scorer"

            def compute_score(self, chunk) -> float:
                # Your custom scoring logic
                return 0.0 to 1.0
    """

    name = "default_quality_scorer"

    @abstractmethod
    def compute_score(self, chunk: Dict[str, Any]) -> float:
        """Compute quality score for a chunk. Returns 0.0 to 1.0."""
        pass

    def compute_components(self, chunk: Dict[str, Any]) -> Dict[str, float]:
        """Return individual component scores. Override for custom components."""
        score = self.compute_score(chunk)
        return {"overall": score}


class EmbeddingProvider(Plugin):
    """Custom embedding provider plugin.

    Override this to use a different embedding model or provider.
    The default provider uses sentence-transformers.

    Example:
        class OpenAIEmbeddings(EmbeddingProvider):
            name = "openai_embeddings"

            def encode(self, texts) -> np.ndarray:
                # Call OpenAI API
                return embeddings
    """

    name = "default_embedding_provider"

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings. Returns (n, d) array."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a query text. Override if query encoding differs."""
        return self.encode([text])[0]


class StorageBackend(Plugin):
    """Custom storage backend plugin.

    Override this to use a different storage system (Redis, PostgreSQL, etc.).
    The default backend uses SQLite + JSON files.

    Example:
        class RedisStorage(StorageBackend):
            name = "redis_storage"

            def save_chunk(self, topic, chunk) -> None:
                # Save to Redis
                pass
    """

    name = "default_storage_backend"

    @abstractmethod
    def save_chunk(self, topic: str, chunk: Dict[str, Any]) -> None:
        """Save a memory chunk."""
        pass

    @abstractmethod
    def load_chunks(self, topic: str) -> List[Dict[str, Any]]:
        """Load all chunks for a topic."""
        pass

    @abstractmethod
    def delete_chunk(self, topic: str, chunk_id: str) -> None:
        """Delete a memory chunk."""
        pass

    @abstractmethod
    def search_chunks(self, query_embedding: np.ndarray, k: int, topics: Optional[List[str]] = None) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Search chunks by embedding similarity."""
        pass

    def save_topic_metadata(self, topic: str, metadata: Dict[str, Any]) -> None:
        """Save topic metadata. Optional override."""
        pass

    def load_topic_metadata(self, topic: str) -> Dict[str, Any]:
        """Load topic metadata. Optional override."""
        return {}


class VerificationStrategy(Plugin):
    """Custom verification strategy plugin.

    Override this to implement your own memory verification logic.
    The default strategy uses cross-referencing across topics.

    Example:
        class LLMVerification(VerificationStrategy):
            name = "llm_verification"

            def verify(self, query, results) -> List[VerificationResult]:
                # Use an LLM to verify results
                pass
    """

    name = "default_verification_strategy"

    @abstractmethod
    def verify(self, query_text: str, results: List[Tuple[float, str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Verify query results. Returns list of verification dicts."""
        pass
