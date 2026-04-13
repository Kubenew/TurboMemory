"""Core TurboMemory engine with lazy loading and async support."""

import os
import re
import json
import time
import logging
import threading
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple, Iterator, Set, Callable
import importlib.util
from pathlib import Path
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import for sentence transformers
SentenceTransformer: Optional[Callable] = None


def _lazy_load_model():
    """Lazy load the sentence transformer model."""
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    return SentenceTransformer


class LazyModel:
    """Lazy loading wrapper for the embedding model."""

    _instance: Optional["LazyModel"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> "LazyModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name)
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    ST = _lazy_load_model()
                    logger.info(f"Loading model: {self.model_name}")
                    self._model = ST(self.model_name)
                    logger.info("Model loaded successfully")
        return self._model

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        return self.model.encode(texts, **kwargs)

    def encode_async(self, texts: List[str], **kwargs) -> asyncio.Future:
        """Async wrapper for encode."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, self.encode, texts, **kwargs)

    def reset(self) -> None:
        """Reset the model (for testing or memory management)."""
        with self._load_lock:
            if self._model is not None:
                del self._model
                self._model = None


@dataclass
class ExclusionRules:
    """Configurable rules for what NOT to store in memory."""
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r"^(debug|trace|log|info|warn|error)\s*[:\[]",
        r"^\s*(at\s+\w+|File\s+\".+\", line \d+)",
        r"^\s*(git\s|git commit|git push|git pull)",
        r"^\s*(PR\s*#?\d+|pull request\s*#?\d+)",
        r"^\s*(import |from .+ import |require\(|package )",
        r"^\s*(class |def |function |const |let |var )",
        r"^\s*(export |module\.exports)",
        r"^\s*(<\?xml|<!DOCTYPE|<html|<head|<body)",
        r"^\s*(-----BEGIN|-----END)",
    ])
    blocked_topics: List[str] = field(default_factory=lambda: [
        "debug", "trace", "logs", "stack_trace", "error_dump",
        "git_history", "pr_history", "code_structure",
        "dependencies", "file_tree",
    ])
    min_text_length: int = 10
    max_text_length: int = 5000
    block_code_snippets: bool = True
    block_debug_output: bool = True
    block_file_paths: bool = False
    block_secrets: bool = True

    def should_exclude(self, text: str, topic: str = "") -> Tuple[bool, str]:
        """Check if text should be excluded. Returns (should_exclude, reason)."""
        text_lower = text.lower().strip()
        topic_lower = topic.lower().strip()

        if len(text_lower) < self.min_text_length:
            return True, f"text too short ({len(text_lower)} chars)"

        if len(text_lower) > self.max_text_length:
            return True, f"text too long ({len(text_lower)} chars)"

        if topic_lower in self.blocked_topics:
            return True, f"blocked topic: {topic}"

        if self.block_secrets:
            secret_patterns = [
                r"(password|passwd|pwd)\s*[=:]\s*\S+",
                r"(api[_-]?key|apikey)\s*[=:]\s*\S+",
                r"(secret|token)\s*[=:]\s*\S+",
                r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
                r"(bearer\s+)?token\s*[=:]\s*[A-Za-z0-9_\-.]+",
                r"sk-[A-Za-z0-9]{20,}",
                r"ghp_[A-Za-z0-9]{36}",
            ]
            for pattern in secret_patterns:
                if re.search(pattern, text_lower):
                    return True, "potential secret detected"

        if self.block_debug_output:
            debug_patterns = [
                r"^(debug|trace|log|info|warn|error)\s*[:\[]",
                r"^\s*(at\s+\w+|File\s+\".+\", line \d+)",
                r"^\s*stack trace",
                r"^\s*traceback",
                r"^\s*exception\s+in\s+thread",
            ]
            for pattern in debug_patterns:
                if re.search(pattern, text_lower):
                    return True, "debug/trace output detected"

        if self.block_code_snippets:
            code_indicators = [
                r"^\s*(import |from .+ import |require\(|package )",
                r"^\s*(class |def |function |const |let |var )\s+\w+",
                r"^\s*(export |module\.exports)",
                r"^\s*(public |private |protected |static )\s+\w+",
            ]
            code_matches = sum(1 for p in code_indicators if re.search(p, text_lower))
            if code_matches >= 2:
                return True, "code snippet detected"

        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower):
                return True, f"blocked pattern: {pattern}"

        return False, ""


@dataclass
class QualityScore:
    """Quality assessment for a memory chunk."""
    overall: float = 0.0
    confidence_component: float = 0.0
    freshness_component: float = 0.0
    specificity_component: float = 0.0
    verification_component: float = 0.0
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall, 3),
            "confidence": round(self.confidence_component, 3),
            "freshness": round(self.freshness_component, 3),
            "specificity": round(self.specificity_component, 3),
            "verification": round(self.verification_component, 3),
            "flags": self.flags,
        }


@dataclass
class VerificationResult:
    """Result of memory verification."""
    chunk_id: str
    topic: str
    verified: bool = False
    verification_score: float = 0.0
    cross_references: int = 0
    agreement_score: float = 0.0
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "topic": self.topic,
            "verified": self.verified,
            "verification_score": round(self.verification_score, 3),
            "cross_references": self.cross_references,
            "agreement_score": round(self.agreement_score, 3),
            "flags": self.flags,
        }


@dataclass
class MemoryMetrics:
    """Comprehensive memory system metrics."""
    total_topics: int = 0
    total_chunks: int = 0
    avg_confidence: float = 0.0
    avg_staleness: float = 0.0
    avg_quality: float = 0.0
    expired_chunks: int = 0
    contradicted_chunks: int = 0
    verified_chunks: int = 0
    storage_bytes: int = 0
    consolidation_runs: int = 0
    last_consolidation: Optional[str] = None
    chunks_removed_by_consolidation: int = 0
    topic_health: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_topics": self.total_topics,
            "total_chunks": self.total_chunks,
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_staleness": round(self.avg_staleness, 3),
            "avg_quality": round(self.avg_quality, 3),
            "expired_chunks": self.expired_chunks,
            "contradicted_chunks": self.contradicted_chunks,
            "verified_chunks": self.verified_chunks,
            "storage_bytes": self.storage_bytes,
            "consolidation_runs": self.consolidation_runs,
            "last_consolidation": self.last_consolidation,
            "chunks_removed_by_consolidation": self.chunks_removed_by_consolidation,
            "topic_health": {k: round(v, 3) for k, v in self.topic_health.items()},
        }


@dataclass
class TurboMemoryConfig:
    """Configuration for TurboMemory."""
    root: str = "turbomemory_data"
    model_name: str = "all-MiniLM-L6-v2"
    quantization: str = "q6"  # q4, q6, q8, fp16, fp32
    use_gpu: bool = False
    default_bits: int = 6
    pool_size: int = 5
    default_ttl_days: Optional[float] = None
    max_chunks_per_topic: int = 300
    contradiction_decay: float = 0.6
    min_confidence: float = 0.0
    lazy_load_model: bool = True
    enable_verification: bool = True
    verification_threshold: float = 0.7
    min_cross_refs: int = 2
    enable_exclusions: bool = True
    exclusion_rules: Optional[ExclusionRules] = None
    min_quality_threshold: float = 0.3
    quality_decay_rate: float = 0.01
    use_faiss: bool = True
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    enable_consolidator: bool = False

    def set_embedding_backend(self, backend: "EmbeddingBackend") -> None:
        """Set custom embedding backend."""
        self._embedding_backend = backend


class EmbeddingBackend:
    """Abstract embedding backend - implement for custom providers."""
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        raise NotImplementedError

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text."""
        return self.encode([text])[0]


class SentenceTransformerBackend(EmbeddingBackend):
    """Default sentence-transformers backend."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = False):
        from sentence_transformers import SentenceTransformer
        device = "cuda" if use_gpu and importlib.util.find_spec("torch") else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    @classmethod
    def from_file(cls, path: str) -> "TurboMemoryConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_file(self, path: str) -> None:
        data = {k: v for k, v in self.__dict__.items() 
                if k not in ('exclusion_rules', 'encryption_key')}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_topic_filename(topic: str) -> str:
    return topic.replace(".", "_").replace("/", "_").replace("\\", "_")


def sha1_text(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def compute_quality_score(
    confidence: float,
    staleness: float,
    text: str,
    verified: bool = False,
    verification_score: float = 0.0,
) -> QualityScore:
    flags = []
    conf_score = max(0.0, min(1.0, confidence))
    fresh_score = max(0.0, 1.0 - staleness)

    words = text.split()
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))
    lexical_diversity = unique_words / max(1, word_count)

    has_numbers = bool(re.search(r'\d+\.?\d*', text))
    has_dates = bool(re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', text))
    has_specific_terms = word_count >= 5 and lexical_diversity > 0.5

    specific_score = 0.0
    if has_specific_terms:
        specific_score += 0.4
    if has_numbers:
        specific_score += 0.3
    if has_dates:
        specific_score += 0.3
    specific_score = min(1.0, specific_score)

    if word_count < 5:
        flags.append("too_short")
    if lexical_diversity < 0.3:
        flags.append("low_diversity")
    if not has_specific_terms and not has_numbers:
        flags.append("vague")

    verif_score = verification_score if verified else 0.0

    overall = (
        conf_score * 0.30 +
        fresh_score * 0.25 +
        specific_score * 0.25 +
        verif_score * 0.20
    )

    return QualityScore(
        overall=overall,
        confidence_component=conf_score,
        freshness_component=fresh_score,
        specificity_component=specific_score,
        verification_component=verif_score,
        flags=flags,
    )


class EncryptionManager:
    """Optional encryption for sensitive data at rest."""

    def __init__(self, key: Optional[str] = None):
        self.key = key
        self._cipher = None

    def is_enabled(self) -> bool:
        return self.key is not None

    def encrypt(self, data: bytes) -> bytes:
        if not self.is_enabled():
            return data
        try:
            from cryptography.fernet import Fernet
            if self._cipher is None:
                self._cipher = Fernet(self.key.encode())
            return self._cipher.encrypt(data)
        except ImportError:
            logger.warning("cryptography not installed, storing unencrypted")
            return data

    def decrypt(self, data: bytes) -> bytes:
        if not self.is_enabled():
            return data
        try:
            from cryptography.fernet import Fernet
            if self._cipher is None:
                self._cipher = Fernet(self.key.encode())
            return self._cipher.decrypt(data)
        except Exception:
            return data


class AuditLogger:
    """Audit logging for sensitive operations."""

    def __init__(self, storage_manager):
        self.storage = storage_manager

    def log(
        self,
        operation: str,
        topic: Optional[str] = None,
        details: Optional[str] = None,
        user_id: Optional[str] = None,
        success: bool = True
    ) -> None:
        try:
            with self.storage.get_conn() as conn:
                conn.execute(
                    """INSERT INTO audit_log(ts, operation, user_id, topic, details, success)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (now_iso(), operation, user_id, topic, details, int(success))
                )
        except Exception as e:
            logger.warning(f"Failed to log audit: {e}")

    def get_logs(
        self,
        operation: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            with self.storage.get_conn() as conn:
                if operation:
                    cur = conn.execute(
                        "SELECT * FROM audit_log WHERE operation = ? ORDER BY ts DESC LIMIT ?",
                        (operation, limit)
                    )
                else:
                    cur = conn.execute(
                        "SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?",
                        (limit,)
                    )
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get audit logs: {e}")
            return []


class GracefulShutdown:
    """Graceful shutdown handler for the memory system."""

    def __init__(self):
        self._running = True
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []

    def register_callback(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
            for cb in self._callbacks:
                try:
                    cb()
                except Exception as e:
                    logger.warning(f"Shutdown callback failed: {e}")

    def is_running(self) -> bool:
        with self._lock:
            return self._running


from .quantization import quantize_packed, dequantize_packed
from .storage import StorageManager


class TurboMemory:
    """TurboMemory v0.5 - Refactored with modular architecture."""

    def __init__(
        self,
        root: str = "turbomemory_data",
        model_name: str = "all-MiniLM-L6-v2",
        quantization: str = "q6",
        use_gpu: bool = False,
        config: Optional[TurboMemoryConfig] = None,
    ):
        """Initialize TurboMemory.
        
        Args:
            root: base data directory
            model_name: name of embedding model to use
            quantization: quantization type (q4|q6|q8|fp16|fp32)
            use_gpu: whether to use GPU for embeddings
            config: TurboMemoryConfig (optional)
        """
        if config:
            self.config = config
        else:
            self.config = TurboMemoryConfig(
                root=root, 
                model_name=model_name,
                quantization=quantization,
                use_gpu=use_gpu,
            )
        if self.config.exclusion_rules is None:
            self.config.exclusion_rules = ExclusionRules()

        self.root = self.config.root
        self.model_name = self.config.model_name
        self.quantization = self.config.quantization
        self.use_gpu = self.config.use_gpu

        # Lazy model loading
        if self.config.lazy_load_model:
            self._model = LazyModel.get_instance(model_name)
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)

        # Storage and encryption
        self.db_dir = os.path.join(self.root, "db")
        self.db_path = os.path.join(self.db_dir, "index.sqlite")
        ensure_dir(self.db_dir)
        
        self._storage = StorageManager(self.db_path, pool_size=self.config.pool_size)
        self._encryption = EncryptionManager(self.config.encryption_key)
        self._audit = AuditLogger(self._storage)
        self._shutdown = GracefulShutdown()

        # File paths
        self.topics_dir = os.path.join(self.root, "topics")
        self.sessions_dir = os.path.join(self.root, "sessions")
        self.memory_index_path = os.path.join(self.root, "MEMORY.md")
        self.config_path = os.path.join(self.root, "config.json")
        
        ensure_dir(self.topics_dir)
        ensure_dir(self.sessions_dir)

        if not os.path.exists(self.memory_index_path):
            with open(self.memory_index_path, "w", encoding="utf-8") as f:
                f.write("# TurboMemory Index (v0.5)\n")

        if not os.path.exists(self.config_path):
            self.config.to_file(self.config_path)

        # Register shutdown callback
        self._shutdown.register_callback(self.close)

    @property
    def model(self):
        if isinstance(self._model, LazyModel):
            return self._model
        return self._model

    def _check_exclusions(self, text: str, topic: str) -> Tuple[bool, str]:
        if not self.config.enable_exclusions:
            return False, ""
        return self.config.exclusion_rules.should_exclude(text, topic)

    def _log_exclusion(self, topic: str, reason: str, text: str) -> None:
        with self._storage.get_conn() as conn:
            conn.execute(
                "INSERT INTO exclusion_log(ts, topic, reason, text_preview) VALUES (?, ?, ?, ?)",
                (now_iso(), topic, reason, text[:200])
            )

    def _topic_path(self, topic: str) -> str:
        return os.path.join(self.topics_dir, safe_topic_filename(topic) + ".tmem")

    def load_topic(self, topic: str) -> Dict[str, Any]:
        path = self._topic_path(topic)
        if not os.path.exists(path):
            return {"topic": topic, "created": now_iso(), "updated": now_iso(), "centroid_q": None, "chunks": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load topic {topic}: {e}")
            return {"topic": topic, "created": now_iso(), "updated": now_iso(), "centroid_q": None, "chunks": []}

    def save_topic(self, topic_data: Dict[str, Any]) -> None:
        topic_data["updated"] = now_iso()
        path = self._topic_path(topic_data["topic"])
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(topic_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Failed to save topic {topic_data['topic']}: {e}")
            raise

    def _is_contradiction(self, old_text: str, new_text: str) -> bool:
        old_l = old_text.lower()
        new_l = new_text.lower()
        
        neg_words = ["not", "never", "no", "without", "can't", "cannot", "won't", "doesn't", "isn't", "aren't", "wasn't", "weren't"]
        old_neg = any(w in old_l for w in neg_words)
        new_neg = any(w in new_l for w in neg_words)

        def extract_numbers(s: str) -> List[float]:
            return [float(x) for x in re.findall(r'\b\d+\.?\d*\b', s)]

        o_nums = extract_numbers(old_l)
        n_nums = extract_numbers(new_l)

        if o_nums and n_nums:
            for on in o_nums:
                for nn in n_nums:
                    if abs(on - nn) / (max(abs(on), abs(nn)) + 1e-9) > 0.1:
                        return True

        if old_neg != new_neg:
            ow = set(w for w in old_l.split() if len(w) > 3)
            nw = set(w for w in new_l.split() if len(w) > 3)
            if len(ow.intersection(nw)) >= 3:
                return True

        date_pattern = r'\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b'
        old_dates = re.findall(date_pattern, old_l)
        new_dates = re.findall(date_pattern, new_l)
        if old_dates and new_dates and set(old_dates) != set(new_dates):
            ow = set(w for w in old_l.split() if len(w) > 3)
            nw = set(w for w in new_l.split() if len(w) > 3)
            if len(ow.intersection(nw)) >= 2:
                return True

        return False

    def add_memory(
        self,
        topic: str,
        text: str,
        confidence: float = 0.8,
        bits: Optional[int] = None,
        source_ref: Optional[str] = None,
        ttl_days: Optional[float] = None,
    ) -> Optional[str]:
        excluded, reason = self._check_exclusions(text, topic)
        if excluded:
            logger.info(f"Excluded memory: {reason}")
            self._log_exclusion(topic, reason, text)
            self._audit.log("add_memory_excluded", topic, reason)
            return None

        bits = bits or self.config.default_bits
        if bits not in (4, 6, 8):
            raise ValueError("bits must be 4, 6, or 8")

        # Batch encoding support
        emb = self.model.encode([text])[0].astype(np.float32)
        emb_q = quantize_packed(emb, bits=bits)

        topic_data = self.load_topic(topic)

        for c in topic_data["chunks"]:
            if self._is_contradiction(c.get("text", ""), text):
                c["confidence"] = float(c.get("confidence", 0.5) * self.config.contradiction_decay)
                c["staleness"] = float(min(1.0, c.get("staleness", 0.0) + 0.25))

        chunk_id = f"c{len(topic_data['chunks'])+1:04d}"
        entropy_score = float(min(1.0, len(text) / 500.0))

        ttl_ts = None
        ttl = ttl_days or self.config.default_ttl_days
        if ttl:
            ttl_ts = (datetime.now(timezone.utc) + timedelta(days=ttl)).isoformat()

        quality = compute_quality_score(confidence, 0.0, text)

        chunk = {
            "chunk_id": chunk_id,
            "timestamp": now_iso(),
            "confidence": float(confidence),
            "importance": float(confidence),
            "access_count": 0,
            "last_accessed": None,
            "staleness": 0.0,
            "entropy": entropy_score,
            "text": text,
            "embedding_q": emb_q,
            "source_refs": [source_ref] if source_ref else [],
            "ttl_ts": ttl_ts,
            "quality_score": quality.overall,
            "verified": False,
            "verification_score": 0.0,
            "cross_refs": 0,
        }

        topic_data["chunks"].append(chunk)

        if topic_data["chunks"]:
            all_embs = [dequantize_packed(c["embedding_q"]) for c in topic_data["chunks"]]
            centroid = np.mean(np.vstack(all_embs), axis=0)
            topic_data["centroid_q"] = quantize_packed(centroid, bits=8)

        self.save_topic(topic_data)

        file_ref = os.path.relpath(self._topic_path(topic), self.root)
        self._upsert_topic_to_db(topic_data, file_ref)
        self._upsert_chunk_to_db(topic, chunk)

        self._audit.log("add_memory", topic, f"chunk_id={chunk_id}")
        return chunk_id

    def _upsert_topic_to_db(self, topic_data: Dict[str, Any], file_ref: str) -> None:
        import base64
        import sqlite3
        
        centroid_q = topic_data.get("centroid_q")
        if centroid_q is None:
            return

        blob = base64.b64decode(centroid_q["data"])
        shape_str = json.dumps(centroid_q.get("shape", [384]))

        with self._storage.get_conn() as conn:
            conn.execute("""
                INSERT INTO topics(topic, file_ref, updated, chunk_count, centroid_bits, centroid_scale, centroid_qmax, centroid_blob, centroid_shape)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    file_ref=excluded.file_ref, updated=excluded.updated,
                    chunk_count=excluded.chunk_count, centroid_bits=excluded.centroid_bits,
                    centroid_scale=excluded.centroid_scale, centroid_qmax=excluded.centroid_qmax,
                    centroid_blob=excluded.centroid_blob, centroid_shape=excluded.centroid_shape
            """, (
                topic_data["topic"], file_ref,
                topic_data.get("updated", now_iso()),
                len(topic_data.get("chunks", [])),
                int(centroid_q["bits"]), float(centroid_q["scale"]),
                int(centroid_q["qmax"]), sqlite3.Binary(blob), shape_str
            ))

    def _upsert_chunk_to_db(self, topic: str, chunk: Dict[str, Any]) -> None:
        import base64
        import sqlite3
        
        emb_q = chunk["embedding_q"]
        blob = base64.b64decode(emb_q["data"])
        text = chunk.get("text", "")
        preview = text[:200]
        th = sha1_text(text)
        shape_str = json.dumps(emb_q.get("shape", [384]))

        with self._storage.get_conn() as conn:
            conn.execute("""
                INSERT INTO chunks(
                    topic, chunk_id, confidence, staleness, entropy, ts, text_hash, text_preview,
                    emb_bits, emb_scale, emb_qmax, emb_blob, emb_shape, ttl_ts,
                    quality_score, verified, verification_score, cross_refs, source_ref
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic, chunk_id) DO UPDATE SET
                    confidence=excluded.confidence, staleness=excluded.staleness,
                    entropy=excluded.entropy, ts=excluded.ts, text_hash=excluded.text_hash,
                    text_preview=excluded.text_preview, emb_bits=excluded.emb_bits,
                    emb_scale=excluded.emb_scale, emb_qmax=excluded.emb_qmax,
                    emb_blob=excluded.emb_blob, emb_shape=excluded.emb_shape, ttl_ts=excluded.ttl_ts,
                    quality_score=excluded.quality_score, verified=excluded.verified,
                    verification_score=excluded.verification_score, cross_refs=excluded.cross_refs,
                    source_ref=excluded.source_ref
            """, (
                topic, chunk["chunk_id"],
                float(chunk.get("confidence", 0.5)),
                float(chunk.get("staleness", 0.0)),
                float(chunk.get("entropy", 0.0)),
                chunk.get("timestamp", now_iso()),
                th, preview,
                int(emb_q["bits"]), float(emb_q["scale"]), int(emb_q["qmax"]),
                sqlite3.Binary(blob), shape_str,
                chunk.get("ttl_ts"),
                float(chunk.get("quality_score", 0.5)),
                int(chunk.get("verified", False)),
                float(chunk.get("verification_score", 0.0)),
                int(chunk.get("cross_refs", 0)),
                chunk.get("source_refs", [None])[0] if chunk.get("source_refs") else None,
            ))

    def _get_all_topic_centroids(self) -> List[Tuple[str, np.ndarray]]:
        import base64
        import json
        
        centroids = []
        try:
            with self._storage.get_conn() as conn:
                cur = conn.execute(
                    "SELECT topic, centroid_bits, centroid_scale, centroid_qmax, centroid_blob, centroid_shape FROM topics"
                )
                for topic, bits, scale, qmax, blob, shape_str in cur.fetchall():
                    try:
                        shape = json.loads(shape_str)
                    except (json.JSONDecodeError, TypeError):
                        shape = [384]
                    qobj = {
                        "bits": int(bits), "scale": float(scale), "qmax": int(qmax),
                        "shape": shape, "data": base64.b64encode(blob).decode("utf-8")
                    }
                    centroids.append((topic, dequantize_packed(qobj)))
        except Exception as e:
            logger.warning(f"Failed to get centroids: {e}")
        return centroids

    def _select_topics(self, qemb: np.ndarray, top_t: int = 5) -> List[str]:
        from .retrieval import cosine_similarity
        
        centroids = self._get_all_topic_centroids()
        if not centroids:
            return []

        scored = [(cosine_similarity(qemb, cemb), topic) for topic, cemb in centroids]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in scored[:top_t]]

    def query(
        self,
        query_text: str,
        k: int = 5,
        top_topics: int = 5,
        topic: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None,
        require_verification: bool = False,
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        from .retrieval import cosine_similarity
        
        min_confidence = min_confidence if min_confidence is not None else self.config.min_confidence
        qemb = self.model.encode([query_text])[0].astype(np.float32)

        candidate_topics = self._select_topics(qemb, top_t=top_topics)

        if not candidate_topics:
            return self.query_slow(query_text, k=k)

        import base64
        import json
        import sqlite3
        
        with self._storage.get_conn() as conn:
            placeholders = ",".join("?" for _ in candidate_topics)
            cur = conn.execute(f"""
                SELECT topic, chunk_id, confidence, staleness, entropy, ts, text_preview,
                       emb_bits, emb_scale, emb_qmax, emb_blob, emb_shape,
                       quality_score, verified, verification_score, cross_refs
                FROM chunks
                WHERE topic IN ({placeholders})
            """, candidate_topics)
            rows = cur.fetchall()

        results = []
        for row in rows:
            (topic, chunk_id, conf, stale, ent, ts, preview,
             bits, scale, qmax, blob, shape_str,
             quality, verified, verif_score, cross_refs) = row

            if conf < min_confidence:
                continue

            if require_verification and not verified:
                continue

            try:
                shape = json.loads(shape_str)
            except (json.JSONDecodeError, TypeError):
                shape = [384]

            qobj = {
                "bits": int(bits), "scale": float(scale), "qmax": int(qmax),
                "shape": shape, "data": base64.b64encode(blob).decode("utf-8")
            }
            emb = dequantize_packed(qobj)
            sim = cosine_similarity(qemb, emb)

            chunk = {
                "chunk_id": chunk_id,
                "confidence": float(conf),
                "staleness": float(stale),
                "entropy": float(ent),
                "timestamp": ts,
                "text": preview,
                "quality_score": float(quality),
                "verified": bool(verified),
                "verification_score": float(verif_score),
                "cross_refs": int(cross_refs),
                "topic": topic,
            }

            # Apply tag filter
            if tags:
                chunk_tags = set(chunk.get("tags", []))
                if not any(t in chunk_tags for t in tags):
                    continue

            # Apply source filter
            if source and chunk.get("source_ref") != source:
                continue

            # Apply topic filter
            if topic and topic != chunk.get("topic"):
                continue

            final_score = sim * (0.5 + conf) * (1.0 - 0.5 * stale) * (0.5 + 0.5 * quality)
            results.append((final_score, topic, chunk))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def query_slow(self, query_text: str, k: int = 5, topic: Optional[str] = None) -> List[Tuple[float, str, Dict[str, Any]]]:
        from .retrieval import cosine_similarity
        
        qemb = self.model.encode([query_text])[0].astype(np.float32)
        results = []

        if topic:
            topic_data = self.load_topic(topic)
            for c in topic_data["chunks"]:
                emb = dequantize_packed(c["embedding_q"])
                sim = cosine_similarity(qemb, emb)
                results.append((sim, topic, c))
        else:
            for fn in os.listdir(self.topics_dir):
                if not fn.endswith(".tmem"):
                    continue
                path = os.path.join(self.topics_dir, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        topic_data = json.load(f)
                    tname = topic_data["topic"]
                    for c in topic_data["chunks"]:
                        emb = dequantize_packed(c["embedding_q"])
                        sim = cosine_similarity(qemb, emb)
                        results.append((sim, tname, c))
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Skipping corrupted topic file {fn}: {e}")
                    continue

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def verify_query_results(
        self,
        query_text: str,
        results: List[Tuple[float, str, Dict[str, Any]]],
    ) -> List[VerificationResult]:
        from .retrieval import cosine_similarity
        
        if not self.config.enable_verification:
            return [
                VerificationResult(
                    chunk_id=r[2].get("chunk_id", ""),
                    topic=r[1],
                    verified=False,
                    flags=["verification_disabled"]
                )
                for r in results
            ]

        qemb = self.model.encode([query_text])[0].astype(np.float32)
        verifications = []

        for score, topic, chunk in results:
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("text", "")

            cross_refs = 0
            agreements = []

            for fn in os.listdir(self.topics_dir):
                if not fn.endswith(".tmem"):
                    continue
                tname = fn[:-5]
                if tname == topic:
                    continue

                topic_data = self.load_topic(tname)
                for c in topic_data.get("chunks", []):
                    other_emb = dequantize_packed(c["embedding_q"])
                    sim = cosine_similarity(qemb, other_emb)
                    if sim > 0.6:
                        cross_refs += 1
                        agreements.append(sim)

            agreement_score = np.mean(agreements) if agreements else 0.0

            has_contradiction = False
            topic_data = self.load_topic(topic)
            for c in topic_data.get("chunks", []):
                if c["chunk_id"] != chunk_id:
                    if self._is_contradiction(c.get("text", ""), chunk_text):
                        has_contradiction = True
                        break

            verif_score = (
                agreement_score * 0.5 +
                min(1.0, cross_refs / self.config.min_cross_refs) * 0.3 +
                (0.2 if not has_contradiction else 0.0)
            )

            verified = verif_score >= self.config.verification_threshold

            flags = []
            if cross_refs == 0:
                flags.append("no_cross_refs")
            if has_contradiction:
                flags.append("contradiction_detected")
            if agreement_score < 0.5:
                flags.append("low_agreement")

            verifications.append(VerificationResult(
                chunk_id=chunk_id,
                topic=topic,
                verified=verified,
                verification_score=verif_score,
                cross_references=cross_refs,
                agreement_score=agreement_score,
                flags=flags,
            ))

        return verifications

    def get_metrics(self) -> MemoryMetrics:
        now = datetime.now(timezone.utc).isoformat()
        
        with self._storage.get_conn() as conn:
            t = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            c = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM chunks").fetchone()[0] or 0
            avg_stale = conn.execute("SELECT AVG(staleness) FROM chunks").fetchone()[0] or 0
            avg_quality = conn.execute("SELECT AVG(quality_score) FROM chunks").fetchone()[0] or 0
            expired = conn.execute("SELECT COUNT(*) FROM chunks WHERE ttl_ts IS NOT NULL AND ttl_ts < ?", (now,)).fetchone()[0]
            contradicted = conn.execute("SELECT COUNT(*) FROM chunks WHERE staleness > 0.5").fetchone()[0]
            verified = conn.execute("SELECT COUNT(*) FROM chunks WHERE verified = 1").fetchone()[0]
            cons_runs = conn.execute("SELECT COUNT(*) FROM consolidation_log").fetchone()[0]
            last_cons = conn.execute("SELECT ts FROM consolidation_log ORDER BY id DESC LIMIT 1").fetchone()
            last_cons_ts = last_cons[0] if last_cons else None
            cons_removed = conn.execute("SELECT COALESCE(SUM(chunks_affected), 0) FROM consolidation_log WHERE action = 'prune'").fetchone()[0]

        total_size = 0
        for dirpath, _, filenames in os.walk(self.root):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)

        topic_health = {}
        for fn in os.listdir(self.topics_dir):
            if not fn.endswith(".tmem"):
                continue
            try:
                with open(os.path.join(self.topics_dir, fn), "r", encoding="utf-8") as f:
                    td = json.load(f)
                chunks = td.get("chunks", [])
                if not chunks:
                    topic_health[td["topic"]] = 0.0
                    continue
                avg_c = np.mean([c.get("confidence", 0.5) for c in chunks])
                avg_s = np.mean([c.get("staleness", 0.0) for c in chunks])
                avg_q = np.mean([c.get("quality_score", 0.5) for c in chunks])
                health = (avg_c * 0.4 + (1 - avg_s) * 0.3 + avg_q * 0.3)
                topic_health[td["topic"]] = float(health)
            except (json.JSONDecodeError, IOError):
                continue

        return MemoryMetrics(
            total_topics=int(t),
            total_chunks=int(c),
            avg_confidence=float(avg_conf),
            avg_staleness=float(avg_stale),
            avg_quality=float(avg_quality),
            expired_chunks=int(expired),
            contradicted_chunks=int(contradicted),
            verified_chunks=int(verified),
            storage_bytes=total_size,
            consolidation_runs=int(cons_runs),
            last_consolidation=last_cons_ts,
            chunks_removed_by_consolidation=int(cons_removed),
            topic_health=topic_health,
        )

    def stats(self) -> Dict[str, Any]:
        return self.get_metrics().to_dict()

    def reinforce(self, chunk_id: str, topic: str, delta: float = 0.1) -> None:
        """Increase confidence for an existing memory chunk."""
        topic_data = self.load_topic(topic)
        for c in topic_data.get("chunks", []):
            if c.get("chunk_id") == chunk_id:
                c["confidence"] = min(float(c.get("confidence", 0.5)) + delta, 1.0)
                break
        self.save_topic(topic_data)
        self._audit.log("reinforce", topic, f"chunk_id={chunk_id}, delta={delta}")

    def penalize(self, chunk_id: str, topic: str, delta: float = 0.1) -> None:
        """Decrease confidence for an existing memory chunk."""
        topic_data = self.load_topic(topic)
        for c in topic_data.get("chunks", []):
            if c.get("chunk_id") == chunk_id:
                c["confidence"] = max(float(c.get("confidence", 0.5)) - delta, 0.0)
                break
        self.save_topic(topic_data)
        self._audit.log("penalize", topic, f"chunk_id={chunk_id}, delta={delta}")

    def forget(self, chunk_id: str, topic: str) -> None:
        """Soft forget: decay confidence to near zero."""
        self.penalize(chunk_id, topic, delta=1.0)

    def recall_topic(self, topic: str, top_k: int = 10) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Recall all memories for a topic."""
        return self.query_slow("", k=top_k, topic=topic)

    def get_chunk(self, chunk_id: str, topic: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID."""
        topic_data = self.load_topic(topic)
        for c in topic_data.get("chunks", []):
            if c.get("chunk_id") == chunk_id:
                return c
        return None

    def list_topics(self) -> List[str]:
        """List all available topics."""
        topics = []
        for fn in os.listdir(self.topics_dir):
            if fn.endswith(".tmem"):
                topic_data = self.load_topic(fn[:-5])
                topics.append(topic_data.get("topic"))
        return topics

    def close(self) -> None:
        self._shutdown.shutdown()
        self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False