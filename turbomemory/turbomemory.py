#!/usr/bin/env python3
"""TurboMemory v0.4 - Enhanced layered memory system with quantization, verification, and observability."""

import os
import re
import json
import time
import base64
import sqlite3
import hashlib
import shutil
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple, Iterator, Set
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ----------------------------
# Utilities
# ----------------------------
def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def safe_topic_filename(topic: str) -> str:
    """Convert topic name to safe filename."""
    return topic.replace(".", "_").replace("/", "_").replace("\\", "_")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 0.0
    a = a / (a_norm + 1e-9)
    b = b / (b_norm + 1e-9)
    return float(np.dot(a, b))


def sha1_text(s: str) -> str:
    """Compute SHA1 hash of text."""
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# ----------------------------
# Packed quantization (4-bit / 6-bit / 8-bit)
# ----------------------------
def quantize_packed(vec: np.ndarray, bits: int = 6) -> Dict[str, Any]:
    """Symmetric uniform quantization with packed storage."""
    if bits not in (4, 6, 8):
        raise ValueError("bits must be 4, 6, or 8")

    vec = vec.astype(np.float32)
    qmax = (2 ** (bits - 1)) - 1

    vmax = float(np.max(np.abs(vec)) + 1e-9)
    scale = vmax / qmax

    q = np.round(vec / scale).astype(np.int16)
    q = np.clip(q, -qmax, qmax).astype(np.int16)

    uq = (q + qmax).astype(np.uint16)
    packed = pack_unsigned(uq, bits)

    return {
        "bits": bits,
        "scale": float(scale),
        "qmax": int(qmax),
        "shape": list(vec.shape),
        "data": base64.b64encode(packed).decode("utf-8")
    }


def dequantize_packed(qobj: Dict[str, Any]) -> np.ndarray:
    """Dequantize a packed quantization object back to float32 vector."""
    bits = int(qobj["bits"])
    scale = float(qobj["scale"])
    qmax = int(qobj["qmax"])
    shape = tuple(qobj["shape"])

    raw = base64.b64decode(qobj["data"])
    uq = unpack_unsigned(raw, bits, int(np.prod(shape))).astype(np.int16)

    q = (uq - qmax).astype(np.float32)
    return (q * scale).reshape(shape)


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    """Packs uint values into bitstream."""
    values = values.astype(np.uint32).ravel()
    out = bytearray()
    buf = 0
    buf_bits = 0
    mask = (1 << bits) - 1

    for v in values:
        buf |= (int(v) & mask) << buf_bits
        buf_bits += bits
        while buf_bits >= 8:
            out.append(buf & 0xFF)
            buf >>= 8
            buf_bits -= 8

    if buf_bits > 0:
        out.append(buf & 0xFF)

    return bytes(out)


def unpack_unsigned(data: bytes, bits: int, n_values: int) -> np.ndarray:
    """Unpacks bitstream into uint array."""
    out = np.zeros(n_values, dtype=np.uint16)
    buf = 0
    buf_bits = 0
    idx = 0
    mask = (1 << bits) - 1

    for b in data:
        buf |= int(b) << buf_bits
        buf_bits += 8
        while buf_bits >= bits and idx < n_values:
            out[idx] = buf & mask
            buf >>= bits
            buf_bits -= bits
            idx += 1
        if idx >= n_values:
            break

    return out


# ----------------------------
# Exclusion Rules
# ----------------------------
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
        r"^\s*(password|secret|token|api_key|apikey)\s*[=:]",
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


# ----------------------------
# Memory Quality Scoring
# ----------------------------
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


def compute_quality_score(
    confidence: float,
    staleness: float,
    text: str,
    verified: bool = False,
    verification_score: float = 0.0,
) -> QualityScore:
    """Compute comprehensive quality score for a memory chunk."""
    flags = []

    # Confidence component (0-1)
    conf_score = max(0.0, min(1.0, confidence))

    # Freshness component (0-1, inverse of staleness)
    fresh_score = max(0.0, 1.0 - staleness)

    # Specificity component (0-1)
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

    # Verification component (0-1)
    verif_score = verification_score if verified else 0.0

    # Weighted overall
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


# ----------------------------
# Retrieval Verification
# ----------------------------
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


# ----------------------------
# Observability Metrics
# ----------------------------
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


# ----------------------------
# SQLite Index Schema
# ----------------------------
INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    topic TEXT PRIMARY KEY,
    file_ref TEXT NOT NULL,
    updated TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    centroid_bits INTEGER NOT NULL,
    centroid_scale REAL NOT NULL,
    centroid_qmax INTEGER NOT NULL,
    centroid_blob BLOB NOT NULL,
    centroid_shape TEXT NOT NULL DEFAULT '[384]'
);

CREATE TABLE IF NOT EXISTS chunks (
    topic TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    staleness REAL NOT NULL,
    entropy REAL NOT NULL,
    ts TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    text_preview TEXT NOT NULL,
    emb_bits INTEGER NOT NULL,
    emb_scale REAL NOT NULL,
    emb_qmax INTEGER NOT NULL,
    emb_blob BLOB NOT NULL,
    emb_shape TEXT NOT NULL DEFAULT '[384]',
    ttl_ts TEXT,
    quality_score REAL DEFAULT 0.5,
    verified INTEGER DEFAULT 0,
    verification_score REAL DEFAULT 0.0,
    cross_refs INTEGER DEFAULT 0,
    PRIMARY KEY(topic, chunk_id)
);

CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    topic TEXT,
    action TEXT NOT NULL,
    details TEXT,
    chunks_affected INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS exclusion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    topic TEXT,
    reason TEXT NOT NULL,
    text_preview TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic);
CREATE INDEX IF NOT EXISTS idx_chunks_confidence ON chunks(confidence);
CREATE INDEX IF NOT EXISTS idx_chunks_ttl ON chunks(ttl_ts);
CREATE INDEX IF NOT EXISTS idx_chunks_quality ON chunks(quality_score);
CREATE INDEX IF NOT EXISTS idx_consolidation_ts ON consolidation_log(ts);
"""


# ----------------------------
# SQLite Connection Pool
# ----------------------------
class SQLitePool:
    """Thread-safe SQLite connection pool."""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: List[sqlite3.Connection] = []

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool (thread-local)."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            with self._lock:
                if len(self._connections) < self.pool_size:
                    conn = self._create_connection()
                    self._connections.append(conn)
                    self._local.conn = conn
                else:
                    conn = self._connections.pop(0)
                    self._connections.append(conn)
                    self._local.conn = conn
        return self._local.conn

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
        self._local.conn = None


# ----------------------------
# Configuration
# ----------------------------
@dataclass
class TurboMemoryConfig:
    """Configuration for TurboMemory."""
    root: str = "turbomemory_data"
    model_name: str = "all-MiniLM-L6-v2"
    default_bits: int = 6
    pool_size: int = 5
    default_ttl_days: Optional[float] = None
    max_chunks_per_topic: int = 300
    contradiction_decay: float = 0.6
    min_confidence: float = 0.0

    # Verification settings
    enable_verification: bool = True
    verification_threshold: float = 0.7
    min_cross_refs: int = 2

    # Exclusion settings
    enable_exclusions: bool = True
    exclusion_rules: Optional[ExclusionRules] = None

    # Quality settings
    min_quality_threshold: float = 0.3
    quality_decay_rate: float = 0.01

    @classmethod
    def from_file(cls, path: str) -> "TurboMemoryConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_file(self, path: str) -> None:
        """Save configuration to a JSON file."""
        data = {k: v for k, v in self.__dict__.items() if k != 'exclusion_rules'}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ----------------------------
# TurboMemory Engine
# ----------------------------
class TurboMemory:
    """
    TurboMemory v0.4

    Layer 1: MEMORY.md (always loaded index pointers)
    Layer 2: topics/*.tmem (structured topic files)
    Layer 3: sessions/*.jsonl (immutable logs)

    Enhancements:
    - SQLite index with connection pooling
    - packed 4/6/8-bit quantization
    - centroid prefilter for topic selection
    - contradiction detection + confidence decay
    - TTL support for memory chunks
    - backup/restore functionality
    - bulk import/export
    - retrieval verification with cross-checking
    - configurable exclusion rules
    - memory quality scoring
    - comprehensive observability metrics
    - improved consolidation with semantic merging
    """

    def __init__(
        self,
        root: str = "turbomemory_data",
        model_name: str = "all-MiniLM-L6-v2",
        config: Optional[TurboMemoryConfig] = None,
    ):
        self.config = config or TurboMemoryConfig(root=root, model_name=model_name)
        if self.config.exclusion_rules is None:
            self.config.exclusion_rules = ExclusionRules()

        self.root = self.config.root
        self.model_name = self.config.model_name
        self.model = SentenceTransformer(model_name)

        self.memory_index_path = os.path.join(self.root, "MEMORY.md")
        self.topics_dir = os.path.join(self.root, "topics")
        self.sessions_dir = os.path.join(self.root, "sessions")
        self.lock_dir = os.path.join(self.root, "lock")
        self.db_dir = os.path.join(self.root, "db")
        self.db_path = os.path.join(self.db_dir, "index.sqlite")
        self.config_path = os.path.join(self.root, "config.json")
        self.metrics_path = os.path.join(self.root, "metrics.json")

        ensure_dir(self.root)
        ensure_dir(self.topics_dir)
        ensure_dir(self.sessions_dir)
        ensure_dir(self.lock_dir)
        ensure_dir(self.db_dir)

        if not os.path.exists(self.memory_index_path):
            with open(self.memory_index_path, "w", encoding="utf-8") as f:
                f.write("# TurboMemory Index (v0.4)\n")

        if not os.path.exists(self.config_path):
            self.config.to_file(self.config_path)

        self._pool = SQLitePool(self.db_path, pool_size=self.config.pool_size)
        self._init_db()

    # ----------------------------
    # DB helpers
    # ----------------------------
    @contextmanager
    def _get_conn(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = self._pool.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript(INDEX_SCHEMA)

    # ----------------------------
    # Exclusion checking
    # ----------------------------
    def _check_exclusions(self, text: str, topic: str) -> Tuple[bool, str]:
        """Check if text should be excluded based on rules."""
        if not self.config.enable_exclusions:
            return False, ""
        return self.config.exclusion_rules.should_exclude(text, topic)

    def _log_exclusion(self, topic: str, reason: str, text: str) -> None:
        """Log an exclusion event."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO exclusion_log(ts, topic, reason, text_preview) VALUES (?, ?, ?, ?)",
                (now_iso(), topic, reason, text[:200])
            )

    # ----------------------------
    # Layer 3: Session logs
    # ----------------------------
    def add_turn(self, role: str, text: str, session_file: Optional[str] = None) -> str:
        """Add a conversation turn to session logs."""
        if session_file is None:
            session_file = datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".jsonl"

        path = os.path.join(self.sessions_dir, session_file)
        record = {"ts": now_iso(), "role": role, "text": text}

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"Failed to write session log: {e}")
            raise

        return os.path.relpath(path, self.root)

    # ----------------------------
    # Layer 2: Topic file IO
    # ----------------------------
    def _topic_path(self, topic: str) -> str:
        """Get the file path for a topic."""
        return os.path.join(self.topics_dir, safe_topic_filename(topic) + ".tmem")

    def load_topic(self, topic: str) -> Dict[str, Any]:
        """Load topic data from file."""
        path = self._topic_path(topic)
        if not os.path.exists(path):
            return {
                "topic": topic,
                "created": now_iso(),
                "updated": now_iso(),
                "centroid_q": None,
                "chunks": []
            }
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load topic {topic}: {e}")
            return {
                "topic": topic,
                "created": now_iso(),
                "updated": now_iso(),
                "centroid_q": None,
                "chunks": []
            }

    def save_topic(self, topic_data: Dict[str, Any]) -> None:
        """Save topic data to file."""
        topic_data["updated"] = now_iso()
        path = self._topic_path(topic_data["topic"])
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(topic_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Failed to save topic {topic_data['topic']}: {e}")
            raise

    # ----------------------------
    # Layer 1: MEMORY.md
    # ----------------------------
    def read_index_lines(self) -> List[str]:
        """Read lines from MEMORY.md."""
        try:
            with open(self.memory_index_path, "r", encoding="utf-8") as f:
                return f.readlines()
        except IOError as e:
            logger.error(f"Failed to read MEMORY.md: {e}")
            return ["# TurboMemory Index (v0.4)\n"]

    def write_index_lines(self, lines: List[str]) -> None:
        """Write lines to MEMORY.md."""
        content = "".join(lines)
        if len(content.encode("utf-8")) > 4096:
            header = [lines[0]] if lines else ["# TurboMemory Index (v0.4)\n"]
            body = [ln for ln in lines[1:] if ln.strip()][:30]
            content = "".join(header + body)

        try:
            with open(self.memory_index_path, "w", encoding="utf-8") as f:
                f.write(content)
        except IOError as e:
            logger.error(f"Failed to write MEMORY.md: {e}")
            raise

    def update_memory_md(self, topic: str, summary: str, score: float, file_ref: str) -> None:
        """Update MEMORY.md with topic entry."""
        lines = self.read_index_lines()
        if not lines:
            lines = ["# TurboMemory Index (v0.4)\n"]

        new_lines = []
        found = False

        for line in lines:
            if line.strip().startswith("#") or line.strip() == "":
                new_lines.append(line)
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6 and parts[1] == topic:
                found = True
                entry_id = parts[0]
                new_line = (
                    f"{entry_id} | {topic} | {summary[:150]} | "
                    f"{score:.2f} | {datetime.now(timezone.utc).date()} | {file_ref}\n"
                )
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        if not found:
            entry_id = safe_topic_filename(topic)[:5]
            new_line = (
                f"{entry_id} | {topic} | {summary[:150]} | "
                f"{score:.2f} | {datetime.now(timezone.utc).date()} | {file_ref}\n"
            )
            new_lines.append(new_line)

        self.write_index_lines(new_lines)

    # ----------------------------
    # Contradiction detection
    # ----------------------------
    def _is_contradiction(self, old_text: str, new_text: str) -> bool:
        """Improved contradiction detection."""
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

    # ----------------------------
    # Add memory chunk
    # ----------------------------
    def add_memory(
        self,
        topic: str,
        text: str,
        confidence: float = 0.8,
        bits: Optional[int] = None,
        source_ref: Optional[str] = None,
        ttl_days: Optional[float] = None,
    ) -> Optional[str]:
        """Add a memory chunk to a topic. Returns chunk_id or None if excluded."""
        # Check exclusions first
        excluded, reason = self._check_exclusions(text, topic)
        if excluded:
            logger.info(f"Excluded memory: {reason}")
            self._log_exclusion(topic, reason, text)
            return None

        bits = bits or self.config.default_bits
        if bits not in (4, 6, 8):
            raise ValueError("bits must be 4, 6, or 8")

        emb = self.model.encode([text])[0].astype(np.float32)
        emb_q = quantize_packed(emb, bits=bits)

        topic_data = self.load_topic(topic)

        # Contradiction scan
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

        # Compute initial quality score
        quality = compute_quality_score(confidence, 0.0, text)

        chunk = {
            "chunk_id": chunk_id,
            "timestamp": now_iso(),
            "confidence": float(confidence),
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

        all_embs = [dequantize_packed(c["embedding_q"]) for c in topic_data["chunks"]]
        centroid = np.mean(np.vstack(all_embs), axis=0)
        topic_data["centroid_q"] = quantize_packed(centroid, bits=8)

        self.save_topic(topic_data)

        file_ref = os.path.relpath(self._topic_path(topic), self.root)
        self.update_memory_md(topic, summary=text, score=confidence, file_ref=file_ref)

        self._upsert_topic_to_db(topic_data, file_ref)
        self._upsert_chunk_to_db(topic, chunk)

        for c in topic_data["chunks"][:-1]:
            self._upsert_chunk_to_db(topic, c)

        return chunk_id

    def _upsert_topic_to_db(self, topic_data: Dict[str, Any], file_ref: str) -> None:
        """Upsert topic data to SQLite."""
        centroid_q = topic_data.get("centroid_q")
        if centroid_q is None:
            return

        blob = base64.b64decode(centroid_q["data"])
        shape_str = json.dumps(centroid_q.get("shape", [384]))

        with self._get_conn() as conn:
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
        """Upsert chunk data to SQLite."""
        emb_q = chunk["embedding_q"]
        blob = base64.b64decode(emb_q["data"])
        text = chunk.get("text", "")
        preview = text[:200]
        th = sha1_text(text)
        shape_str = json.dumps(emb_q.get("shape", [384]))

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO chunks(
                    topic, chunk_id, confidence, staleness, entropy, ts, text_hash, text_preview,
                    emb_bits, emb_scale, emb_qmax, emb_blob, emb_shape, ttl_ts,
                    quality_score, verified, verification_score, cross_refs
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic, chunk_id) DO UPDATE SET
                    confidence=excluded.confidence, staleness=excluded.staleness,
                    entropy=excluded.entropy, ts=excluded.ts, text_hash=excluded.text_hash,
                    text_preview=excluded.text_preview, emb_bits=excluded.emb_bits,
                    emb_scale=excluded.emb_scale, emb_qmax=excluded.emb_qmax,
                    emb_blob=excluded.emb_blob, emb_shape=excluded.emb_shape, ttl_ts=excluded.ttl_ts,
                    quality_score=excluded.quality_score, verified=excluded.verified,
                    verification_score=excluded.verification_score, cross_refs=excluded.cross_refs
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
            ))

    # ----------------------------
    # Fast topic prefilter
    # ----------------------------
    def _get_all_topic_centroids(self) -> List[Tuple[str, np.ndarray]]:
        """Get all topic centroids from SQLite."""
        with self._get_conn() as conn:
            cur = conn.execute("SELECT topic, centroid_bits, centroid_scale, centroid_qmax, centroid_blob, centroid_shape FROM topics")
            rows = cur.fetchall()

        out = []
        for topic, bits, scale, qmax, blob, shape_str in rows:
            try:
                shape = json.loads(shape_str)
            except (json.JSONDecodeError, TypeError):
                shape = [384]
            qobj = {
                "bits": int(bits), "scale": float(scale), "qmax": int(qmax),
                "shape": shape, "data": base64.b64encode(blob).decode("utf-8")
            }
            out.append((topic, dequantize_packed(qobj)))
        return out

    def _select_topics(self, qemb: np.ndarray, top_t: int = 5) -> List[str]:
        """Select top topics by centroid similarity."""
        centroids = self._get_all_topic_centroids()
        if not centroids:
            return []

        scored = [(cosine_sim(qemb, cemb), topic) for topic, cemb in centroids]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in scored[:top_t]]

    # ----------------------------
    # Query with verification
    # ----------------------------
    def query(
        self,
        query_text: str,
        k: int = 5,
        top_topics: int = 5,
        min_confidence: Optional[float] = None,
        require_verification: bool = False,
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Query memory with semantic search."""
        min_confidence = min_confidence if min_confidence is not None else self.config.min_confidence
        qemb = self.model.encode([query_text])[0].astype(np.float32)

        candidate_topics = self._select_topics(qemb, top_t=top_topics)

        if not candidate_topics:
            return self.query_slow(query_text, k=k)

        with self._get_conn() as conn:
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
            sim = cosine_sim(qemb, emb)

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
            }

            final_score = sim * (0.5 + conf) * (1.0 - 0.5 * stale) * (0.5 + 0.5 * quality)
            results.append((final_score, topic, chunk))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def query_slow(self, query_text: str, k: int = 5, topic: Optional[str] = None) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Fallback query that scans topic files directly."""
        qemb = self.model.encode([query_text])[0].astype(np.float32)
        results = []

        if topic:
            topic_data = self.load_topic(topic)
            for c in topic_data["chunks"]:
                emb = dequantize_packed(c["embedding_q"])
                sim = cosine_sim(qemb, emb)
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
                        sim = cosine_sim(qemb, emb)
                        results.append((sim, tname, c))
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Skipping corrupted topic file {fn}: {e}")
                    continue

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    # ----------------------------
    # Retrieval Verification
    # ----------------------------
    def verify_query_results(
        self,
        query_text: str,
        results: List[Tuple[float, str, Dict[str, Any]]],
    ) -> List[VerificationResult]:
        """Verify query results through cross-referencing."""
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

            # Cross-reference: find similar chunks in OTHER topics
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
                    sim = cosine_sim(qemb, other_emb)
                    if sim > 0.6:
                        cross_refs += 1
                        agreements.append(sim)

            # Compute agreement score
            agreement_score = np.mean(agreements) if agreements else 0.0

            # Check for contradictions in same topic
            has_contradiction = False
            topic_data = self.load_topic(topic)
            for c in topic_data.get("chunks", []):
                if c["chunk_id"] != chunk_id:
                    if self._is_contradiction(c.get("text", ""), chunk_text):
                        has_contradiction = True
                        break

            # Verification decision
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

    def verify_and_score(
        self,
        query_text: str,
        k: int = 5,
        top_topics: int = 5,
        min_confidence: Optional[float] = None,
    ) -> List[Tuple[float, str, Dict[str, Any], VerificationResult]]:
        """Query with built-in verification. Returns (score, topic, chunk, verification)."""
        results = self.query(query_text, k=k, top_topics=top_topics, min_confidence=min_confidence)
        verifications = self.verify_query_results(query_text, results)

        combined = []
        for (score, topic, chunk), verif in zip(results, verifications):
            # Update chunk with verification info
            chunk["verified"] = verif.verified
            chunk["verification_score"] = verif.verification_score
            chunk["cross_refs"] = verif.cross_references
            chunk["verification_flags"] = verif.flags

            # Adjust score based on verification
            verif_multiplier = 0.5 + 0.5 * verif.verification_score
            adjusted_score = score * verif_multiplier

            combined.append((adjusted_score, topic, chunk, verif))

        combined.sort(key=lambda x: x[0], reverse=True)
        return combined

    # ----------------------------
    # Quality scoring
    # ----------------------------
    def get_chunk_quality(self, topic: str, chunk_id: str) -> QualityScore:
        """Get quality score for a specific chunk."""
        topic_data = self.load_topic(topic)
        for c in topic_data.get("chunks", []):
            if c["chunk_id"] == chunk_id:
                return compute_quality_score(
                    confidence=c.get("confidence", 0.5),
                    staleness=c.get("staleness", 0.0),
                    text=c.get("text", ""),
                    verified=c.get("verified", False),
                    verification_score=c.get("verification_score", 0.0),
                )
        return QualityScore()

    def decay_quality(self) -> int:
        """Apply quality decay to all chunks. Returns count of decayed chunks."""
        decayed = 0
        for fn in os.listdir(self.topics_dir):
            if not fn.endswith(".tmem"):
                continue
            path = os.path.join(self.topics_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    topic_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            changed = False
            for c in topic_data.get("chunks", []):
                old_quality = c.get("quality_score", 0.5)
                new_quality = max(0.0, old_quality - self.config.quality_decay_rate)
                if new_quality != old_quality:
                    c["quality_score"] = new_quality
                    c["staleness"] = min(1.0, c.get("staleness", 0.0) + self.config.quality_decay_rate * 0.5)
                    changed = True
                    decayed += 1

            if changed:
                self.save_topic(topic_data)
                file_ref = os.path.relpath(path, self.root)
                self._upsert_topic_to_db(topic_data, file_ref)
                for c in topic_data.get("chunks", []):
                    self._upsert_chunk_to_db(topic_data["topic"], c)

        return decayed

    # ----------------------------
    # Rebuild SQLite from topic files
    # ----------------------------
    def rebuild_index(self) -> None:
        """Rebuild SQLite index from topic files."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM chunks;")
            conn.execute("DELETE FROM topics;")

        for fn in os.listdir(self.topics_dir):
            if not fn.endswith(".tmem"):
                continue
            path = os.path.join(self.topics_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    topic_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping corrupted topic file {fn}: {e}")
                continue

            file_ref = os.path.relpath(path, self.root)
            self._upsert_topic_to_db(topic_data, file_ref)
            for c in topic_data.get("chunks", []):
                self._upsert_chunk_to_db(topic_data["topic"], c)

    # ----------------------------
    # TTL Management
    # ----------------------------
    def expire_ttl(self) -> int:
        """Remove expired chunks based on TTL."""
        now = datetime.now(timezone.utc).isoformat()
        removed = 0

        with self._get_conn() as conn:
            cur = conn.execute("SELECT topic, chunk_id FROM chunks WHERE ttl_ts IS NOT NULL AND ttl_ts < ?", (now,))
            expired = cur.fetchall()

        for topic, chunk_id in expired:
            topic_data = self.load_topic(topic)
            original_count = len(topic_data["chunks"])
            topic_data["chunks"] = [c for c in topic_data["chunks"] if c["chunk_id"] != chunk_id]
            if len(topic_data["chunks"]) < original_count:
                self.save_topic(topic_data)
                removed += 1

        with self._get_conn() as conn:
            conn.execute("DELETE FROM chunks WHERE ttl_ts IS NOT NULL AND ttl_ts < ?", (now,))

        logger.info(f"Expired {removed} chunks")
        return removed

    # ----------------------------
    # Backup/Restore
    # ----------------------------
    def backup(self, backup_path: str) -> str:
        """Create a backup of the entire memory store."""
        ensure_dir(backup_path)

        for dirname in ["topics", "sessions", "db"]:
            src = os.path.join(self.root, dirname)
            dst = os.path.join(backup_path, dirname)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

        if os.path.exists(self.memory_index_path):
            shutil.copy2(self.memory_index_path, os.path.join(backup_path, "MEMORY.md"))

        if os.path.exists(self.config_path):
            shutil.copy2(self.config_path, os.path.join(backup_path, "config.json"))

        metrics = self.get_metrics()
        with open(os.path.join(backup_path, "backup.json"), "w") as f:
            json.dump({
                "backup_ts": now_iso(),
                "root": self.root,
                "topics": metrics.total_topics,
                "chunks": metrics.total_chunks,
                "avg_quality": metrics.avg_quality,
            }, f, indent=2)

        logger.info(f"Backup created at {backup_path}")
        return backup_path

    def restore(self, backup_path: str) -> None:
        """Restore from a backup."""
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        self._pool.close_all()

        for dirname in ["topics", "sessions", "db"]:
            src = os.path.join(backup_path, dirname)
            dst = os.path.join(self.root, dirname)
            if os.path.exists(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        src_md = os.path.join(backup_path, "MEMORY.md")
        if os.path.exists(src_md):
            shutil.copy2(src_md, self.memory_index_path)

        src_config = os.path.join(backup_path, "config.json")
        if os.path.exists(src_config):
            shutil.copy2(src_config, self.config_path)

        self._pool = SQLitePool(self.db_path, pool_size=self.config.pool_size)
        self._init_db()

        logger.info(f"Restored from backup at {backup_path}")

    # ----------------------------
    # Bulk Import/Export
    # ----------------------------
    def bulk_import(self, items: List[Dict[str, Any]], topic_field: str = "topic", text_field: str = "text") -> Dict[str, int]:
        """Import multiple memory items. Returns {imported, excluded, failed}."""
        imported = 0
        excluded = 0
        failed = 0

        for item in items:
            try:
                topic = item[topic_field]
                text = item[text_field]
                confidence = item.get("confidence", 0.8)
                bits = item.get("bits", self.config.default_bits)
                ttl_days = item.get("ttl_days")
                source_ref = item.get("source_ref")

                result = self.add_memory(topic, text, confidence, bits, source_ref, ttl_days)
                if result:
                    imported += 1
                else:
                    excluded += 1
            except Exception as e:
                logger.warning(f"Failed to import item: {e}")
                failed += 1

        return {"imported": imported, "excluded": excluded, "failed": failed}

    def export_topic(self, topic: str, include_embeddings: bool = False) -> Dict[str, Any]:
        """Export a topic's data."""
        topic_data = self.load_topic(topic)
        if not include_embeddings:
            for chunk in topic_data.get("chunks", []):
                chunk.pop("embedding_q", None)
            topic_data.pop("centroid_q", None)
        return topic_data

    def export_all(self, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Export all topics."""
        topics = []
        for fn in os.listdir(self.topics_dir):
            if fn.endswith(".tmem"):
                topic_name = fn[:-5]
                topics.append(self.export_topic(topic_name, include_embeddings))
        return topics

    # ----------------------------
    # Topic Management
    # ----------------------------
    def merge_topics(self, source_topic: str, target_topic: str) -> int:
        """Merge source topic into target topic."""
        source_data = self.load_topic(source_topic)
        target_data = self.load_topic(target_topic)

        existing_hashes = {sha1_text(c.get("text", "")) for c in target_data["chunks"]}
        merged = 0

        for chunk in source_data["chunks"]:
            text_hash = sha1_text(chunk.get("text", ""))
            if text_hash not in existing_hashes:
                chunk["chunk_id"] = f"c{len(target_data['chunks'])+1:04d}"
                target_data["chunks"].append(chunk)
                existing_hashes.add(text_hash)
                merged += 1

        if target_data["chunks"]:
            all_embs = [dequantize_packed(c["embedding_q"]) for c in target_data["chunks"]]
            target_data["centroid_q"] = quantize_packed(np.mean(np.vstack(all_embs), axis=0), bits=8)

        self.save_topic(target_data)

        file_ref = os.path.relpath(self._topic_path(target_topic), self.root)
        self._upsert_topic_to_db(target_data, file_ref)
        for c in target_data["chunks"][-merged:] if merged > 0 else []:
            self._upsert_chunk_to_db(target_topic, c)

        logger.info(f"Merged {merged} chunks from {source_topic} into {target_topic}")
        return merged

    def split_topic(self, topic: str, chunk_ids: List[str], new_topic: str) -> int:
        """Split chunks from a topic into a new topic."""
        source_data = self.load_topic(topic)
        target_data = {
            "topic": new_topic, "created": now_iso(), "updated": now_iso(),
            "centroid_q": None, "chunks": []
        }

        chunk_id_set = set(chunk_ids)
        moved = 0
        kept = []

        for chunk in source_data["chunks"]:
            if chunk["chunk_id"] in chunk_id_set:
                chunk["chunk_id"] = f"c{len(target_data['chunks'])+1:04d}"
                target_data["chunks"].append(chunk)
                moved += 1
            else:
                kept.append(chunk)

        source_data["chunks"] = kept
        if kept:
            all_embs = [dequantize_packed(c["embedding_q"]) for c in kept]
            source_data["centroid_q"] = quantize_packed(np.mean(np.vstack(all_embs), axis=0), bits=8)

        if target_data["chunks"]:
            all_embs = [dequantize_packed(c["embedding_q"]) for c in target_data["chunks"]]
            target_data["centroid_q"] = quantize_packed(np.mean(np.vstack(all_embs), axis=0), bits=8)

        self.save_topic(source_data)
        self.save_topic(target_data)

        file_ref = os.path.relpath(self._topic_path(topic), self.root)
        self._upsert_topic_to_db(source_data, file_ref)
        file_ref = os.path.relpath(self._topic_path(new_topic), self.root)
        self._upsert_topic_to_db(target_data, file_ref)

        logger.info(f"Moved {moved} chunks from {topic} to {new_topic}")
        return moved

    # ----------------------------
    # Observability
    # ----------------------------
    def get_metrics(self) -> MemoryMetrics:
        """Get comprehensive memory metrics."""
        with self._get_conn() as conn:
            t = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            c = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM chunks").fetchone()[0] or 0
            avg_stale = conn.execute("SELECT AVG(staleness) FROM chunks").fetchone()[0] or 0
            avg_quality = conn.execute("SELECT AVG(quality_score) FROM chunks").fetchone()[0] or 0
            expired = conn.execute("SELECT COUNT(*) FROM chunks WHERE ttl_ts IS NOT NULL AND ttl_ts < ?",
                                  (datetime.now(timezone.utc).isoformat(),)).fetchone()[0]
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

        # Per-topic health scores
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

    def log_consolidation(self, topic: str, action: str, details: str, chunks_affected: int) -> None:
        """Log a consolidation event."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO consolidation_log(ts, topic, action, details, chunks_affected) VALUES (?, ?, ?, ?, ?)",
                (now_iso(), topic, action, details, chunks_affected)
            )

    # ----------------------------
    # Stats
    # ----------------------------
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        metrics = self.get_metrics()
        return metrics.to_dict()

    def close(self) -> None:
        """Close all database connections."""
        self._pool.close_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
