#!/usr/bin/env python3
# TurboMemory v0.2
# - SQLite index
# - packed 4/6-bit quantization
# - centroid prefilter
# - contradiction detection + confidence decay
# - safe file layout + locking helpers

import os
import json
import time
import base64
import sqlite3
import hashlib
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


# ----------------------------
# Utilities
# ----------------------------
def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_topic_filename(topic: str) -> str:
    return topic.replace(".", "_").replace("/", "_").replace("\\", "_")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# ----------------------------
# Packed quantization (4-bit / 6-bit)
# ----------------------------
def quantize_packed(vec: np.ndarray, bits: int = 6) -> Dict[str, Any]:
    """
    Symmetric uniform quantization with packed storage.
    bits supported: 4 or 6 or 8 (8 stored as uint8 raw).
    Range: signed [-qmax, qmax]
    """
    if bits not in (4, 6, 8):
        raise ValueError("bits must be 4, 6, or 8")

    vec = vec.astype(np.float32)
    qmax = (2 ** (bits - 1)) - 1  # signed

    vmax = float(np.max(np.abs(vec)) + 1e-9)
    scale = vmax / qmax

    q = np.round(vec / scale).astype(np.int16)
    q = np.clip(q, -qmax, qmax).astype(np.int16)

    # shift signed -> unsigned
    uq = (q + qmax).astype(np.uint16)  # range [0, 2*qmax]

    packed = pack_unsigned(uq, bits)

    return {
        "bits": bits,
        "scale": float(scale),
        "qmax": int(qmax),
        "shape": list(vec.shape),
        "data": base64.b64encode(packed).decode("utf-8")
    }


def dequantize_packed(qobj: Dict[str, Any]) -> np.ndarray:
    bits = int(qobj["bits"])
    scale = float(qobj["scale"])
    qmax = int(qobj["qmax"])
    shape = tuple(qobj["shape"])

    raw = base64.b64decode(qobj["data"])
    uq = unpack_unsigned(raw, bits, int(np.prod(shape))).astype(np.int16)

    q = (uq - qmax).astype(np.float32)
    return (q * scale).reshape(shape)


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    """
    Packs uint values into bitstream.
    """
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
    """
    Unpacks bitstream into uint array.
    """
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
# SQLite Index
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
    centroid_blob BLOB NOT NULL
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
    PRIMARY KEY(topic, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic);
"""


# ----------------------------
# TurboMemory Engine
# ----------------------------
class TurboMemory:
    """
    TurboMemory v0.2

    Layer 1: MEMORY.md (always loaded index pointers)
    Layer 2: topics/*.tmem (structured topic files)
    Layer 3: sessions/*.jsonl (immutable logs)

    Enhancements:
    - SQLite index for fast retrieval
    - packed 4/6-bit quantization
    - centroid prefilter for topic selection
    - contradiction detection + confidence decay
    - safe consolidator lock helpers
    """

    def __init__(
        self,
        root: str = "turbomemory_data",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.root = root
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.memory_index_path = os.path.join(root, "MEMORY.md")
        self.topics_dir = os.path.join(root, "topics")
        self.sessions_dir = os.path.join(root, "sessions")
        self.lock_dir = os.path.join(root, "lock")
        self.db_dir = os.path.join(root, "db")
        self.db_path = os.path.join(self.db_dir, "index.sqlite")

        ensure_dir(root)
        ensure_dir(self.topics_dir)
        ensure_dir(self.sessions_dir)
        ensure_dir(self.lock_dir)
        ensure_dir(self.db_dir)

        if not os.path.exists(self.memory_index_path):
            with open(self.memory_index_path, "w", encoding="utf-8") as f:
                f.write("# TurboMemory Index (v0.2)\n")

        self._init_db()

    # ----------------------------
    # DB helpers
    # ----------------------------
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        conn = self._connect()
        try:
            conn.executescript(INDEX_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # ----------------------------
    # Layer 3: Session logs
    # ----------------------------
    def add_turn(self, role: str, text: str, session_file: Optional[str] = None) -> str:
        if session_file is None:
            session_file = datetime.utcnow().strftime("%Y-%m-%d") + ".jsonl"

        path = os.path.join(self.sessions_dir, session_file)

        record = {"ts": now_iso(), "role": role, "text": text}

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return os.path.relpath(path, self.root)

    # ----------------------------
    # Layer 2: Topic file IO
    # ----------------------------
    def _topic_path(self, topic: str) -> str:
        return os.path.join(self.topics_dir, safe_topic_filename(topic) + ".tmem")

    def load_topic(self, topic: str) -> Dict[str, Any]:
        path = self._topic_path(topic)
        if not os.path.exists(path):
            return {
                "topic": topic,
                "created": now_iso(),
                "updated": now_iso(),
                "centroid_q": None,
                "chunks": []
            }
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_topic(self, topic_data: Dict[str, Any]):
        topic_data["updated"] = now_iso()
        path = self._topic_path(topic_data["topic"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(topic_data, f, ensure_ascii=False, indent=2)

    # ----------------------------
    # Layer 1: MEMORY.md
    # ----------------------------
    def read_index_lines(self) -> List[str]:
        with open(self.memory_index_path, "r", encoding="utf-8") as f:
            return f.readlines()

    def write_index_lines(self, lines: List[str]):
        content = "".join(lines)
        if len(content.encode("utf-8")) > 4096:
            header = [lines[0]] if lines else ["# TurboMemory Index (v0.2)\n"]
            body = [ln for ln in lines[1:] if ln.strip()][:30]
            content = "".join(header + body)

        with open(self.memory_index_path, "w", encoding="utf-8") as f:
            f.write(content)

    def update_memory_md(self, topic: str, summary: str, score: float, file_ref: str):
        lines = self.read_index_lines()
        if not lines:
            lines = ["# TurboMemory Index (v0.2)\n"]

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
                    f"{score:.2f} | {datetime.utcnow().date()} | {file_ref}\n"
                )
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        if not found:
            entry_id = safe_topic_filename(topic)[:5]
            new_line = (
                f"{entry_id} | {topic} | {summary[:150]} | "
                f"{score:.2f} | {datetime.utcnow().date()} | {file_ref}\n"
            )
            new_lines.append(new_line)

        self.write_index_lines(new_lines)

    # ----------------------------
    # Contradiction detection (simple heuristic)
    # ----------------------------
    def _is_contradiction(self, old_text: str, new_text: str) -> bool:
        """
        v0.2 heuristic contradiction detection:
        - if they are highly similar but contain different numeric values
        - or contains negation patterns with same keywords
        """
        old_l = old_text.lower()
        new_l = new_text.lower()

        neg_words = ["not", "never", "no", "without", "can't", "cannot", "won't", "doesn't"]
        old_neg = any(w in old_l for w in neg_words)
        new_neg = any(w in new_l for w in neg_words)

        # numeric conflict check
        def extract_numbers(s):
            nums = []
            buf = ""
            for ch in s:
                if ch.isdigit() or ch == ".":
                    buf += ch
                else:
                    if buf:
                        try:
                            nums.append(float(buf))
                        except:
                            pass
                        buf = ""
            if buf:
                try:
                    nums.append(float(buf))
                except:
                    pass
            return nums

        o_nums = extract_numbers(old_l)
        n_nums = extract_numbers(new_l)

        # if both have numbers and sets differ
        if o_nums and n_nums and set(o_nums) != set(n_nums):
            return True

        # negation mismatch + overlapping keywords
        if old_neg != new_neg:
            # if share at least 3 words
            ow = set([w for w in old_l.split() if len(w) > 3])
            nw = set([w for w in new_l.split() if len(w) > 3])
            if len(ow.intersection(nw)) >= 3:
                return True

        return False

    # ----------------------------
    # Add memory chunk + update DB
    # ----------------------------
    def add_memory(
        self,
        topic: str,
        text: str,
        confidence: float = 0.8,
        bits: int = 6,
        source_ref: Optional[str] = None,
        contradiction_decay: float = 0.6
    ):
        if bits not in (4, 6, 8):
            raise ValueError("bits must be 4, 6, or 8")

        emb = self.model.encode([text])[0].astype(np.float32)
        emb_q = quantize_packed(emb, bits=bits)

        topic_data = self.load_topic(topic)

        # contradiction scan (topic-local)
        for c in topic_data["chunks"]:
            if self._is_contradiction(c.get("text", ""), text):
                # decay old chunk confidence
                c["confidence"] = float(c.get("confidence", 0.5) * contradiction_decay)
                c["staleness"] = float(min(1.0, c.get("staleness", 0.0) + 0.25))

        chunk_id = f"c{len(topic_data['chunks'])+1:04d}"
        entropy_score = float(min(1.0, len(text) / 500.0))

        chunk = {
            "chunk_id": chunk_id,
            "timestamp": now_iso(),
            "confidence": float(confidence),
            "staleness": 0.0,
            "entropy": entropy_score,
            "text": text,
            "embedding_q": emb_q,
            "source_refs": [source_ref] if source_ref else []
        }

        topic_data["chunks"].append(chunk)

        # centroid recompute
        all_embs = [dequantize_packed(c["embedding_q"]) for c in topic_data["chunks"]]
        centroid = np.mean(np.vstack(all_embs), axis=0)
        topic_data["centroid_q"] = quantize_packed(centroid, bits=8)

        self.save_topic(topic_data)

        file_ref = os.path.relpath(self._topic_path(topic), self.root)
        self.update_memory_md(topic, summary=text, score=confidence, file_ref=file_ref)

        # update sqlite
        self._upsert_topic_to_db(topic_data, file_ref)
        self._upsert_chunk_to_db(topic, chunk)

        # also update decayed chunks if contradictions occurred
        for c in topic_data["chunks"][:-1]:
            self._upsert_chunk_to_db(topic, c)

    def _upsert_topic_to_db(self, topic_data: Dict[str, Any], file_ref: str):
        centroid_q = topic_data.get("centroid_q")
        if centroid_q is None:
            return

        blob = base64.b64decode(centroid_q["data"])
        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO topics(topic, file_ref, updated, chunk_count, centroid_bits, centroid_scale, centroid_qmax, centroid_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    file_ref=excluded.file_ref,
                    updated=excluded.updated,
                    chunk_count=excluded.chunk_count,
                    centroid_bits=excluded.centroid_bits,
                    centroid_scale=excluded.centroid_scale,
                    centroid_qmax=excluded.centroid_qmax,
                    centroid_blob=excluded.centroid_blob
            """, (
                topic_data["topic"],
                file_ref,
                topic_data.get("updated", now_iso()),
                len(topic_data.get("chunks", [])),
                int(centroid_q["bits"]),
                float(centroid_q["scale"]),
                int(centroid_q["qmax"]),
                sqlite3.Binary(blob)
            ))
            conn.commit()
        finally:
            conn.close()

    def _upsert_chunk_to_db(self, topic: str, chunk: Dict[str, Any]):
        emb_q = chunk["embedding_q"]
        blob = base64.b64decode(emb_q["data"])
        text = chunk.get("text", "")
        preview = text[:200]
        th = sha1_text(text)

        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO chunks(
                    topic, chunk_id, confidence, staleness, entropy, ts, text_hash, text_preview,
                    emb_bits, emb_scale, emb_qmax, emb_blob
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic, chunk_id) DO UPDATE SET
                    confidence=excluded.confidence,
                    staleness=excluded.staleness,
                    entropy=excluded.entropy,
                    ts=excluded.ts,
                    text_hash=excluded.text_hash,
                    text_preview=excluded.text_preview,
                    emb_bits=excluded.emb_bits,
                    emb_scale=excluded.emb_scale,
                    emb_qmax=excluded.emb_qmax,
                    emb_blob=excluded.emb_blob
            """, (
                topic,
                chunk["chunk_id"],
                float(chunk.get("confidence", 0.5)),
                float(chunk.get("staleness", 0.0)),
                float(chunk.get("entropy", 0.0)),
                chunk.get("timestamp", now_iso()),
                th,
                preview,
                int(emb_q["bits"]),
                float(emb_q["scale"]),
                int(emb_q["qmax"]),
                sqlite3.Binary(blob)
            ))
            conn.commit()
        finally:
            conn.close()

    # ----------------------------
    # Fast topic prefilter (centroid search)
    # ----------------------------
    def _get_all_topic_centroids(self) -> List[Tuple[str, np.ndarray]]:
        conn = self._connect()
        try:
            cur = conn.execute("SELECT topic, centroid_bits, centroid_scale, centroid_qmax, centroid_blob FROM topics")
            rows = cur.fetchall()
        finally:
            conn.close()

        out = []
        for topic, bits, scale, qmax, blob in rows:
            qobj = {
                "bits": int(bits),
                "scale": float(scale),
                "qmax": int(qmax),
                "shape": [384],  # MiniLM default
                "data": base64.b64encode(blob).decode("utf-8")
            }
            out.append((topic, dequantize_packed(qobj)))
        return out

    def _select_topics(self, qemb: np.ndarray, top_t: int = 5) -> List[str]:
        centroids = self._get_all_topic_centroids()
        if not centroids:
            return []

        scored = []
        for topic, cemb in centroids:
            scored.append((cosine_sim(qemb, cemb), topic))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in scored[:top_t]]

    # ----------------------------
    # Query (fast mode via DB)
    # ----------------------------
    def query(
        self,
        query_text: str,
        k: int = 5,
        top_topics: int = 5,
        min_confidence: float = 0.0
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        qemb = self.model.encode([query_text])[0].astype(np.float32)

        candidate_topics = self._select_topics(qemb, top_t=top_topics)

        if not candidate_topics:
            # fallback to scanning topic files
            return self.query_slow(query_text, k=k)

        conn = self._connect()
        try:
            placeholders = ",".join("?" for _ in candidate_topics)
            cur = conn.execute(f"""
                SELECT topic, chunk_id, confidence, staleness, entropy, ts, text_preview,
                       emb_bits, emb_scale, emb_qmax, emb_blob
                FROM chunks
                WHERE topic IN ({placeholders})
            """, candidate_topics)

            rows = cur.fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            topic, chunk_id, conf, stale, ent, ts, preview, bits, scale, qmax, blob = row

            if conf < min_confidence:
                continue

            qobj = {
                "bits": int(bits),
                "scale": float(scale),
                "qmax": int(qmax),
                "shape": [384],
                "data": base64.b64encode(blob).decode("utf-8")
            }
            emb = dequantize_packed(qobj)
            sim = cosine_sim(qemb, emb)

            chunk = {
                "chunk_id": chunk_id,
                "confidence": float(conf),
                "staleness": float(stale),
                "entropy": float(ent),
                "timestamp": ts,
                "text": preview
            }

            # rank boost for confidence, penalty for staleness
            final_score = sim * (0.5 + conf) * (1.0 - 0.5 * stale)
            results.append((final_score, topic, chunk))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def query_slow(self, query_text: str, k: int = 5, topic: Optional[str] = None):
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
                with open(path, "r", encoding="utf-8") as f:
                    topic_data = json.load(f)
                tname = topic_data["topic"]
                for c in topic_data["chunks"]:
                    emb = dequantize_packed(c["embedding_q"])
                    sim = cosine_sim(qemb, emb)
                    results.append((sim, tname, c))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    # ----------------------------
    # Rebuild SQLite from topic files (repair tool)
    # ----------------------------
    def rebuild_index(self):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM chunks;")
            conn.execute("DELETE FROM topics;")
            conn.commit()
        finally:
            conn.close()

        for fn in os.listdir(self.topics_dir):
            if not fn.endswith(".tmem"):
                continue

            path = os.path.join(self.topics_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                topic_data = json.load(f)

            file_ref = os.path.relpath(path, self.root)
            self._upsert_topic_to_db(topic_data, file_ref)

            for c in topic_data.get("chunks", []):
                self._upsert_chunk_to_db(topic_data["topic"], c)

    # ----------------------------
    # Stats
    # ----------------------------
    def stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            t = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            c = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()

        return {
            "root": self.root,
            "model": self.model_name,
            "topics": int(t),
            "chunks": int(c),
            "db_path": self.db_path
        }
