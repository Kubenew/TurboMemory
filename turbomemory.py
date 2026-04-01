import os
import json
import base64
import time
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_topic_filename(topic: str):
    return topic.replace(".", "_").replace("/", "_")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def quantize_vector(vec: np.ndarray, bits: int = 6) -> Dict[str, Any]:
    """
    Symmetric quantization for embedding vectors.
    Stores int16 packed bytes for simplicity (v0.1).
    """
    vec = vec.astype(np.float32)

    qmax = (2 ** (bits - 1)) - 1  # signed
    vmax = float(np.max(np.abs(vec)) + 1e-9)

    scale = vmax / qmax
    q = np.round(vec / scale).astype(np.int16)

    return {
        "bits": bits,
        "scale": float(scale),
        "zero": 0,
        "dtype": "int16",
        "shape": list(vec.shape),
        "data": base64.b64encode(q.tobytes()).decode("utf-8")
    }


def dequantize_vector(qobj: Dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(qobj["data"])
    dtype = np.dtype(qobj.get("dtype", "int16"))
    shape = tuple(qobj["shape"])
    q = np.frombuffer(raw, dtype=dtype).reshape(shape)

    scale = float(qobj["scale"])
    return q.astype(np.float32) * scale


class TurboMemory:
    """
    TurboMemory-v0.1
    Layer 1: MEMORY.md (index, always loaded)
    Layer 2: topics/*.tmem (structured topic knowledge)
    Layer 3: sessions/*.jsonl (immutable logs, evidence)
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

        ensure_dir(root)
        ensure_dir(self.topics_dir)
        ensure_dir(self.sessions_dir)
        ensure_dir(self.lock_dir)

        if not os.path.exists(self.memory_index_path):
            with open(self.memory_index_path, "w", encoding="utf-8") as f:
                f.write("# TurboMemory Index (v0.1)\n")

    def add_turn(self, role: str, text: str, session_file: Optional[str] = None) -> str:
        if session_file is None:
            session_file = datetime.utcnow().strftime("%Y-%m-%d") + ".jsonl"

        path = os.path.join(self.sessions_dir, session_file)

        record = {
            "ts": now_iso(),
            "role": role,
            "text": text
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return os.path.relpath(path, self.root)

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

    def read_index_lines(self) -> List[str]:
        with open(self.memory_index_path, "r", encoding="utf-8") as f:
            return f.readlines()

    def write_index_lines(self, lines: List[str]):
        content = "".join(lines)

        if len(content.encode("utf-8")) > 4096:
            header = [lines[0]] if lines else ["# TurboMemory Index (v0.1)\n"]
            body = [ln for ln in lines[1:] if ln.strip()][:30]
            content = "".join(header + body)

        with open(self.memory_index_path, "w", encoding="utf-8") as f:
            f.write(content)

    def update_memory_md(self, topic: str, summary: str, score: float, file_ref: str):
        lines = self.read_index_lines()

        if not lines:
            lines = ["# TurboMemory Index (v0.1)\n"]

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
            entry_id = hex(int(time.time() * 1000))[-5:]
            new_line = (
                f"{entry_id} | {topic} | {summary[:150]} | "
                f"{score:.2f} | {datetime.utcnow().date()} | {file_ref}\n"
            )
            new_lines.append(new_line)

        self.write_index_lines(new_lines)

    def add_memory(
        self,
        topic: str,
        text: str,
        confidence: float = 0.8,
        bits: int = 6,
        source_ref: Optional[str] = None
    ):
        emb = self.model.encode([text])[0].astype(np.float32)
        emb_q = quantize_vector(emb, bits=bits)

        topic_data = self.load_topic(topic)

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

        all_embs = [dequantize_vector(c["embedding_q"]) for c in topic_data["chunks"]]
        centroid = np.mean(np.vstack(all_embs), axis=0)

        topic_data["centroid_q"] = quantize_vector(centroid, bits=8)

        self.save_topic(topic_data)

        file_ref = os.path.relpath(self._topic_path(topic), self.root)
        self.update_memory_md(topic, summary=text, score=confidence, file_ref=file_ref)

    def query(
        self,
        query_text: str,
        k: int = 5,
        topic: Optional[str] = None
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        qemb = self.model.encode([query_text])[0].astype(np.float32)

        results = []

        if topic:
            topic_data = self.load_topic(topic)
            for c in topic_data["chunks"]:
                emb = dequantize_vector(c["embedding_q"])
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
                    emb = dequantize_vector(c["embedding_q"])
                    sim = cosine_sim(qemb, emb)
                    results.append((sim, tname, c))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def stats(self) -> Dict[str, Any]:
        topics = [f for f in os.listdir(self.topics_dir) if f.endswith(".tmem")]
        total_chunks = 0

        for t in topics:
            with open(os.path.join(self.topics_dir, t), "r", encoding="utf-8") as f:
                td = json.load(f)
                total_chunks += len(td.get("chunks", []))

        return {
            "root": self.root,
            "model": self.model_name,
            "topics": len(topics),
            "chunks": total_chunks
        }
