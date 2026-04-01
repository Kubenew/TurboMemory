import pytest
import os
import json
import shutil
import tempfile
import numpy as np
from datetime import datetime, timezone, timedelta

from turbomemory.turbomemory import (
    TurboMemory,
    TurboMemoryConfig,
    quantize_packed,
    dequantize_packed,
    cosine_sim,
    pack_unsigned,
    unpack_unsigned,
    now_iso,
    sha1_text,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tm(temp_dir):
    with TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2") as memory:
        yield memory


class TestQuantization:
    def test_quantize_dequantize_6bit(self):
        vec = np.random.randn(384).astype(np.float32)
        q = quantize_packed(vec, bits=6)
        assert q["bits"] == 6
        assert q["scale"] > 0
        assert q["qmax"] == 31
        assert q["shape"] == [384]
        assert "data" in q

        reconstructed = dequantize_packed(q)
        assert reconstructed.shape == vec.shape
        sim = cosine_sim(vec, reconstructed)
        assert sim > 0.95

    def test_quantize_dequantize_4bit(self):
        vec = np.random.randn(384).astype(np.float32)
        q = quantize_packed(vec, bits=4)
        assert q["bits"] == 4
        assert q["qmax"] == 7

        reconstructed = dequantize_packed(q)
        assert reconstructed.shape == vec.shape
        sim = cosine_sim(vec, reconstructed)
        assert sim > 0.90

    def test_quantize_dequantize_8bit(self):
        vec = np.random.randn(384).astype(np.float32)
        q = quantize_packed(vec, bits=8)
        assert q["bits"] == 8
        assert q["qmax"] == 127

        reconstructed = dequantize_packed(q)
        assert reconstructed.shape == vec.shape
        sim = cosine_sim(vec, reconstructed)
        assert sim > 0.99

    def test_invalid_bits(self):
        vec = np.random.randn(384)
        with pytest.raises(ValueError):
            quantize_packed(vec, bits=5)

    def test_pack_unpack_roundtrip(self):
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint16)
        packed = pack_unsigned(values, bits=4)
        unpacked = unpack_unsigned(packed, bits=4, n_values=8)
        np.testing.assert_array_equal(values, unpacked)

    def test_pack_unpack_6bit(self):
        values = np.array([0, 10, 20, 30, 40, 50, 60, 63], dtype=np.uint16)
        packed = pack_unsigned(values, bits=6)
        unpacked = unpack_unsigned(packed, bits=6, n_values=8)
        np.testing.assert_array_equal(values, unpacked)


class TestCosineSim:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_sim(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_sim(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        assert cosine_sim(a, b) < -0.99

    def test_zero_vector(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_sim(a, b) == 0.0


class TestTurboMemory:
    def test_add_memory(self, tm):
        chunk_id = tm.add_memory("test.topic", "Python is a programming language")
        assert chunk_id == "c0001"

    def test_add_multiple_chunks(self, tm):
        tm.add_memory("test.topic", "Python is a programming language")
        tm.add_memory("test.topic", "JavaScript is used for web development")
        tm.add_memory("test.topic", "Rust is a systems programming language")

        stats = tm.stats()
        assert stats["chunks"] == 3
        assert stats["topics"] == 1

    def test_query(self, tm):
        tm.add_memory("python", "Python is a programming language created by Guido van Rossum")
        tm.add_memory("javascript", "JavaScript is used for web development")

        results = tm.query("programming language", k=2)
        assert len(results) > 0

        scores = [score for score, _, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_min_confidence(self, tm):
        tm.add_memory("test", "Some text here", confidence=0.3)
        tm.add_memory("test", "Important text here", confidence=0.9)

        results = tm.query("text", min_confidence=0.5)
        for score, _, chunk in results:
            assert chunk["confidence"] >= 0.5

    def test_topic_isolation(self, tm):
        tm.add_memory("topic_a", "Content about topic A")
        tm.add_memory("topic_b", "Content about topic B")

        stats = tm.stats()
        assert stats["topics"] == 2

    def test_add_turn(self, tm):
        ref = tm.add_turn("user", "Hello, how are you?")
        assert ref.endswith(".jsonl")

        session_path = os.path.join(tm.sessions_dir, ref)
        with open(session_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["role"] == "user"
        assert record["text"] == "Hello, how are you?"

    def test_stats(self, tm):
        tm.add_memory("test", "Test content")
        stats = tm.stats()

        assert stats["topics"] == 1
        assert stats["chunks"] == 1
        assert "root" in stats
        assert "model" in stats
        assert "db_path" in stats
        assert "storage_bytes" in stats

    def test_rebuild_index(self, tm):
        tm.add_memory("test", "Test content for rebuild")

        stats_before = tm.stats()
        tm.rebuild_index()
        stats_after = tm.stats()

        assert stats_before["chunks"] == stats_after["chunks"]
        assert stats_before["topics"] == stats_after["topics"]

    def test_context_manager(self, temp_dir):
        with TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2") as tm:
            tm.add_memory("test", "Test content")
            assert tm.stats()["chunks"] == 1


class TestTTL:
    def test_ttl_expiration(self, temp_dir):
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        try:
            tm.add_memory("test", "Temporary content", ttl_days=0.00001)

            expired = tm.expire_ttl()
            assert expired == 1

            stats = tm.stats()
            assert stats["chunks"] == 0
        finally:
            tm.close()

    def test_no_expiration(self, temp_dir):
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        try:
            tm.add_memory("test", "Permanent content", ttl_days=365)

            expired = tm.expire_ttl()
            assert expired == 0

            stats = tm.stats()
            assert stats["chunks"] == 1
        finally:
            tm.close()


class TestBackupRestore:
    def test_backup_and_restore(self, temp_dir):
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        try:
            tm.add_memory("test", "Content to backup")
            tm.add_memory("test2", "More content")

            backup_path = os.path.join(temp_dir, "backup")
            tm.backup(backup_path)

            assert os.path.exists(os.path.join(backup_path, "topics"))
            assert os.path.exists(os.path.join(backup_path, "db"))
            assert os.path.exists(os.path.join(backup_path, "MEMORY.md"))
            assert os.path.exists(os.path.join(backup_path, "backup.json"))
        finally:
            tm.close()

    def test_restore(self, temp_dir):
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        try:
            tm.add_memory("test", "Content to backup")

            backup_path = os.path.join(temp_dir, "backup")
            tm.backup(backup_path)

            tm.restore(backup_path)
            stats = tm.stats()
            assert stats["chunks"] == 1
        finally:
            tm.close()


class TestBulkOperations:
    def test_bulk_import(self, temp_dir):
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        try:
            items = [
                {"topic": "python", "text": "Python is great", "confidence": 0.9},
                {"topic": "python", "text": "Python has many libraries", "confidence": 0.8},
                {"topic": "rust", "text": "Rust is fast", "confidence": 0.7},
            ]
            count = tm.bulk_import(items)
            assert count == 3

            stats = tm.stats()
            assert stats["chunks"] == 3
            assert stats["topics"] == 2
        finally:
            tm.close()

    def test_export_topic(self, temp_dir):
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        try:
            tm.add_memory("test", "Export test content")

            data = tm.export_topic("test", include_embeddings=False)
            assert data["topic"] == "test"
            assert len(data["chunks"]) == 1

            for chunk in data["chunks"]:
                assert "embedding_q" not in chunk
        finally:
            tm.close()

    def test_export_all(self, temp_dir):
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        try:
            tm.add_memory("topic1", "Content 1")
            tm.add_memory("topic2", "Content 2")

            all_data = tm.export_all(include_embeddings=False)
            assert len(all_data) == 2
        finally:
            tm.close()


class TestTopicManagement:
    def test_merge_topics(self, temp_dir):
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        try:
            tm.add_memory("python_old", "Python is a language")
            tm.add_memory("python_new", "Python has many uses")

            merged = tm.merge_topics("python_old", "python_new")
            assert merged >= 1

            stats = tm.stats()
            assert stats["topics"] == 2
        finally:
            tm.close()

    def test_split_topic(self, temp_dir):
        tm = TurboMemory(root=temp_dir, model_name="all-MiniLM-L6-v2")
        try:
            tm.add_memory("mixed", "Python content")
            tm.add_memory("mixed", "Rust content")

            moved = tm.split_topic("mixed", ["c0001"], "python")
            assert moved == 1

            stats = tm.stats()
            assert stats["topics"] == 2
        finally:
            tm.close()


class TestConfig:
    def test_config_from_file(self, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        config = TurboMemoryConfig(root=temp_dir, default_bits=4, default_ttl_days=30)
        config.to_file(config_path)

        loaded = TurboMemoryConfig.from_file(config_path)
        assert loaded.root == temp_dir
        assert loaded.default_bits == 4
        assert loaded.default_ttl_days == 30

    def test_config_in_memory(self, temp_dir):
        config = TurboMemoryConfig(
            root=temp_dir,
            model_name="all-MiniLM-L6-v2",
            default_bits=8,
            pool_size=3,
        )
        tm = TurboMemory(config=config)
        try:
            assert tm.config.default_bits == 8
            assert tm.config.pool_size == 3
        finally:
            tm.close()


class TestContradictionDetection:
    def test_numeric_contradiction(self, tm):
        tm.add_memory("test", "The temperature is 25 degrees")
        tm.add_memory("test", "The temperature is 30 degrees")

        topic_data = tm.load_topic("test")
        decayed_chunks = [c for c in topic_data["chunks"] if c["confidence"] < 0.8]
        assert len(decayed_chunks) > 0

    def test_negation_contradiction(self, tm):
        tm.add_memory("test", "Python is not a compiled language")
        tm.add_memory("test", "Python is a compiled language")

        topic_data = tm.load_topic("test")
        decayed_chunks = [c for c in topic_data["chunks"] if c["confidence"] < 0.8]
        assert len(decayed_chunks) > 0
