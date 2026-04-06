"""Tests for the retrieval module."""

import pytest
import numpy as np

from turbomemory.retrieval import (
    cosine_similarity,
    NumpyIndex,
    RetrievalEngine,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        assert cosine_similarity(a, b) < -0.99

    def test_zero_vector(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == 0.0


class TestNumpyIndex:
    def test_empty_search(self):
        index = NumpyIndex(dimension=10)
        query = np.random.randn(10)
        results = index.search(query, k=5)
        assert results == []

    def test_add_and_search(self):
        index = NumpyIndex(dimension=10)
        
        vectors = np.random.randn(5, 10)
        ids = ["vec0", "vec1", "vec2", "vec3", "vec4"]
        index.add_vectors(vectors, ids)
        
        query = vectors[2]  # Search for similar to third vector
        results = index.search(query, k=3)
        
        assert len(results) <= 3
        assert results[0][0] == "vec2"  # Should find itself first

    def test_ids_tracking(self):
        index = NumpyIndex(dimension=5)
        vectors = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        index.add_vectors(vectors, ["a", "b"])
        
        results = index.search(np.array([1, 0, 0, 0, 0]), k=2)
        ids = [r[0] for r in results]
        assert "a" in ids


class TestRetrievalEngine:
    def test_creation_numpy(self):
        engine = RetrievalEngine(dimension=10, use_faiss=False)
        assert engine.dimension == 10

    def test_search_without_faiss(self):
        engine = RetrievalEngine(dimension=10, use_faiss=False)
        
        vectors = np.random.randn(5, 10)
        ids = ["vec0", "vec1", "vec2", "vec3", "vec4"]
        engine.add_vectors(vectors, ids)
        
        results = engine.search(vectors[0], k=3)
        assert len(results) <= 3

    def test_filter_ids(self):
        engine = RetrievalEngine(dimension=10, use_faiss=False)
        
        vectors = np.random.randn(5, 10)
        ids = ["vec0", "vec1", "vec2", "vec3", "vec4"]
        engine.add_vectors(vectors, ids)
        
        results = engine.search(vectors[0], k=5, filter_ids=["vec0", "vec2"])
        ids = [r[0] for r in results]
        
        for id_ in ids:
            assert id_ in ["vec0", "vec2"]