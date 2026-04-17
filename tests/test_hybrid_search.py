import pytest
from turbomemory.hybrid_search import HybridSearchEngine, BM25


class TestHybridSearch:
    @pytest.fixture
    def temp_root(self, tmp_path):
        return str(tmp_path / "hybrid_test")

    @pytest.fixture
    def engine(self, temp_root):
        return HybridSearchEngine(temp_root)

    def test_bm25_basic(self):
        bm25 = BM25()
        docs = [
            "TurboMemory stores semantic chunks efficiently",
            "Vector search with compression",
            "SQLite for metadata indexing",
        ]
        bm25.fit(docs)
        
        scores = bm25.search("semantic storage", top_k=2)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores.values())

    def test_engine_init(self, temp_root):
        engine = HybridSearchEngine(temp_root)
        assert engine.root == temp_root

    def test_engine_search_empty(self, temp_root):
        engine = HybridSearchEngine(temp_root)
        results = engine.search("test query", top_k=5)
        assert isinstance(results, list)