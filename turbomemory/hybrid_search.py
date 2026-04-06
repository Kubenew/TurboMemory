"""Hybrid search combining BM25 + vector fusion."""

import math
import os
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class BM25Config:
    """Configuration for BM25 ranking."""
    k1: float = 1.5
    b: float = 0.75
    avgdl: float = 0.0
    idf_cache: Optional[Dict[str, float]] = None


class BM25:
    """BM25 ranking implementation for keyword search."""
    
    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        self.doc_lengths: Dict[str, int] = {}
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.avgdl: float = 0.0
        self.N: int = 0
        self.corpus: Dict[str, List[str]] = {}
    
    def index_documents(self, documents: Dict[str, str]) -> None:
        """Index documents for BM25 search."""
        self.corpus = {k: self._tokenize(v) for k, v in documents.items()}
        self.N = len(self.corpus)
        
        # Calculate document frequencies
        self.doc_freqs = defaultdict(int)
        for doc_tokens in self.corpus.values():
            for token in set(doc_tokens):
                self.doc_freqs[token] += 1
        
        # Calculate average document length
        if self.corpus:
            self.doc_lengths = {k: len(v) for k, v in self.corpus.items()}
            self.avgdl = sum(self.doc_lengths.values()) / self.N
            self.config.avgdl = self.avgdl
        
        # Calculate IDF
        self.config.idf_cache = {}
        for term, df in self.doc_freqs.items():
            self.config.idf_cache[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def score(self, query: str, doc_id: str) -> float:
        """Calculate BM25 score for a document."""
        if doc_id not in self.corpus:
            return 0.0
        
        query_terms = self._tokenize(query)
        doc_terms = self.corpus[doc_id]
        doc_len = self.doc_lengths.get(doc_id, 0)
        
        score = 0.0
        doc_tf = defaultdict(int)
        for term in doc_terms:
            doc_tf[term] += 1
        
        for term in query_terms:
            if term not in self.config.idf_cache:
                continue
            
            tf = doc_tf.get(term, 0)
            idf = self.config.idf_cache[term]
            
            # BM25 formula
            numerator = tf * (self.config.k1 + 1)
            denominator = tf + self.config.k1 * (1 - self.config.b + self.config.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for top-k documents."""
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in self.corpus]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridSearch:
    """Combines BM25 + vector search with configurable fusion."""
    
    FUSION_METHODS = ["linear", "rrf", "colbert"]
    
    def __init__(
        self,
        bm25: Optional[BM25] = None,
        vector_search_fn: Optional[callable] = None,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
    ):
        self.bm25 = bm25 or BM25()
        self.vector_search_fn = vector_search_fn
        self.fusion_method = fusion_method
        self.alpha = alpha  # Weight for vector vs BM25
    
    def index_documents(self, documents: Dict[str, str]) -> None:
        """Index documents for hybrid search."""
        self.bm25.index_documents(documents)
    
    def search(
        self,
        query: str,
        vector_query: Optional[np.ndarray] = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float, str]]:
        """
        Perform hybrid search.
        
        Returns: List of (doc_id, score, source) where source is "bm25", "vector", or "hybrid"
        """
        results = []
        
        # BM25 search
        if self.fusion_method == "linear":
            bm25_results = self.bm25.search(query, top_k * 2)
            for doc_id, score in bm25_results:
                results.append((doc_id, score * self.alpha, "bm25"))
        
        # Vector search
        if vector_query is not None and self.vector_search_fn is not None:
            vector_results = self.vector_search_fn(query, top_k * 2)
            for doc_id, score in vector_results:
                if self.fusion_method == "linear":
                    # Linear combination
                    existing = next((i for i, (d, _, s) in enumerate(results) if d == doc_id), None)
                    if existing is not None:
                        results[existing] = (doc_id, results[existing][1] + score * (1 - self.alpha), "hybrid")
                    else:
                        results.append((doc_id, score * (1 - self.alpha), "vector"))
                else:
                    results.append((doc_id, score, "vector"))
        
        # RRF (Reciprocal Rank Fusion)
        if self.fusion_method == "rrf":
            rrf_scores = defaultdict(float)
            k = 60  # RRF parameter
            
            # BM25 rankings
            for rank, (doc_id, score) in enumerate(self.bm25.search(query, top_k * 2)):
                rrf_scores[doc_id] += 1 / (k + rank)
            
            # Vector rankings
            if vector_query is not None and self.vector_search_fn is not None:
                for rank, (doc_id, score) in enumerate(self.vector_search_fn(query, top_k * 2)):
                    rrf_scores[doc_id] += 1 / (k + rank)
            
            results = [(doc_id, score, "hybrid") for doc_id, score in rrf_scores.items()]
        
        # Filter and sort
        results = [(d, s, src) for d, s, src in results if s >= min_score]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


class HybridSearchEngine:
    """Full hybrid search engine with TMF integration."""
    
    def __init__(self, tmf_root: str):
        self.tmf_root = tmf_root
        self.hybrid = HybridSearch()
        self._load_index()
    
    def _load_index(self) -> None:
        """Load or build the BM25 index."""
        index_path = os.path.join(self.tmf_root, "bm25_index.json")
        
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                data = json.load(f)
                docs = data.get("documents", {})
                self.hybrid.index_documents(docs)
        else:
            # Build from topic files
            topics_dir = os.path.join(self.tmf_root, "topics")
            documents = {}
            
            if os.path.exists(topics_dir):
                for fn in os.listdir(topics_dir):
                    if fn.endswith(".tmem"):
                        topic = fn[:-5]
                        topic_path = os.path.join(topics_dir, fn)
                        try:
                            with open(topic_path, "r") as f:
                                topic_data = json.load(f)
                                for chunk in topic_data.get("chunks", []):
                                    key = f"{topic}:{chunk.get('chunk_id', '')}"
                                    documents[key] = chunk.get("text", "")
                        except (json.JSONDecodeError, IOError):
                            continue
            
            self.hybrid.index_documents(documents)
            self._save_index()
    
    def _save_index(self) -> None:
        """Save BM25 index."""
        index_path = os.path.join(self.tmf_root, "bm25_index.json")
        with open(index_path, "w") as f:
            json.dump({"documents": self.hybrid.bm25.corpus}, f)
    
    def search(
        self,
        query: str,
        vector_fn: Optional[callable] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search with both keyword and vector."""
        # Build vector search function if provided
        vector_search_fn = None
        if vector_fn:
            vector_search_fn = lambda q, k: [(d, s) for d, s, src in vector_fn(q, k) if src != "bm25"]
        
        results = self.hybrid.search(
            query=query,
            vector_query=None,  # Would pass embedding here
            top_k=top_k,
        )
        
        return [
            {
                "doc_id": doc_id,
                "score": score,
                "source": source,
            }
            for doc_id, score, source in results
        ]
    
    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        self.hybrid.bm25.corpus[doc_id] = self.hybrid.bm25._tokenize(text)
        self.hybrid.bm25.doc_lengths[doc_id] = len(self.hybrid.bm25.corpus[doc_id])
        self._save_index()
    
    def rebuild_index(self, documents: Dict[str, str]) -> None:
        """Rebuild the entire BM25 index."""
        self.hybrid = HybridSearch()
        self.hybrid.index_documents(documents)
        self._save_index()