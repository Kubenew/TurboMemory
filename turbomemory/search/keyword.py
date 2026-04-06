"""BM25 keyword search implementation."""

import math
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class BM25Result:
    """BM25 search result."""
    chunk_key: str
    topic: str
    score: float
    text: str


class BM25Search:
    """BM25 keyword search for TurboMemory.
    
    Implements Okapi BM25 ranking function with standard parameters:
    - k1 = 1.5 (term frequency saturation)
    - b = 0.75 (document length normalization)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # Inverted index: term -> list of (doc_id, tf)
        self._inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        
        # Document storage
        self._docs: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._avg_doc_len: float = 0.0
        self._doc_count: int = 0
        self._doc_lens: Dict[str, int] = {}
        
        # IDF cache
        self._idf: Dict[str, float] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        
        # Remove stop words (simple list)
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
            'that', 'the', 'to', 'was', 'will', 'with'
        }
        
        return [t for t in tokens if t not in stop_words and len(t) > 1]
    
    def index_document(self, doc_id: str, text: str, **metadata) -> None:
        """Index a document for BM25 search."""
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        
        # Store document
        self._docs[doc_id] = {
            "text": text,
            "tokens": tokens,
            "term_freq": Counter(tokens),
            "metadata": metadata,
        }
        
        # Update document length
        self._doc_lens[doc_id] = doc_len
        
        # Build inverted index
        for term, tf in Counter(tokens).items():
            self._inverted_index[term].append((doc_id, tf))
        
        # Update statistics
        self._doc_count = len(self._docs)
        self._avg_doc_len = sum(self._doc_lens.values()) / max(1, self._doc_count)
        
        # Invalidate IDF cache
        self._idf.clear()
    
    def _compute_idf(self, term: str) -> float:
        """Compute IDF for a term."""
        if term in self._idf:
            return self._idf[term]
        
        # Count documents containing term
        doc_freq = sum(1 for doc_id, _ in self._inverted_index.get(term, []))
        
        if doc_freq == 0:
            idf = 0.0
        else:
            # Standard BM25 IDF formula
            idf = math.log((self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        
        self._idf[term] = idf
        return idf
    
    def score(self, query: str, doc_id: str) -> float:
        """Compute BM25 score for a single document."""
        if doc_id not in self._docs:
            return 0.0
        
        doc = self._docs[doc_id]
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return 0.0
        
        doc_len = self._doc_lens.get(doc_id, 0)
        term_freq = doc["term_freq"]
        
        score = 0.0
        for term in query_terms:
            if term not in self._inverted_index:
                continue
            
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            
            idf = self._compute_idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self._avg_doc_len))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(
        self, 
        query: str, 
        limit: int = 10,
        topics: Optional[List[str]] = None,
    ) -> List[BM25Result]:
        """Search documents by keyword similarity."""
        if not query or self._doc_count == 0:
            return []
        
        query_terms = self._tokenize(query)
        
        # Score all documents
        scores = []
        for doc_id, doc in self._docs.items():
            # Filter by topic if specified
            if topics and doc.get("metadata", {}).get("topic") not in topics:
                continue
            
            bm25_score = self.score(query, doc_id)
            if bm25_score > 0:
                scores.append((bm25_score, doc_id))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results
        results = []
        for score, doc_id in scores[:limit]:
            doc = self._docs[doc_id]
            results.append(BM25Result(
                chunk_key=doc_id,
                topic=doc.get("metadata", {}).get("topic", ""),
                score=score,
                text=doc["text"],
            ))
        
        return results
    
    def get_relevant_terms(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Get terms from query that have entries in the index."""
        query_terms = self._tokenize(query)
        
        relevant = []
        for term in query_terms:
            if term in self._inverted_index:
                idf = self._compute_idf(term)
                doc_count = len(self._inverted_index[term])
                relevant.append((term, idf, doc_count))
        
        # Sort by IDF (most discriminative first)
        relevant.sort(key=lambda x: x[1], reverse=True)
        
        return [(t, idf) for t, idf, _ in relevant[:limit]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics."""
        return {
            "doc_count": self._doc_count,
            "avg_doc_len": round(self._avg_doc_len, 2),
            "unique_terms": len(self._inverted_index),
            "indexed_docs": len(self._docs),
        }
    
    def clear(self) -> None:
        """Clear the index."""
        self._inverted_index.clear()
        self._docs.clear()
        self._doc_lens.clear()
        self._idf.clear()
        self._doc_count = 0
        self._avg_doc_len = 0.0
