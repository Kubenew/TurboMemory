"""Scoring fusion for hybrid vector + keyword search."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Configuration for scoring fusion."""
    vector_weight: float = 0.6
    keyword_weight: float = 0.3
    recency_weight: float = 0.1
    confidence_weight: float = 0.0
    
    # Optional thresholds
    min_keyword_score: float = 0.0
    min_vector_score: float = 0.0
    
    # Normalization method
    normalize_method: str = "minmax"  # "minmax", "zscore", or "none"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "recency_weight": self.recency_weight,
            "confidence_weight": self.confidence_weight,
            "min_keyword_score": self.min_keyword_score,
            "min_vector_score": self.min_vector_score,
            "normalize_method": self.normalize_method,
        }


@dataclass
class FusionResult:
    """Result of fused search."""
    doc_id: str
    topic: str
    text: str
    final_score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    recency_score: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FusionScorer:
    """Fuses multiple scoring signals into a single ranking.
    
    Combines:
    - Vector similarity (semantic search)
    - BM25 keyword score
    - Recency (time-based boost)
    - Confidence (chunk quality)
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
    
    def _normalize_scores(
        self, 
        scores: List[float], 
        method: str = "minmax"
    ) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []
        
        if method == "minmax":
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [1.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        elif method == "zscore":
            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                return [0.5] * len(scores)
            return [(s - mean) / std for s in scores]
        
        else:
            return scores
    
    def _compute_recency_score(self, ts: str) -> float:
        """Compute recency score based on timestamp.
        
        More recent = higher score (exponential decay).
        """
        from datetime import datetime, timezone, timedelta
        
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            
            # Time since creation (in days)
            age_days = (now - dt).total_seconds() / 86400
            
            # Exponential decay: half-life of 30 days
            score = np.exp(-0.023 * age_days)  # ln(0.5) / 30
            
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default to middle if parsing fails
    
    def fuse(
        self,
        vector_results: List[Tuple[float, str, Dict[str, Any]]],
        keyword_results: List[Dict[str, Any]],
        timestamps: Optional[Dict[str, str]] = None,
        confidences: Optional[Dict[str, float]] = None,
    ) -> List[FusionResult]:
        """Fuse multiple search results into a single ranking."""
        
        # Build score dictionaries
        vector_scores = {(r[1], r[2].get("chunk_key", "")): r[0] for r in vector_results}
        keyword_scores = {(r.topic, r.chunk_key): r.score for r in keyword_results}
        
        # Get all unique documents
        all_docs = set(vector_scores.keys()) | set(keyword_scores.keys())
        
        # Build input lists for normalization
        vec_list = [vector_scores.get(d, 0.0) for d in all_docs]
        kw_list = [keyword_scores.get(d, 0.0) for d in all_docs]
        
        # Normalize
        norm_vec = self._normalize_scores(vec_list, self.config.normalize_method)
        norm_kw = self._normalize_scores(kw_list, self.config.normalize_method)
        
        # Build score dictionaries
        doc_vector = {d: s for d, s in zip(all_docs, norm_vec)}
        doc_keyword = {d: s for d, s in zip(all_docs, norm_kw)}
        
        # Compute combined scores
        results = []
        
        for doc in all_docs:
            topic, chunk_key = doc
            
            vec_s = doc_vector.get(doc, 0.0)
            kw_s = doc_keyword.get(doc, 0.0)
            
            # Apply minimum thresholds
            if kw_s < self.config.min_keyword_score:
                kw_s = 0.0
            if vec_s < self.config.min_vector_score:
                vec_s = 0.0
            
            # Compute recency
            ts = timestamps.get(chunk_key, "") if timestamps else ""
            recency_s = self._compute_recency_score(ts) if ts else 0.5
            
            # Compute confidence
            conf = confidences.get(chunk_key, 0.5) if confidences else 0.5
            
            # Weighted combination
            final = (
                vec_s * self.config.vector_weight +
                kw_s * self.config.keyword_weight +
                recency_s * self.config.recency_score +
                conf * self.config.confidence_weight
            )
            
            # Get text from results
            text = ""
            for r in vector_results:
                if r[1] == topic and r[2].get("chunk_key") == chunk_key:
                    text = r[2].get("text", "")
                    break
            
            if not text:
                for r in keyword_results:
                    if r.topic == topic and r.chunk_key == chunk_key:
                        text = r.text
                        break
            
            results.append(FusionResult(
                doc_id=chunk_key,
                topic=topic,
                text=text,
                final_score=final,
                vector_score=vec_s,
                keyword_score=kw_s,
                recency_score=recency_s,
                confidence_score=conf,
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def explain_score(
        self,
        result: FusionResult,
    ) -> Dict[str, Any]:
        """Explain how a result's score was computed."""
        return {
            "doc_id": result.doc_id,
            "topic": result.topic,
            "final_score": round(result.final_score, 4),
            "breakdown": {
                "vector": {
                    "score": round(result.vector_score, 4),
                    "weight": self.config.vector_weight,
                    "contribution": round(result.vector_score * self.config.vector_weight, 4),
                },
                "keyword": {
                    "score": round(result.keyword_score, 4),
                    "weight": self.config.keyword_weight,
                    "contribution": round(result.keyword_score * self.config.keyword_weight, 4),
                },
                "recency": {
                    "score": round(result.recency_score, 4),
                    "weight": self.config.recency_weight,
                    "contribution": round(result.recency_score * self.config.recency_weight, 4),
                },
                "confidence": {
                    "score": round(result.confidence_score, 4),
                    "weight": self.config.confidence_weight,
                    "contribution": round(result.confidence_score * self.config.confidence_weight, 4),
                },
            },
        }
