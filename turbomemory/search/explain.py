"""Query explanation for debugging and transparency."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class QueryPlan:
    """Query execution plan."""
    query_text: str
    mode: str  # "vector_only", "keyword_only", "hybrid"
    topics_filter: Optional[List[str]] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    k: int = 10
    top_topics: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "mode": self.mode,
            "topics_filter": self.topics_filter,
            "metadata_filters": self.metadata_filters,
            "k": self.k,
            "top_topics": self.top_topics,
        }


@dataclass
class ScoreExplanation:
    """Explanation of a single result's score."""
    doc_id: str
    topic: str
    text_preview: str
    final_score: float
    
    # Score components
    vector_score: float = 0.0
    vector_weight: float = 0.0
    vector_contribution: float = 0.0
    
    keyword_score: float = 0.0
    keyword_weight: float = 0.0
    keyword_contribution: float = 0.0
    
    recency_score: float = 0.0
    recency_weight: float = 0.0
    recency_contribution: float = 0.0
    
    confidence_score: float = 0.0
    confidence_weight: float = 0.0
    confidence_contribution: float = 0.0
    
    # Why matched / why not matched
    matched_terms: List[str] = field(default_factory=list)
    missing_terms: List[str] = field(default_factory=list)
    filter_reasons: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = ""
    confidence: float = 0.0
    staleness: float = 0.0
    quality_score: float = 0.0
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "topic": self.topic,
            "text_preview": self.text_preview[:200],
            "final_score": round(self.final_score, 4),
            "vector": {
                "score": round(self.vector_score, 4),
                "weight": round(self.vector_weight, 2),
                "contribution": round(self.vector_contribution, 4),
            },
            "keyword": {
                "score": round(self.keyword_score, 4),
                "weight": round(self.keyword_weight, 2),
                "contribution": round(self.keyword_contribution, 4),
                "matched_terms": self.matched_terms,
                "missing_terms": self.missing_terms,
            },
            "recency": {
                "score": round(self.recency_score, 4),
                "weight": round(self.recency_weight, 2),
                "contribution": round(self.recency_contribution, 4),
            },
            "confidence": {
                "score": round(self.confidence_score, 4),
                "weight": round(self.confidence_weight, 2),
                "contribution": round(self.confidence_contribution, 4),
            },
            "metadata": {
                "timestamp": self.timestamp,
                "confidence": round(self.confidence, 3),
                "staleness": round(self.staleness, 3),
                "quality_score": round(self.quality_score, 3),
                "verified": self.verified,
            },
            "filter_reasons": self.filter_reasons,
        }


@dataclass
class SearchExplanation:
    """Complete explanation of a search operation."""
    query: str
    timestamp: str
    plan: QueryPlan
    
    results_count: int
    results: List[ScoreExplanation]
    
    # Execution info
    execution_time_ms: float = 0.0
    topics_considered: int = 0
    chunks_considered: int = 0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "plan": self.plan.to_dict(),
            "results_count": self.results_count,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "topics_considered": self.topics_considered,
            "chunks_considered": self.chunks_considered,
            "warnings": self.warnings,
            "results": [r.to_dict() for r in self.results],
        }


class QueryExplainer:
    """Provides detailed explanations for query results."""
    
    def __init__(self):
        self._last_explanation: Optional[SearchExplanation] = None
    
    def explain_search(
        self,
        query: str,
        results: List[Any],
        vector_weights: Optional[Dict[str, float]] = None,
        execution_time_ms: float = 0.0,
    ) -> SearchExplanation:
        """Generate explanation for search results."""
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create score explanations
        explanations = []
        
        for result in results:
            if isinstance(result, tuple):
                score, topic, chunk = result
                text = chunk.get("text", "")
                chunk_key = chunk.get("chunk_id", "")
            else:
                score = result.get("score", 0.0)
                topic = result.get("topic", "")
                text = result.get("text", "")
                chunk_key = result.get("chunk_id", "")
            
            # Build explanation
            exp = ScoreExplanation(
                doc_id=chunk_key,
                topic=topic,
                text_preview=text,
                final_score=score,
                vector_score=score * (vector_weights.get("vector", 0.6) if vector_weights else 0.6),
                keyword_score=score * (vector_weights.get("keyword", 0.3) if vector_weights else 0.3),
                recency_score=score * (vector_weights.get("recency", 0.1) if vector_weights else 0.1),
                timestamp=chunk.get("timestamp", ""),
                confidence=chunk.get("confidence", 0.5),
                staleness=chunk.get("staleness", 0.0),
                quality_score=chunk.get("quality_score", 0.5),
                verified=chunk.get("verified", False),
            )
            
            explanations.append(exp)
        
        explanation = SearchExplanation(
            query=query,
            timestamp=timestamp,
            plan=QueryPlan(query_text=query, mode="hybrid", k=len(results)),
            results_count=len(results),
            results=explanations,
            execution_time_ms=execution_time_ms,
        )
        
        self._last_explanation = explanation
        return explanation
    
    def explain_score(
        self,
        chunk: Dict[str, Any],
        config: Optional[Dict[str, float]] = None,
    ) -> ScoreExplanation:
        """Explain a single chunk's score components."""
        
        config = config or {"vector": 0.6, "keyword": 0.3, "recency": 0.1, "confidence": 0.0}
        
        text = chunk.get("text", "")
        topic = chunk.get("topic", "")
        chunk_key = chunk.get("chunk_id", "")
        
        # Get scores
        vector = chunk.get("vector_score", 0.0)
        keyword = chunk.get("keyword_score", 0.0)
        recency = self._calc_recency(chunk.get("timestamp", ""))
        confidence = chunk.get("confidence", 0.5)
        
        # Calculate contributions
        vec_contrib = vector * config["vector"]
        kw_contrib = keyword * config["keyword"]
        rec_contrib = recency * config["recency"]
        conf_contrib = confidence * config["confidence"]
        
        final = vec_contrib + kw_contrib + rec_contrib + conf_contrib
        
        return ScoreExplanation(
            doc_id=chunk_key,
            topic=topic,
            text_preview=text,
            final_score=final,
            vector_score=vector,
            vector_weight=config["vector"],
            vector_contribution=vec_contrib,
            keyword_score=keyword,
            keyword_weight=config["keyword"],
            keyword_contribution=kw_contrib,
            recency_score=recency,
            recency_weight=config["recency"],
            recency_contribution=rec_contrib,
            confidence_score=confidence,
            confidence_weight=config["confidence"],
            confidence_contribution=conf_contrib,
            timestamp=chunk.get("timestamp", ""),
            confidence=confidence,
            staleness=chunk.get("staleness", 0.0),
            quality_score=chunk.get("quality_score", 0.5),
            verified=chunk.get("verified", False),
        )
    
    def _calc_recency(self, ts: str) -> float:
        """Calculate recency score."""
        import numpy as np
        from datetime import datetime, timezone, timedelta
        
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_days = (now - dt).total_seconds() / 86400
            return max(0.0, min(1.0, np.exp(-0.023 * age_days)))
        except:
            return 0.5
    
    def get_last_explanation(self) -> Optional[SearchExplanation]:
        """Get the last search explanation."""
        return self._last_explanation
    
    def format_explanation(self, explanation: SearchExplanation) -> str:
        """Format explanation as human-readable string."""
        lines = [
            f"Query: {explanation.query}",
            f"Timestamp: {explanation.timestamp}",
            f"Results: {explanation.results_count}",
            f"Execution time: {explanation.execution_time_ms:.2f}ms",
            "",
            "Results:",
        ]
        
        for i, result in enumerate(explanation.results, 1):
            lines.append(f"\n{i}. [{result.topic}] score={result.final_score:.4f}")
            lines.append(f"   {result.text_preview[:100]}...")
            lines.append(f"   vector={result.vector_contribution:.4f}, keyword={result.keyword_contribution:.4f}")
            if result.filter_reasons:
                lines.append(f"   filtered: {', '.join(result.filter_reasons)}")
        
        return "\n".join(lines)
