"""Query endpoints."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    k: int = Field(5, ge=1, le=100, description="Number of results")
    top_topics: int = Field(5, ge=1, le=50, description="Number of topics to consider")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)
    require_verified: bool = Field(False, description="Only return verified results")
    enable_verification: bool = Field(False, description="Enable cross-reference verification")


class QueryResult(BaseModel):
    score: float
    topic: str
    chunk_id: str
    text: str
    confidence: float
    verified: bool


class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    execution_time_ms: float


_memory_store = {}


def get_memory_store():
    return _memory_store


@router.post("/query", response_model=QueryResponse)
async def query_memory(request: QueryRequest, tenant_id: str = "default"):
    """Query memory with semantic search."""
    import time
    
    store = get_memory_store()
    
    if tenant_id not in store:
        from turbomemory import TurboMemory
        store[tenant_id] = TurboMemory(root=f"/tmp/turbomemory_{tenant_id}")
    
    tm = store[tenant_id]
    
    start = time.perf_counter()
    
    try:
        if request.enable_verification:
            results = tm.verify_and_score(
                request.query,
                k=request.k,
                top_topics=request.top_topics,
                min_confidence=request.min_confidence,
            )
            
            query_results = []
            for score, topic, chunk, verif in results:
                if request.require_verified and not verif.verified:
                    continue
                
                query_results.append(QueryResult(
                    score=score,
                    topic=topic,
                    chunk_id=chunk.get("chunk_id", ""),
                    text=chunk.get("text", ""),
                    confidence=chunk.get("confidence", 0.0),
                    verified=verif.verified,
                ))
        else:
            results = tm.query(
                request.query,
                k=request.k,
                top_topics=request.top_topics,
                min_confidence=request.min_confidence,
            )
            
            query_results = []
            for score, topic, chunk in results:
                if request.require_verified and not chunk.get("verified", False):
                    continue
                
                query_results.append(QueryResult(
                    score=score,
                    topic=topic,
                    chunk_id=chunk.get("chunk_id", ""),
                    text=chunk.get("text", ""),
                    confidence=chunk.get("confidence", 0.0),
                    verified=chunk.get("verified", False),
                ))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return QueryResponse(
            query=request.query,
            results=query_results,
            execution_time_ms=elapsed,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid_query")
async def hybrid_query(
    query: str = Query(...),
    k: int = Query(5),
    min_confidence: float = Query(0.0),
    tenant_id: str = "default",
):
    """Hybrid search (vector + keyword)."""
    # Placeholder - implement with v0.6 hybrid search
    return {"query": query, "mode": "hybrid", "message": "Coming in v0.6"}
