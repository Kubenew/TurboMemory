"""Stats endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class StatsResponse(BaseModel):
    total_topics: int
    total_chunks: int
    avg_confidence: float
    avg_quality: float
    storage_bytes: int
    verified_chunks: int
    expired_chunks: int


_memory_store = {}


def get_memory_store():
    return _memory_store


@router.get("/stats", response_model=StatsResponse)
async def get_stats(tenant_id: str = "default"):
    """Get memory statistics."""
    store = get_memory_store()
    
    if tenant_id not in store:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    tm = store[tenant_id]
    
    try:
        stats = tm.stats()
        
        return StatsResponse(
            total_topics=stats.get("total_topics", 0),
            total_chunks=stats.get("total_chunks", 0),
            avg_confidence=stats.get("avg_confidence", 0.0),
            avg_quality=stats.get("avg_quality", 0.0),
            storage_bytes=stats.get("storage_bytes", 0),
            verified_chunks=stats.get("verified_chunks", 0),
            expired_chunks=stats.get("expired_chunks", 0),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(tenant_id: str = "default"):
    """Get detailed metrics."""
    store = get_memory_store()
    
    if tenant_id not in store:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    tm = store[tenant_id]
    
    try:
        metrics = tm.get_metrics()
        return metrics.to_dict()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
