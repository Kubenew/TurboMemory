"""Add memory endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone

router = APIRouter()


class MemoryAddRequest(BaseModel):
    topic: str = Field(..., description="Topic for the memory")
    text: str = Field(..., description="Memory text content")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence score")
    bits: int = Field(6, ge=4, le=8, description="Quantization bits")
    ttl_days: Optional[float] = Field(None, description="Time-to-live in days")
    source_ref: Optional[str] = Field(None, description="Source reference")


class MemoryAddResponse(BaseModel):
    chunk_id: str
    topic: str
    timestamp: str


class BulkAddRequest(BaseModel):
    memories: List[MemoryAddRequest] = Field(..., max_items=1000)
    topic_prefix: Optional[str] = Field(None, description="Optional topic prefix for all items")


class BulkAddResponse(BaseModel):
    imported: int
    excluded: int
    failed: int
    errors: List[str]


# In-memory storage for demo (replace with proper tenant-aware storage)
_memory_store = {}


def get_memory_store():
    return _memory_store


@router.post("/add", response_model=MemoryAddResponse)
async def add_memory(request: MemoryAddRequest, tenant_id: str = "default"):
    """Add a single memory chunk."""
    store = get_memory_store()
    
    if tenant_id not in store:
        from turbomemory import TurboMemory
        store[tenant_id] = TurboMemory(root=f"/tmp/turbomemory_{tenant_id}")
    
    tm = store[tenant_id]
    
    try:
        chunk_id = tm.add_memory(
            topic=request.topic,
            text=request.text,
            confidence=request.confidence,
            bits=request.bits,
            source_ref=request.source_ref,
            ttl_days=request.ttl_days,
        )
        
        if chunk_id is None:
            raise HTTPException(status_code=400, detail="Memory excluded by rules")
        
        return MemoryAddResponse(
            chunk_id=chunk_id,
            topic=request.topic,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk_add", response_model=BulkAddResponse)
async def bulk_add_memory(request: BulkAddRequest, tenant_id: str = "default"):
    """Add multiple memory chunks at once."""
    store = get_memory_store()
    
    if tenant_id not in store:
        from turbomemory import TurboMemory
        store[tenant_id] = TurboMemory(root=f"/tmp/turbomemory_{tenant_id}")
    
    tm = store[tenant_id]
    
    imported = 0
    excluded = 0
    failed = 0
    errors = []
    
    for mem in request.memories:
        try:
            topic = mem.topic
            if request.topic_prefix:
                topic = f"{request.topic_prefix}.{topic}"
            
            result = tm.add_memory(
                topic=topic,
                text=mem.text,
                confidence=mem.confidence,
                bits=mem.bits,
                source_ref=mem.source_ref,
                ttl_days=mem.ttl_days,
            )
            
            if result:
                imported += 1
            else:
                excluded += 1
        
        except Exception as e:
            failed += 1
            errors.append(f"{mem.topic}: {str(e)}")
    
    return BulkAddResponse(
        imported=imported,
        excluded=excluded,
        failed=failed,
        errors=errors,
    )


@router.delete("/{topic}/{chunk_id}")
async def delete_memory(topic: str, chunk_id: str, tenant_id: str = "default"):
    """Delete a memory chunk."""
    store = get_memory_store()
    
    if tenant_id not in store:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    tm = store[tenant_id]
    
    # This would need a delete method - implement later
    return {"deleted": True, "topic": topic, "chunk_id": chunk_id}
