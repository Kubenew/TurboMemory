"""FastAPI server for TurboMemory."""

import os
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core import TurboMemory, TurboMemoryConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Application state holder."""
    tm: TurboMemory


class AddMemoryRequest(BaseModel):
    topic: str = Field(..., description="Topic for the memory")
    text: str = Field(..., description="Text content to store")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    bits: Optional[int] = Field(default=None, description="Quantization bits (4, 6, or 8)")
    source_ref: Optional[str] = Field(default=None, description="Source reference")
    ttl_days: Optional[float] = Field(default=None, description="Time to live in days")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    k: int = Field(default=5, ge=1, le=100, description="Number of results")
    top_topics: int = Field(default=5, ge=1, le=50, description="Number of topics to search")
    min_confidence: Optional[float] = Field(default=None, description="Minimum confidence filter")
    require_verification: bool = Field(default=False, description="Require verified results")


class BulkImportRequest(BaseModel):
    items: List[Dict[str, Any]] = Field(..., description="List of items to import")
    topic_field: str = Field(default="topic", description="Field name for topic")
    text_field: str = Field(default="text", description="Field name for text")


class ConfigUpdate(BaseModel):
    default_bits: Optional[int] = None
    default_ttl_days: Optional[float] = None
    enable_verifications: Optional[bool] = None
    enable_exclusions: Optional[bool] = None


def create_app(tm: TurboMemory) -> FastAPI:
    """Create and configure FastAPI application."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.tm = tm
        logger.info("TurboMemory server started")
        yield
        tm.close()
        logger.info("TurboMemory server stopped")

    app = FastAPI(
        title="TurboMemory API",
        description="Lightweight semantic storage engine with compressed embeddings",
        version="0.5.1",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "0.5.1"}

    @app.post("/memory")
    async def add_memory(
        request: AddMemoryRequest,
        background_tasks: BackgroundTasks,
    ):
        """Add a memory chunk."""
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
                raise HTTPException(status_code=400, detail="Memory was excluded by rules")
            return {"chunk_id": chunk_id, "topic": request.topic}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/memory/bulk")
    async def bulk_import(request: BulkImportRequest):
        """Bulk import memory items."""
        try:
            result = tm.bulk_import(
                items=request.items,
                topic_field=request.topic_field,
                text_field=request.text_field,
            )
            return result
        except Exception as e:
            logger.error(f"Error in bulk import: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/memory/{topic}")
    async def get_topic(topic: str):
        """Get all memories in a topic."""
        try:
            topic_data = tm.load_topic(topic)
            if not topic_data.get("chunks"):
                raise HTTPException(status_code=404, detail=f"Topic '{topic}' not found")
            return topic_data
        except Exception as e:
            logger.error(f"Error getting topic: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/memory/{topic}/{chunk_id}")
    async def delete_chunk(topic: str, chunk_id: str):
        """Delete a specific chunk."""
        try:
            topic_data = tm.load_topic(topic)
            original_count = len(topic_data.get("chunks", []))
            topic_data["chunks"] = [c for c in topic_data["chunks"] if c["chunk_id"] != chunk_id]
            if len(topic_data["chunks"]) == original_count:
                raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found in topic '{topic}'")
            tm.save_topic(topic_data)
            return {"status": "deleted", "topic": topic, "chunk_id": chunk_id}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting chunk: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query")
    async def query(request: QueryRequest):
        """Query memory."""
        try:
            results = tm.query(
                query_text=request.query,
                k=request.k,
                top_topics=request.top_topics,
                min_confidence=request.min_confidence,
                require_verification=request.require_verification,
            )
            return {
                "query": request.query,
                "results": [
                    {"score": score, "topic": topic, "chunk": chunk}
                    for score, topic, chunk in results
                ]
            }
        except Exception as e:
            logger.error(f"Error querying: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query/verify")
    async def query_with_verification(request: QueryRequest):
        """Query with verification."""
        try:
            results = tm.verify_and_score(
                query_text=request.query,
                k=request.k,
                top_topics=request.top_topics,
                min_confidence=request.min_confidence,
            )
            return {
                "query": request.query,
                "results": [
                    {
                        "score": score,
                        "topic": topic,
                        "chunk": chunk,
                        "verification": verif.to_dict(),
                    }
                    for score, topic, chunk, verif in results
                ]
            }
        except Exception as e:
            logger.error(f"Error in verified query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/topics")
    async def list_topics():
        """List all topics."""
        try:
            topics = []
            for fn in os.listdir(tm.topics_dir):
                if fn.endswith(".tmem"):
                    topic_name = fn[:-5]
                    topic_data = tm.load_topic(topic_name)
                    topics.append({
                        "name": topic_name,
                        "chunk_count": len(topic_data.get("chunks", [])),
                        "updated": topic_data.get("updated"),
                    })
            return {"topics": topics}
        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics."""
        try:
            return tm.get_metrics().to_dict()
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    async def get_stats():
        """Get quick stats."""
        try:
            return tm.stats()
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/audit")
    async def get_audit_logs(
        operation: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ):
        """Get audit logs."""
        try:
            logs = tm._audit.get_logs(operation=operation, limit=limit)
            return {"logs": logs}
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/export/topic/{topic}")
    async def export_topic(topic: str, include_embeddings: bool = False):
        """Export a topic."""
        try:
            return tm.export_topic(topic, include_embeddings=include_embeddings)
        except Exception as e:
            logger.error(f"Error exporting topic: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/export")
    async def export_all(include_embeddings: bool = False):
        """Export all topics."""
        try:
            return {"topics": tm.export_all(include_embeddings=include_embeddings)}
        except Exception as e:
            logger.error(f"Error exporting all: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/backup")
    async def create_backup(backup_path: str):
        """Create a backup."""
        try:
            path = tm.backup(backup_path)
            return {"backup_path": path}
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/restore")
    async def restore_backup(backup_path: str):
        """Restore from a backup."""
        try:
            tm.restore(backup_path)
            return {"status": "restored", "backup_path": backup_path}
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/consolidate")
    async def run_consolidation(topic: Optional[str] = None):
        """Run consolidation."""
        from ..consolidator import run_consolidation as run_consol
        try:
            result = run_consol(tm, topic=topic)
            return {"status": "completed", **result}
        except Exception as e:
            logger.error(f"Error running consolidation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/config")
    async def get_config():
        """Get current configuration."""
        return tm.config.__dict__

    @app.patch("/config")
    async def update_config(config: ConfigUpdate):
        """Update configuration."""
        try:
            for key, value in config.__dict__.items():
                if value is not None and hasattr(tm.config, key):
                    setattr(tm.config, key, value)
            tm.config.to_file(tm.config_path)
            return {"status": "updated", "config": tm.config.__dict__}
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    root: str = "turbomemory_data",
    model_name: str = "all-MiniLM-L6-v2",
    reload: bool = False,
):
    """Run the TurboMemory server."""
    import uvicorn
    
    config = TurboMemoryConfig(root=root, model_name=model_name)
    tm = TurboMemory(config=config)
    app = create_app(tm)
    
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()