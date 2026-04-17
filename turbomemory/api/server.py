"""FastAPI server for TurboMemory with multi-tenant support."""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from turbomemory import TurboMemory, TurboMemoryConfig
from turbomemory.formats import TMFFormat
from turbomemory.replication import create_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    root: str = "turbomemory_data"
    model_name: str = "all-MiniLM-L6-v2"
    max_memory_mb: int = 500
    enable_namespaces: bool = True
    default_namespace: str = "default"


class Namespaces:
    """Multi-tenant namespace manager."""
    
    def __init__(self, base_root: str):
        self.base_root = base_root
        self._instances: Dict[str, TurboMemory] = {}
    
    def get_tm(self, namespace: str = "default") -> TurboMemory:
        if namespace not in self._instances:
            ns_root = os.path.join(self.base_root, "namespaces", namespace)
            self._instances[namespace] = TurboMemory(root=ns_root)
        return self._instances[namespace]
    
    def close_all(self) -> None:
        for tm in self._instances.values():
            tm.close()
        self._instances.clear()


namespaces = None


def get_namespace(authorization: Optional[str] = Header(None)) -> str:
    """Extract namespace from Authorization header or use default."""
    if not authorization:
        return "default"
    # In production, decode JWT or API key to get namespace
    return authorization.split(":")[-1] if ":" in authorization else "default"


def get_tm(namespace: str = Depends(get_namespace)) -> TurboMemory:
    """Get TurboMemory instance for namespace."""
    return namespaces.get_tm(namespace)


class AddMemoryRequest(BaseModel):
    topic: str
    text: str
    confidence: float = 0.8
    bits: Optional[int] = None
    source_ref: Optional[str] = None
    ttl_days: Optional[float] = None


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    top_topics: int = 5
    min_confidence: Optional[float] = None


class BulkImportRequest(BaseModel):
    items: list[Dict[str, Any]]
    topic_field: str = "topic"
    text_field: str = "text"


class SyncRequest(BaseModel):
    remote_url: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global namespaces
    config = app.state.config
    namespaces = Namespaces(config.root)
    logger.info(f"TurboMemory server started on {config.host}:{config.port}")
    yield
    namespaces.close_all()
    logger.info("TurboMemory server stopped")


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    config = config or ServerConfig()
    
    app = FastAPI(
        title="TurboMemory API",
        description="SQLite for semantic data + zstd for embeddings",
        version="0.5.1",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.config = config

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "0.5.1", "namespaces_enabled": config.enable_namespaces}

    @app.post("/memory")
    async def add_memory(request: AddMemoryRequest, tm: TurboMemory = Depends(get_tm)):
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
            return {"chunk_id": chunk_id, "topic": request.topic}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/memory/bulk")
    async def bulk_import(request: BulkImportRequest, tm: TurboMemory = Depends(get_tm)):
        try:
            result = tm.bulk_import(request.items, request.topic_field, request.text_field)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/memory/{topic}")
    async def get_topic(topic: str, tm: TurboMemory = Depends(get_tm)):
        try:
            topic_data = tm.load_topic(topic)
            if not topic_data.get("chunks"):
                raise HTTPException(status_code=404, detail=f"Topic '{topic}' not found")
            return topic_data
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/memory/{topic}/{chunk_id}")
    async def delete_chunk(topic: str, chunk_id: str, tm: TurboMemory = Depends(get_tm)):
        try:
            topic_data = tm.load_topic(topic)
            original = len(topic_data.get("chunks", []))
            topic_data["chunks"] = [c for c in topic_data["chunks"] if c["chunk_id"] != chunk_id]
            if len(topic_data["chunks"]) == original:
                raise HTTPException(status_code=404, detail="Chunk not found")
            tm.save_topic(topic_data)
            return {"status": "deleted", "topic": topic, "chunk_id": chunk_id}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query")
    async def query(request: QueryRequest, tm: TurboMemory = Depends(get_tm)):
        try:
            results = tm.query(
                query_text=request.query,
                k=request.k,
                top_topics=request.top_topics,
                min_confidence=request.min_confidence,
            )
            return {"query": request.query, "results": [{"score": s, "topic": t, "chunk": c} for s, t, c in results]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/topics")
    async def list_topics(tm: TurboMemory = Depends(get_tm)):
        try:
            topics = []
            for fn in os.listdir(tm.topics_dir):
                if fn.endswith(".tmem"):
                    name = fn[:-5]
                    data = tm.load_topic(name)
                    topics.append({"name": name, "chunks": len(data.get("chunks", [])), "updated": data.get("updated")})
            return {"topics": topics}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics")
    async def get_metrics(tm: TurboMemory = Depends(get_tm)):
        return tm.get_metrics().to_dict()

    @app.get("/stats")
    async def get_stats(tm: TurboMemory = Depends(get_tm)):
        return tm.stats()

    @app.get("/export/topic/{topic}")
    async def export_topic(topic: str, include_embeddings: bool = False, tm: TurboMemory = Depends(get_tm)):
        return tm.export_topic(topic, include_embeddings)

    @app.get("/export")
    async def export_all(include_embeddings: bool = False, tm: TurboMemory = Depends(get_tm)):
        return {"topics": tm.export_all(include_embeddings)}

    @app.post("/backup")
    async def create_backup(backup_path: str, tm: TurboMemory = Depends(get_tm)):
        path = tm.backup(backup_path)
        return {"backup_path": path}

    @app.post("/restore")
    async def restore_backup(backup_path: str, tm: TurboMemory = Depends(get_tm)):
        tm.restore(backup_path)
        return {"status": "restored"}

    @app.post("/sync")
    async def sync(request: SyncRequest, tm: TurboMemory = Depends(get_tm)):
        sync = create_sync(tm.root, request.remote_url)
        result = sync.sync()
        return result

    @app.get("/namespaces")
    async def list_namespaces():
        return {"namespaces": list(namespaces._instances.keys())} if namespaces else {"namespaces": ["default"]}

    # Sync endpoints
    @app.get("/sync/events")
    async def get_sync_events(
        since: Optional[str] = None,
        tm: TurboMemory = Depends(get_tm),
    ):
        from turbomemory.sync import SyncNode
        node = SyncNode(tm.root)
        events = node.event_log.get_since(since) if since else node.event_log.get_all()
        return {"events": [e.to_dict() for e in events], "node_id": node.node_id}

    @app.post("/sync/events")
    async def post_sync_events(
        request: dict,
        tm: TurboMemory = Depends(get_tm),
    ):
        from turbomemory.sync import SyncNode, SyncEvent
        node = SyncNode(tm.root)
        imported = 0
        for evt_dict in request.get("events", []):
            evt = SyncEvent.from_dict(evt_dict)
            if not node.event_log.get(evt.event_id):
                node.event_log._events.append(evt)
                imported += 1
        node.event_log._save()
        return {"imported": imported}

    @app.get("/sync/status")
    async def get_sync_status(tm: TurboMemory = Depends(get_tm)):
        from turbomemory.sync import SyncNode
        node = SyncNode(tm.root)
        return node.get_status()

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
    
    config = ServerConfig(host=host, port=port, root=root, model_name=model_name)
    app = create_app(config)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        limit_concurrency=100,
        limit_max_requests=10000,
    )


if __name__ == "__main__":
    run_server()