"""TurboMemory FastAPI server."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .routes import add, query, stats, health
from .auth import APIKeyMiddleware
from .tenants import TenantManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_tenants: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TurboMemory server...")
    app.state.tenant_manager = TenantManager()
    yield
    logger.info("Shutting down TurboMemory server...")
    for tm in app.state.tenant_manager.list_memories():
        tm.close()


app = FastAPI(
    title="TurboMemory API",
    description="Lightweight semantic storage engine for compressed embedding archives",
    version="0.7.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(add.router, prefix="/api/v1", tags=["Add"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(stats.router, prefix="/api/v1", tags=["Stats"])


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "ValueError", "detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "NotFound", "detail": str(exc)})


@app.get("/")
async def root():
    return {"name": "TurboMemory API", "version": "0.7.0", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
