"""API routes."""

from .add import router as add_router
from .query import router as query_router
from .stats import router as stats_router
from .health import router as health_router

__all__ = ["add_router", "query_router", "stats_router", "health_router"]
