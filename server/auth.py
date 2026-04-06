"""API Key authentication."""

from fastapi import Request, HTTPException
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
import os


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""
    
    def __init__(self, app, api_keys: list = None):
        super().__init__(app)
        self.api_keys = api_keys or self._load_keys()
    
    def _load_keys(self) -> list:
        """Load API keys from environment."""
        keys = os.environ.get("TURBOMEMORY_API_KEYS", "")
        if not keys:
            return []
        return [k.strip() for k in keys.split(",") if k.strip()]
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health endpoints and docs
        if request.url.path.startswith("/health") or request.url.path in ["/", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Check API key
        api_key = request.headers.get("X-API-Key")
        
        if self.api_keys and not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        if api_key and api_key not in self.api_keys:
            raise HTTPException(status_code=403, detail="Invalid API key")
        
        response = await call_next(request)
        return response


def verify_api_key(api_key: str, valid_keys: list) -> bool:
    """Verify an API key."""
    return api_key in valid_keys


def require_api_key(header: str = None) -> bool:
    """Decorator to require API key for a route."""
    # This is a placeholder - FastAPI handles this via dependencies
    pass
