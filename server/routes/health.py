"""Health check routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def health_check():
    return {"status": "healthy", "service": "turbomemory"}


@router.get("/ready")
async def readiness():
    return {"ready": True}
